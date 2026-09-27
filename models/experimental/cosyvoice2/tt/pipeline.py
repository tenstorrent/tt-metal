# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The whole CosyVoice2 model, wired: text -> speech tokens -> mel -> 24 kHz waveform, non-streaming.

    TtQwen2LM                    text ids + prompt           -> speech tokens (25 Hz)   tt/llm/qwen2lm.py
    TtCausalMaskedDiffWithXvec   speech tokens + prompt       -> 80-bin mel (50 Hz)      tt/flow/flow.py
    TtHiFTGenerator              mel                          -> waveform (24 kHz)       tt/hifigan/generator.py

`synthesize` is upstream's `CosyVoice2.inference_zero_shot(..., stream=False)` segment for segment. It normalizes
and splits the text (tt/text.py), then runs `text_to_tokens` -> `tokens_to_mel` -> `mel_to_wav` per segment and
concatenates the audio. The prompt side (prompt transcript ids, prompt speech tokens, prompt mel, speaker
embedding) arrives precomputed in a `PromptContext` (tt/prompt.py).

**Configuration is explicit.** `CosyVoice2Config` holds every switch the pipeline sets, `reported()` is the
configuration behind published numbers, and every `Synthesis` carries the configuration that produced it. The
modules below still read a few environment variables at construction (`ENV_SWITCHES`); the pipeline refuses to
build while any of them is set, so the environment cannot change a pipeline run behind the caller's back.

**Traces.** The LLM decode trace (on in `reported()`) lives inside one `generate()` call, which releases it before
returning, so it is gone before the flow runs; a decode trace kept alive across the flow and the vocoder hung the
card (2026-09-21). The flow's CFM trace is off: it holds one slot keyed on the exact mel length, so distinct
utterances capture on every call, and a capturing solve costs about what an eager solve does (697.7 vs 675.3 ms,
2026-09-22). The flow encoder cannot be traced at all (a host round trip in its relative-position shift).
`live_traces()` names any trace still held; after `synthesize` returns it is empty.

**Context budget.** `ModelArgs.max_seq_len` (the LLM's KV cache and RoPE tables) is fixed at construction from the
config's budget: the longest prompt transcript, prompt speech and segment the pipeline accepts, plus upstream's
token limit of 20 speech tokens per text token. `synthesize` refuses an input over budget before running anything.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

import ttnn

from .prompt import PromptContext, RandomSources
from .text import MODEL_REPO_ID, TextFrontend

SAMPLE_RATE = 24000

# Environment variables the modules read at construction. Setting any of them would change a pipeline run without
# the config knowing, so `CosyVoice2TTNN` refuses to build while one is set; use the modules directly to
# experiment with them.
ENV_SWITCHES = (
    "COSYVOICE2_FLOW_SDPA",
    "COSYVOICE2_FLOW_FUSED_QKV",
    "COSYVOICE2_FLOW_MATMUL_CC",
    "COSYVOICE2_FLOW_CFM_TRACE",
    "COSYVOICE2_FLOW_ENCODER_TRACE",
    "COSYVOICE2_CFM_TRACE_CACHE",
    "COSYVOICE2_CFM_TRACE_CACHE_CAPACITY",
    "COSYVOICE2_ENCODER_TRACE_CACHE",
    "COSYVOICE2_CONV_CONFIG_IN_DRAM",
    "COSYVOICE2_DRAM_FREE_THRESHOLD_MB",
)

# What the pipeline does not switch but the published numbers depend on: the modules' defaults, stated here so
# that `CosyVoice2Config.describe()` reports them.
FIXED = {
    # ModelArgs' default, tt_transformers DecodersPrecision.accuracy: for Qwen2, attention weights and KV cache
    # bf16 at HiFi4; MLP weights bfp8
    "llm_decoder_precision": "tt_transformers DecodersPrecision.accuracy",
    "euler_steps": 10,  # tt/flow/flow.py N_TIMESTEPS, as upstream's flow.inference hardcodes it
    "flow_fused_sdpa": True,
    "flow_fused_qkv": True,
    "flow_matmul_compute_config": "ttnn default",
    "flow_encoder_trace": False,  # not traceable, see the module docstring
    "conv_config_tensors_in_dram": True,
    "frontend": "upstream text_normalize without ttsfrd/wetext (tt/text.py); prompt features from "
    "scripts/prepare_inputs.py",
}


@dataclass(frozen=True)
class CosyVoice2Config:
    """Every switch `CosyVoice2TTNN` sets. The defaults are the configuration behind published numbers."""

    # dtypes. The HiFT decoder runs fp32: with real weights, bf16 collapses conv_post's output before its exp
    # (PCC ~0.49). The F0 predictor and NSF source default to fp32: bf16 roughly triples the F0 error (mean
    # |df| 0.70 vs 0.23 Hz over 931 voiced frames of real speech) and adds voiced/unvoiced flips.
    llm_dtype: str = "bfloat16"
    flow_dtype: str = "bfloat16"
    hift_decoder_dtype: str = "float32"
    hift_source_dtype: str = "float32"
    # traces (see the module docstring)
    llm_decode_trace: bool = True
    cfm_trace: bool = False
    # LLM sampling: upstream cosyvoice2.yaml (ras_sampling) and Qwen2LM.inference's token/text ratios
    sampler: str = "ras"  # "ras" | "greedy"
    top_p: float = 0.8
    top_k: int = 25
    win_size: int = 10
    tau_r: float = 0.1
    min_token_text_ratio: float = 2.0
    max_token_text_ratio: float = 20.0
    # context budget, fixed at construction (see the module docstring)
    max_prompt_text_tokens: int = 128
    max_prompt_speech_tokens: int = 750  # upstream refuses prompt audio over 30 s: 750 tokens at 25 Hz
    max_segment_text_tokens: int = 100  # split_paragraph aims at <= 80 tokens; one long sentence can exceed it

    @classmethod
    def reported(cls) -> "CosyVoice2Config":
        return cls()

    @classmethod
    def eager(cls) -> "CosyVoice2Config":
        """No traces at all: the baseline the traced configuration is compared against."""
        return cls(llm_decode_trace=False, cfm_trace=False)

    def max_seq_len(self) -> int:
        from .llm.qwen2lm import TtQwen2LM, required_max_seq_len

        prefix = TtQwen2LM.prefix_len(
            torch.zeros(1, self.max_prompt_text_tokens + self.max_segment_text_tokens),
            torch.zeros(1, self.max_prompt_speech_tokens),
        )
        return required_max_seq_len(prefix, self.max_tokens_for(self.max_segment_text_tokens))

    def min_tokens_for(self, n_text: int) -> int:
        return int(n_text * self.min_token_text_ratio)

    def max_tokens_for(self, n_text: int) -> int:
        return int(n_text * self.max_token_text_ratio)

    def describe(self) -> dict:
        return {**asdict(self), "max_seq_len": self.max_seq_len(), **FIXED}


@dataclass
class SegmentResult:
    """One segment's output and device-synchronized stage times (seconds)."""

    text: str
    text_ids: list[int]
    tokens: list[int]
    mel_frames: int
    audio_samples: int
    timings: dict[str, float]  # llm_prefill, llm_decode, flow_encoder, flow_cfm, hift, total


@dataclass
class Synthesis:
    """What `synthesize` returns. `wall_s` spans the whole call, text normalization included."""

    audio: np.ndarray  # float32, SAMPLE_RATE, all segments concatenated
    segments: list[SegmentResult]
    wall_s: float
    config: dict
    sample_rate: int = SAMPLE_RATE
    notes: list[str] = field(default_factory=list)

    @property
    def audio_s(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.wall_s / self.audio_s if self.audio_s > 0 else float("inf")

    @property
    def tokens(self) -> list[int]:
        return [t for s in self.segments for t in s.tokens]

    def stage_totals(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.segments:
            for k, v in s.timings.items():
                out[k] = out.get(k, 0.0) + v
        return out


def device_memory(device) -> dict[str, int]:
    """Bytes allocated per bank, by buffer type."""
    kinds = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1, "l1_small": ttnn.BufferType.L1_SMALL}
    return {k: int(ttnn.get_memory_view(device, v).total_bytes_allocated_per_bank) for k, v in kinds.items()}


def _refuse_env_switches() -> None:
    set_now = {k: os.environ[k] for k in ENV_SWITCHES if k in os.environ}
    if set_now:
        raise RuntimeError(
            f"CosyVoice2TTNN takes its configuration from CosyVoice2Config only, but {set_now} is set in the "
            "environment; unset it, or drive the modules directly to experiment with it"
        )


def _qwen2_backbone_dir(llm_state_dict: dict, llm_pt_path: str, cache_dir: str) -> str:
    """tt_transformers' `ModelArgs` loads a Qwen2 backbone from a Hugging Face-format directory. llm.pt's `llm.*`
    keys are that backbone's state dict (tt/checkpoint.py), so write it out once and reuse it while llm.pt is the
    same file."""
    from .checkpoint import build_local_qwen2_checkpoint_dir

    out = os.path.join(cache_dir, "qwen2_backbone")
    marker = os.path.join(out, "source.json")
    source = {"llm_pt": os.path.realpath(llm_pt_path), "size": os.path.getsize(llm_pt_path)}
    try:
        with open(marker) as fh:
            if json.load(fh) == source and os.path.exists(os.path.join(out, "model.safetensors")):
                return out
    except (OSError, ValueError):
        pass
    build_local_qwen2_checkpoint_dir(llm_state_dict, out)
    with open(marker, "w") as fh:
        json.dump(source, fh)
    return out


class _StageClock:
    """Device-synchronized wall time, accumulated per stage name while `active`."""

    def __init__(self, device):
        self.device = device
        self.totals: dict[str, float] = {}

    def wrap(self, name: str, fn):
        def timed(*args, **kwargs):
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                ttnn.synchronize_device(self.device)
                self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

        return timed

    def take(self) -> dict[str, float]:
        out, self.totals = self.totals, {}
        return out


class CosyVoice2TTNN:
    """The three stages behind one call. The device must be open with `l1_small_size` >= 65536 and, for the LLM
    decode trace, a `trace_region_size` (50 MB is enough)."""

    def __init__(self, device, config: CosyVoice2Config | None = None, *, cache_dir: str | None = None):
        from huggingface_hub import hf_hub_download

        from models.tt_transformers.tt.model_config import ModelArgs

        from .checkpoint import sub_state_dict
        from .flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
        from .hifigan.f0_predictor import TorchConvRNNF0PredictorRef
        from .hifigan.generator import (
            TorchHiFTDecodeRef,
            TorchHiFTGeneratorInferenceRef,
            TtHiFTDecoder,
            TtHiFTGenerator,
        )
        from .llm.qwen2lm import TtQwen2LM

        _refuse_env_switches()
        self.device = device
        self.config = config or CosyVoice2Config.reported()
        cfg = self.config
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "cosyvoice2_ttnn")

        def load(name):
            path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=name)
            return path, torch.load(path, map_location="cpu")

        # LLM: the Qwen2 backbone through tt_transformers (HF_MODEL names the directory ModelArgs loads), plus
        # CosyVoice2's own speech embedding, sos/task embedding and output head from llm.pt.
        llm_path, llm_sd = load("llm.pt")
        os.environ["HF_MODEL"] = _qwen2_backbone_dir(llm_sd, llm_path, cache_dir)
        args = ModelArgs(device, max_batch_size=1, max_seq_len=cfg.max_seq_len(), dummy_weights=False, use_hf_rope=True)
        self.llm = TtQwen2LM(
            args,
            device,
            args.load_state_dict(),
            dtype=getattr(ttnn, cfg.llm_dtype),
            cosyvoice_state_dict=llm_sd,
            use_decode_trace=cfg.llm_decode_trace,
        )
        del llm_sd

        _, flow_sd = load("flow.pt")
        flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
        flow_ref.eval()
        self.flow = TtCausalMaskedDiffWithXvec(device, flow_ref, dtype=getattr(ttnn, cfg.flow_dtype))
        self.flow.decoder.use_trace = cfg.cfm_trace
        self.flow.encoder.use_trace = False
        del flow_sd, flow_ref

        _, hift_sd = load("hift.pt")
        decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
        hift_ref = TorchHiFTGeneratorInferenceRef(
            decode_ref,
            TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor.")),
            hift_sd["m_source.l_linear.weight"],
            hift_sd["m_source.l_linear.bias"],
        )
        self.hift = TtHiFTGenerator(
            device,
            hift_ref,
            TtHiFTDecoder(device, decode_ref, dtype=getattr(ttnn, cfg.hift_decoder_dtype)),
            dtype=getattr(ttnn, cfg.hift_source_dtype),
        )
        self.harmonics = hift_ref.harmonic_num + 1
        del hift_sd

        self.text = TextFrontend()

        # Stage clocks. `generate()` and `flow.inference` call these through their instances, so wrapping the
        # instance attributes times the sub-stages without changing either module.
        self._clock = _StageClock(device)
        self.llm.prefill = self._clock.wrap("llm_prefill", self.llm.prefill)
        self.flow.decoder.forward = self._clock.wrap("flow_cfm", self.flow.decoder.forward)

    # ------------------------------------------------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------------------------------------------------
    def text_to_tokens(self, ctx: PromptContext, text_ids: list[int], *, seed: int | None = None) -> list[int]:
        """Stage 1, the LLM: upstream's `Qwen2LM.inference`. The prompt transcript precedes the segment's ids, the
        prompt speech tokens follow the task token, and min/max length are 2x / 20x the segment's text tokens."""
        cfg = self.config
        ids = torch.tensor([text_ids], dtype=torch.long)
        if ctx.prompt_text_ids is not None:
            ids = torch.cat([ctx.prompt_text_ids.long(), ids], dim=1)
        speech = ctx.llm_prompt_speech_tokens.long() if ctx.llm_prompt_speech_tokens is not None else None
        sampling = {}
        if cfg.sampler == "ras":
            sampling = dict(top_p=cfg.top_p, top_k=cfg.top_k, win_size=cfg.win_size, tau_r=cfg.tau_r)
        return self.llm.generate(
            ids,
            speech,
            max_tokens=cfg.max_tokens_for(len(text_ids)),
            min_tokens=cfg.min_tokens_for(len(text_ids)),
            sampler=cfg.sampler,
            seed=seed,
            **sampling,
        )

    def tokens_to_mel(self, tokens: list[int], ctx: PromptContext) -> torch.Tensor:
        """Stage 2, the flow: upstream's `CausalMaskedDiffWithXvec.inference(streaming=False, finalize=True)`.
        Returns the host mel of the generated tokens only, `[1, 2 x len(tokens), 80]`."""
        token = torch.tensor([tokens], dtype=torch.int32)
        return self.flow.inference(token, ctx.flow_prompt_speech_tokens, ctx.prompt_feat, ctx.embedding)

    def mel_to_wav(self, mel: torch.Tensor, rng: RandomSources) -> torch.Tensor:
        """Stage 3, HiFT: F0 predictor, NSF source (SineGen2's noise from `rng`), decoder with the iSTFT head.
        Returns the host waveform, `mel frames x 480` samples."""
        mel_frames = int(mel.shape[1])
        audio_len = mel_frames * self.hift.upsample_scale
        mel_dev = ttnn.from_torch(
            mel, dtype=getattr(ttnn, self.config.hift_source_dtype), layout=ttnn.TILE_LAYOUT, device=self.device
        )
        wav_dev = self.hift.inference(mel_dev, mel_frames, 1, sine_noise=rng.sine_noise_for(audio_len, self.harmonics))
        wav = ttnn.to_torch(wav_dev).float().reshape(-1)[:audio_len]
        ttnn.deallocate(wav_dev)
        ttnn.deallocate(mel_dev)
        return wav

    # ------------------------------------------------------------------------------------------------------------
    def synthesize(self, ctx: PromptContext, text: str, *, rng: RandomSources | None = None) -> Synthesis:
        """`text` in the prompt's voice. `rng.llm_seed` seeds the LLM's sampling once, before the first segment,
        as upstream's callers seed once per call; later segments continue the same random stream."""
        if ctx.mode != "zero_shot":
            raise NotImplementedError(f"mode {ctx.mode!r}: only zero_shot is wired end to end so far")
        rng = rng or RandomSources()
        self._check_prompt_budget(ctx)
        t_call = time.perf_counter()
        segments = self.text.normalize(text, split=True)
        seg_ids = [self.text.encode(s) for s in segments]
        for s, ids in zip(segments, seg_ids):
            if len(ids) > self.config.max_segment_text_tokens:
                raise ValueError(
                    f"segment of {len(ids)} text tokens exceeds max_segment_text_tokens="
                    f"{self.config.max_segment_text_tokens}: {s[:80]!r}"
                )

        results, audio, notes = [], [], []
        for i, (seg, ids) in enumerate(zip(segments, seg_ids)):
            self._clock.take()
            t0 = time.perf_counter()
            llm = self._clock.wrap("llm", self.text_to_tokens)
            tokens = llm(ctx, ids, seed=rng.llm_seed if i == 0 else None)
            if tokens:
                mel = self._clock.wrap("flow", self.tokens_to_mel)(tokens, ctx)
                wav = self._clock.wrap("hift", self.mel_to_wav)(mel, rng)
                mel_frames = int(mel.shape[1])
            else:
                notes.append(f"segment {i}: the LLM stopped before any speech token; no audio")
                wav, mel_frames = torch.zeros(0), 0
            t = self._clock.take()
            timings = {
                "llm_prefill": t.get("llm_prefill", 0.0),
                "llm_decode": t["llm"] - t.get("llm_prefill", 0.0),
                "flow_encoder": t.get("flow", 0.0) - t.get("flow_cfm", 0.0),
                "flow_cfm": t.get("flow_cfm", 0.0),
                "hift": t.get("hift", 0.0),
                "total": time.perf_counter() - t0,
            }
            results.append(SegmentResult(seg, ids, tokens, mel_frames, int(wav.numel()), timings))
            audio.append(wav.numpy().astype(np.float32))
        return Synthesis(
            audio=np.concatenate(audio) if audio else np.zeros(0, np.float32),
            segments=results,
            wall_s=time.perf_counter() - t_call,
            config=self.config.describe(),
            notes=notes,
        )

    def warmup(self, ctx: PromptContext, text: str = "Hello there.") -> Synthesis:
        """One throwaway call: compiles kernels and builds the device-side state every call reuses, so the calls
        that follow are not first calls. It cannot pre-resolve the flow's and vocoder's geometries for later
        utterances -- non-streaming lengths are exact, and a new length meets the conv resolver's first-sight
        verification whenever it comes. Returns the warm-up's own result, the process's cold figure."""
        return self.synthesize(ctx, text, rng=RandomSources(llm_seed=0))

    def live_traces(self) -> list[str]:
        live = []
        if self.llm._trace_id is not None:
            live.append("llm_decode")
        if self.flow.decoder._traces:
            live.append("flow_cfm")
        if self.flow.encoder._trace_id is not None:
            live.append("flow_encoder")
        return live

    def release(self) -> None:
        """Free every trace (and its persistent buffers). Safe to call at any time."""
        self.llm.release_decode_trace()
        self.flow.release_traces()

    def _check_prompt_budget(self, ctx: PromptContext) -> None:
        cfg = self.config
        n_text = 0 if ctx.prompt_text_ids is None else int(ctx.prompt_text_ids.shape[1])
        n_speech = 0 if ctx.llm_prompt_speech_tokens is None else int(ctx.llm_prompt_speech_tokens.shape[1])
        if n_text > cfg.max_prompt_text_tokens or n_speech > cfg.max_prompt_speech_tokens:
            raise ValueError(
                f"prompt of {n_text} text / {n_speech} speech tokens exceeds the budget "
                f"({cfg.max_prompt_text_tokens} / {cfg.max_prompt_speech_tokens}); see CosyVoice2Config"
            )
