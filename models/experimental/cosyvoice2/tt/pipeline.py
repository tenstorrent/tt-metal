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
config's budget: the longest prompt transcript, prompt speech and segment the pipeline accepts, plus the speech-token
limit per segment. The limit is upstream's 20 speech tokens per text token, capped at the HiFT single-pass limit
(`CosyVoice2Config.max_segment_speech_tokens`). `synthesize` refuses an input over budget before running anything.
A segment whose speech runs past the cap raises `SegmentTooLong`, naming its length, instead of being truncated;
so do `tokens_to_mel` and `mel_to_wav` given more than the cap.

**Bucketing (on in `reported()`).** Non-streaming lengths are otherwise exact, so every distinct utterance would meet
new device geometries. Each new geometry means JIT kernel compiles and the conv resolver's first-sight checks:
minutes per utterance on a cold kernel cache (docs/VALIDATION.md). Instead:
- the flow runs at the smallest token bucket strictly above prompt + generated tokens, with the padding masked;
- HiFT runs at the smallest frame bucket at or above its mel length, padded with silence and trimmed;
- the LLM already pads its prefill to multiples of 128;
- `warmup_buckets()` runs every one of these geometries once, at start-up, in a fixed order.

The per-geometry conv caches never evict in this mode: the set is finite and warmed, and an eviction would make
the next request re-prepare and re-verify weights mid-request.

**Fragility, stated once.** The conv and halo kernels take their config tensors' DRAM addresses as compile-time
arguments, so a process reuses the disk kernel cache only if it allocates exactly as the process that compiled it.
`warmup_buckets()` running first, in a fixed order, is what makes that so. Any change to the code, the
configuration, the checkpoint or the warm-up sequence shifts those addresses. That costs one full recompile of every
bucket, after which the cache is warm again.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

import ttnn

from .prompt import PromptContext, RandomSources
from .text import MODEL_REPO_ID, TextFrontend

SAMPLE_RATE = 24000
TOKEN_RATE_HZ = 25
TOKEN_MEL_RATIO = 2
# HiFT's bucket padding: CosyVoice2's mel is log(clamp(x, 1e-5)), so this is digital silence. The quietest frames of
# real LibriSpeech prompt mels sit exactly here.
MEL_SILENCE = math.log(1e-5)

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
    # bf16 at HiFi4; MLP weights bfp8 at HiFi2 with fp16 accumulation
    "llm_decoder_precision": "tt_transformers DecodersPrecision.accuracy",
    "euler_steps": 10,  # tt/flow/flow.py N_TIMESTEPS, as upstream's flow.inference hardcodes it
    "flow_fused_sdpa": True,
    "flow_fused_qkv": True,
    "flow_matmul_compute_config": "ttnn default",
    "flow_encoder_trace": False,  # not traceable, see the module docstring
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
    max_segment_text_tokens: int = 80  # upstream split_paragraph's token_max_n; the bucket sets derive from it
    # Upstream allows 20 speech tokens per text token (1,600 for an 80-token segment, 64 s). Single-pass HiFT on
    # an N150 is capped at 1,024 tokens (2,048 mel frames, 41 s). Past ~2,900 frames a ttnn.concat row exceeds one
    # core's L1, and with every bucket warmed and resident, a 2,560-frame bucket found no contiguous DRAM
    # (docs/VALIDATION.md). The cap binds only when a segment of more than 51 text tokens has not ended after 41 s;
    # the pipeline then raises `SegmentTooLong` rather than truncate the speech.
    max_segment_speech_tokens: int = 1024
    # geometry: bucketing runs the flow and HiFT at a finite set of lengths, all warmed at start-up (`warmup_buckets`)
    bucketing: bool = True
    conv_config_tensors_in_dram: bool = True

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
        return required_max_seq_len(prefix, self.llm_steps_for(self.max_segment_text_tokens))

    def min_tokens_for(self, n_text: int) -> int:
        return int(n_text * self.min_token_text_ratio)

    def max_tokens_for(self, n_text: int) -> int:
        """The most speech tokens a segment of `n_text` text tokens can yield."""
        return min(int(n_text * self.max_token_text_ratio), self.max_segment_speech_tokens)

    def llm_steps_for(self, n_text: int) -> int:
        """The LLM's step limit: upstream's 20 per text token, or, where the cap binds, one step past it. The extra
        step tells a segment of exactly `max_segment_speech_tokens` from a longer one (`SegmentTooLong`)."""
        return min(int(n_text * self.max_token_text_ratio), self.max_segment_speech_tokens + 1)

    def flow_token_buckets(self) -> list[int]:
        """Flow lengths (prompt + generated speech tokens) up to the budget's maximum: the longest prompt plus the
        longest segment's token limit. Every length maps to a bucket STRICTLY above it (`bucket_for`)."""
        return tiered_buckets(self.max_prompt_speech_tokens + self.max_tokens_for(self.max_segment_text_tokens), 64)

    def hift_frame_buckets(self) -> list[int]:
        """HiFT lengths in mel frames (2 per generated token) up to the longest segment's token limit, inclusive:
        HiFT has one code path, so an exact fit needs no padding (`bucket_at_least`)."""
        return tiered_buckets(TOKEN_MEL_RATIO * self.max_tokens_for(self.max_segment_text_tokens), 128, strict=False)

    def llm_prefill_lengths(self) -> list[int]:
        """The LLM pads its prefill to multiples of 128 already; these are every length the budget allows."""
        from .llm.qwen2lm import PREFILL_SEQ_MULTIPLE

        longest = 1 + self.max_prompt_text_tokens + self.max_segment_text_tokens + 1 + self.max_prompt_speech_tokens
        return list(
            range(
                PREFILL_SEQ_MULTIPLE,
                -(-longest // PREFILL_SEQ_MULTIPLE) * PREFILL_SEQ_MULTIPLE + 1,
                PREFILL_SEQ_MULTIPLE,
            )
        )

    def describe(self) -> dict:
        out = {**asdict(self), "max_seq_len": self.max_seq_len(), **FIXED}
        if self.bucketing:
            out.update(flow_token_buckets=self.flow_token_buckets(), hift_frame_buckets=self.hift_frame_buckets())
        return out


def tiered_buckets(cap: int, unit: int, strict: bool = True) -> list[int]:
    """`unit, 2 unit, ... 8 unit`, then steps of 2 `unit` to 16 `unit`, then 4 `unit`, and so on, up to the first
    bucket above `cap` (strictly, or at least `cap` with `strict=False`). Past the first 8 steps a length is padded
    by at most a quarter of itself (about an eighth on average); the number of buckets grows with the log of `cap`."""
    out, step, b = [], unit, unit
    while b < cap or (strict and b == cap):
        out.append(b)
        if b >= 8 * step:
            step *= 2
        b += step
    out.append(b)
    return out


def bucket_at_least(length: int, buckets: list[int]) -> int:
    """The smallest bucket >= `length` (HiFT: padding is only data there, so an exact fit is the same path)."""
    for b in buckets:
        if b >= length:
            return b
    raise ValueError(f"length {length} exceeds the largest bucket {buckets[-1]}; see CosyVoice2Config's budget")


def bucket_for(length: int, buckets: list[int]) -> int:
    """The smallest bucket STRICTLY above `length`. Strict, so an utterance always has at least one padded position
    and always takes the masked (bucketed) code path that `warmup_buckets` warmed; an exact fit would take the
    unmasked path, whose programs are different."""
    for b in buckets:
        if b > length:
            return b
    raise ValueError(f"length {length} exceeds the largest bucket {buckets[-1]}; see CosyVoice2Config's budget")


class SegmentTooLong(ValueError):
    """A segment's speech runs past `CosyVoice2Config.max_segment_speech_tokens`, the single-pass HiFT limit."""

    @classmethod
    def past_cap(cls, what: str, cfg: CosyVoice2Config) -> "SegmentTooLong":
        cap = cfg.max_segment_speech_tokens
        return cls(
            f"{what}: past max_segment_speech_tokens={cap} ({cap / TOKEN_RATE_HZ:.1f} s, {TOKEN_MEL_RATIO * cap} mel "
            "frames), the most single-pass HiFT runs; split the text into shorter segments"
        )


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


def _find(root, kind, _seen=None) -> list:
    """Every `kind` instance reachable through `root`'s attributes, lists and dicts (the modules own their caches)."""
    seen = set() if _seen is None else _seen
    if id(root) in seen:
        return []
    seen.add(id(root))
    if isinstance(root, kind):
        return [root]
    if isinstance(root, (list, tuple)):
        children = root
    elif isinstance(root, dict):
        children = root.values()
    elif hasattr(root, "__dict__") and type(root).__module__.startswith("models.experimental.cosyvoice2"):
        children = vars(root).values()
    else:
        return []
    return [x for c in children for x in _find(c, kind, seen)]


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
        from .geometry_cache import threshold_override
        from .hifigan.conv import config_tensors_in_dram_override
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

        # The conv modules read two switches at construction; the pipeline sets both from its config. In bucketed
        # mode the per-geometry conv caches never evict: the geometry set is finite and warmed at start-up, and an
        # eviction would make the next request at that geometry re-prepare and re-verify its weights.
        with (
            config_tensors_in_dram_override(cfg.conv_config_tensors_in_dram),
            threshold_override(0 if cfg.bucketing else None),
        ):
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
        prompt speech tokens follow the task token, and min/max length are 2x / 20x the segment's text tokens.
        Raises `SegmentTooLong` if the speech has not ended by `max_segment_speech_tokens`."""
        cfg = self.config
        ids = torch.tensor([text_ids], dtype=torch.long)
        if ctx.prompt_text_ids is not None:
            ids = torch.cat([ctx.prompt_text_ids.long(), ids], dim=1)
        speech = ctx.llm_prompt_speech_tokens.long() if ctx.llm_prompt_speech_tokens is not None else None
        sampling = {}
        if cfg.sampler == "ras":
            sampling = dict(top_p=cfg.top_p, top_k=cfg.top_k, win_size=cfg.win_size, tau_r=cfg.tau_r)
        tokens = self.llm.generate(
            ids,
            speech,
            max_tokens=cfg.llm_steps_for(len(text_ids)),
            min_tokens=cfg.min_tokens_for(len(text_ids)),
            sampler=cfg.sampler,
            seed=seed,
            **sampling,
        )
        if len(tokens) > cfg.max_segment_speech_tokens:
            raise SegmentTooLong.past_cap(
                f"a segment of {len(text_ids)} text tokens had not ended after {len(tokens)} speech tokens", cfg
            )
        return tokens

    def tokens_to_mel(self, tokens: list[int], ctx: PromptContext) -> torch.Tensor:
        """Stage 2, the flow: upstream's `CausalMaskedDiffWithXvec.inference(streaming=False, finalize=True)`.
        Returns the host mel of the generated tokens only, `[1, 2 x len(tokens), 80]`. Bucketed, it runs at the
        smallest flow bucket above prompt + generated tokens, with the padding masked (tt/flow/flow.py)."""
        if len(tokens) > self.config.max_segment_speech_tokens:
            raise SegmentTooLong.past_cap(f"a segment of {len(tokens)} speech tokens", self.config)
        token = torch.tensor([tokens], dtype=torch.int32)
        bucket = None
        if self.config.bucketing:
            bucket = bucket_for(ctx.n_prompt_tokens + len(tokens), self.config.flow_token_buckets())
        return self.flow.inference(token, ctx.flow_prompt_speech_tokens, ctx.prompt_feat, ctx.embedding, bucket)

    def mel_to_wav(self, mel: torch.Tensor, rng: RandomSources) -> torch.Tensor:
        """Stage 3, HiFT: F0 predictor, NSF source (SineGen2's noise from `rng`), decoder with the iSTFT head.
        Returns the host waveform, `mel frames x 480` samples.

        Bucketed, the mel is padded to the smallest HiFT bucket above its length with `MEL_SILENCE` (silence, not
        zeros: 0 is a loud mel), the source noise with zeros, and the audio is trimmed back. HiFT's convs look
        ahead, so the padding reaches the last few tens of milliseconds; docs/VALIDATION.md has the measured tail
        error."""
        mel_frames = int(mel.shape[1])
        if mel_frames > TOKEN_MEL_RATIO * self.config.max_segment_speech_tokens:
            raise SegmentTooLong.past_cap(f"a mel of {mel_frames} frames", self.config)
        audio_len = mel_frames * self.hift.upsample_scale
        noise = rng.sine_noise_for(audio_len, self.harmonics)
        run_frames = mel_frames
        if self.config.bucketing:
            run_frames = bucket_at_least(mel_frames, self.config.hift_frame_buckets())
            mel = torch.cat([mel, torch.full((1, run_frames - mel_frames, mel.shape[2]), MEL_SILENCE)], dim=1)
            noise = torch.cat(
                [noise, torch.zeros(1, (run_frames - mel_frames) * self.hift.upsample_scale, self.harmonics)], 1
            )
        mel_dev = ttnn.from_torch(
            mel, dtype=getattr(ttnn, self.config.hift_source_dtype), layout=ttnn.TILE_LAYOUT, device=self.device
        )
        wav_dev = self.hift.inference(mel_dev, run_frames, 1, sine_noise=noise)
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

    def warmup_buckets(self, on_geometry=None) -> dict[str, float]:
        """Run every geometry a request can meet, once, in a fixed order, and return the seconds each took:
        - the LLM's prefill lengths and prefix lookups, then one decode;
        - every flow bucket, with one padded position, so it is the masked path every request takes;
        - every HiFT bucket.

        Call it right after construction, before anything else touches the device. Its order and its dummy inputs
        never change, so each process allocates exactly as the previous one did. The conv kernels, whose
        compile-time arguments carry DRAM addresses (docs/VALIDATION.md), then load from the disk cache instead of
        compiling. Any change to the code or to this sequence costs one full recompile, and then the cache is warm
        again. `on_geometry(name)`, if given, is called after each geometry (measurement hooks)."""
        cfg, clock = self.config, {}

        def timed(name, fn):
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(self.device)
            clock[name] = time.perf_counter() - t0
            if on_geometry is not None:
                on_geometry(name)

        dim = self.llm.args.dim
        for n in cfg.llm_prefill_lengths():
            timed(f"llm_prefill_{n}", lambda n=n: self.llm.prefill(torch.zeros(1, n, dim)))
            ids = torch.zeros(1, n, dtype=torch.long)
            timed(
                f"llm_lookup_{n}",
                lambda ids=ids: (self.llm.embed_text_tokens_host(ids), self.llm.embed_speech_tokens_host(ids)),
            )
        timed(
            "llm_decode",
            lambda: self.llm.generate(torch.zeros(1, 4, dtype=torch.long), None, max_tokens=4, sampler="greedy"),
        )

        no_prompt, no_feat, emb = torch.zeros(1, 0, dtype=torch.int32), torch.zeros(1, 0, 80), torch.ones(1, 192)
        for b in cfg.flow_token_buckets():
            tokens = torch.zeros(1, b - 1, dtype=torch.int32)
            timed(f"flow_{b}", lambda b=b, t=tokens: self.flow.inference(t, no_prompt, no_feat, emb, bucket_tokens=b))
        for m in cfg.hift_frame_buckets():
            timed(f"hift_{m}", lambda m=m: self._hift_at(m))
        return clock

    def _hift_at(self, frames: int) -> None:
        """HiFT once at exactly `frames` (a bucket), on a silent mel."""
        mel = ttnn.from_torch(
            torch.full((1, frames, 80), MEL_SILENCE),
            dtype=getattr(ttnn, self.config.hift_source_dtype),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        noise = torch.zeros(1, frames * self.hift.upsample_scale, self.harmonics)
        ttnn.deallocate(self.hift.inference(mel, frames, 1, sine_noise=noise))
        ttnn.deallocate(mel)

    def conv_cache_evictions(self) -> int:
        """Entries the per-geometry conv caches have evicted for DRAM pressure. Always 0 in bucketed mode."""
        from .geometry_cache import GeometryWeightCache

        return sum(c.evictions for c in _find(self, GeometryWeightCache))

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
