# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The stage sequencing (tokenize -> autoregressive frames -> 200-frame chunk denoising with the previous
# window's carry -> vocoder decode -> crop 86 / 258 latents -> stitch) transcribes diffusers
# ``modular_pipelines/minimax_music3/modular_blocks_minimax_music3.py`` and its blocks (Apache-2.0,
# Copyright 2026 The MiniMax Team and The HuggingFace Team).
"""MiniMax-Music3 end to end on one Blackhole chip: (caption, lyrics) -> stereo 44.1 kHz waveform.

``MiniMaxMusic3Pipeline.load(mesh_device)`` puts the Qwen3 backbone (stage 02), the RVQ depth decoder
(stage 03), the flow-matching DiT and its condition encoder (stage 05) on the chip, keeps the Flow-VAE
vocoder (``reference/vocoder_ref.py``, vendored torch) on the host, and warms the AR traces and the DiT
program cache. ``generate`` then runs diffusers' ``MiniMaxMusic3Blocks`` sequence:

1. ``MiniMaxMusic3TokenizeStep`` + ``MiniMaxMusic3AutoregressiveStep`` -> ``ARGenerator.generate``
   (``frame_hiddens [1, F, 32768]``, ``max_frames = min(int(audio_duration * 25), 9000)``);
2. ``MiniMaxMusic3PrepareChunksStep`` -> ``chunk_starts_for(F)``;
3. per window ``MiniMaxMusic3ChunkDenoiseStep`` -> ``ChunkDenoiser.denoise_chunk`` with the window's noise drawn
   from the *same* ``torch.Generator`` the AR sampling used (``randn_tensor`` on the CPU generator), exactly as
   the reference threads one generator through the whole pipeline;
4. ``MiniMaxMusic3VocoderDecodeStep`` -> vocoder per window, crop (86 latents left / 258 right, x 512 samples),
   concatenate, clamp to ``[-1, 1]``.

Device memory: the three transformers stay resident (about 21 GB in the functional policy: backbone 12.0 GB +
3.3 GB KV cache, depth decoder 1.35 GB, DiT 4.6 GB - measured in ``doc/pipeline/README.md``). Trace-lifetime
rule inherited from stage 04: the AR traces are captured after every weight tensor exists; anything the
DiT allocates later (its per-shape RoPE / mask / selector caches, the per-chunk condition projection) is freed
before ``generate`` returns, so no buffer allocated after a capture is alive during a later trace replay.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import vocoder_ref as V
from models.autoports.minimaxai_minimax_music3.tt.ar_generator import ARGenerator
from models.autoports.minimaxai_minimax_music3.tt.condition_encoder import (
    CONDITION_HIDDEN,
    NUM_CONDITION_LAYERS,
    ConditionEncoder,
    latent_length,
)
from models.autoports.minimaxai_minimax_music3.tt.constants import FRAME_RATE, MAX_AUDIO_FRAMES
from models.autoports.minimaxai_minimax_music3.tt.denoiser import (
    CHUNK_FRAMES,
    DEFAULT_STEPS,
    ChunkDenoiser,
    ChunkResult,
    chunk_starts_for,
)
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DepthDecoder
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import FlowTransformer
from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM

NUM_LATENT_CHANNELS = 128
WARMUP_PROMPT = "Genre: ambient. A quiet pad."
WARMUP_LYRICS = "[verse]\nla la la"

DIT_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


def dram_usage_bytes(mesh_device) -> Dict[str, int]:
    """Allocated / free DRAM of the chip from the allocator's memory view (per-bank figures times the bank count)."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        banks = int(view.num_banks)
        return {
            "allocated_bytes": int(view.total_bytes_allocated_per_bank) * banks,
            "free_bytes": int(view.total_bytes_free_per_bank) * banks,
            "total_bytes": int(view.total_bytes_per_bank) * banks,
            "largest_free_block_bytes_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
            "banks": banks,
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


class MiniMaxMusic3Pipeline:
    """The whole model on one chip. Build it with :meth:`load`."""

    def __init__(
        self,
        mesh_device,
        *,
        llm: MusicLLM,
        depth: DepthDecoder,
        ar: ARGenerator,
        transformer: FlowTransformer,
        condition_encoder: ConditionEncoder,
        vocoder: V.MiniMaxMusic3Vocoder,
        weights_dir: Path,
        dtype_policy: str,
        load_log: Optional[dict] = None,
    ):
        self.mesh_device = mesh_device
        self.llm = llm
        self.depth = depth
        self.ar = ar
        self.transformer = transformer
        self.condition_encoder = condition_encoder
        self.vocoder = vocoder
        self.weights_dir = Path(weights_dir)
        self.dtype_policy = dtype_policy
        self.denoiser = ChunkDenoiser(transformer, condition_encoder)
        self.sampling_rate = int(vocoder.sampling_rate)
        self.latent_hop_length = int(vocoder.hop_length)
        self.frame_rate = FRAME_RATE
        self.load_log = load_log or {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(
        cls,
        mesh_device,
        weights_dir: Optional[str] = None,
        *,
        dtype_policy: str = "functional",
        dit_dtype: str = "bf16",
        warm: bool = True,
        vocoder_threads: Optional[int] = None,
    ) -> "MiniMaxMusic3Pipeline":
        """Load every component onto ``mesh_device`` (1x1 mesh) and warm the traces.

        Args:
            weights_dir: the HF snapshot directory (default ``$MM3_WEIGHTS``).
            dtype_policy: the backbone's ``MusicLLM`` policy (``"functional"``: bf16 attention / KV, bfp8 MLP).
            dit_dtype: ``"bf16"`` (default) or ``"bfp8"`` for the DiT weights (the fallback if DRAM is short).
            warm: run a 1-frame song and one full-length DiT step so the AR traces exist and the DiT
                programs are compiled before the first real request.
            vocoder_threads: torch intra-op threads for the host vocoder (default: leave torch's setting).
        """
        from models.autoports.minimaxai_minimax_music3.reference.hf_llm import weights_dir as default_weights_dir

        weights_dir = Path(weights_dir) if weights_dir is not None else default_weights_dir()
        if dit_dtype not in DIT_DTYPES:
            raise ValueError(f"dit_dtype must be one of {sorted(DIT_DTYPES)}, got {dit_dtype!r}")
        if vocoder_threads:
            torch.set_num_threads(int(vocoder_threads))
        log: Dict[str, object] = {"weights_dir": str(weights_dir), "dtype_policy": dtype_policy, "dit_dtype": dit_dtype}
        t_all = time.perf_counter()

        # 1. weights on device, largest first; every persistent buffer exists before any trace is captured.
        t0 = time.perf_counter()
        llm = MusicLLM(mesh_device, dtype_policy=dtype_policy, hf_model_dir=str(weights_dir / "language_model"))
        log["llm_load_s"] = time.perf_counter() - t0
        log["dram_after_llm"] = dram_usage_bytes(mesh_device)
        t0 = time.perf_counter()
        depth = DepthDecoder.from_pretrained(mesh_device, weights_dir)
        log["depth_load_s"] = time.perf_counter() - t0
        log["dram_after_depth"] = dram_usage_bytes(mesh_device)
        t0 = time.perf_counter()
        transformer = FlowTransformer.from_pretrained(mesh_device, weights_dir, dtype=DIT_DTYPES[dit_dtype])
        condition_encoder = ConditionEncoder.from_pretrained(mesh_device, weights_dir)
        log["dit_load_s"] = time.perf_counter() - t0
        log["dram_after_dit"] = dram_usage_bytes(mesh_device)
        t0 = time.perf_counter()
        vocoder = V.load_vocoder(weights_dir, dtype=torch.float32)
        log["vocoder_load_s"] = time.perf_counter() - t0

        # 2. AR generator: decode inputs + depth traces (the backbone trace is captured on the first decode).
        t0 = time.perf_counter()
        ar = ARGenerator(llm, depth, tokenizer_dir=str(weights_dir / "tokenizer"))
        log["ar_init_s"] = time.perf_counter() - t0
        pipe = cls(
            mesh_device,
            llm=llm,
            depth=depth,
            ar=ar,
            transformer=transformer,
            condition_encoder=condition_encoder,
            vocoder=vocoder,
            weights_dir=weights_dir,
            dtype_policy=dtype_policy,
            load_log=log,
        )
        if warm:
            pipe.warm()
        log["dram_resident"] = dram_usage_bytes(mesh_device)
        log["load_total_s"] = time.perf_counter() - t_all
        logger.info(f"MiniMaxMusic3Pipeline ready in {log['load_total_s']:.0f}s; DRAM {log['dram_resident']}")
        return pipe

    def warm(self) -> None:
        """Capture the backbone trace (1-frame song) and compile the DiT at the full 200-frame window."""
        t0 = time.perf_counter()
        self.ar.generate(WARMUP_PROMPT, WARMUP_LYRICS, max_frames=1, seed=0)
        self.load_log["warm_ar_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        frames = CHUNK_FRAMES
        fake_hiddens = torch.zeros(1, frames, NUM_CONDITION_LAYERS * CONDITION_HIDDEN)
        noise = torch.randn(1, NUM_LATENT_CHANNELS, latent_length(frames), generator=torch.Generator().manual_seed(0))
        try:
            self.denoiser.denoise_chunk(fake_hiddens, None, None, noise, 1)
        finally:
            self.transformer.clear_caches()
        self.load_log["warm_dit_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        lyrics: str,
        *,
        audio_duration: float = 60.0,
        seed: Optional[int] = None,
        num_inference_steps: int = DEFAULT_STEPS,
        max_frames: Optional[int] = None,
        teacher_codes: Optional[torch.Tensor] = None,
        teacher_frame0_codes: Optional[torch.Tensor] = None,
        noises: Optional[Sequence[torch.Tensor]] = None,
        text_ids: Optional[torch.Tensor] = None,
        keep_latents: bool = True,
    ) -> dict:
        """Generate a song.

        Args:
            prompt, lyrics: caption and lyrics (``text_ids`` overrides them).
            audio_duration: seconds; ``max_frames = min(int(audio_duration * 25), 9000)`` like the reference.
                The song may end earlier through the end token.
            seed: CPU ``torch.Generator`` seed for every random draw (AR top-k sampling and the per-window noise,
                one generator threaded through, as in diffusers); ``None`` draws a seed and reports it.
            num_inference_steps: Euler steps per window (30 in the reference).
            max_frames: explicit frame cap (overrides ``audio_duration``).
            teacher_codes / teacher_frame0_codes / noises: replay hooks (golden ``sampled_codes.pt`` layout,
                frame-0 codes, one ``[1, 128, L_k]`` noise per window) for the golden-replay test.
            keep_latents: also return the per-window uncropped latents and the frame hiddens.
        Returns:
            ``audio`` ``np.ndarray [2, samples]`` float32 in ``[-1, 1]``, ``sampling_rate``, ``frames``, ``seed``,
            ``codes [F, 8]``, ``chunk_starts``, ``stopped_by`` (``"max_frames"`` / ``"end_token"`` / ``"context"`` / ``"teacher_codes"``),
            ``context_frames`` (the most frames the 10240-position backbone context can hold after this prompt) and
            ``truncated_by_context``, ``timings`` (seconds; ``prefill``, ``ar``,
            ``ar_frames_per_s``, ``dit_per_chunk``, ``vocoder_per_chunk``, ``total`` ...), and with
            ``keep_latents`` also ``latents`` (list) and ``frame_hiddens``.
        """
        if audio_duration is None or audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {audio_duration}")
        if max_frames is None:
            max_frames = min(int(audio_duration * self.frame_rate), MAX_AUDIO_FRAMES)
        if max_frames == 0:
            raise ValueError(
                f"`audio_duration` {audio_duration} is shorter than one audio frame (1 / {self.frame_rate} s)"
            )
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        seed = int(seed)
        steps = int(num_inference_steps)
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {steps}")

        # Context cap: the backbone runs at the checkpoint's advertised 10240 positions (stage 02), so a song can hold
        # at most 10240 - prompt_len - 1 frames; the reference's separate caps (5000 tokens + 9000 frames) exceed that.
        # The AR loop stops with stopped_by == "context" when it is reached; say so up front instead of only at the end.
        if text_ids is None:
            text_ids = self.ar.build_text_ids(prompt, lyrics)
        prompt_len = int(text_ids.shape[1])
        context_frames = self.llm.max_seq_len - prompt_len - 1
        if max_frames > context_frames:
            logger.warning(
                f"generate: {max_frames} frames requested but the {self.llm.max_seq_len}-position context leaves room for "
                f"{context_frames} after the {prompt_len}-token prompt; the song will be cut there (stopped_by='context')"
            )

        timings: Dict[str, object] = {}
        t_all = time.perf_counter()
        # One CPU generator for the whole song, threaded through the AR top-k draws and then the per-window
        # noise (``randn_tensor`` on the same generator), exactly like the reference's ``generator`` input.
        generator = torch.Generator().manual_seed(seed)

        # ---- 1. autoregressive stage
        t0 = time.perf_counter()
        ar_out = self.ar.generate(
            prompt,
            lyrics,
            max_frames=max_frames,
            seed=seed,
            teacher_codes=teacher_codes,
            teacher_frame0_codes=teacher_frame0_codes,
            text_ids=text_ids,
            generator=generator,
        )
        timings["ar"] = time.perf_counter() - t0
        timings["prefill"] = ar_out["timings"]["prefill"]
        frames = int(ar_out["frames"])
        timings["ar_frames_per_s"] = frames / max(1e-9, timings["ar"] - timings["prefill"])
        frame_hiddens = ar_out["frame_hiddens"]
        # ---- 2/3. chunk denoising
        starts = chunk_starts_for(frames)
        if noises is not None:
            assert len(noises) == len(starts), (len(noises), len(starts))
        results: List[ChunkResult] = []
        dit_times: List[float] = []
        prev_lat = prev_cond = None
        try:
            for k, start in enumerate(starts):
                end = min(start + CHUNK_FRAMES, frames)
                num_latents = latent_length(end - start)
                if noises is not None:
                    noise = torch.as_tensor(noises[k]).float()
                    assert tuple(noise.shape) == (1, NUM_LATENT_CHANNELS, num_latents), (
                        tuple(noise.shape),
                        num_latents,
                    )
                else:
                    noise = torch.randn((1, NUM_LATENT_CHANNELS, num_latents), generator=generator, dtype=torch.float32)
                t0 = time.perf_counter()
                r = self.denoiser.denoise_chunk(frame_hiddens[:, start:end], prev_lat, prev_cond, noise, steps)
                dit_times.append(time.perf_counter() - t0)
                prev_lat, prev_cond = r.previous_latent, r.previous_condition
                results.append(r)
        finally:
            self.transformer.clear_caches()
        timings["dit_per_chunk"] = dit_times
        timings["dit"] = float(sum(dit_times))
        timings["dit_per_step_ms"] = [1e3 * t / steps for t in dit_times]

        # ---- 4. vocoder + crop + stitch
        waveforms: List[torch.Tensor] = []
        voc_times: List[float] = []
        for r in results:
            t0 = time.perf_counter()
            waveforms.append(self.vocoder(r.latents.float()))
            voc_times.append(time.perf_counter() - t0)
        audio = V.stitch_waveforms(waveforms, self.latent_hop_length)  # [1, 2, S]
        timings["vocoder_per_chunk"] = voc_times
        timings["vocoder"] = float(sum(voc_times))
        timings["total"] = time.perf_counter() - t_all
        timings["ar_detail"] = {k: v for k, v in ar_out["timings"].items() if k != "per_frame"}

        out = {
            "audio": audio[0].numpy().astype(np.float32),
            "sampling_rate": self.sampling_rate,
            "frames": frames,
            "seed": seed,
            "num_inference_steps": steps,
            "codes": ar_out["codes"],
            "frame0_codes": ar_out["frame0_codes"],
            "chunk_starts": starts,
            "stopped_by": ar_out["stopped_by"],
            "prompt_len": ar_out["prompt_len"],
            "max_frames": max_frames,
            "context_frames": context_frames,
            "truncated_by_context": ar_out["stopped_by"] == "context",
            "text_ids": ar_out["text_ids"],
            "timings": timings,
        }
        if keep_latents:
            out["latents"] = [r.latents for r in results]
            out["conditions"] = [r.condition for r in results]
            out["frame_hiddens"] = frame_hiddens
        logger.info(
            f"generate: {frames} frames ({frames / self.frame_rate:.1f} s, stop={out['stopped_by']}), {len(starts)} windows, "
            f"{audio.shape[-1] / self.sampling_rate:.2f} s of audio; AR {timings['ar']:.1f} s ({timings['ar_frames_per_s']:.1f} frames/s), "
            f"DiT {timings['dit']:.1f} s, vocoder {timings['vocoder']:.1f} s, total {timings['total']:.1f} s"
        )
        return out

    # ------------------------------------------------------------------ housekeeping
    def release(self) -> None:
        self.ar.release()
        self.transformer.release()
        self.condition_encoder.release()
        self.llm.release()
