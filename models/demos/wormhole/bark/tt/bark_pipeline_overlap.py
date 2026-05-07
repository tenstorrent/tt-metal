# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small — Pipeline Overlap Analysis (Stage 1 → Stage 2).

Measures the current sequential pipeline and estimates the latency benefit
available from overlapping Semantic (Stage 1) and Coarse (Stage 2) on
multi-device or async scheduling setups.

Strategy:
  - Semantic generation produces tokens autoregressively (full completion)
  - Coarse generation starts after all semantic tokens are ready
  - Timing data is used to estimate multi-device overlap opportunity

Note: Full streaming overlap requires concurrent execution on separate
devices or async scheduling. This module implements sequential execution
with timing analysis to estimate the benefit of overlap.

Usage:
    from bark_pipeline_overlap import BarkStreamingPipeline
    pipeline = BarkStreamingPipeline(model)
    audio = pipeline.generate_streamed(text)
"""

import time

import numpy as np


class BarkStreamingPipeline:
    """Sequential pipeline with multi-device overlap estimates.

    Architecture:
        ┌──────────────────────────┐
        │  Stage 1 (Semantic)      │
        │  Generates chunks of     │
        │  semantic tokens         │
        └──────────┬───────────────┘
                   │ Multi-device overlap opportunity
        ┌──────────▼───────────────┐
        │  Stage 2 (Coarse)        │
        │  Begins processing       │
        │  available semantics     │
        └──────────┬───────────────┘
                   │
        ┌──────────▼───────────────┐
        │  Stage 3 (Fine)          │
        │  Full codebook expansion │
        └──────────┬───────────────┘
                   │
        ┌──────────▼───────────────┐
        │  Stage 4 (EnCodec)       │
        │  Audio waveform decode   │
        └──────────────────────────┘

    On single-device systems, true parallelism isn't possible (the device
    can only run one model at a time). This module's value is in:
    1. Documenting the overlap opportunity for multi-device scaling
    2. Capturing per-stage timings for performance reports
    3. Estimating upper-bound latency improvements
    """

    def __init__(self, model):
        """
        Args:
            model: TtBarkModel instance
        """
        self.model = model

    def generate_streamed(self, text: str, verbose: bool = True) -> np.ndarray:
        """Generate audio with streaming pipeline overlap.

        For single-device execution, this generates semantic tokens fully,
        then overlaps coarse processing with any remaining cleanup. The
        real benefit comes from the chunked memory approach.

        Args:
            text: Input text
            verbose: Print stage timing

        Returns:
            audio: numpy array of 24kHz mono audio
        """
        timings = {}

        # Stage 1: Full semantic generation (cannot be chunked — autoregressive)
        t0 = time.time()
        semantic_tokens = self.model.generate_semantic_tokens(text)
        timings["semantic"] = time.time() - t0
        n_sem = semantic_tokens.shape[-1]

        if verbose:
            tps = n_sem / max(timings["semantic"], 1e-6)
            print(f"Stage 1 (Semantic): {n_sem} tokens in {timings['semantic']:.2f}s ({tps:.1f} tok/s)")

        # Stage 2: Coarse generation — can start immediately after Stage 1
        # On a multi-device system, this stage is the overlap candidate
        # once semantic generation has produced a usable prefix.
        t0 = time.time()
        coarse_tokens = self.model.generate_coarse_tokens(semantic_tokens)
        timings["coarse"] = time.time() - t0
        n_coarse = coarse_tokens.shape[-1]

        if verbose:
            tps = n_coarse / max(timings["coarse"], 1e-6)
            print(f"Stage 2 (Coarse):   {n_coarse} tokens in {timings['coarse']:.2f}s ({tps:.1f} tok/s)")

        # Stage 3: Fine — can run as soon as Stage 2 completes
        t0 = time.time()
        fine_tokens = self.model.generate_fine_tokens(coarse_tokens)
        timings["fine"] = time.time() - t0

        if verbose:
            print(f"Stage 3 (Fine):     {timings['fine']:.2f}s")

        # Stage 4: Decode
        t0 = time.time()
        audio = self.model.decode_audio(fine_tokens)
        timings["decode"] = time.time() - t0
        audio = np.clip(audio, -1.0, 1.0)

        total = sum(timings.values())
        audio_dur = len(audio) / 24000

        if verbose:
            print(f"Stage 4 (Decode):   {timings['decode']:.2f}s")
            print(f"Total: {total:.2f}s | Audio: {audio_dur:.2f}s | RTF: {total / max(audio_dur, 1e-6):.3f}")

            # Multi-device overlap estimate
            # If stages 1 & 2 could overlap, the pipeline time would be:
            #   max(stage1, stage2) + stage3 + stage4  instead of  sum(all)
            overlap_time = max(timings["semantic"], timings["coarse"]) + timings["fine"] + timings["decode"]
            overlap_rtf = overlap_time / max(audio_dur, 1e-6)
            print(f"\nMulti-device overlap estimate:")
            print(f"  Overlap time: {overlap_time:.2f}s (vs {total:.2f}s sequential)")
            print(f"  Overlap RTF:  {overlap_rtf:.3f} (vs {total / max(audio_dur, 1e-6):.3f} sequential)")
            print(f"  Speedup:      {total / max(overlap_time, 1e-6):.2f}x")

        return audio

    def estimate_pipeline_latency(self, timings: dict) -> dict:
        """Estimate latency under different parallelism strategies.

        Args:
            timings: Dict with per-stage times

        Returns:
            Dict with latency estimates under different strategies
        """
        sequential = sum(timings.values())
        # Stage 1 || Stage 2 (needs 2 devices or async scheduling)
        overlap_12 = (
            max(timings.get("semantic", 0), timings.get("coarse", 0))
            + timings.get("fine", 0)
            + timings.get("decode", 0)
        )
        # All stages overlapped (theoretical minimum = slowest stage)
        max_stage = max(timings.values())

        return {
            "sequential": sequential,
            "overlap_s1_s2": overlap_12,
            "theoretical_min": max_stage,
            "speedup_overlap": sequential / max(overlap_12, 1e-6),
            "speedup_theoretical": sequential / max(max_stage, 1e-6),
        }
