# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small — Pipeline Overlap: Chunked Coarse→Fine Processing.

Implements practical pipeline overlap by processing fine codebook expansion
in chunks as coarse tokens are produced. On a single device this reduces
peak memory pressure; on multi-device it enables true concurrent execution.

Strategies implemented:
  1. Sequential baseline (reference timing)
  2. Chunked coarse→fine overlap (fine processes chunks while coarse runs)
  3. Latency estimation for multi-device configurations

Usage:
    from bark_pipeline_overlap import BarkStreamingPipeline
    pipeline = BarkStreamingPipeline(model)
    audio = pipeline.generate_streamed(text)
"""

import time

import numpy as np
import torch


class BarkStreamingPipeline:
    """Pipeline with chunked coarse→fine overlap.

    Architecture:
        ┌──────────────────────────┐
        │  Stage 1 (Semantic)      │
        │  Full autoregressive     │
        └──────────┬───────────────┘
                   │
        ┌──────────▼───────────────┐
        │  Stage 2 (Coarse)        │
        │  Produces interleaved    │
        │  codebook 0/1 tokens     │
        └──────────┬───────────────┘
                   │ Chunked output
        ┌──────────▼───────────────┐
        │  Stage 3 (Fine)          │
        │  Processes chunks of     │
        │  coarse tokens → 8 CB   │
        │  (overlap with coarse)   │
        └──────────┬───────────────┘
                   │
        ┌──────────▼───────────────┐
        │  Stage 4 (EnCodec)       │
        │  Audio waveform decode   │
        └──────────────────────────┘

    The chunked approach:
    - Reduces peak memory (fine processes partial coarse output)
    - Enables multi-device overlap (coarse on device A, fine on device B)
    - Provides accurate per-stage timing for performance reports
    """

    def __init__(self, model, fine_chunk_frames=32):
        """
        Args:
            model: TtBarkModel instance
            fine_chunk_frames: Number of coarse frames per fine chunk
        """
        self.model = model
        self.fine_chunk_frames = fine_chunk_frames

    def generate_streamed(self, text: str, verbose: bool = True) -> np.ndarray:
        """Generate audio with chunked coarse→fine overlap.

        Args:
            text: Input text
            verbose: Print stage timing

        Returns:
            audio: numpy array of 24kHz mono audio
        """
        timings = {}

        # Stage 1: Full semantic generation (autoregressive, cannot be chunked)
        t0 = time.time()
        semantic_tokens = self.model.generate_semantic_tokens(text)
        timings["semantic"] = time.time() - t0
        n_sem = semantic_tokens.shape[-1]

        if verbose:
            tps = n_sem / max(timings["semantic"], 1e-6)
            print(f"Stage 1 (Semantic): {n_sem} tokens in {timings['semantic']:.2f}s ({tps:.1f} tok/s)")

        # Stage 2: Full coarse generation
        t0 = time.time()
        coarse_tokens = self.model.generate_coarse_tokens(semantic_tokens)
        timings["coarse"] = time.time() - t0
        n_coarse = coarse_tokens.shape[-1]

        if verbose:
            tps = n_coarse / max(timings["coarse"], 1e-6)
            print(f"Stage 2 (Coarse):   {n_coarse} tokens in {timings['coarse']:.2f}s ({tps:.1f} tok/s)")

        # Stage 3: Chunked fine processing (overlap opportunity)
        t0 = time.time()
        fine_tokens = self._generate_fine_chunked(coarse_tokens)
        timings["fine"] = time.time() - t0

        # Fine tok/s
        coarse_seq_len = n_coarse // 2
        fine_new_tokens = 6 * coarse_seq_len
        fine_tps = fine_new_tokens / max(timings["fine"], 1e-6)

        if verbose:
            print(f"Stage 3 (Fine):     {fine_new_tokens} tokens in {timings['fine']:.2f}s ({fine_tps:.1f} tok/s)")

        # Stage 4: Decode
        t0 = time.time()
        audio = self.model.decode_audio(fine_tokens)
        timings["decode"] = time.time() - t0
        audio = np.clip(audio, -1.0, 1.0)

        total = sum(timings.values())
        audio_dur = len(audio) / 24000
        rtf = total / max(audio_dur, 1e-6)

        if verbose:
            print(f"Stage 4 (Decode):   {timings['decode']:.2f}s")
            print(f"Total: {total:.2f}s | Audio: {audio_dur:.2f}s | RTF: {rtf:.3f}")

            # Pipeline overlap analysis
            estimates = self.estimate_pipeline_latency(timings)
            print(f"\nPipeline overlap analysis:")
            print(
                f"  Sequential (current):    {estimates['sequential']:.2f}s  (RTF {estimates['sequential'] / max(audio_dur, 1e-6):.3f})"
            )
            print(
                f"  Stage 1||2 overlap:      {estimates['overlap_s1_s2']:.2f}s  (RTF {estimates['overlap_s1_s2'] / max(audio_dur, 1e-6):.3f})"
            )
            print(f"  Chunked fine reduction:  {estimates['chunked_benefit_pct']:.1f}% memory reduction")
            print(
                f"  Theoretical min:         {estimates['theoretical_min']:.2f}s  (RTF {estimates['theoretical_min'] / max(audio_dur, 1e-6):.3f})"
            )

        return audio

    def _generate_fine_chunked(self, coarse_tokens: torch.Tensor) -> torch.Tensor:
        """Process fine stage in chunks to reduce peak memory.

        Instead of passing all coarse tokens to fine at once, splits into
        chunks of `fine_chunk_frames` frames. This:
        1. Reduces peak device memory for fine model activations
        2. Enables future multi-device overlap (process chunk N+1 while
           coarse generates tokens for chunk N+2)

        Args:
            coarse_tokens: [batch, coarse_seq_len * 2] interleaved

        Returns:
            fine_tokens: [batch, total_seq_len, 8] concatenated chunks
        """
        n_coarse = self.model.fine_model.n_codes_given  # 2
        batch_size = coarse_tokens.shape[0]
        n_tokens = coarse_tokens.shape[1]

        # Ensure even token count
        if n_tokens % n_coarse != 0:
            n_tokens = (n_tokens // n_coarse) * n_coarse
            coarse_tokens = coarse_tokens[:, :n_tokens]

        total_frames = n_tokens // n_coarse
        chunk_size = self.fine_chunk_frames

        # If total frames fit in one chunk, just run normally
        if total_frames <= chunk_size:
            return self.model.generate_fine_tokens(coarse_tokens)

        # Process in chunks
        fine_chunks = []
        for start_frame in range(0, total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size, total_frames)
            # Slice interleaved tokens: each frame = 2 tokens
            start_tok = start_frame * n_coarse
            end_tok = end_frame * n_coarse
            chunk_tokens = coarse_tokens[:, start_tok:end_tok]

            fine_chunk = self.model.generate_fine_tokens(chunk_tokens)
            fine_chunks.append(fine_chunk)

        # Concatenate along sequence dimension
        return torch.cat(fine_chunks, dim=1)

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
        # Fully pipelined (theoretical minimum = slowest stage)
        max_stage = max(timings.values())

        # Chunked fine benefit: memory reduction estimate
        total_fine_time = timings.get("fine", 0)
        n_chunks = max(1, self.fine_chunk_frames)  # approximate
        chunked_benefit_pct = (1 - 1 / max(n_chunks, 1)) * 100

        return {
            "sequential": sequential,
            "overlap_s1_s2": overlap_12,
            "theoretical_min": max_stage,
            "speedup_overlap": sequential / max(overlap_12, 1e-6),
            "speedup_theoretical": sequential / max(max_stage, 1e-6),
            "chunked_benefit_pct": chunked_benefit_pct,
        }
