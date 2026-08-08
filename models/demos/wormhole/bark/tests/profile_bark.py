# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-stage throughput profiler for Bark Small on TT hardware."""

import time

import ttnn
from models.demos.wormhole.bark.tt.bark_model import TtBarkModel


def profile_bark():
    device = ttnn.open_device(device_id=0)

    try:
        print("Initializing Bark Model...")
        model = TtBarkModel(device)

        text = "Hello, I am profiling Bark Small on Tenstorrent hardware."

        print(f"Generating audio for: '{text}'")

        start_time = time.time()

        # 1. Semantic
        print("Starting Stage 1: Semantic...")
        s1_start = time.time()
        semantic_tokens = model.generate_semantic_tokens(text)
        s1_end = time.time()
        n_sem = semantic_tokens.shape[-1]
        s1_tps = n_sem / (s1_end - s1_start) if (s1_end - s1_start) > 0 else 0
        print(f"Stage 1 done in {s1_end - s1_start:.2f}s ({n_sem} tokens, {s1_tps:.1f} tok/s)")

        # 2. Coarse
        print("Starting Stage 2: Coarse...")
        s2_start = time.time()
        coarse_tokens = model.generate_coarse_tokens(semantic_tokens)
        s2_end = time.time()
        n_coarse = coarse_tokens.shape[-1]
        s2_tps = n_coarse / (s2_end - s2_start) if (s2_end - s2_start) > 0 else 0
        print(f"Stage 2 done in {s2_end - s2_start:.2f}s ({n_coarse} tokens, {s2_tps:.1f} tok/s)")

        # 3. Fine
        print("Starting Stage 3: Fine...")
        s3_start = time.time()
        fine_tokens = model.generate_fine_tokens(coarse_tokens)
        s3_end = time.time()
        # Fine generates 6 new codebooks × coarse_seq_len tokens
        coarse_seq_len = n_coarse // 2  # De-interleaved length
        fine_new_tokens = 6 * coarse_seq_len
        s3_tps = fine_new_tokens / (s3_end - s3_start) if (s3_end - s3_start) > 0 else 0
        print(f"Stage 3 done in {s3_end - s3_start:.2f}s ({fine_new_tokens} tokens, {s3_tps:.1f} tok/s)")

        # 4. Decode audio
        print("Starting Stage 4: Decode...")
        s4_start = time.time()
        audio = model.decode_audio(fine_tokens)
        s4_end = time.time()
        audio_duration = len(audio) / 24000
        print(f"Stage 4 done in {s4_end - s4_start:.2f}s ({audio_duration:.2f}s audio)")

        total_end = time.time()
        total = total_end - start_time
        rtf = total / audio_duration if audio_duration > 0 else float("inf")
        print(f"\nTotal generation time: {total:.2f}s")
        print(f"  Stage 1 (Semantic): {s1_end - s1_start:.2f}s  ({s1_tps:.1f} tok/s)")
        print(f"  Stage 2 (Coarse):   {s2_end - s2_start:.2f}s  ({s2_tps:.1f} tok/s)")
        print(f"  Stage 3 (Fine):     {s3_end - s3_start:.2f}s  ({s3_tps:.1f} tok/s)")
        print(f"  Stage 4 (Decode):   {s4_end - s4_start:.2f}s")
        print(f"  Audio duration:     {audio_duration:.2f}s")
        print(f"  RTF:                {rtf:.3f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    profile_bark()
