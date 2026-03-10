# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time

import ttnn
from models.demos.wormhole.bark.tt.bark_model import TtBarkModel


def profile_bark():
    device = ttnn.open_device(device_id=0)

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
    print(f"Stage 1 done in {s1_end - s1_start:.2f}s")

    # 2. Coarse
    print("Starting Stage 2: Coarse...")
    s2_start = time.time()
    coarse_tokens = model.generate_coarse_tokens(semantic_tokens)
    s2_end = time.time()
    print(f"Stage 2 done in {s2_end - s2_start:.2f}s")

    # 3. Fine
    print("Starting Stage 3: Fine...")
    s3_start = time.time()
    fine_tokens = model.generate_fine_tokens(coarse_tokens)
    s3_end = time.time()
    print(f"Stage 3 done in {s3_end - s3_start:.2f}s")

    total_end = time.time()
    print(f"Total generation time: {total_end - start_time:.2f}s")

    ttnn.close_device(device)


if __name__ == "__main__":
    profile_bark()
