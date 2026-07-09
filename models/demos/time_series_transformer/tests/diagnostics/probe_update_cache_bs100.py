# SPDX-License-Identifier: Apache-2.0
"""Does update_cache work when BS=100 (this model's S=100 sample-generation
case), or does it hard-fail past batch=32? Isolated check before touching
production code -- test_sample_generation_under_1s uses BS=100 and must
not regress if update_cache is adopted for the smaller-batch tests."""
import torch

import ttnn


def main():
    device = ttnn.open_device(device_id=0)
    try:
        for BS in (1, 4, 32, 33, 100):
            H, T_max, D = 2, 24, 32
            try:
                v_cache = ttnn.from_torch(
                    torch.zeros(BS, H, T_max, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                v_input = ttnn.from_torch(
                    torch.zeros(BS, H, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                ttnn.update_cache(v_cache, v_input, update_idx=0, batch_offset=0)
                print(f"BS={BS}: OK")
            except Exception as e:
                print(f"BS={BS}: FAILED - {str(e)[:200]}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
