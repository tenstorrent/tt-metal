# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does the in-place cache write take its index as a DEVICE TENSOR?

If it does, one trace suffices for the write and only the positional slice still varies
per sub-step. If it is a baked Python int, the design needs 32 traces. That single fact
is the difference between a small change and a large one.
"""
import torch

import ttnn

H, DK, W = 16, 64, 288
dev = ttnn.open_device(device_id=0, l1_small_size=131072)
try:
    mk = lambda x, d=ttnn.bfloat16, l=ttnn.TILE_LAYOUT: ttnn.from_torch(x, dtype=d, layout=l, device=dev)
    cache = mk(torch.zeros(1, H, W, DK))
    tok = mk(torch.randn(1, H, 1, DK))

    for label, call in [
        (
            "paged_update_cache(idx tensor int32 RM)",
            lambda: ttnn.experimental.paged_update_cache(
                cache,
                tok,
                update_idxs_tensor=mk(torch.tensor([260], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ),
        ),
        (
            "paged_update_cache(update_idxs=[260])",
            lambda: ttnn.experimental.paged_update_cache(cache, tok, update_idxs=[260]),
        ),
    ]:
        try:
            call()
            back = ttnn.to_torch(cache).float()
            rows = (back.abs().sum(dim=(0, 1, 3)) > 0).nonzero().flatten().tolist()
            print(f"  {label:44s} OK, rows {rows[:4]}")
        except Exception as e:
            print(f"  {label:44s} {type(e).__name__}: {str(e)[:110]}")
finally:
    ttnn.close_device(dev)
