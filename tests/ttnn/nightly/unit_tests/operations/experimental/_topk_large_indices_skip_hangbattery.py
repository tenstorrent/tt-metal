# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunk-skip hang battery: repeated launches alternating max-skip
(descending rows -> nearly every tested chunk skips) and zero-skip
(ascending rows) inputs across shapes, program cache on. Any hang shows up
as the outer timeout killing this process. Underscore-prefixed."""

import torch
import ttnn

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

shapes = [
    (2, 65536, 32),
    (2, 65536, 512),
    (8, 65536, 32),
    (2, 51200, 1536),
    (2, 65536, 1024),
]


def make(mode, rows, n):
    base = torch.linspace(-90.0, 90.0, n, dtype=torch.float32).to(torch.bfloat16)
    if mode == "maxskip":
        base = torch.flip(base, dims=[0]).contiguous()
    return base.unsqueeze(0).repeat(rows, 1)


launches = 0
for rep in range(20):
    mode = "maxskip" if rep % 2 == 0 else "zeroskip"
    rows, n, k = shapes[rep % len(shapes)]
    x = make(mode, rows, n)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.experimental.topk_large_indices(tt, k=k)
    idx = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64)
    got = torch.gather(x.float(), -1, idx)
    ref, _ = torch.topk(x.float(), k, dim=-1)
    assert torch.equal(got.sort(-1).values, ref.sort(-1).values), f"rep{rep} {mode} mismatch"
    launches += 1
    print(f"rep {rep} {mode} rows={rows} n={n} k={k} OK", flush=True)

print(f"HANGBATTERY OK: {launches} launches")
ttnn.close_device(device)
