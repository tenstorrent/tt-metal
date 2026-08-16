# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunk-skip decision tracer: 4-chunk rows with exactly computable
max/threshold per tested chunk. Run with CHUNK_SKIP_DEBUG enabled in
topk_large_indices_chunk_skip.hpp and DPRINT env:
    TT_METAL_DPRINT_CORES=worker TT_METAL_DPRINT_RISCVS=TR1 \
    TT_METAL_DPRINT_FILE=<f> python <this>
"""

import os

import torch
import ttnn

mode = os.environ.get("SKIP_DBG_MODE", "asc")
k = int(os.environ.get("TOPK_K", "32"))
n = 4 * 512  # 4 chunks at llk 512

device = ttnn.open_device(device_id=0)

bits = (torch.arange(n, dtype=torch.int32) + 0x3800).to(torch.int16)
row = bits.view(torch.bfloat16)
if mode == "desc":
    row = torch.flip(row, dims=[0]).contiguous()
torch_input = torch.stack([row, row])

tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = ttnn.experimental.topk_large_indices(tt_input, k=k)
indices = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64)
actual = torch.gather(torch_input.float(), dim=-1, index=indices)
ref, _ = torch.topk(torch_input.float(), k, dim=-1, largest=True, sorted=True)
exact = torch.equal(actual.sort(dim=-1).values, ref.sort(dim=-1).values)


# Host-side expectations for the two tested chunks (2 and 3)
def bf16_word(j):
    return (0x3800 + j) << 16


if mode == "asc":
    print("EXPECT chunk2: max", bf16_word(1535), "thr", bf16_word(1024 - 1 - (k - 1)), "skip 0")
    print("EXPECT chunk3: max", bf16_word(2047), "thr", bf16_word(1536 - 1 - (k - 1)), "skip 0")
else:
    print("EXPECT chunk2: max", bf16_word(n - 1 - 1024), "thr", bf16_word(n - 1 - (k - 1)), "skip 1")
    print("EXPECT chunk3: max", bf16_word(n - 1 - 1536), "thr", bf16_word(n - 1 - (k - 1)), "skip 1")
print("EXACT" if exact else "MISMATCH", "mode=", mode, "k=", k)
ttnn.close_device(device)
