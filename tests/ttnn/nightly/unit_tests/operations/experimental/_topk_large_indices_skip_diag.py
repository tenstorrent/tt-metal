# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunk-skip calibration driver: runs one row-parallel topk_large_indices
call with a crafted distinct-bf16 input so the CHUNK_SKIP_DIAG dump in
compute.cpp (post-rebuild DST values region over DPRINT) can be matched
against torch rank order. Underscore-prefixed: not collected by pytest.

Env: TOPK_K in {512, 1024, 2048} (llk window == k here). Run with
    TT_METAL_DPRINT_CORES=worker TT_METAL_DPRINT_FILE=<file> python <this>
"""

import os

import torch
import ttnn

k = int(os.environ.get("TOPK_K", "512"))
n = 2 * k  # two chunks -> one merge+rebuild before the dump

device = ttnn.open_device(device_id=0)

# Distinct, monotone-increasing positive bf16 values: bits 0x3800 + j.
bits = (torch.arange(n, dtype=torch.int32) + 0x3800).to(torch.int16)
row = bits.view(torch.bfloat16)
torch_input = torch.stack([row, row])  # 2 identical rows -> row-parallel path

tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = ttnn.experimental.topk_large_indices(tt_input, k=k)
indices = ttnn.to_torch(out, dtype=torch.uint32)
print("OK diag k=", k, "first indices:", indices[0, :8].tolist())
ttnn.close_device(device)
