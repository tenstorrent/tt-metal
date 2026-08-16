# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Adversarial correctness battery for the chunk-skip early-out in the
row-parallel topk_large_indices kernels. Bit-exact comparison against torch
top-k value multisets in an int-bits domain (stimuli avoid bf16 specials:
no NaN/-0/subnormals). Underscore-prefixed: not collected by pytest.

Covers:
  * iid randn (the sweep stimulus; skip fires at small k)
  * all-equal rows (strict-< keeps every boundary tie -> 0% skip)
  * ascending rows (every chunk raises the max -> 0% skip)
  * global top-k entirely in the LAST chunk (max skip pressure beforehand;
    the winning chunk must never be skipped)
  * boundary ties straddling chunks (k-th value duplicated across chunks)
  * descending rows (maximum skip: all winners in chunk 0)
  * valid_length prefix cell
  * return_values variant (compute_with_values.cpp) on the same stimuli
"""

import os

import torch
import ttnn

torch.manual_seed(1234)

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

failures = []


def bits16(t):
    return t.view(torch.int16).to(torch.int64) & 0xFFFF


def check(name, torch_input, k, valid_length=None, return_values=False, iters=2):
    n = torch_input.shape[-1]
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for it in range(iters):  # program-cache on: exercise cache-hit path too
        if return_values:
            vals, out = ttnn.experimental.topk_large_indices(
                tt_input, k=k, valid_length=valid_length, return_values=True
            )
        else:
            out = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length=valid_length)
        indices = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64)
        ref_in = torch_input if valid_length is None else torch_input[:, :valid_length]
        ref, _ = torch.topk(ref_in.float(), k, dim=-1, largest=True, sorted=True)
        # Bit-exact multiset compare in the bf16-bits domain (values are
        # bf16-exact end to end; gather in int64 index domain).
        sentinel = indices == 0xFFFFFFFF
        gathered = torch.gather(torch_input, dim=-1, index=indices.clamp(max=n - 1))
        got_bits = bits16(gathered)
        got_bits[sentinel] = -1  # sentinel lanes: only legal when ref lane is -inf
        ref_bf16 = ref.to(torch.bfloat16)
        ref_bits = bits16(ref_bf16)
        ref_bits[ref_bf16 == float("-inf")] = -1
        ok_vals = torch.equal(got_bits.sort(dim=-1).values, ref_bits.sort(dim=-1).values)
        # Indices must be distinct per row (excluding sentinels)
        ok_idx = True
        for r in range(indices.shape[0]):
            real = indices[r][~sentinel[r]]
            if real.numel() != real.unique().numel():
                ok_idx = False
        if return_values:
            v = ttnn.to_torch(vals, dtype=torch.bfloat16)
            vb = bits16(v)
            vb[v == float("-inf")] = -1
            ok_vals = ok_vals and torch.equal(vb.sort(dim=-1).values, ref_bits.sort(dim=-1).values)
        status = "PASS" if (ok_vals and ok_idx) else "FAIL"
        if status == "FAIL":
            failures.append(f"{name} iter{it}")
        print(
            f"{status} {name} iter{it} k={k} shape={tuple(torch_input.shape)} valid={valid_length} rv={return_values}"
        )


N = 65536
LLK = 512  # window for k<=512

# 1. iid randn (multi-row -> row-parallel), small and large k
x = torch.randn(2, N, dtype=torch.bfloat16)
check("randn_k32", x, 32)
check("randn_k512", x, 512)
check("randn_k32_rv", x, 32, return_values=True)
check("randn_k1536_llk2048", torch.randn(2, 51200, dtype=torch.bfloat16), 1536)

# 2. all-equal rows
check("allequal_k32", torch.full((2, N), 1.5, dtype=torch.bfloat16), 32)
check("allequal_k512", torch.full((2, N), -2.0, dtype=torch.bfloat16), 512)

# 3. ascending rows (0% skip: every chunk's max beats the running k-th)
asc = (torch.arange(N, dtype=torch.float32) / N * 200.0 - 100.0).to(torch.bfloat16).unsqueeze(0).repeat(2, 1)
check("ascending_k32", asc, 32)
check("ascending_k512", asc, 512)

# 4. descending rows (max skip: all winners in chunk 0; every later chunk skippable)
desc = torch.flip(asc, dims=[-1]).contiguous()
check("descending_k32", desc, 32)
check("descending_k512", desc, 512)

# 5. global top-k entirely in the LAST chunk, preceded by a long low plateau
last = torch.full((2, N), -50.0, dtype=torch.bfloat16)
last[:, -LLK:] = torch.linspace(60.0, 90.0, LLK, dtype=torch.float32).to(torch.bfloat16)
check("topk_in_last_chunk_k32", last, 32)
check("topk_in_last_chunk_k512", last, 512)

# 5b. top-k split between chunk 0 and the last chunk (middle all-skippable)
split = torch.full((2, N), -50.0, dtype=torch.bfloat16)
split[:, :16] = 80.0
split[:, -16:] = 85.0
check("topk_split_first_last_k32", split, 32)

# 6. boundary ties straddling chunks: k-th value duplicated in many chunks
ties = torch.randn(2, N, dtype=torch.bfloat16).clamp(max=0.0)  # background <= 0
ties[:, 100] = 10.0  # a few clear winners in chunk 0
ties[:, 200] = 9.0
# tie value 5.0 appears once per chunk for 64 chunks -> exactly at/around rank k
for c in range(64):
    ties[:, c * LLK + 7] = 5.0
check("boundary_ties_k32", ties, 32)
check("boundary_ties_k64", ties, 64)

# 7. valid_length prefix (bounded_cache shape)
vx = torch.randn(2, 102400, dtype=torch.bfloat16)
check("valid56320_k1536", vx, 1536, valid_length=56320)
check("valid56320_k1536_rv", vx, 1536, valid_length=56320, return_values=True)

# 8. k=1024 window coverage
check("randn_k1024", torch.randn(2, N, dtype=torch.bfloat16), 1024)

print("FAILURES:", failures if failures else "none")
ttnn.close_device(device)
raise SystemExit(1 if failures else 0)
