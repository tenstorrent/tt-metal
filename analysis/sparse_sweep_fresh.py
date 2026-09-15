# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only sparse-MLA SDPA sweep (TEN-4716). ttnn.transformer.sparse_sdpa: each of S queries
attends its TOP-K selected latent KV entries (indices [1,1,S,TOPK]); asymmetric MLA dims
(K_DIM=576 scores, V_DIM=512 output), shared latent (nkv=1). Compute is K_eff = TOPK selected
entries per query -> predictable from the indices shape. Sweeps TOPK to check K_eff scaling.
Captures per-engine counters under tracy multipass. No golden.

Env: SP_H, SP_S, SP_T, SP_TOPKS (csv), SP_KDIM, SP_VDIM, SP_KC, SP_ITERS.
"""
from __future__ import annotations
import os
import pytest


@pytest.mark.parametrize("topk", [int(x) for x in os.environ.get("SP_TOPKS", "512,1024,2048").split(",")])
def test_sparse(device, topk):
    import torch
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import make_inputs, to_dev

    H = int(os.environ.get("SP_H", "16"))
    S = int(os.environ.get("SP_S", "2048"))
    T = int(os.environ.get("SP_T", "8192"))
    K_DIM = int(os.environ.get("SP_KDIM", "576"))
    V_DIM = int(os.environ.get("SP_VDIM", "512"))
    kc = int(os.environ.get("SP_KC", "128"))
    q, kv, indices = make_inputs(H, S, T, topk, K_DIM, lambda s: topk)
    tt_q = to_dev(q, device, ttnn.bfloat16)
    tt_kv = to_dev(kv, device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    scale = K_DIM**-0.5
    print(f"\n[sparse] H={H} S={S} T={T} TOPK={topk} K_DIM={K_DIM} V_DIM={V_DIM}", flush=True)
    for _ in range(int(os.environ.get("SP_ITERS", "2"))):
        out = ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_kv,
            tt_idx,
            V_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=scale,
            k_chunk_size=kc,
            compute_kernel_config=ck,
        )
        ttnn.synchronize_device(device)
        out.deallocate()
    for t in (tt_q, tt_kv, tt_idx):
        t.deallocate()
    print(f"[sparse] OK TOPK={topk}", flush=True)
