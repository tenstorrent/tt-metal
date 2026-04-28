# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for issue #43270: ``nlp_create_qkv_heads_decode`` produced
wrong output on Blackhole when its bf16 input lived in DRAM-interleaved
memory. The interleaved reader kernel reads each face row as a 16-element
(``16 * element_size``) noc_async_read transaction. For bf16 that is 32
bytes — equal to Wormhole's 32-byte DRAM read alignment but below
Blackhole's 64-byte alignment. With sub-alignment reads, every odd-indexed
Q/K/V head silently came back with the *previous* user's row of data.

Test exercises the full QKV-projection chain (matmul → bias add →
nlp_create_qkv_heads_decode) at the gpt-oss-20b shape so the regression is
caught in the calling pattern that originally exposed it. The op's own
existing parametrized DRAM tests (test_nlp_create_qkv_heads_decode.py)
provide broader head/batch/dtype coverage.

Configuration:

    activation : [1, 1, batch_padded=32, hidden=2880] bfloat8_b in DRAM
    weight     : [1, 1, hidden=2880, total_qkv=5120]  bfloat8_b in DRAM
                 with total_qkv = (num_q + 2*num_kv) * head_dim
                                = (64 + 2*8) * 64 = 5120
    bias       : [1, 1, 1, total_qkv=5120]            bfloat16  in DRAM
    matmul out : bf16 in DRAM (tested) and bf16 in L1 (control)
    head split : nlp_create_qkv_heads_decode → height-sharded L1 outputs

Both parametrize cases must pass PCC ≥ 0.99 vs a torch fp32 reference. On
the un-fixed Blackhole codebase, the DRAM case dropped to ~0.75; on
Wormhole both cases always passed. With the fix in place (op promotes the
input to L1 when the per-face-row read would be sub-DRAM-alignment), both
cases pass on both architectures.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


M_BATCH_PADDED = 32
K_HIDDEN = 2880
HEAD_DIM = 64
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 4096
K_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
V_DIM = NUM_KV_HEADS * HEAD_DIM  # 512
N_QKV = Q_DIM + K_DIM + V_DIM  # 5120


@pytest.mark.parametrize(
    "matmul_output_memory_config",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    ids=["matmul_out_dram", "matmul_out_l1"],
)
def test_qkv_pipeline_pcc_vs_torch(device, matmul_output_memory_config):
    """Replays the gpt-oss-20b QKV pipeline (matmul → bias add →
    nlp_create_qkv_heads_decode) and compares Q, K, V to a torch fp32
    reference using PCC.

    On Blackhole single chip:
        matmul_out_dram : FAIL — Q/K/V PCC drops well below 0.99 because the
                          DRAM matmul output is striped along the qkv tile
                          dimension.
        matmul_out_l1   : PASS — Q/K/V PCC ≥ 0.99; only bf8 round-trip noise.
    """
    torch.manual_seed(0)

    # Activation has a small +1.0 mean shift; weight is unit-normal scaled
    # so the per-output-element magnitude stays small. The mean shift is
    # important — purely zero-mean random inputs don't surface the bug, but
    # any non-trivial mean does. Magnitudes are kept small to keep bf8
    # round-trip noise low so the L1 path passes PCC ≥ 0.99 cleanly.
    a_torch = (torch.randn(1, 1, M_BATCH_PADDED, K_HIDDEN, dtype=torch.bfloat16) + 1.0) * 0.1
    b_torch = torch.randn(1, 1, K_HIDDEN, N_QKV, dtype=torch.bfloat16) * 0.1
    bias_torch = torch.randn(1, 1, 1, N_QKV, dtype=torch.bfloat16) * 0.1

    # ---- torch fp32 reference ---------------------------------------------
    qkv_ref = a_torch.float() @ b_torch.float() + bias_torch.float()  # [1, 1, B, N_QKV]
    qkv_ref = qkv_ref.squeeze(0)  # [1, B, N_QKV]
    q_ref = qkv_ref[..., :Q_DIM].reshape(1, M_BATCH_PADDED, NUM_Q_HEADS, HEAD_DIM)
    k_ref = qkv_ref[..., Q_DIM : Q_DIM + K_DIM].reshape(1, M_BATCH_PADDED, NUM_KV_HEADS, HEAD_DIM)
    v_ref = qkv_ref[..., Q_DIM + K_DIM :].reshape(1, M_BATCH_PADDED, NUM_KV_HEADS, HEAD_DIM)

    # ---- ttnn pipeline ----------------------------------------------------
    a_tt = ttnn.from_torch(
        a_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_tt = ttnn.from_torch(
        bias_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    qkv_tt = ttnn.matmul(a_tt, b_tt, dtype=ttnn.bfloat16, memory_config=matmul_output_memory_config)
    ttnn.add(qkv_tt, bias_tt, output_tensor=qkv_tt)

    tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt,
        num_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    q_tt = ttnn.to_torch(tt_q).float()
    k_tt = ttnn.to_torch(tt_k).float()
    v_tt = ttnn.to_torch(tt_v).float()

    # ---- PCC comparison ---------------------------------------------------
    print(f"\nQ shape: ttnn {tuple(q_tt.shape)} vs ref {tuple(q_ref.shape)}")
    print(f"K shape: ttnn {tuple(k_tt.shape)} vs ref {tuple(k_ref.shape)}")
    print(f"V shape: ttnn {tuple(v_tt.shape)} vs ref {tuple(v_ref.shape)}")

    assert_with_pcc(q_ref, q_tt, pcc=0.99)
    assert_with_pcc(k_ref, k_tt, pcc=0.99)
    assert_with_pcc(v_ref, v_tt, pcc=0.99)
