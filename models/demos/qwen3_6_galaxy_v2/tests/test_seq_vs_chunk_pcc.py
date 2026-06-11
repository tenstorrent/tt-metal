# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone op-level PCC check: the ported C++ `gated_delta_attn_seq` path
(chunk_gated_delta_rule_seq + the [B,T,H,X]->[BH,T,X] adapter) vs galaxy's
already-HF-validated pure-TTNN `chunk_gated_delta_rule_ttnn`, on identical random
inputs at the galaxy DeltaNet head config (B=1, H=n_v_per_row=6, K=V=128).

Runs on whatever mesh is available (uses TT_VISIBLE_DEVICES), not the full 8x4
galaxy. Inputs are replicated; results are identical per device.

Run:
  TT_VISIBLE_DEVICES=0,1,2,3 python models/demos/qwen3_6_galaxy_v2/tests/test_seq_vs_chunk_pcc.py
"""
import torch
from loguru import logger

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq
from models.demos.qwen3_6_galaxy_v2.tt.qwen35_chunk_delta_rule_ops import chunk_gated_delta_rule_ttnn, l2_norm_ttnn


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _seq_path(q_exp, k_exp, v_h, beta, g, B, T, H, K, V, mesh, chunk_size):
    """Mirror of TtQwen36DeltaAttention._chunk_gdr_seq."""
    _dram = ttnn.DRAM_MEMORY_CONFIG if T > 512 else ttnn.L1_MEMORY_CONFIG
    BH = B * H
    q = l2_norm_ttnn(q_exp, dim=-1)
    k = l2_norm_ttnn(k_exp, dim=-1)
    q = ttnn.reshape(
        ttnn.typecast(ttnn.transpose(q, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T, K]
    )
    k = ttnn.reshape(
        ttnn.typecast(ttnn.transpose(k, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T, K]
    )
    v = ttnn.reshape(
        ttnn.typecast(ttnn.transpose(v_h, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T, V]
    )
    beta3 = ttnn.reshape(
        ttnn.typecast(ttnn.transpose(beta, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T, 1]
    )
    g3 = ttnn.reshape(
        ttnn.typecast(ttnn.transpose(g, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T]
    )
    out, final_state = chunk_gated_delta_rule_seq(
        q, k, v, beta3, g3, chunk_size=chunk_size, scale=None, initial_state=None, mesh_device=mesh, cached_masks=None
    )
    if out.shape[1] != T:
        out = ttnn.slice(out, (0, 0, 0), (BH, T, V))
    core_out = ttnn.transpose(ttnn.reshape(out, [B, H, T, V]), 1, 2)  # [B,T,H,V]
    new_state = ttnn.reshape(final_state, [B, H, K, V])
    return core_out, new_state


def main():
    torch.manual_seed(0)
    import os

    B, T, H, K, V = 1, 256, 6, 128, 128
    CHUNK = int(os.environ.get("QWEN36_SEQ_CHUNK", "128"))

    dev_ids = ttnn.get_device_ids()
    ncols = min(len(dev_ids), 4) or 1
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ncols))
    logger.info(f"opened mesh (1,{ncols}); config B={B} T={T} H={H} K={K} V={V} chunk={CHUNK}")

    try:
        q_t = torch.randn(B, T, H, K)
        k_t = torch.randn(B, T, H, K)
        v_t = torch.randn(B, T, H, V)
        beta_t = torch.sigmoid(torch.randn(B, T, H))
        g_t = -0.05 * torch.rand(B, T, H)  # small negative log-decay

        def to_dev(x):
            return ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # Reference: pure-TTNN chunk (normalizes + scales internally)
        ref_out, ref_state = chunk_gated_delta_rule_ttnn(
            q=to_dev(q_t),
            k=to_dev(k_t),
            v=to_dev(v_t),
            beta=to_dev(beta_t),
            g=to_dev(g_t),
            chunk_size=CHUNK,
            initial_state=None,
            device=mesh,
            cached_masks=None,
        )
        # Seq path (adapter normalizes; seq scales internally)
        seq_out, seq_state = _seq_path(
            to_dev(q_t), to_dev(k_t), to_dev(v_t), to_dev(beta_t), to_dev(g_t), B, T, H, K, V, mesh, CHUNK
        )

        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        ref_o = ttnn.to_torch(ref_out, mesh_composer=comp)[:B]
        seq_o = ttnn.to_torch(seq_out, mesh_composer=comp)[:B]
        ref_s = ttnn.to_torch(ref_state, mesh_composer=comp)[:B]
        seq_s = ttnn.to_torch(seq_state, mesh_composer=comp)[:B]

        pcc_out = _pcc(ref_o, seq_o)
        pcc_state = _pcc(ref_s, seq_s)
        logger.info(f"core_out shapes ref={tuple(ref_o.shape)} seq={tuple(seq_o.shape)}")
        logger.info(f"PCC core_out   = {pcc_out:.6f}")
        logger.info(f"PCC final_state= {pcc_state:.6f}")
        ok = pcc_out > 0.99 and pcc_state > 0.99
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  (core_out PCC={pcc_out:.5f}, state PCC={pcc_state:.5f})")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
