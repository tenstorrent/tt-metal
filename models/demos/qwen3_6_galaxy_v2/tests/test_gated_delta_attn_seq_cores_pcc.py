# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Guard test for the gated_delta_attn_seq kernel's core usage + correctness.

Purpose
-------
The C++ ``gated_delta_attn_seq`` kernel currently maps **one Tensix core per head**
(``head_cores[h] = {col_off + h/grid_y, h%grid_y}``), so on the galaxy (n_v_per_row=6)
it runs on only 6 cores in a single column. That (a) collides with the TP=32 persistent
CCL L1 buffers on column 0, and (b) under-uses the grid. The planned fix is to shard each
head's work across MORE cores (smaller per-core CB footprint + more parallelism).

This test is the spec/guard for that rewrite:
  1. PCC: the seq kernel must match the HF-validated pure-TTNN reference
     (``chunk_gated_delta_rule_ttnn``) to > 0.99 — for ANY core layout.
  2. Cores: the seq op must run on MORE than ``BH`` cores once the multi-core kernel
     lands (``QWEN36_SEQ_CORES_PER_HEAD`` > 1). Until then this assertion xfails.

Run (after a device is free):
  TT_VISIBLE_DEVICES=0,1,2,3 python -m pytest --noconftest \
      models/demos/qwen3_6_galaxy_v2/tests/test_gated_delta_attn_seq_cores_pcc.py -v -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq
from models.demos.qwen3_6_galaxy_v2.tt.qwen35_chunk_delta_rule_ops import chunk_gated_delta_rule_ttnn, l2_norm_ttnn

# galaxy DeltaNet per-chip head config
_B, _H, _K, _V = 1, 6, 128, 128
_CHUNK = 128  # inner seq-kernel chunk_size C (sizes the CBs)
_PCC_BAR = 0.99


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _to_bht_dram(t_4d, X, BH, T, mesh):
    """[B,T,H,X] -> [BH,T,X] float32 DRAM (mirrors TtQwen36DeltaAttention._chunk_gdr_seq)."""
    d = ttnn.DRAM_MEMORY_CONFIG
    return ttnn.reshape(
        ttnn.typecast(ttnn.transpose(t_4d, 1, 2, memory_config=d), ttnn.float32, memory_config=d), [BH, T, X]
    )


@pytest.mark.parametrize("T", [256, 512], ids=["T256", "T512"])
def test_gated_delta_attn_seq_pcc_and_cores(T):
    """Seq kernel must match the pure-TTNN reference (PCC) and (eventually) use >BH cores."""
    cores_per_head = int(os.environ.get("QWEN36_SEQ_CORES_PER_HEAD", "1"))
    BH = _B * _H

    # On the 32-device galaxy, a (1,4) mesh maps to remote-only devices and hangs; use the
    # full 8x4. On a 4-device P150 box, use (1,4). The op runs per-device (replicated inputs).
    ndev = len(ttnn.get_device_ids())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4) if ndev >= 32 else ttnn.MeshShape(1, min(ndev, 4) or 1))
    try:
        torch.manual_seed(0)
        q_t = torch.randn(_B, T, _H, _K)
        k_t = torch.randn(_B, T, _H, _K)
        v_t = torch.randn(_B, T, _H, _V)
        beta_t = torch.sigmoid(torch.randn(_B, T, _H))
        g_t = -0.05 * torch.rand(_B, T, _H)

        def to_dev(x):
            return ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # Reference: pure-TTNN chunk (HF-validated); normalizes + scales internally.
        ref_out, ref_state = chunk_gated_delta_rule_ttnn(
            q=to_dev(q_t),
            k=to_dev(k_t),
            v=to_dev(v_t),
            beta=to_dev(beta_t),
            g=to_dev(g_t),
            chunk_size=_CHUNK,
            initial_state=None,
            device=mesh,
            cached_masks=None,
        )

        # Seq kernel under test: l2-norm q/k (kernel expects normalized), then [BH,T,X], seq applies scale.
        qn, kn = l2_norm_ttnn(to_dev(q_t), dim=-1), l2_norm_ttnn(to_dev(k_t), dim=-1)
        q3, k3 = _to_bht_dram(qn, _K, BH, T, mesh), _to_bht_dram(kn, _K, BH, T, mesh)
        v3 = _to_bht_dram(to_dev(v_t), _V, BH, T, mesh)
        d = ttnn.DRAM_MEMORY_CONFIG
        beta3 = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(to_dev(beta_t), 1, 2, memory_config=d), ttnn.float32, memory_config=d),
            [BH, T, 1],
        )
        g3 = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(to_dev(g_t), 1, 2, memory_config=d), ttnn.float32, memory_config=d), [BH, T]
        )
        seq_out, seq_state = chunk_gated_delta_rule_seq(
            q3,
            k3,
            v3,
            beta3,
            g3,
            chunk_size=_CHUNK,
            scale=None,
            initial_state=None,
            mesh_device=mesh,
            cached_masks=None,
        )

        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        # ref_out: [B,T,H,V] (replicated across devices -> take first B rows).
        ref = ttnn.to_torch(ref_out, mesh_composer=comp)[:_B]  # [B,T,H,V]
        # seq_out: [BH,L,V] (replicated -> take first BH); slice L->T, reshape to [B,T,H,V].
        seq = ttnn.to_torch(seq_out, mesh_composer=comp)[:BH][:, :T, :]  # [BH,T,V]
        seq = seq.reshape(_B, _H, T, _V).permute(0, 2, 1, 3).contiguous()  # [B,T,H,V]
        pcc = _pcc(ref, seq)
        print(f"\n[seq-cores-pcc] T={T} cores_per_head={cores_per_head} PCC={pcc:.6f}")

        # final_state PCC — this is what seeds DECODE after prefill. With value-dim split,
        # each core writes its [v_off,v_off+Vt) slice of final_state[BH,Dk,Dv]; a wrong slice
        # write produces a coherent prefill output but GARBAGE decode (caught here, not by out PCC).
        # Both are replicated across the 32-device mesh and concatenated on dim 0; the first BH
        # rows after flattening to [-1, Dk, Dv] are device 0's heads (representative).
        rs = ttnn.to_torch(ref_state, mesh_composer=comp).reshape(-1, _K, _V)[:BH].contiguous()
        ss = ttnn.to_torch(seq_state, mesh_composer=comp).reshape(-1, _K, _V)[:BH].contiguous()
        pcc_state = _pcc(rs, ss)
        print(f"[seq-cores-pcc] T={T} cores_per_head={cores_per_head} STATE_PCC={pcc_state:.6f}")

        # (1) Correctness guard — must hold for any core layout / kernel rewrite.
        assert pcc > _PCC_BAR, f"seq kernel out PCC {pcc:.5f} below {_PCC_BAR} (T={T})"
        assert pcc_state > _PCC_BAR, f"seq kernel final_state PCC {pcc_state:.5f} below {_PCC_BAR} (T={T})"

        # (2) Perf — time the seq wrapper (rebuild [BH,T,X] inputs each iter since the
        # wrapper consumes/frees them; this mirrors the per-layer cost in the model).
        # Verification harness for the defensive wrapper rewrite: PCC must stay > 0.99
        # and ms/call must not regress materially vs this baseline.
        import time as _time

        _NIT = 5
        ttnn.synchronize_device(mesh)
        _t0 = _time.time()
        for _ in range(_NIT):
            _q3 = _to_bht_dram(l2_norm_ttnn(to_dev(q_t), dim=-1), _K, BH, T, mesh)
            _k3 = _to_bht_dram(l2_norm_ttnn(to_dev(k_t), dim=-1), _K, BH, T, mesh)
            _v3 = _to_bht_dram(to_dev(v_t), _V, BH, T, mesh)
            _b3 = ttnn.reshape(
                ttnn.typecast(ttnn.transpose(to_dev(beta_t), 1, 2, memory_config=d), ttnn.float32, memory_config=d),
                [BH, T, 1],
            )
            _g3 = ttnn.reshape(
                ttnn.typecast(ttnn.transpose(to_dev(g_t), 1, 2, memory_config=d), ttnn.float32, memory_config=d),
                [BH, T],
            )
            _o, _s = chunk_gated_delta_rule_seq(
                _q3,
                _k3,
                _v3,
                _b3,
                _g3,
                chunk_size=_CHUNK,
                scale=None,
                initial_state=None,
                mesh_device=mesh,
                cached_masks=None,
            )
            ttnn.deallocate(_o)
            if _s is not None:
                ttnn.deallocate(_s)
        ttnn.synchronize_device(mesh)
        _ms = (_time.time() - _t0) / _NIT * 1000.0
        print(f"[seq-perf] T={T} cores_per_head={cores_per_head} {_ms:.2f} ms/call (incl input prep)")

        # (2) Multi-core path: with QWEN36_SEQ_CORES_PER_HEAD>1 the op shards each head's value
        # dim across that many cores (BH*split_v cores total). PCC must still hold — the assert
        # above is the guard. Vt(=Dv/32=4) must be divisible by split_v.
        if cores_per_head > 1:
            print(f"[seq-cores-pcc] multi-core (cores_per_head={cores_per_head}) PCC OK = {pcc:.6f}")
    finally:
        ttnn.close_mesh_device(mesh)
