# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Can a stock ttnn op read decode's per-bank expert streams (``fused/moe/layout.py``, tp8 share) as a matmul operand?

One chip, random weights, decode's own stream builders and placement (``layout.gate_up_stream`` / ``down_stream`` /
``to_device_stream``: BFP4, WIDTH_SHARDED over the 8 DRAM banks, ``[rows, 32]`` per bank). Each probe records
whether the op runs and, if so, its PCC against the expert's math (``HY4_TP8G4_PROBE`` lines; nothing asserted on
the probes themselves: the answer is the record).

  gate_up (an expert's block is [192 K tiles x (gate, up)] rows x 8 bank columns = [12288, 256]: tile row 2k is the
  gate tile of K tile k, 2k + 1 the up tile; bank b = intermediate tile column b):
    A  ttnn.matmul(x', W) with W the whole one-expert stream and x' = x with zero K tiles interleaved ([gate | 0]
       per K tile, and [0 | up] stacked below it in M): gate and up as one matmul at 2x the MACs
    B  ttnn.reshape of a 2-expert stream to [1, 2, 12288, 256] (an expert per batch), batched matmul
    C  ttnn.slice of the 2-expert stream to expert 1's rows, then A
  down (an expert's block: bank b holds output tile columns 24 b .. 24 b + 23, each as its 8 K tiles, i.e. tile rows
  [8 c, 8 c + 8) x 8 banks = [256, 256] whose output tile column b is the model's column 24 b + c):
    D  ttnn.reshape of the one-expert down stream to [1, 24, 256, 256] and a batched matmul (24 x [M,256]@[256,256])
"""

from __future__ import annotations

import json

import torch
from loguru import logger

import ttnn


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _rec(name, **kw):
    logger.info(f"HY4_TP8G4_PROBE {json.dumps(dict(probe=name, **kw))}")


def _try(name, fn):
    try:
        kw = fn()
        _rec(name, ok=True, **(kw or {}))
    except Exception as e:  # the probe's answer is whether the op accepts the operand
        msg = str(e).strip().splitlines()
        _rec(
            name,
            ok=False,
            error=(msg[0] if msg else repr(e))[:400],
            error_tail=" | ".join(m for m in msg if "TT_FATAL" in m or "info:" in m)[:600],
        )


def test_stream_as_matmul_operand(device):
    from models.experimental.hy4_preview.fused.moe import layout as L

    share = L.tp_share(8)
    g = torch.Generator().manual_seed(0)
    E, M, H, Ic = 2, 64, L.HIDDEN, share.channels  # 256 channels per chip
    gate = torch.randn(E, Ic, H, generator=g) / H**0.5
    up = torch.randn(E, Ic, H, generator=g) / H**0.5
    down = torch.randn(E, H, Ic, generator=g) / Ic**0.5
    gq, uq = L.bfp4_round(gate.transpose(1, 2)).transpose(1, 2), L.bfp4_round(up.transpose(1, 2)).transpose(1, 2)
    dq = L.bfp4_round(down.transpose(1, 2)).transpose(1, 2)
    x = torch.randn(M, H, generator=g).to(torch.bfloat16).float()

    gu_stream = L.gate_up_stream(gate, up, share)  # [E * 12288, 256]
    dn_stream = L.down_stream(down, share)  # [E * 6144, 256]
    rows_e = gu_stream.shape[0] // E
    _rec("shapes", gate_up_stream=list(gu_stream.shape), down_stream=list(dn_stream.shape), rows_per_expert=rows_e)
    W2 = L.to_device_stream(device, gu_stream, host_round=True)
    W1 = L.to_device_stream(device, gu_stream[:rows_e].contiguous(), host_round=True)
    D1 = L.to_device_stream(device, dn_stream[: dn_stream.shape[0] // E].contiguous(), host_round=True)
    _rec("placement", memory_config=str(W2.memory_config())[:300])

    # x' [2M, 12288]: rows 0..M-1 have x's K tile k at tile column 2k (gate), rows M..2M-1 at 2k + 1 (up)
    xt = x.reshape(M, H // 32, 32)
    xp = torch.zeros(2 * M, H // 32, 2, 32)
    xp[:M, :, 0] = xt
    xp[M:, :, 1] = xt
    xp = xp.reshape(2 * M, 2 * H)
    x_tt = ttnn.from_torch(xp[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def expect_gu(e):
        return torch.cat((x @ gq[e].T, x @ uq[e].T), dim=0)

    def probe_a():
        y = ttnn.to_torch(ttnn.matmul(x_tt, W1)).float()[0, 0]
        return dict(pcc=_pcc(y, expect_gu(0)), note="whole one-expert stream as in1, x zero-interleaved (2x MACs)")

    _try("A_matmul_one_expert_stream", probe_a)

    def probe_b():
        Wb = ttnn.reshape(W2, (1, E, rows_e, 256))
        xb = ttnn.from_torch(
            xp[None, None].expand(1, E, -1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        y = ttnn.to_torch(ttnn.matmul(xb, Wb)).float()[0]
        return dict(
            pcc=min(_pcc(y[e], expect_gu(e)) for e in range(E)), view_memory_config=str(Wb.memory_config())[:200]
        )

    _try("B_reshape_batched_matmul", probe_b)

    def probe_c():
        Ws = ttnn.slice(W2, [0, 0, rows_e, 0], [1, 1, 2 * rows_e, 256])
        y = ttnn.to_torch(ttnn.matmul(x_tt, Ws)).float()[0, 0]
        return dict(pcc=_pcc(y, expect_gu(1)), slice_memory_config=str(Ws.memory_config())[:200])

    _try("C_slice_then_matmul", probe_c)

    act = torch.randn(M, Ic, generator=g).to(torch.bfloat16).float()

    def probe_d():
        Db = ttnn.reshape(D1, (1, 24, 256, 256))
        ab = ttnn.from_torch(
            act[None, None].expand(1, 24, -1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        y = ttnn.to_torch(ttnn.matmul(ab, Db)).float()[0]  # [24 (c), M, 256 (b tile, 32)]
        y = y.reshape(24, M, 8, 32).permute(1, 2, 0, 3).reshape(M, H)  # column 24 b + c
        return dict(pcc=_pcc(y, act @ dq[0].T))

    _try("D_down_reshape_batched_matmul", probe_d)

    def probe_e():  # the down stream as one matmul with N = 256 per c block: only c = 0's rows
        Ds = ttnn.slice(D1, [0, 0, 0, 0], [1, 1, 256, 256])
        a_tt = ttnn.from_torch(act[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        y = ttnn.to_torch(ttnn.matmul(a_tt, Ds)).float()[0, 0]
        cols = torch.cat([torch.arange(32) + 32 * (24 * b) for b in range(8)])
        return dict(pcc=_pcc(y, (act @ dq[0].T)[:, cols]))

    _try("E_down_slice_c0_matmul", probe_e)

    def probe_f():  # the decode-style DRAM-sharded matmul: the only stock config that takes a width-sharded DRAM in1
        m = 32
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        k = 2 * H  # the whole one-expert stream is K = 12288
        in0_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (m, k // 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (m, 256 // 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        xs = ttnn.from_torch(
            xp[None, None, :m], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mc
        )
        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k // 32 // 8 // 4, per_core_M=1, per_core_N=1, fused_activation=None
        )
        y = ttnn.to_torch(ttnn.matmul(xs, W1, program_config=cfg, memory_config=out_mc)).float()[0, 0]
        return dict(pcc=_pcc(y, (x @ gq[0].T)[:m]), note="M = 32 rows in L1, whole stream tensor = one expert")

    _try("F_dram_sharded_matmul_one_expert", probe_f)

    def probe_g():  # the same config on the 2-expert stream: K would be both experts' rows; no per-expert offset
        Ws = ttnn.reshape(W2, (1, 1, 2 * rows_e, 256))
        return dict(
            note="stock matmul has no in1 row offset; K of a multi-expert stream is every expert", shape=list(Ws.shape)
        )

    _try("G_multi_expert_offset", probe_g)
