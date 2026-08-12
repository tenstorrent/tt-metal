# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated probes for the op-merging decisions, at the layer's real shapes.

Three questions the end-to-end A/B cannot answer cleanly on its own:

1. Where should the SiLU / sigmoid go — a matmul pack-time ``activation=``, or
   an input activation on the binary that consumes the result?
2. Is ``ttnn.experimental.minimal_matmul(fuse_swiglu=True)`` (one kernel for
   gate+up+silu+mul) worth the second, differently-laid-out copy of the
   531 MB gate/up weight it needs?
3. Does the RoPE dedicated op actually beat the spelled-out rotate-half?

Output: ``doc/fused_decoder/logs/op_merge_probes.log``.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

HIDDEN = 6656
INTERMEDIATE = 19968
HEAD_DIM = 128
N_HEADS = 32
PREFILL_ROWS = 8192
DECODE_ROWS = 32


def timed(mesh, label, fn, iters):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(iters):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - start) / iters * 1e3
    print(f"PROBE {label:52s} {dt:9.3f} ms", flush=True)
    return dt


def probe_activation_placement(mesh):
    """SiLU on the matmul vs on the multiply that consumes it."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, PREFILL_ROWS, INTERMEDIATE).to(torch.bfloat16)
    b = torch.randn(1, 1, PREFILL_ROWS, INTERMEDIATE).to(torch.bfloat16)
    ta, tb = (
        ttnn.from_torch(
            t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for t in (a, b)
    )
    ref = (a.float() * torch.sigmoid(a.float())) * b.float()

    def unfused():
        s = ttnn.silu(ta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.mul(s, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        return out

    def merged():
        return ttnn.mul(
            ta, tb, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for label, fn in (("silu + mul (unfused)", unfused), ("mul(input_a_activations=[SILU])", merged)):
        out = fn()
        print(f"PROBE {label:52s} PCC={comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]}", flush=True)
        ttnn.deallocate(out)
        timed(mesh, label, fn, 10)

    # The matmul's own pack-time activation, same shape as the MLP gate matmul.
    w = ttnn.from_torch(
        (torch.randn(1, 1, HIDDEN, INTERMEDIATE) * 0.02).to(torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, PREFILL_ROWS, HIDDEN).to(torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    timed(mesh, "ttnn.linear plain", lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG), 5)
    timed(
        mesh,
        'ttnn.linear activation="silu"',
        lambda: ttnn.linear(x, w, activation="silu", memory_config=ttnn.DRAM_MEMORY_CONFIG),
        5,
    )
    for t in (ta, tb, w, x):
        ttnn.deallocate(t)


def probe_fused_swiglu(mesh):
    """minimal_matmul(fuse_swiglu=True) vs two matmuls + a merged multiply."""
    torch.manual_seed(0)
    wg = (torch.randn(HIDDEN, INTERMEDIATE) * 0.02).to(torch.bfloat16)
    wu = (torch.randn(HIDDEN, INTERMEDIATE) * 0.02).to(torch.bfloat16)
    packed = prepare_for_fused_swiglu(torch.cat([wg, wu], dim=-1).float(), 1, gate_is_first=True).to(torch.bfloat16)
    tp = ttnn.from_torch(
        packed.reshape(1, 1, HIDDEN, 2 * INTERMEDIATE),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tg = ttnn.from_torch(
        wg.reshape(1, 1, HIDDEN, INTERMEDIATE),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tu = ttnn.from_torch(
        wu.reshape(1, 1, HIDDEN, INTERMEDIATE),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for rows, iters in ((PREFILL_ROWS, 3), (DECODE_ROWS, 20)):
        x = (torch.randn(1, 1, rows, HIDDEN) * 0.1).to(torch.bfloat16)
        tx = ttnn.from_torch(
            x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g = x.float().reshape(rows, HIDDEN) @ wg.float()
        ref = (g * torch.sigmoid(g)) * (x.float().reshape(rows, HIDDEN) @ wu.float())

        def separate(shipped: bool):
            def run():
                mm = (
                    (lambda a, b: ttnn.experimental.minimal_matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    if shipped
                    else (lambda a, b: ttnn.linear(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                )
                gate, up = mm(tx, tg), mm(tx, tu)
                out = ttnn.mul(
                    gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                return out

            return run

        def fused():
            return ttnn.experimental.minimal_matmul(tx, tp, fuse_swiglu=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        for label, fn in (
            (f"rows={rows} linear x2 + mul(SILU)  [decode path]", separate(False)),
            (f"rows={rows} minimal x2 + mul(SILU)  [prefill path]", separate(True)),
            (f"rows={rows} minimal fuse_swiglu", fused),
        ):
            out = fn()
            pcc = comp_pcc(ref, ttnn.to_torch(out).reshape(rows, -1).float(), 0.99)[1]
            ttnn.deallocate(out)
            print(f"PROBE {label:52s} PCC={pcc}", flush=True)
            timed(mesh, label, fn, iters)
        ttnn.deallocate(tx)
    for t in (tp, tg, tu):
        ttnn.deallocate(t)


def probe_rope(mesh):
    """rotary_embedding_hf vs the spelled-out rotate-half, prefill shapes."""
    seq = 1024
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)] * 2, dim=-1)
    cos_t, sin_t = emb.cos(), emb.sin()
    torch.manual_seed(0)
    x = torch.randn(1, N_HEADS, seq, HEAD_DIM).to(torch.bfloat16)
    half = HEAD_DIM // 2
    rot = torch.cat((-x.float()[..., half:], x.float()[..., :half]), dim=-1)
    ref = x.float() * cos_t.reshape(1, 1, seq, HEAD_DIM) + rot * sin_t.reshape(1, 1, seq, HEAD_DIM)

    tx = ttnn.from_torch(
        x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tc, ts = (
        ttnn.from_torch(
            t.reshape(1, 1, seq, HEAD_DIM).to(torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in (cos_t, sin_t)
    )

    def unfused():
        x1, x2 = tx[..., :half], tx[..., half:]
        neg = ttnn.neg(x2)
        ttnn.deallocate(x2)
        rotated = ttnn.concat([neg, x1], dim=-1)
        ttnn.deallocate(neg)
        ttnn.deallocate(x1)
        out = ttnn.add(ttnn.mul(tx, tc), ttnn.mul(rotated, ts))
        ttnn.deallocate(rotated)
        return out

    def fused():
        return ttnn.experimental.rotary_embedding_hf(tx, tc, ts, is_decode_mode=False)

    for label, fn in (("rope spelled out (7 ops)", unfused), ("rotary_embedding_hf (1 op)", fused)):
        out = fn()
        print(f"PROBE {label:52s} PCC={comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]}", flush=True)
        ttnn.deallocate(out)
        timed(mesh, label, fn, 50)
    for t in (tx, tc, ts):
        ttnn.deallocate(t)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        print("--- activation placement ---", flush=True)
        probe_activation_placement(mesh)
        print("--- fused swiglu ---", flush=True)
        probe_fused_swiglu(mesh)
        print("--- rope ---", flush=True)
        probe_rope(mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
