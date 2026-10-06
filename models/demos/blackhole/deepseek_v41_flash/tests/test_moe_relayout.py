"""Single-layer check of tt/moe_relayout.py: unified prefill weights -> moe_compute decode layout (tile-exact vs the decode cache) and back; copy times.
Env: DSV41_RL_LAYER (3), DSV41_RL_DEVS (devices compared, '0,9,31')."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import moe_relayout as rl
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import build_expert_weights


def cmp(a, b, devs, name):
    ok = True
    for d in devs:
        ta = ttnn.to_torch(ttnn.get_device_tensors(a)[d]).float()
        tb = ttnn.to_torch(ttnn.get_device_tensors(b)[d]).float()
        bad = (ta != tb).sum().item()
        print(f"  {name} dev {d}: shape {tuple(ta.shape)} mismatching elements {bad} / {ta.numel()}", flush=True)
        ok &= bad == 0
    return ok


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.timeout(3600)
def test_moe_relayout(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("DSV41_RL_LAYER", "3"))
    devs = [int(x) for x in os.environ.get("DSV41_RL_DEVS", "0,9,31").split(",")]
    gate, up, down = build_expert_weights(md, layer)
    mc = rl.decode_mem_configs(md)
    d = mw.expert_cache_dir(layer)
    ref0 = ttnn.to_device(ttnn.load_tensor(f"{d}/moe_w0_w1_bfp8.tensorbin"), md, memory_config=mc[0])
    ref1 = ttnn.to_device(ttnn.load_tensor(f"{d}/moe_w2_bfp8.tensorbin"), md, memory_config=mc[1])
    out0, out1 = rl.alloc_decode(md)
    print("decode shapes", out0.shape, out1.shape, ref0.shape, ref1.shape, flush=True)
    rl.relayout(md, 0, gate, up, down, out0, out1)
    ttnn.synchronize_device(md)
    assert cmp(out0, ref0, devs, "w0_w1") and cmp(
        out1, ref1, devs, "w2"
    ), "unified -> decode differs from the decode cache"
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        rl.relayout(md, 0, gate, up, down, out0, out1)
    ttnn.synchronize_device(md)
    print(f"unified -> decode: {(time.perf_counter() - t0) / n * 1e3:.2f} ms per layer", flush=True)
    # decode -> unified into fresh tensors, compare against the originals
    g2 = [
        ttnn.empty(
            gate[0].shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=gate[0].memory_config(),
        )
        for _ in gate
    ]
    u2 = [
        ttnn.empty(
            gate[0].shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=gate[0].memory_config(),
        )
        for _ in gate
    ]
    d2 = [
        ttnn.empty(
            down[0].shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=down[0].memory_config(),
        )
        for _ in gate
    ]
    rl.relayout(md, 1, g2, u2, d2, ref0, ref1)
    ttnn.synchronize_device(md)
    ok = True
    for e in (0, 5, 11):
        ok &= (
            cmp(g2[e], gate[e], devs[:1], f"gate{e}")
            and cmp(u2[e], up[e], devs[:1], f"up{e}")
            and cmp(d2[e], down[e], devs[:1], f"down{e}")
        )
    t0 = time.perf_counter()
    for _ in range(n):
        rl.relayout(md, 1, g2, u2, d2, ref0, ref1)
    ttnn.synchronize_device(md)
    print(f"decode -> unified: {(time.perf_counter() - t0) / n * 1e3:.2f} ms per layer", flush=True)
    assert ok
    # address stability across a free + re-allocate cycle (traces capture weight addresses)
    a_old = [t.buffer_address() for t in gate + up + down]
    b_old = (out0.buffer_address(), out1.buffer_address())
    for t in gate + up + down:
        ttnn.deallocate(t)
    out0.deallocate(True)
    out1.deallocate(True)
    new0, new1 = rl.alloc_decode(md)
    gate2 = [
        ttnn.empty(
            g2[0].shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=g2[0].memory_config()
        )
        for _ in range(12)
    ]
    print(
        "decode addr old/new",
        b_old,
        (new0.buffer_address(), new1.buffer_address()),
        "unified gate0 old/new",
        a_old[0],
        gate2[0].buffer_address(),
        flush=True,
    )


def _empty_like_unified(md, gate0, down0):
    return ttnn.empty(
        gate0.shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=gate0.memory_config()
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.timeout(3600)
def test_inplace_cycle(mesh_device):
    """One persistent DRAM region per layer, rewritten at each phase switch: unified (36 tensors) + spacer <-> decode (2 tensors), via a one-layer staging buffer.
    Checks that every tensor comes back at its ORIGINAL address (what a trace captured on it needs)."""
    md = mesh_device
    layer = int(os.environ.get("DSV41_RL_LAYER", "3"))
    stg = rl.alloc_decode(
        md
    )  # ONE persistent staging buffer (one layer), allocated first so it never sits inside the per-layer regions
    gate, up, down = build_expert_weights(md, layer)
    sh = rl.decode_shapes()
    d_bytes = (sh[0][3] * sh[0][4] // 32 * 4 * sh[0][2] * 8 + sh[1][3] * sh[1][4] // 32 * 4 * sh[1][2] * 8) * rl.TB
    u_bytes = 12 * 3 * 160 * 72 * rl.TB
    spacer_bytes = d_bytes - u_bytes
    PAGE = 4096

    def spacer():
        return ttnn.empty(
            (spacer_bytes // PAGE // 1, PAGE // 4),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def alloc_unified():
        g, u, d = [], [], []
        for _ in range(12):
            g.append(_empty_like_unified(md, gate[0], down[0]))
            u.append(_empty_like_unified(md, gate[0], down[0]))
            d.append(
                ttnn.empty(
                    down[0].shape,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=down[0].memory_config(),
                )
            )
        return g, u, d

    addr_u = [t.buffer_address() for t in gate + up + down]
    sp = spacer()
    print(
        "U first/last",
        min(addr_u),
        max(addr_u),
        "spacer",
        sp.buffer_address(),
        "d_bytes/chip MiB",
        d_bytes / 2**20,
        "u",
        u_bytes / 2**20,
        flush=True,
    )
    ref = {}
    for e in (0, 7):
        ref[e] = [ttnn.to_torch(ttnn.get_device_tensors(t[e])[5]).float() for t in (gate, up, down)]
    for cyc in range(3):
        # unified -> decode
        t0 = time.perf_counter()
        rl.relayout(md, 0, gate, up, down, *stg)
        for t in gate + up + down:
            ttnn.deallocate(t)
        ttnn.deallocate(sp)
        D = rl.alloc_decode(md)
        ttnn.copy(stg[0], D[0])
        ttnn.copy(stg[1], D[1])
        ttnn.synchronize_device(md)
        t1 = time.perf_counter()
        print(
            f"switch unified->decode (staged, incl. alloc): {(t1 - t0) * 1e3:.1f} ms; decode addrs {[t.buffer_address() for t in D]} (hole start {min(addr_u)})",
            flush=True,
        )
        # decode -> unified
        t0 = time.perf_counter()
        ttnn.copy(D[0], stg[0])
        ttnn.copy(D[1], stg[1])
        for t in D:
            ttnn.deallocate(t)
        g2, u2, d2 = alloc_unified()
        sp2 = spacer()
        rl.relayout(md, 1, g2, u2, d2, *stg)
        ttnn.synchronize_device(md)
        print(f"switch decode->unified (staged, incl. alloc): {(time.perf_counter() - t0) * 1e3:.1f} ms", flush=True)
        addr_u2 = [t.buffer_address() for t in g2 + u2 + d2]
        same = addr_u2 == addr_u
        print("unified addresses identical after the cycle:", same, "spacer", sp2.buffer_address(), flush=True)
        for e in (0, 7):
            for i, t in enumerate((g2, u2, d2)):
                assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(t[e])[5]).float(), ref[e][i])
        assert same
        gate, up, down, sp = g2, u2, d2, sp2
