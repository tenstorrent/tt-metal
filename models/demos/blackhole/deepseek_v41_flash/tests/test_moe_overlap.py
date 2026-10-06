"""Single-layer stage timing + PCC of the MoE collectives and their fused / overlapped alternatives (pf_moeoverlap).
Env: DSV41_UM_LAYER (3), DSV41_UM_N tokens per mesh row (4096), DSV41_UM_REAL=1 (N=512: real post-norm FFN inputs of the S=128 dump), DSV41_UM_RING (1),
DSV41_UM_STAGES (1: per-stage times), DSV41_MO_VARIANTS (comma list of reduce modes to compare, default hidden,a2a)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import moe_weights as mw
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import (
    COLS,
    ROWS,
    DSV41UnifiedMoE,
    get_shared,
    reduce_scatter_tokens,
    route_cols,
)
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 400_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_overlap(mesh_device):
    md = mesh_device
    layer = int(os.environ.get("DSV41_UM_LAYER", "3"))
    N = int(os.environ.get("DSV41_UM_N", "4096"))
    w = mw.load_moe_layer(layer)
    blk = DSV41MoEBlock(md, w, batch_per_device=32, gate_bias_shift=0.0)
    blk.warmup()
    get_shared(md, N)
    ring = os.environ.get("DSV41_UM_RING", "1") == "1"
    um = (
        DSV41UnifiedMoE(md, layer, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
        if ring
        else DSV41UnifiedMoE(md, layer)
    )
    torch.manual_seed(0)
    x = torch.randn(ROWS, N, D).to(torch.bfloat16)
    if os.environ.get(
        "DSV41_UM_REAL"
    ):  # real post-norm FFN inputs of the CPU dumps: N=512 (S=128 x 16 users) or N=4096 (one 4096-token user, rolled per row)
        if N == 512:
            d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s128/layer_{layer}.pt", mmap=True)["prefill"]["ffn_in"]
            x = d_.float().reshape(ROWS, N, D).to(torch.bfloat16)
        else:
            assert N == 4096
            d_ = torch.load(f"/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1f/layer_{layer}.pt", mmap=True)["prefill"][
                "ffn_in"
            ]
            x = torch.stack([d_.float().reshape(N, D).roll(r * 1024, 0) for r in range(ROWS)]).to(torch.bfloat16)
    n_own = N // COLS
    xo = x.reshape(ROWS, N // 256, COLS, 32, D).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, D)
    xod = ttnn.from_torch(
        xo,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
    )
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    dev = lambda t, r=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS + c])
    free = lambda *ts: [ttnn.deallocate(t) for t in ts]

    # ---- intermediates shared by the stage timings
    h = mc.allgather(xod, ccl, axis=1, dim=2)
    sc_all, ix_all, junk = route_cols(blk.gate, xod, h, mc, ccl)
    x_rm = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
    part = um.forward(x_rm, sc_all, ix_all)
    ttnn.synchronize_device(md)

    def T(name, fn):
        print(f"MO time N={N} {name:34s} {chain_ms(md, fn):.3f} ms", flush=True)

    # ---- router variants: batched own-row router must be bit-identical to the 32-row slices; AG in RM must equal AG(tile) + untilize
    def run_route(mode):
        os.environ["DSV41_UM_ROUTER_TMP"] = mode
        os.environ["DSV41_UNI_ROUTER"] = mode
        a, b, j = route_cols(blk.gate, xod, h, mc, ccl)
        ttnn.synchronize_device(md)
        return a, b

    ra = run_route("slices")
    rb = run_route("batched")
    for r in (0, 3):
        for c in (0, 5):
            ia, ib = dev(ra[1], r, c).long().reshape(-1, 6), dev(rb[1], r, c).long().reshape(-1, 6)
            wa, wb = dev(ra[0], r, c).float().reshape(-1, 6), dev(rb[0], r, c).float().reshape(-1, 6)
            print(
                f"MO router batched vs slices row {r} col {c}: idx equal {bool((ia == ib).all())}, weights max|d| {(wa - wb).abs().max():.6f}",
                flush=True,
            )
    for mode in ("slices", "batched"):
        os.environ["DSV41_UNI_ROUTER"] = mode

        def f_route():
            a, b, j = route_cols(blk.gate, xod, h, mc, ccl)
            free(a, b)

        print(f"MO time N={N} router+packed AG [{mode}]: {chain_ms(md, f_route):.3f} ms", flush=True)
    os.environ["DSV41_UNI_ROUTER"] = "slices"
    xr_a = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
    xr_b = ttnn.reshape(mc.allgather(ttnn.to_layout(xod, ttnn.ROW_MAJOR_LAYOUT), ccl, axis=1, dim=2), [1, N, D])
    ttnn.synchronize_device(md)
    print(f"MO AG(RM) equal to AG(tile)+untilize: {bool(torch.equal(dev(xr_a, 1, 2), dev(xr_b, 1, 2)))}", flush=True)
    T(
        "AG tile + untilize",
        lambda: ttnn.deallocate(
            ttnn.reshape(ttnn.to_layout(mc.allgather(xod, ccl, axis=1, dim=2), ttnn.ROW_MAJOR_LAYOUT), [1, N, D])
        ),
    )
    T(
        "untilize own + AG RM",
        lambda: ttnn.deallocate(
            ttnn.reshape(mc.allgather(ttnn.to_layout(xod, ttnn.ROW_MAJOR_LAYOUT), ccl, axis=1, dim=2), [1, N, D])
        ),
    )

    # ---- PCC of the reduction variants (same partial sums) against the host fp32 sum of the 8 column partials
    modes = os.environ.get("DSV41_MO_VARIANTS", "hidden").split(",")
    outs = {}
    for m in modes:
        outs[m] = reduce_scatter_tokens(part, ccl, mode=m, free=False)
    ttnn.synchronize_device(md)
    for r in (0, 3):
        pt = torch.stack([dev(part, r, c).float().reshape(N, D) for c in range(COLS)])  # [col, N, D] partials
        for c in (0, 5):
            ref = pt[:, c * n_own : (c + 1) * n_own].sum(0)
            for m in modes:
                got = dev(outs[m], r, c).float().reshape(n_own, D)
                print(f"MO pcc N={N} row {r} col {c} mode {m}: vs host fp32 sum {pcc(got, ref):.6f}", flush=True)
            if len(modes) > 1:
                a = dev(outs[modes[0]], r, c).float().reshape(n_own, D)
                b = dev(outs[modes[1]], r, c).float().reshape(n_own, D)
                print(f"MO pcc N={N} row {r} col {c}: {modes[0]} vs {modes[1]} {pcc(a, b):.6f}", flush=True)

    # ---- timings (traced chains, per call)
    T("all_gather hidden", lambda: ttnn.deallocate(mc.allgather(xod, ccl, axis=1, dim=2)))

    T("to RM", lambda: ttnn.deallocate(ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [1, N, D])))
    if os.environ.get("DSV41_UM_STAGES", "1") == "1":
        prev = 0.0
        for k, name in (
            (1, "bincount+cumsum"),
            (2, "+dispatch"),
            (3, "+experts"),
            (4, "+combine"),
            (99, "+post_combine_reduce"),
        ):

            def f(k=k):
                r_ = um.forward(x_rm, sc_all, ix_all, upto=k)
                for t_ in r_ if isinstance(r_, tuple) else (r_,):
                    ttnn.deallocate(t_)

            t_ = chain_ms(md, f)
            print(f"MO stage N={N} {name:24s} cumulative {t_:.3f} (+{t_ - prev:.3f}) ms", flush=True)
            prev = t_
    shd = get_shared(md, N)
    if os.environ.get("DSV41_MO_WSWEEP", "1") == "1":
        for wk in (1, 2, 3, 4):
            shd.workers = wk

            def fd(k=2):
                r_ = um.forward(x_rm, sc_all, ix_all, upto=2)
                for t_ in r_ if isinstance(r_, tuple) else (r_,):
                    ttnn.deallocate(t_)

            try:
                print(f"MO time N={N} bincount+dispatch workers/sender={wk}: {chain_ms(md, fd):.3f} ms", flush=True)
            except Exception as e:  # noqa
                print(f"MO workers={wk} failed: {str(e)[:120]}", flush=True)
        shd.workers = 2
    if os.environ.get("DSV41_MO_TOPO", "1") == "1":
        # dispatch / combine topology over the 4 rows: Linear (default) vs Ring (wrap link); outputs compared bit-for-bit
        base_part = um.forward(x_rm, sc_all, ix_all)
        ttnn.synchronize_device(md)
        for tp_name in ("ring", "linear"):
            os.environ["DSV41_UNI_TOPO"] = tp_name

            def fk():
                ttnn.deallocate(um.forward(x_rm, sc_all, ix_all))

            try:
                got = um.forward(x_rm, sc_all, ix_all)
                ttnn.synchronize_device(md)
                same = all(torch.equal(dev(base_part, r, c), dev(got, r, c)) for r in (0, 3) for c in (0, 5))
                print(f"MO topology {tp_name}: output equal to default {same}", flush=True)
                print(
                    f"MO time N={N} dispatch+combine topology={tp_name} full MoE: {chain_ms(md, fk):.3f} ms", flush=True
                )
            except Exception as e:  # noqa
                print(f"MO topology {tp_name} failed: {str(e)[:200]}", flush=True)
        os.environ["DSV41_UNI_TOPO"] = "linear"
    for m in modes:
        T(f"reduce mode {m}", lambda m=m: ttnn.deallocate(reduce_scatter_tokens(part, ccl, mode=m, free=False)))
    T(
        "reduce_scatter(hidden) only",
        lambda: ttnn.deallocate(
            ttnn.reduce_scatter(
                part,
                dim=3,
                cluster_axis=1,
                num_links=ccl.num_links,
                topology=ccl.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ),
    )
    r_ = ttnn.reduce_scatter(
        part,
        dim=3,
        cluster_axis=1,
        num_links=ccl.num_links,
        topology=ccl.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    T(
        "all_to_all(hidden->tokens) only",
        lambda: ttnn.deallocate(
            ttnn.experimental.all_to_all_async_generic(
                r_, in_dim=3, out_dim=2, num_links=ccl.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
            )
        ),
    )
    T(
        "all_to_all(tokens->hidden) only",
        lambda: ttnn.deallocate(
            ttnn.experimental.all_to_all_async_generic(
                part, in_dim=2, out_dim=3, num_links=ccl.num_links, topology=ttnn.Topology.Ring, cluster_axis=1
            )
        ),
    )
