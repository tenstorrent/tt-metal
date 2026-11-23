# Batch prefill layout benchmark: DP4 x TP8 (today: one prompt per 8-chip mesh column, collectives on axis 0) vs
# DP8 x TP4 (one prompt per 4-chip mesh row, collectives on axis 1, 8 prompts per step). Runs the shipped DP4 layer's op
# sequence with its kernels and formats (rms_norm -> all_gather_async -> QKV minimal_matmul -> heads -> rope -> causal
# SDPA -> concat -> WO matmul_reduce_scatter -> add -> rms_norm -> all_gather_async -> FF13 + SwiGLU -> FF2
# matmul_reduce_scatter -> add) with random weights, all 32 chips busy, LAYERS-layer forwards back to back so the chips
# sit at the power cap, the two layouts alternating. Reports ms per layer per step and the modeled 32-prompt TTFT mean
# (all arrive at once: prompts finish in waves of 4 / 8).
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp8tp4_bench.py -s
#   env: BT_M (4096,16384), BT_LAYERS (80), BT_ROUNDS (3), BT_CFGS (tp8,tp4,tp4kv), BT_TOPO4 (Ring)
# tp4kv: tp4 plus moving each layer's KV into the decode layout (row all-gather + masked column reduce-scatter).
# tp4h: TP4 on half-columns (two 4x4 submeshes, rows 0-3 and 4-7; linear collectives on axis 0), the layout that keeps a
# prompt's KV cache inside its lane's column.
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, D, NQ_ALL, NKV_ALL, FF_ALL = 8192, 128, 64, 8, 28672
DRAM = ttnn.DRAM_MEMORY_CONFIG
START_S = {4096: 0.115, 16384: 0.22}  # measured request-to-first-step offsets
STEP_OVERHEAD_S = 0.03  # measured per-step time outside the forward (logits, sampling, scheduling)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp8tp4_bench(mesh_device, device_params):
    mesh = mesh_device
    layers = int(os.environ.get("BT_LAYERS") or 80)
    rounds = int(os.environ.get("BT_ROUNDS") or 3)
    cfgs = (os.environ.get("BT_CFGS") or "tp8,tp4,tp4kv").split(",")
    topo4 = getattr(ttnn.Topology, os.environ.get("BT_TOPO4") or "Ring")
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    subs = None

    def ck(fid, fp32=False):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
        )

    lofi, norm_ck = ck(ttnn.MathFidelity.LoFi), ck(ttnn.MathFidelity.HiFi4, True)
    sdpa_ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mmrs_ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)

    def dev(shape, dt, scale=1.0, device=None):
        return ttnn.from_torch(
            torch.randn(shape) * scale, device=device or mesh, layout=ttnn.TILE_LAYOUT, dtype=dt, memory_config=DRAM
        )

    def mm(a, w, dtype, swiglu=False, kb=16):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=kb,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=dtype,
            fuse_swiglu=swiglu,
        )

    tm = torch.zeros(1, 1, 32, 32)
    tm[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    tm[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1

    def build_one(tp, M, d, axis, topo):
        nq, nkv, ff = NQ_ALL // tp, NKV_ALL // tp, FF_ALL // tp
        return {
            "tp": tp,
            "axis": axis,
            "topo": topo,
            "nq": nq,
            "nkv": nkv,
            "M": M,
            "dev": d,
            "sems": [[ttnn.create_global_semaphore(d, crs, 0) for _ in range(2)] for _ in range(2)],
            "bars": [ttnn.create_global_semaphore(d, crs, 0) for _ in range(2)],
            "agi": 0,
            "trans": ttnn.from_torch(tm, device=d, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            "wqkv": dev((1, 1, H, (nq + 2 * nkv) * D), ttnn.bfloat8_b, 0.02, d),
            "wo": dev((nq * D, H), ttnn.bfloat8_b, 0.02, d),  # 2D, as matmul_reduce_scatter_prefill needs
            "w13": dev((1, 1, H, 2 * ff), ttnn.bfloat4_b, 0.02, d),
            "w2": dev((ff, H), ttnn.bfloat8_b, 0.02, d),
            "x": dev((1, 1, M // tp, H), ttnn.bfloat16, 1.0, d),
            "cos": dev((1, 1, M, D), ttnn.bfloat16, 1.0, d),
            "sin": dev((1, 1, M, D), ttnn.bfloat16, 1.0, d),
        }

    def build(cfg, M):
        nonlocal subs
        if cfg == "tp8":
            return [build_one(8, M, mesh, 0, ttnn.Topology.Ring)]
        if cfg in ("tp4", "tp4kv"):
            st = build_one(4, M, mesh, 1, topo4)
            st["kvmove"] = cfg == "tp4kv"
            return [st]
        if subs is None:
            subs = mesh.create_submeshes(ttnn.MeshShape(4, 4))
            logger.info(f"BENCH submeshes: {[tuple(s_.shape) for s_ in subs]}")
        return [build_one(4, M, s_, 0, ttnn.Topology.Linear) for s_ in subs]

    def ag(st, x):
        i = st["agi"]
        st["agi"] ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=st["sems"][i],
            num_links=2,
            barrier_semaphore=st["bars"][i],
            memory_config=DRAM,
            topology=st["topo"],
            cluster_axis=st["axis"],
            num_workers_per_link=8,
        )

    def mmrs(st, a, w):
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            a,
            w,
            cluster_axis=st["axis"],
            scatter_dim=-2,
            topology=st["topo"],
            num_links=2,
            compute_kernel_config=mmrs_ck,
            memory_config=DRAM,
        )

    def norm(x):
        return ttnn.rms_norm(x, epsilon=1e-5, memory_config=DRAM, compute_kernel_config=norm_ck, dtype=ttnn.bfloat8_b)

    def layer(st, x):
        M = st["M"]
        h = norm(x)
        hg = ag(st, h)
        ttnn.deallocate(h)
        qkv = mm(hg, st["wqkv"], ttnn.bfloat16)
        ttnn.deallocate(hg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=st["nq"], num_kv_heads=st["nkv"], transpose_k_heads=False, memory_config=DRAM
        )
        ttnn.deallocate(qkv)
        qd = ttnn.bfloat8_b if M >= 8192 else None
        q2 = ttnn.experimental.rotary_embedding_llama(
            q, st["cos"], st["sin"], st["trans"], is_decode_mode=False, **({"dtype": qd} if qd else {})
        )
        k8 = ttnn.experimental.rotary_embedding_llama(
            k, st["cos"], st["sin"], st["trans"], is_decode_mode=False, dtype=ttnn.bfloat8_b
        )
        ttnn.deallocate(q), ttnn.deallocate(k)
        v8 = ttnn.typecast(v, ttnn.bfloat8_b)
        ttnn.deallocate(v)
        if st.get("kvmove"):
            # KV into the decode layout (one kv head per chip on the lane's column): all-gather the row's kv heads
            # (axis 1), keep them only on the lane's column, slot the lane's 2 prompts, reduce-scatter over heads down
            # the column (axis 0). Zeros elsewhere keep the sum exact. Masks are constants here (same cost).
            kv = ttnn.concat([k8, v8], dim=3)  # [1, nkv, M, 256]
            g = ttnn.all_gather(kv, 1, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=2, memory_config=DRAM)
            ttnn.deallocate(kv)
            a0, a1 = ttnn.multiply(g, 1.0, memory_config=DRAM), ttnn.multiply(g, 0.0, memory_config=DRAM)
            ttnn.deallocate(g)
            t = ttnn.concat([a0, a1], dim=2)  # [1, 8, 2M, 256]
            ttnn.deallocate(a0), ttnn.deallocate(a1)
            mine = ttnn.reduce_scatter(
                t, 1, cluster_axis=0, topology=ttnn.Topology.Ring, num_links=2, memory_config=DRAM
            )
            ttnn.deallocate(t)
            ttnn.deallocate(mine)  # [1, 1, 2M, 256]: head k of both of the lane's prompts, ready for paged_fill_cache
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512, exp_approx_mode=True
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            q2, k8, v8, is_causal=True, program_config=cfg, compute_kernel_config=sdpa_ck, memory_config=DRAM
        )
        ttnn.deallocate(q2), ttnn.deallocate(k8), ttnn.deallocate(v8)
        a = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        a8 = a if a.dtype == ttnn.bfloat8_b else ttnn.typecast(a, ttnn.bfloat8_b)
        if a8 is not a:
            ttnn.deallocate(a)
        r1 = mmrs(st, a8, st["wo"])
        ttnn.deallocate(a8)
        x1 = ttnn.add(x, r1, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r1)
        h = norm(x1)
        hg = ag(st, h)
        ttnn.deallocate(h)
        try:
            f = mm(hg, st["w13"], ttnn.bfloat8_b, swiglu=True, kb=32)
        except RuntimeError:
            f = mm(hg, st["w13"], ttnn.bfloat8_b, swiglu=True, kb=16)
        ttnn.deallocate(hg)
        r2 = mmrs(st, f, st["w2"])
        ttnn.deallocate(f)
        x2 = ttnn.add(x1, r2, dtype=ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(r2), ttnn.deallocate(x1)
        return x2

    def forward(group):
        # one layer per state in turn, so submeshes (tp4h) run side by side
        xs = [st["x"] for st in group]
        for i in range(layers):
            for j, st in enumerate(group):
                y = layer(st, xs[j])
                if xs[j] is not st["x"]:
                    ttnn.deallocate(xs[j])
                xs[j] = y
        for x in xs:
            ttnn.deallocate(x)

    def sync():
        ttnn.synchronize_device(mesh)
        for s_ in subs or []:
            ttnn.synchronize_device(s_)

    for M in [int(m) for m in (os.environ.get("BT_M") or "4096,16384").split(",")]:
        sts = {}
        for c in cfgs:
            try:
                sts[c] = build(c, M)
                forward(sts[c])  # compile
                sync()
                logger.info(f"BENCH M={M} {c}: compiled")
            except Exception as e:
                logger.warning(f"BENCH M={M} {c} ERROR {str(e)[:400]}")
                sts.pop(c, None)
        times = {c: [] for c in sts}
        for r in range(rounds):
            for c, st in sts.items():
                sync()
                t0 = time.perf_counter()
                forward(st)
                sync()
                times[c].append((time.perf_counter() - t0) / layers * 1e3)
        for c, ts in times.items():
            tp = 8 if c == "tp8" else 4
            per_step = sum(ts[1:] or ts) / len(ts[1:] or ts)  # steady state: skip the first (cool) round
            prompts = 32 // tp
            waves = 32 // prompts
            step_s = per_step * layers / 1e3 + STEP_OVERHEAD_S
            mean = START_S.get(M, 0.115) + step_s * (waves + 1) / 2
            logger.info(
                f"BENCH M={M} {c}: rounds " + " ".join(f"{t:.3f}" for t in ts) + f" ms/layer; {prompts} prompts "
                f"per step, {per_step / prompts:.3f} ms per prompt-layer; step {step_s:.3f} s; modeled 32-prompt "
                f"TTFT mean {mean:.3f} s"
            )
        for group in sts.values():
            for st in group:
                for k in ("wqkv", "wo", "w13", "w2", "x", "cos", "sin", "trans"):
                    ttnn.deallocate(st[k])
