# Gemma4-31B row-parallel projections (BH Galaxy CP8 x TP4 per-device shapes), on a QB2 1x4 ring or a BH Galaxy 8x4:
# today's Gemma4 path (its ttnn.linear configs + ttnn.reduce_scatter rows, from kmabee/gemma4-fused-oproj-rs) vs
# ttnn.experimental.matmul_reduce_scatter_prefill(scatter_dim=-2) as that branch calls it.
# Traced; each measured replay runs the path CALLS times back to back; REPLAYS replays in one profiler window.
# Device time per path call = per chip, sum of the path's program durations; slowest chip; median over calls.
# Writes one JSON line per case to $GEMMA_OUT.
#
# Env:
#   GEMMA_OUT       results file (JSON lines, appended); required
#   GEMMA_MESH      mesh to open: 1x4 (QB2, default) or 8x4 (Galaxy: CP8 x TP4, TP on axis 1 as Gemma4 runs it)
#   GEMMA_TP_AXIS   cluster axis of the TP group (default 1)
#   GEMMA_FABRIC    ring (FABRIC_1D_RING + 8 KiB router payload, Gemma4's Galaxy setting; default on 8x4) or
#                   torus (FABRIC_2D_TORUS_XY, the op's QB2 test setting; default on 1x4)
#   GEMMA_GRID      fused op grid: GRID_2D / FOLDED_N / FOLDED_M (unset: the op's default; main's op has no grid=)
#   GEMMA_HEADROOM  fused op l1_headroom in bytes (default 65536)
#   GEMMA_CHECK     1: check the last output on every device against a torch reference (PCC 0.99)
#   GEMMA_BUILD     label stored with each result
# Example (Galaxy): GEMMA_MESH=8x4 GEMMA_GRID=FOLDED_N GEMMA_OUT=/path/out.jsonl \
#   scripts/run_safe_pytest.sh --run-all <dir>/test_zz_gemma_bench.py -s

import json
import os
import statistics

import pytest
import torch

import ttnn
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    ring_device_params,
)

CALLS = 10
REPLAYS = 5
N = 5376
MESH_SHAPE = tuple(int(v) for v in os.environ.get("GEMMA_MESH", "1x4").lower().split("x"))
TP_AXIS = int(os.environ.get("GEMMA_TP_AXIS", "1"))
TP = MESH_SHAPE[TP_AXIS]
NUM_LINKS = 2
FABRIC = os.environ.get("GEMMA_FABRIC", "ring" if MESH_SHAPE[0] * MESH_SHAPE[1] > 4 else "torus")


def device_params():
    if FABRIC == "torus":
        params = ring_device_params()
    else:  # Gemma4's Galaxy fabric (models/demos/gemma4_d_p/tests/test_factory.py)
        router = ttnn._ttnn.fabric.FabricRouterConfig()
        router.max_packet_payload_size_bytes = 8192
        params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "fabric_router_config": router}
    return {**params, "trace_region_size": 8 << 20}


# ---- copied from models/demos/gemma4_d_p/tt/matmul_config.py (kmabee/gemma4-fused-oproj-rs 694f861fbf4) ----
_PER_CORE_N_1D = 2
_MAX_PER_CORE_M = 4
_MAX_SHORT_M_TILES = 8
_SHARD_K_TILES = (8, 7, 6, 4)


def _in0_block_w(k_tiles):
    return max(d for d in range(1, min(k_tiles, 16) + 1) if k_tiles % d == 0)


def prefill_matmul_program_config(hidden_states, weight, grid_x, grid_y, fused_activation=None, fp32_dest_acc=False):
    tile = ttnn.TILE_SIZE
    m_tiles = hidden_states.padded_shape[-2] // tile
    k_tiles = hidden_states.padded_shape[-1] // tile
    n_tiles = weight.padded_shape[-1] // tile
    per_core_m = ttnn.core.divup(m_tiles, grid_y)
    if per_core_m > _MAX_PER_CORE_M:
        return None
    per_core_n = ttnn.core.divup(n_tiles, grid_x)
    subblock_w = 2 if per_core_n % 2 == 0 else 1
    max_subblock_tiles = 4 if fp32_dest_acc else 8
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=_in0_block_w(k_tiles),
        out_subblock_h=next(h for h in (4, 3, 2, 1) if per_core_m % h == 0 and h * subblock_w <= max_subblock_tiles),
        out_subblock_w=subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def prefill_1d_matmul_program_config(
    hidden_states, weight, grid, fused_activation=None, per_core_n=None, max_m_tiles=None
):
    per_core_n = per_core_n or _PER_CORE_N_1D
    tile = ttnn.TILE_SIZE
    m_tiles = hidden_states.padded_shape[-2] // tile
    k_tiles = hidden_states.padded_shape[-1] // tile
    n_tiles = weight.padded_shape[-1] // tile
    too_tall = m_tiles > (max_m_tiles or _MAX_SHORT_M_TILES)
    if too_tall or n_tiles % per_core_n or n_tiles // per_core_n > grid.x * grid.y:
        return None
    out_subblock_w = per_core_n if per_core_n <= 4 else (2 if per_core_n % 2 == 0 else 1)
    out_subblock_h = 2 if out_subblock_w <= 2 and m_tiles % 2 == 0 else 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=(
            hidden_states.memory_config().shard_spec.shape[1] // tile
            if hidden_states.is_sharded()
            else _in0_block_w(k_tiles)
        ),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def tile_rows(x):
    return x.padded_shape[-2] // ttnn.TILE_SIZE


def is_short_m(x):
    return tile_rows(x) <= _MAX_SHORT_M_TILES


def _width_sharded_l1(device, rows, num_cores, shard_tiles):
    cores = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (rows, shard_tiles * ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
    )


def short_m_sharded_memcfg(device, rows, k_tiles):
    grid = device.compute_with_storage_grid_size()
    shard = next((d for d in _SHARD_K_TILES if k_tiles % d == 0), None)
    assert shard is not None and k_tiles // shard <= grid.x * grid.y
    return _width_sharded_l1(device, rows, k_tiles // shard, shard)


def to_l1_width_sharded(x):
    memcfg = short_m_sharded_memcfg(x.device(), x.padded_shape[-2], x.padded_shape[-1] // ttnn.TILE_SIZE)
    return x if x.memory_config() == memcfg else ttnn.to_memory_config(x, memcfg)


def short_m_output_memcfg(x, weight, per_core_n=_PER_CORE_N_1D):
    n_tiles = weight.padded_shape[-1] // ttnn.TILE_SIZE
    return _width_sharded_l1(x.device(), x.padded_shape[-2], n_tiles // per_core_n, per_core_n)


# ---- attention/operations.py, mlp.py, ccl.py ----
_LOFI_PROJECTION_MIN_ROWS = 512
_MAX_SHARDED_OUTPUT_M_TILES = 16
_MAX_TUNED_REDUCE_SCATTER_IN_ROWS = 512
_SHORT_SLAB_REDUCE_SCATTER_TRANSPORT = dict(num_workers_per_link=4, chunks_per_sync=2, num_buffers_per_channel=2)
_MAX_L1_REDUCE_SCATTER_OUT_ROWS = 256
_DOWN_PER_CORE_N = 4


def projection_math_fidelity(rows):
    return ttnn.MathFidelity.LoFi if rows >= _LOFI_PROJECTION_MIN_ROWS else ttnn.MathFidelity.HiFi2


def compute_cfg(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def o_proj_today(x, w, fidelity_rows):
    """apply_output_projection: apply_attn_projection(into_reduce_scatter=True)."""
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    sharded_output = tile_rows(x) <= _MAX_SHARDED_OUTPUT_M_TILES
    xs = to_l1_width_sharded(x) if sharded_output or is_short_m(x) else x
    max_m = _MAX_SHARDED_OUTPUT_M_TILES if sharded_output else None
    pc = prefill_1d_matmul_program_config(xs, w, grid, max_m_tiles=max_m) or prefill_matmul_program_config(
        xs, w, grid.x, grid.y, fp32_dest_acc=True
    )
    cc = compute_cfg(device, projection_math_fidelity(fidelity_rows)) if pc is not None else None
    mc = short_m_output_memcfg(xs, w) if sharded_output else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.linear(xs, w, memory_config=mc, program_config=pc, compute_kernel_config=cc)
    if xs is not x:
        xs.deallocate(True)
    return out


def down_today(x, w, core_grid):
    """MLP.__call__'s down projection (_project with per_core_n=4, LoFi)."""
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    short_m = is_short_m(x)
    xs = to_l1_width_sharded(x) if short_m else x
    mc = short_m_output_memcfg(xs, w, per_core_n=_DOWN_PER_CORE_N) if short_m else ttnn.DRAM_MEMORY_CONFIG
    n_tiles = w.padded_shape[-1] // ttnn.TILE_SIZE
    grid_x = max(gx for gx in range(1, grid.x + 1) if n_tiles % gx == 0)
    pc = prefill_1d_matmul_program_config(xs, w, grid, None, _DOWN_PER_CORE_N) or prefill_matmul_program_config(
        xs, w, grid_x, grid.y, None, fp32_dest_acc=True
    )
    cc = compute_cfg(device, ttnn.MathFidelity.LoFi) if pc is not None else None
    out = ttnn.linear(
        xs,
        w,
        program_config=pc,
        compute_kernel_config=cc,
        core_grid=None if pc is not None else ttnn.CoreGrid(y=grid.y, x=grid.x),
        memory_config=mc,
    )
    if xs is not x:
        xs.deallocate(True)
    return out


def reduce_scatter_rows_today(t):
    out_rows = t.padded_shape[-2] // TP
    mc = ttnn.L1_MEMORY_CONFIG if out_rows <= _MAX_L1_REDUCE_SCATTER_OUT_ROWS else ttnn.DRAM_MEMORY_CONFIG
    transport = {}
    if t.padded_shape[-2] <= _MAX_TUNED_REDUCE_SCATTER_IN_ROWS:
        transport = dict(num_links=NUM_LINKS, **_SHORT_SLAB_REDUCE_SCATTER_TRANSPORT)
    out = ttnn.reduce_scatter(t, dim=2, cluster_axis=TP_AXIS, memory_config=mc, **transport)
    t.deallocate(True)
    return out


def fused(x, w, fidelity, headroom):
    out_rows = x.padded_shape[-2] // TP
    kwargs = {} if headroom is None else dict(l1_headroom=headroom)
    if os.environ.get("GEMMA_GRID"):
        kwargs["grid"] = getattr(ttnn.MatmulReduceScatterPrefillGrid, os.environ["GEMMA_GRID"])
    return ttnn.experimental.matmul_reduce_scatter_prefill(
        x,
        w,
        cluster_axis=TP_AXIS,
        scatter_dim=-2,
        topology=ttnn.Topology.Ring,
        num_links=NUM_LINKS,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity),
        memory_config=ttnn.L1_MEMORY_CONFIG if out_rows <= _MAX_L1_REDUCE_SCATTER_OUT_ROWS else ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    )


def last_plan():
    seen = set()

    def walk(mod, depth):
        if depth > 4 or id(mod) in seen:
            return None
        seen.add(id(mod))
        fn = getattr(mod, "_matmul_reduce_scatter_prefill_last_plan", None)
        if fn is not None:
            return fn
        for name in dir(mod):
            sub = getattr(mod, name, None)
            if type(sub).__name__ == "module" and getattr(sub, "__name__", "").startswith("ttnn"):
                found = walk(sub, depth + 1)
                if found is not None:
                    return found
        return None

    fn = walk(ttnn._ttnn, 0)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(fn()).items()} if fn else {}


# site: K, path kind; rows label -> (rows per device, fidelity rows (one request's chunk))
SITES = {"sliding_o_proj": 2048, "global_o_proj": 4096, "mlp_down": 5376}
STEPS = {"2k_256": (256, 256), "4k_512": (512, 512), "8k_1024": (1024, 1024), "4x2k_1024": (1024, 256)}
CASES = [(s, st, p) for s in SITES for st in STEPS for p in ("today", "fused")]


def run_traced(mesh_device, fn):
    fn()  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(CALLS):
        fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        def replays():
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        _, records = profile_realtime_program(mesh_device, replays, collect_all=True)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return records


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("site,step,path", CASES, ids=[f"{s}-{st}-{p}" for s, st, p in CASES])
def test_gemma_bench(mesh_device, site, step, path):
    K = SITES[site]
    rows, fid_rows = STEPS[step]
    if path == "fused" and site == "mlp_down" and step == "4x2k_1024":
        pytest.skip("same op call as 8k_1024 (MLP down is LoFi at every step)")
    headroom = int(os.environ.get("GEMMA_HEADROOM", 65536))
    torch.manual_seed(0)
    a = torch.randn((1, 1, rows, K), dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn((K, N), dtype=torch.float32) * K**-0.5).to(torch.bfloat16)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ta = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    tw = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    fidelity = ttnn.MathFidelity.LoFi if site == "mlp_down" else projection_math_fidelity(fid_rows)
    grid = mesh_device.compute_with_storage_grid_size()

    outs = []

    def call():
        if path == "fused":
            out = fused(ta, tw, fidelity, headroom)
        else:
            mm = down_today(ta, tw, grid) if site == "mlp_down" else o_proj_today(ta, tw, fid_rows)
            out = reduce_scatter_rows_today(mm)
        if outs:
            ttnn.deallocate(outs.pop())
        outs.append(out)

    records = run_traced(mesh_device, call)
    if os.environ.get("GEMMA_CHECK"):
        from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

        exp = TP * (a.float() @ ttnn.to_torch(ttnn.get_device_tensors(tw)[0]).float())
        cols = MESH_SHAPE[1]
        for i, t in enumerate(ttnn.get_device_tensors(outs[-1])):
            r, c = divmod(i, cols)
            ref = torch.chunk(exp, TP, dim=-2)[c if TP_AXIS == 1 else r]
            ok, msg = comp_pcc(ref, ttnn.to_torch(t).float().reshape(ref.shape), 0.99)
            print("GEMMA_PCC", site, step, path, i, msg)
            assert ok, msg
    plan = last_plan() if path == "fused" else {}
    # per chip, records in arrival order; the measured window holds REPLAYS * CALLS path calls
    per_chip = {}
    for r in records:
        per_chip.setdefault(r["chip_id"], []).append(r)
    nprog = {c: len(v) // (REPLAYS * CALLS) for c, v in per_chip.items()}
    assert len(set(nprog.values())) == 1 and min(nprog.values()) >= 1, {c: len(v) for c, v in per_chip.items()}
    k = next(iter(nprog.values()))
    calls = REPLAYS * CALLS
    per_call = []
    per_op = {}
    for i in range(calls):
        chip_tot = []
        for c, v in per_chip.items():
            tail = v[len(v) - calls * k :]
            grp = tail[i * k : (i + 1) * k]
            chip_tot.append(sum(r["duration_ns"] for r in grp))
            for j, r in enumerate(grp):
                src = os.path.basename(r["kernel_sources"][0]) if r["kernel_sources"] else "?"
                per_op.setdefault(f"{j}:{src}", []).append(r["duration_ns"])
        per_call.append(max(chip_tot))
    res = dict(
        site=site,
        step=step,
        path=path,
        rows=rows,
        K=K,
        N=N,
        programs=k,
        median_us=statistics.median(per_call) / 1e3,
        min_us=min(per_call) / 1e3,
        max_us=max(per_call) / 1e3,
        per_op_median_us={o: statistics.median(v) / 1e3 for o, v in per_op.items()},
        headroom=headroom if path == "fused" else None,
        plan=plan,
        build=os.environ.get("GEMMA_BUILD", "?"),
        grid=os.environ.get("GEMMA_GRID"),
        mesh=list(MESH_SHAPE),
        tp_axis=TP_AXIS,
        fabric=FABRIC,
    )
    with open(os.environ["GEMMA_OUT"], "a") as f:
        f.write(json.dumps(res) + "\n")
    print("GEMMA", json.dumps(res))
