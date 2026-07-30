# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-free bisection of a prefill determinism failure, from collectives down to one core.

The prefill determinism tests compare full-block outputs and demand bit-exactness, so a
failure says only "something on this box is not reproducible". These tests split that claim
apart without weights, a model, or a trace, in the order worth running:

  test_ccl_determinism           each TP collective alone, on one fixed input
  test_ccl_chain_determinism     a realistic chain of them, past the semaphore pool depth
  test_local_compute_determinism no collective at all -- one chip's own arithmetic
  test_local_op_determinism      which subsystem of a chip: DRAM, SFPU/pack, or matmul
  test_local_matmul_core_locality  whether a bad matmul footprint tracks a core or an address
  test_report_device_mapping     shard index -> physical device id, which is not identity

Everything from test_local_compute_determinism down replicates identical inputs to all 32
chips, so a chip that disagrees with the other 31 is the fault, and no fabric is involved.
A failure there is below tt-metal -- one chip, not the code -- but below tt-metal is not
automatically the silicon: firmware owns the operating point (AICLK, VDD, DVFS, harvesting),
so a marginal core can pass under one firmware and drift under another. Rule out the
firmware/KMD pair before calling it a bad die. Only the first two tests implicate tt-metal.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

EMB_DIM = 7168
NUM_ITERATIONS = 5


def _shards(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _first_diff(base, cur):
    """(ndiff, total, maxabs, diverging chip indices) — bit-exactness, not tolerance."""
    ndiff = sum(int((b != c).sum()) for b, c in zip(base, cur))
    total = sum(b.numel() for b in base)
    maxabs = max(float((b.float() - c.float()).abs().max()) for b, c in zip(base, cur))
    bad = [i for i, (b, c) in enumerate(zip(base, cur)) if not torch.equal(b, c)]
    return ndiff, total, maxabs, bad


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_local", [3200, 640, 128], ids=["seq3200", "seq640", "seq128"])
@pytest.mark.parametrize("op", ["legacy_rs", "async_rs", "async_rs_persist", "legacy_ag"])
def test_ccl_determinism(mesh_device, device_params, seq_local, op):
    tp_axis = 1
    tp_factor = mesh_device.shape[tp_axis]
    torch.manual_seed(0)

    # dims=(0, 1) puts a distinct slice on every chip while keeping the full EMB_DIM width
    # per chip, which is what a TP reduce consumes: 4 partials of the same shape.
    host = torch.randn(mesh_device.shape[0], tp_factor, seq_local, EMB_DIM, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
    )

    tt_ccl = get_tt_ccl(mesh_device) if op.startswith("async_rs") else None
    # A persistent intermediate holds the accumulator at one DRAM address for every call. The
    # shared-expert path already relies on that for bit-exactness, so contrasting it against the
    # fresh-per-call intermediate isolates address stability from the kernel itself.
    persist = [tt_ccl.get_shared_rs_intermediate(tt_in)] if op == "async_rs_persist" else None

    def run():
        if op == "legacy_rs":
            return ttnn.reduce_scatter(tt_in, dim=-1, cluster_axis=tp_axis, num_links=2, topology=ttnn.Topology.Linear)
        if op == "legacy_ag":
            return ttnn.all_gather(tt_in, dim=-1, cluster_axis=tp_axis, num_links=2, topology=ttnn.Topology.Linear)
        return ttnn.experimental.reduce_scatter_minimal_async(
            tt_in,
            dim=3,
            persistent_output_buffers=persist,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=tp_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=tp_axis),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
        )

    logger.info(f"{op} seq_local={seq_local} tp={tp_factor}: {NUM_ITERATIONS} iterations on one fixed input")
    baseline = None
    failures = []
    for i in range(NUM_ITERATIONS):
        out = run()
        ttnn.synchronize_device(mesh_device)
        cur = _shards(out)
        addr = out.buffer_address()
        ttnn.deallocate(out)
        if i == 0:
            baseline = cur
            logger.info(f"  iter 0: baseline captured, out_addr=0x{addr:x}")
            continue
        ndiff, total, maxabs, bad = _first_diff(baseline, cur)
        status = "BIT-EXACT" if ndiff == 0 else "DIVERGED"
        logger.info(
            f"  iter {i}: {status} ndiff={ndiff}/{total} maxabs={maxabs:.3e} "
            f"chips={len(bad)}/{len(cur)} {bad[:8]} out_addr=0x{addr:x}"
        )
        if ndiff:
            failures.append((i, ndiff, maxabs, bad))

    if failures:
        msg = "; ".join(f"iter {i}: ndiff={n} maxabs={m:.3e} chips={len(b)}" for i, n, m, b in failures)
        pytest.fail(f"{op} seq_local={seq_local} is not run-to-run deterministic: {msg}")
    logger.success(f"{op} seq_local={seq_local} bit-exact across {NUM_ITERATIONS} iterations")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_local", [3200, 128], ids=["seq3200", "seq128"])
@pytest.mark.parametrize("mode", ["chain", "chain_free", "chain_sync"])
def test_ccl_chain_determinism(mesh_device, device_params, seq_local, mode):
    """One TP collective in isolation is deterministic; a realistic chain of them may not be.

    Two conditions the single-op test could not reach, both present in every real MLA forward:

    * The RS/AG/barrier semaphore pools are depth 2 and cycle modulo 2 (tt_ccl.py), while one
      forward issues six-plus TP collectives. The third op reuses op one's handle, so a handle
      whose previous user has not fully drained is read as if it were fresh.
    * Nothing synchronizes between ops. The single-op test called synchronize_device after every
      call, which drains exactly the state a stale-handle or use-after-free bug needs to survive.

    `chain` keeps every intermediate referenced; `chain_free` deallocates each input the moment the
    op returns, so a still-running kernel's source DRAM becomes claimable; `chain_sync` drains
    between ops. chain/chain_free diverging while chain_sync stays exact isolates missing inter-op
    synchronization from the kernels themselves.
    """
    tp_axis = 1
    tp_factor = mesh_device.shape[tp_axis]
    tt_ccl = get_tt_ccl(mesh_device)
    torch.manual_seed(0)

    host = torch.randn(mesh_device.shape[0], tp_factor, seq_local, EMB_DIM, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
    )

    def rs(t):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=3,
            persistent_output_buffers=None,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=tp_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=tp_axis),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
        )

    def ag(t):
        return ttnn.experimental.all_gather_async(
            t,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=tp_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=tp_axis),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
        )

    # RS/AG alternate so the width returns to EMB_DIM each round-trip, letting the chain run
    # arbitrarily long on one shape. Three round-trips plus the legacy reduce_scatter is six
    # pool-cycled collectives, past the point where the depth-2 pools wrap.
    STEPS = [rs, ag, rs, ag, rs, ag]

    def run():
        t = tt_in
        for step in STEPS:
            prev = t
            t = step(t)
            if mode == "chain_free" and prev is not tt_in:
                ttnn.deallocate(prev)
            if mode == "chain_sync":
                ttnn.synchronize_device(mesh_device)
        # TtFfn's collective: allocates its intermediate fresh and creates its own semaphores, so it
        # closes the chain with the same op the dense FFN actually runs.
        return ttnn.reduce_scatter(t, dim=-1, cluster_axis=tp_axis, num_links=2, topology=ttnn.Topology.Linear)

    logger.info(f"{mode} seq_local={seq_local}: {len(STEPS)+1} chained collectives, {NUM_ITERATIONS} iterations")
    baseline = None
    failures = []
    for i in range(NUM_ITERATIONS):
        out = run()
        ttnn.synchronize_device(mesh_device)
        cur = _shards(out)
        ttnn.deallocate(out)
        if i == 0:
            baseline = cur
            logger.info("  iter 0: baseline captured")
            continue
        ndiff, total, maxabs, bad = _first_diff(baseline, cur)
        status = "BIT-EXACT" if ndiff == 0 else "DIVERGED"
        logger.info(
            f"  iter {i}: {status} ndiff={ndiff}/{total} maxabs={maxabs:.3e} chips={len(bad)}/{len(cur)} {bad[:8]}"
        )
        if ndiff:
            failures.append((i, ndiff, maxabs, bad))

    if failures:
        msg = "; ".join(f"iter {i}: ndiff={n} maxabs={m:.3e} chips={len(b)}" for i, n, m, b in failures)
        pytest.fail(f"{mode} seq_local={seq_local} chain is not run-to-run deterministic: {msg}")
    logger.success(f"{mode} seq_local={seq_local} chain bit-exact across {NUM_ITERATIONS} iterations")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_local", [3200], ids=["seq3200"])
def test_local_compute_determinism(mesh_device, device_params, seq_local):
    """No collectives at all: is a single chip's own arithmetic reproducible?

    Identical input and weights replicated to all 32 chips, a local matmul chain, no CCL. A
    divergence here cannot be an op or fabric bug, because there is nothing left that is shared.

    Two independent comparisons, because they fail for different reasons:
      * run-to-run per chip -- an intermittent fault (marginal DRAM/SRAM cell, weak core).
      * chip-vs-chip within one iteration -- every chip runs the same program on the same bytes, so
        any chip that disagrees with the other 31 is computing differently every time.
    """
    ITERS = 10
    HIDDEN = 4608  # INTERMEDIATE_SIZE / tp_factor: the dense FFN's per-chip hidden width
    torch.manual_seed(0)

    def repl(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # 0.02 std keeps a two-matmul chain inside bf16 range; an overflow to inf would make every
    # comparison meaningless (comp_pcc zeroes non-finite values).
    x = repl(torch.randn(1, 1, seq_local, EMB_DIM) * 0.02)
    w_up = repl(torch.randn(EMB_DIM, HIDDEN) * 0.02)
    w_down = repl(torch.randn(HIDDEN, EMB_DIM) * 0.02)

    logger.info(f"local matmul chain, no CCL: {ITERS} iterations, seq_local={seq_local}")
    baseline = None
    r2r_bad = {}
    xchip_bad = {}
    for i in range(ITERS):
        h = ttnn.linear(x, w_up)
        out = ttnn.linear(h, w_down)
        ttnn.synchronize_device(mesh_device)
        cur = _shards(out)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

        # Replicated input + replicated weights: chip 0 is as valid a reference as any other, so a
        # disagreement localizes to the minority chip.
        odd = [c for c in range(1, len(cur)) if not torch.equal(cur[0], cur[c])]
        for c in odd:
            xchip_bad[c] = xchip_bad.get(c, 0) + int((cur[0] != cur[c]).sum())
        if odd:
            logger.warning(f"  iter {i}: {len(odd)} chip(s) disagree with chip 0: {odd[:8]}")

        if i == 0:
            baseline = cur
            logger.info("  iter 0: baseline captured")
            continue
        ndiff, total, maxabs, bad = _first_diff(baseline, cur)
        status = "BIT-EXACT" if ndiff == 0 else "DIVERGED"
        logger.info(f"  iter {i}: {status} ndiff={ndiff}/{total} maxabs={maxabs:.3e} chips={bad[:8]}")
        for c in bad:
            r2r_bad[c] = r2r_bad.get(c, 0) + int((baseline[c] != cur[c]).sum())

    if r2r_bad:
        logger.error(f"run-to-run divergence by chip (chip: total differing elements): {r2r_bad}")
    if xchip_bad:
        logger.error(f"chip-vs-chip0 divergence (chip: total differing elements): {xchip_bad}")
    assert not r2r_bad and not xchip_bad, (
        f"local matmul chain is not deterministic without any collective: "
        f"run_to_run={r2r_bad} chip_vs_chip0={xchip_bad}"
    )
    logger.success(f"local matmul chain bit-exact across {ITERS} iterations and all 32 chips")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_report_device_mapping(mesh_device, device_params):
    """Shard index -> mesh coordinate -> physical device id.

    Every test here names a *shard index* (the position in get_device_tensors), which is meaningless
    for a hardware claim until it is tied to a physical chip. The mapping is not the identity, so
    reading a shard index as a device id names the wrong chip.
    """
    rows, cols = tuple(mesh_device.shape)
    ids = list(mesh_device.get_device_ids())
    logger.info(f"mesh shape {rows}x{cols}, get_device_ids() order: {ids}")
    for r in range(rows):
        coord_ids = [mesh_device.get_device_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)]
        shard_idx = list(range(r * cols, (r + 1) * cols))
        logger.info(f"  row {r}: shard idx {shard_idx} -> device ids {coord_ids}")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("op", ["readback", "eltwise", "matmul1", "matmul2"])
@pytest.mark.parametrize("seq", [3200, 640], ids=["seq3200", "seq640"])
def test_local_op_determinism(mesh_device, device_params, op, seq):
    """Which subsystem of a chip is wrong: DRAM, the SFPU/pack path, or the matmul array.

    Four rungs over one chip's data path, cheapest first:
      readback -- no compute at all, just DRAM write then repeated read. Fails only if storage or
                  the readback path is bad.
      eltwise  -- one pass through the unpack/SFPU/pack path, no matmul array.
      matmul1  -- one matmul.
      matmul2  -- two chained matmuls, which amplify a single bad tile across the whole output row.
    The first rung that fails names the subsystem. Diff positions are reported tile-aligned, because
    a single bad Tensix core corrupts whole output tiles rather than scattered elements.
    """
    ITERS = 8
    HIDDEN = 4608
    # seq is the per-chip sequence: ISL / SP 8. 3200 is ISL 25600, 640 is a 5120 chunk. A smaller
    # seq gives each core proportionally less work per iteration, and this fault is intermittent,
    # so a pass at 640 bounds the exposure at that size rather than clearing the core.
    torch.manual_seed(0)

    def repl(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    x = repl(torch.randn(1, 1, seq, EMB_DIM) * 0.02)
    w_up = repl(torch.randn(EMB_DIM, HIDDEN) * 0.02) if op.startswith("matmul") else None
    w_down = repl(torch.randn(HIDDEN, EMB_DIM) * 0.02) if op == "matmul2" else None

    def run():
        if op == "readback":
            return x, False  # nothing to free: x is the tensor under test
        if op == "eltwise":
            return ttnn.mul(x, x), True
        if op == "matmul1":
            return ttnn.linear(x, w_up), True
        h = ttnn.linear(x, w_up)
        out = ttnn.linear(h, w_down)
        ttnn.deallocate(h)
        return out, True

    def describe(a, b, chip):
        """Tile-aligned footprint of the disagreement, so one bad core is distinguishable."""
        d = (a != b).nonzero()
        rows = {int(t[-2]) for t in d}
        cols = {int(t[-1]) for t in d}
        return (
            f"chip {chip}: {len(d)} diffs over {len(rows)} rows x {len(cols)} cols, "
            f"row tiles {len(sorted({r // 32 for r in rows}))} (first {sorted({r // 32 for r in rows})[:6]}), "
            f"col tiles {len(sorted({c // 32 for c in cols}))} (first {sorted({c // 32 for c in cols})[:6]}), "
            f"maxabs={float((a.float() - b.float()).abs().max()):.3e}"
        )

    logger.info(f"local {op}: seq={seq} ({seq // 32}x{HIDDEN // 32} output tiles), {ITERS} iterations, no collectives")
    baseline = None
    r2r, xchip = {}, {}
    for i in range(ITERS):
        out, freeable = run()
        ttnn.synchronize_device(mesh_device)
        cur = _shards(out)
        if freeable:
            ttnn.deallocate(out)

        odd = [c for c in range(1, len(cur)) if not torch.equal(cur[0], cur[c])]
        for c in odd:
            xchip[c] = xchip.get(c, 0) + int((cur[0] != cur[c]).sum())
        if odd:
            logger.warning(f"  iter {i}: disagrees with chip 0: {odd[:8]} | {describe(cur[0], cur[odd[0]], odd[0])}")

        if i == 0:
            baseline = cur
            logger.info("  iter 0: baseline captured")
            continue
        ndiff, total, maxabs, bad = _first_diff(baseline, cur)
        logger.info(
            f"  iter {i}: {'BIT-EXACT' if ndiff == 0 else 'DIVERGED'} ndiff={ndiff}/{total} "
            f"maxabs={maxabs:.3e} chips={bad[:8]}"
        )
        if bad:
            logger.warning(f"    run-to-run {describe(baseline[bad[0]], cur[bad[0]], bad[0])}")
        for c in bad:
            r2r[c] = r2r.get(c, 0) + int((baseline[c] != cur[c]).sum())

    assert not r2r and not xchip, f"local {op} seq={seq} not deterministic: run_to_run={r2r} chip_vs_chip0={xchip}"
    logger.success(f"local {op} seq={seq} bit-exact across {ITERS} iterations and all 32 chips")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq, hidden", [(3200, 4608), (3200, 2304), (1600, 4608)], ids=["base", "halfN", "halfM"])
def test_local_matmul_core_locality(mesh_device, device_params, seq, hidden):
    """Is a chip's matmul failure tied to a core, or to an output address range?

    When the disagreements fall inside one tile-aligned rectangle, two readings fit: one bad Tensix
    core, or something address-bound. A shape change separates them, because the output-tile -> core
    mapping scales with the tile counts while an address range does not. Halving N (or M) halves the
    rectangle's offset and extent if it is a core, and leaves it put if it is an address. The
    rectangle is reported as a block index, which is invariant exactly when it is one core.
    """
    ITERS = 6
    torch.manual_seed(0)

    def repl(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    x = repl(torch.randn(1, 1, seq, EMB_DIM) * 0.02)
    w = repl(torch.randn(EMB_DIM, hidden) * 0.02)
    mt, nt = seq // 32, hidden // 32
    logger.info(f"matmul {seq}x{EMB_DIM} @ {EMB_DIM}x{hidden} -> output {mt}x{nt} tiles, {ITERS} iterations")

    def rect(a, b, chip):
        """The disagreement as a tile rectangle, plus the block index it implies."""
        d = (a != b).nonzero()
        rt = sorted({int(t[-2]) // 32 for t in d})
        ct = sorted({int(t[-1]) // 32 for t in d})
        per_m, per_n = rt[-1] - rt[0] + 1, ct[-1] - ct[0] + 1
        return (
            f"chip {chip}: {len(d)} diffs | row tiles {rt[0]}-{rt[-1]} ({per_m} tall), "
            f"col tiles {ct[0]}-{ct[-1]} ({per_n} wide) | block idx "
            f"(M {rt[0] // per_m} of {mt / per_m:.1f}, N {ct[0] // per_n} of {nt / per_n:.1f}) | "
            f"maxabs={float((a.float() - b.float()).abs().max()):.3e}"
        )

    baseline, r2r, xchip = None, {}, {}
    for i in range(ITERS):
        out = ttnn.linear(x, w)
        ttnn.synchronize_device(mesh_device)
        cur = _shards(out)
        ttnn.deallocate(out)

        odd = [c for c in range(1, len(cur)) if not torch.equal(cur[0], cur[c])]
        for c in odd:
            xchip[c] = xchip.get(c, 0) + int((cur[0] != cur[c]).sum())
        if odd:
            logger.warning(f"  iter {i}: vs chip 0 -> {rect(cur[0], cur[odd[0]], odd[0])}")
        if i == 0:
            baseline = cur
            continue
        _, _, _, bad = _first_diff(baseline, cur)
        for c in bad:
            r2r[c] = r2r.get(c, 0) + int((baseline[c] != cur[c]).sum())
        if bad:
            logger.warning(f"  iter {i}: run-to-run -> {rect(baseline[bad[0]], cur[bad[0]], bad[0])}")

    assert not r2r and not xchip, f"matmul {seq}x{hidden} not deterministic: run_to_run={r2r} chip_vs_chip0={xchip}"
    logger.success(f"matmul {seq}x{hidden} bit-exact across {ITERS} iterations and all 32 chips")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_matmul_core_sweep(mesh_device, device_params):
    """Name the core outright: confine a matmul to a known window of cores and sweep the grid.

    test_local_matmul_core_locality infers a core from an output-tile rectangle, which needs the
    program config and transpose_mcast to become a coordinate. This removes the inference:
    allowed_worker_cores confines the whole matmul to a window, so a 1x1 window makes a failure
    name that core directly and a pass exonerates it. The logical coordinate is reported alongside
    the physical (NOC) coordinate as the *mesh* translates it; harvesting is per chip, so confirm
    that translation on the failing chip before filing it as a harvest request.

    The per-core block matches what the failing 3200x4608 matmul hands a single core (10x12 output
    tiles over the full K), so a core that drifts there gets the same amount of work here. A
    smaller block is not equivalent: the fault is intermittent, and less work per core means fewer
    chances to trip it.

    A 1x1 window cannot sit on the last row or column: the 2D factory reads the neighbours at
    start+1 unconditionally when it builds the mcast ranges, so an origin there asks for a core
    that does not exist. Those cores get a second pass with a 2x2 window anchored one back, where
    the output rectangle picks one of four candidates.

    Restrict the sweep with TT_DET_CORES="x,y;x,y", and raise the iteration count with
    TT_DET_SWEEP_ITERS, when a core needs more attempts to trip.
    """
    ITERS = int(os.environ.get("TT_DET_SWEEP_ITERS", "10"))
    PER_CORE_M, PER_CORE_N = 10, 12  # the block one core owns in the failing 3200x4608 matmul
    IN0_BLOCK_W = 8  # must divide Kt = EMB_DIM/32 = 224; keeps the in0 L1 block inside 1.5 MB
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    operands = {}

    def window(ox, oy, gx, gy):
        """Run the matmul on the gx x gy core window at (ox, oy); return {chip: {core: ndiff}}."""
        if (gx, gy) not in operands:
            seq, hidden = gy * PER_CORE_M * 32, gx * PER_CORE_N * 32
            operands[(gx, gy)] = tuple(
                ttnn.from_torch(
                    t,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                for t in (torch.randn(1, 1, seq, EMB_DIM) * 0.02, torch.randn(EMB_DIM, hidden) * 0.02)
            )
        x, w = operands[(gx, gy)]
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=IN0_BLOCK_W,
            out_subblock_h=2,  # h*w must stay within the 8-tile dest register
            out_subblock_w=4,
            per_core_M=PER_CORE_M,
            per_core_N=PER_CORE_N,
            transpose_mcast=False,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + gx - 1, oy + gy - 1))}
            ),
        )

        def blame(a, b):
            """Which cores of this window own the differing elements. transpose_mcast=False, so
            the M block index is the grid y and the N block index is the grid x."""
            d = (a != b).nonzero()
            return {(ox + int(t[-1]) // 32 // PER_CORE_N, oy + int(t[-2]) // 32 // PER_CORE_M) for t in d}

        hits, baseline = {}, None
        for i in range(ITERS):
            out = ttnn.matmul(x, w, program_config=pc)
            ttnn.synchronize_device(mesh_device)
            cur = _shards(out)
            ttnn.deallocate(out)

            # chip-vs-chip0 stands on its own at iteration 0; run-to-run needs the baseline.
            pairs = [(c, cur[0], cur[c]) for c in range(1, len(cur)) if not torch.equal(cur[0], cur[c])]
            if i == 0:
                baseline = cur
            else:
                _, _, _, bad = _first_diff(baseline, cur)
                pairs += [(c, baseline[c], cur[c]) for c in bad]
            for c, a, b in pairs:
                per_chip = hits.setdefault(c, {})
                for core in blame(a, b):
                    per_chip[core] = per_chip.get(core, 0) + int((a != b).sum())
        return hits

    env = os.environ.get("TT_DET_CORES", "").strip()
    if env:
        cores = [tuple(int(v) for v in tok.split(",")) for tok in env.split(";") if tok]
    else:
        cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    pinnable = [(cx, cy) for cx, cy in cores if cx < grid.x - 1 and cy < grid.y - 1]
    edge = sorted({(min(cx, grid.x - 2), min(cy, grid.y - 2)) for cx, cy in cores if (cx, cy) not in pinnable})
    logger.info(
        f"grid {grid.x}x{grid.y}, {PER_CORE_M}x{PER_CORE_N} output tiles per core over K={EMB_DIM}, "
        f"{ITERS} iterations: {len(pinnable)} cores pinned 1x1, "
        f"{len(cores) - len(pinnable)} on the last row/col covered by {len(edge)} 2x2 windows"
    )

    failures = {}
    for (ox, oy), (gx, gy) in [(c, (1, 1)) for c in pinnable] + [(a, (2, 2)) for a in edge]:
        for chip, per_core in window(ox, oy, gx, gy).items():
            for core, ndiff in per_core.items():
                failures.setdefault(core, {})
                failures[core][chip] = failures[core].get(chip, 0) + ndiff
                phys = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
                logger.warning(
                    f"  logical core {core} -> physical ({phys.x},{phys.y}) per the mesh, chip {chip}: "
                    f"NOT deterministic, {ndiff} elements (window {gx}x{gy} at ({ox},{oy}))"
                )

    assert not failures, f"matmul is core-dependent: {failures}"
    logger.success(f"all {len(cores)} cores bit-exact across {ITERS} iterations and all 32 chips")
