# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vsa_sdpa re-runnability: run-to-run determinism, trace replay (the model runs traced), and
program-cache-hit invocations after other ops / with poisoned L1 (regression for a model-block hang
whose root cause was a kreq page word read before it was written)."""

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.sdpa.test_vsa_sdpa_perf import make_inputs
from tests.ttnn.utils_for_testing import comp_pcc


def _run_checks(dev, order):
    q, k, v, idx, counts, _ = make_inputs(dev, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order=order)
    multi = isinstance(dev, ttnn.MeshDevice) and dev.get_num_devices() > 1
    composer = ttnn.ConcatMeshToTensor(dev, dim=0) if multi else None
    host = lambda t: ttnn.to_torch(t, mesh_composer=composer).float()
    ref = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)  # compile + untraced reference
    rep = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)  # untraced repeat
    ttnn.synchronize_device(dev)
    ref_t = host(ref)
    rep_t = host(rep)
    _, pcc_rep = comp_pcc(ref_t, rep_t, 0.9999)
    exact_rep = torch.equal(ref_t, rep_t)

    # Fresh copies at NEW addresses: a program-cache hit must patch every kernel's buffer args
    # (the reader/writer are instantiated per role on disjoint core sets).
    # (clone BEFORE freeing anything so the copies cannot land at the originals' addresses)
    q2, k2, v2, idx2, counts2 = (ttnn.clone(x) for x in (q, k, v, idx, counts))
    assert q2.buffer_address() != q.buffer_address() and v2.buffer_address() != v.buffer_address()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = ttnn.transformer.vsa_sdpa(q2, k2, v2, idx2, counts2)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    out_t = host(out)
    ttnn.release_trace(dev, tid)
    _, pcc_tr = comp_pcc(ref_t, out_t, 0.9999)
    print(f"VSA_DETERMINISM order={order} untraced-repeat exact={exact_rep} pcc={pcc_rep:.6f} traced pcc={pcc_tr:.6f}")
    # Trace must not add error beyond the op's own run-to-run variance (window partitioning is
    # timing-dependent, so untraced repeats already differ at bf16 rounding-order level).
    assert pcc_tr >= pcc_rep - 5e-4, f"traced PCC {pcc_tr} vs untraced-repeat PCC {pcc_rep}"


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20_000_000, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("order", ["topk", "model"])
def test_vsa_sdpa_trace_replay(device, order):
    _run_checks(device, order)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), {"trace_region_size": 20_000_000, "l1_small_size": 65536})],
    indirect=["mesh_device", "device_params"],
)
def test_vsa_sdpa_trace_replay_mesh(mesh_device):
    _run_checks(mesh_device, "model")


def _run_cache_hit(dev, between):
    """Untraced: cache-miss run, then a cache-hit run on fresh tensors (new addresses), optionally
    with other ops in between. Mirrors the model block's second invocation (which hangs)."""
    q, k, v, idx, counts, _ = make_inputs(
        dev, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="model"
    )
    multi = isinstance(dev, ttnn.MeshDevice) and dev.get_num_devices() > 1
    composer = ttnn.ConcatMeshToTensor(dev, dim=0) if multi else None
    host = lambda t: ttnn.to_torch(t, mesh_composer=composer).float()
    ref = host(ttnn.transformer.vsa_sdpa(q, k, v, idx, counts))
    q2, k2, v2, idx2, counts2 = (ttnn.clone(x) for x in (q, k, v, idx, counts))
    if between == "matmul":
        a = ttnn.from_torch(
            torch.randn(1, 1, 4096, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if multi else None,
        )
        b = ttnn.matmul(a, a)
        ttnn.deallocate(b)
        ttnn.deallocate(a)
    ttnn.synchronize_device(dev)
    out = host(ttnn.transformer.vsa_sdpa(q2, k2, v2, idx2, counts2))
    _, pcc = comp_pcc(ref, out, 0.9999)
    print(f"VSA_CACHE_HIT between={between} pcc={pcc:.6f} shape={tuple(out.shape)}")
    if pcc < 0.998:
        # Diagnose the corruption pattern: NaN/Inf counts and which (head, 64-row block) are bad.
        nan = torch.isnan(out).sum().item()
        inf = torch.isinf(out).sum().item()
        bad_hb = []
        for h in range(out.shape[1]):
            for rb in range(out.shape[2] // 64):
                r, o = ref[0, h, rb * 64 : (rb + 1) * 64], out[0, h, rb * 64 : (rb + 1) * 64]
                if not torch.allclose(r, o, atol=2e-2, rtol=2e-2):
                    bad_hb.append((h, rb))
        heads = sorted({h for h, _ in bad_hb})
        rbs = sorted({rb for _, rb in bad_hb})
        print(f"VSA_CACHE_HIT nan={nan} inf={inf} bad_blocks={len(bad_hb)} heads={heads} row_blocks={rbs[:40]}...")
    assert pcc > 0.998


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("between", ["none", "matmul"])
def test_vsa_sdpa_cache_hit(device, between):
    _run_cache_hit(device, between)


def _corruption_pattern(ref, out):
    """Which (head, 64-row block) pairs are wrong, and how (all-zero? scaled? garbage?)."""
    bad = []
    for h in range(out.shape[1]):
        for rb in range(out.shape[2] // 64):
            r, o = ref[0, h, rb * 64 : (rb + 1) * 64], out[0, h, rb * 64 : (rb + 1) * 64]
            if not torch.allclose(r, o, atol=2e-2, rtol=2e-2):
                bad.append((h, rb))
    heads = sorted({h for h, _ in bad})
    rbs = sorted({rb for _, rb in bad})
    n_rb = out.shape[2] // 64
    per_head = {h: sum(1 for hh, _ in bad if hh == h) for h in heads}
    print(f"CORRUPT bad_blocks={len(bad)}/{out.shape[1] * n_rb} heads={heads} per_head={per_head}")
    print(f"CORRUPT row_blocks(first 60)={rbs[:60]}")
    if bad:
        h, rb = bad[0]
        r, o = ref[0, h, rb * 64 : rb * 64 + 4, :8], out[0, h, rb * 64 : rb * 64 + 4, :8]
        print(f"CORRUPT sample head={h} rb={rb} ref={r.tolist()} out={o.tolist()}")
        d = out[0, h, rb * 64 : (rb + 1) * 64] - ref[0, h, rb * 64 : (rb + 1) * 64]
        # a stale-accumulate signature: the error is constant across the 128 head dims of a row
        print(
            f"CORRUPT diff per-row mean(first 8 rows)={d.mean(dim=1)[:8].tolist()} per-row std={d.std(dim=1)[:8].tolist()}"
        )


def _l1_fill(dev, value, mib_per_core=1.0):
    """Blanket every worker core's free L1 with `value` (height-sharded L1 tensor, then free it):
    exposes any L1 the kernels read before writing (NaN fill -> NaNs in the output)."""
    grid = dev.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    rows_per_core = int(mib_per_core * (1 << 20)) // 64  # rows of 32 bf16 = 64 B
    t = torch.full((rows_per_core * ncores, 32), value, dtype=torch.bfloat16)
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
        (rows_per_core, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)
    ttnn.synchronize_device(dev)
    print(f"L1_FILL value={value} addr={x.buffer_address():#x} bytes_per_core={rows_per_core * 64:#x}")
    ttnn.deallocate(x)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_l1_fill_smoke(device):
    _l1_fill(device, 0.0)
    _l1_fill(device, float("nan"))
    _l1_fill(device, 0.0)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("between", ["none", "matmul", "nanfill", "matmul_nanfill", "fill1000", "fill1000_128k"])
def test_vsa_sdpa_cache_hit_loop(device, between):
    """Regression for the kreq window-marker bug (the writer read an L1 word the reader never wrote):
    cache-hit invocations on fresh tensors, with another op or a poisoned L1 in between, must match
    the cache-miss reference. `fill1000_<n>k` poisons only the top n KiB of L1 (buffers allocate
    top-down), which is how the offending CB was bisected."""
    dev = device
    q, k, v, idx, counts, _ = make_inputs(
        dev, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="model"
    )
    ref = ttnn.to_torch(ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)).float()
    a = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    l1_fill = lambda value: _l1_fill(dev, value)

    fails = []
    for it in range(12):
        q2, k2, v2, idx2, counts2 = (ttnn.clone(x) for x in (q, k, v, idx, counts))
        if between.startswith("matmul"):
            b = ttnn.matmul(a, a)
            ttnn.deallocate(b)
        if between == "add":
            b = ttnn.add(a, a)
            ttnn.deallocate(b)
        if between.endswith("nanfill"):
            l1_fill(float("nan"))
        if between.endswith("zerofill"):
            l1_fill(0.0)
        if between.endswith("fill1000"):
            l1_fill(1000.0)
        if "fill1000_" in between:  # fill only the TOP <n>k of each core's L1 (buffers allocate top-down)
            _l1_fill(dev, 1000.0, mib_per_core=int(between.split("fill1000_")[1].rstrip("k")) / 1024)
        out = ttnn.to_torch(ttnn.transformer.vsa_sdpa(q2, k2, v2, idx2, counts2)).float()
        _, pcc = comp_pcc(ref, out, 0.9999)
        nan = torch.isnan(out).sum().item()
        print(f"VSA_CACHE_HIT_LOOP between={between} it={it} pcc={pcc:.6f} nan={nan}")
        if pcc < 0.998:
            fails.append((it, pcc))
            if len(fails) <= 2:
                _corruption_pattern(ref, out)
        for x in (q2, k2, v2, idx2, counts2):
            ttnn.deallocate(x)
    print(f"VSA_CACHE_HIT_LOOP between={between} fails={fails}")
    assert not fails


def _run_followed_by(dev, n_follow):
    """Trace {vsa_sdpa, n_follow x other programs}: inside a multi-program trace the dispatcher
    prefetches the NEXT program's launch data into every worker core's L1 while vsa_sdpa is still
    running -- inbound NOC traffic the op-alone replay never sees (model-trace hang repro)."""
    q, k, v, idx, counts, _ = make_inputs(
        dev, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="model"
    )
    multi = isinstance(dev, ttnn.MeshDevice) and dev.get_num_devices() > 1
    composer = ttnn.ConcatMeshToTensor(dev, dim=0) if multi else None
    host = lambda t: ttnn.to_torch(t, mesh_composer=composer).float()
    a = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if multi else None,
    )

    def graph():
        out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts)
        tails = []
        for _ in range(n_follow):
            tails.append(ttnn.matmul(a, a))  # kernel-heavy follower: many cores, fresh binaries
            tails.append(ttnn.add(tails[-1], 1.0))
        for t in tails:
            ttnn.deallocate(t)
        return out

    ref = host(graph())  # compile run
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = graph()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    out_t = host(out)
    ttnn.release_trace(dev, tid)
    _, pcc = comp_pcc(ref, out_t, 0.9999)
    print(f"VSA_TRACE_FOLLOWED_BY n_follow={n_follow} pcc={pcc:.6f}")
    assert pcc > 0.998


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 40_000_000, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("n_follow", [1, 4])
def test_vsa_sdpa_trace_followed_by(device, n_follow):
    _run_followed_by(device, n_follow)


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), {"trace_region_size": 40_000_000, "l1_small_size": 65536})],
    indirect=["mesh_device", "device_params"],
)
def test_vsa_sdpa_trace_followed_by_mesh(mesh_device):
    _run_followed_by(mesh_device, 4)
