# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level timing of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

Each case times REPS replays of a captured trace (each followed by a synchronize) of the dense ring op, the sparse op
with or without pooled K / V, or the K / V all-gather alone, on synthetic and frame masks. Every case appends one JSON
record to `<basetemp>/sparse_ring_sdpa_perf*/results.jsonl`. Skipped on CI.
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import (
    ALIGNED,
    BLOCK,
    D,
    MASK_STRUCTURES,
    Case,
    Geometry,
    _block_lengths,
    _check,
    _density,
    balance_mask,
    frame_block_mask,
    make_mask,
    make_structured_mask,
    pool,
    sparse_reference,
    window_allow,
)

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")

REPS = 6
TRACE_REGION_SIZE = 200_000_000
WORKING_POINTS = {4736: "5s", 9184: "10s", 13664: "15s"}
DENSE_CHUNKS = {4736: (320, 384), 9184: (256, 512), 13664: (256, 512)}  # dense op (q_chunk, k_chunk)
DENSITIES = [0.20, 0.07, 0.05]  # 5 s block densities at tau 1 / 2 / 3
K_CHUNKS = [256, 384, 512]
FAMILY_DENSITIES = [0.05, 0.1, 0.2, 0.35]
FAMILY_K_CHUNK = 256  # production at 5 / 10 / 15 s
# Video frame shapes: 21 frames padded to 24, rows per device on SP=8 for 3840 / 1920 tokens per frame.
FRAME_POINTS = {11520: ("720p", 3840), 5760: ("480p", 1920)}
FRAMES_REAL, FRAMES_PADDED = 21, 24
DENSE_CHUNKS.update({rows: (320, 384) for rows in FRAME_POINTS})
SHAPE_NAMES = {**WORKING_POINTS, **{rows: name for rows, (name, _) in FRAME_POINTS.items()}}
TAIL_TOKENS = [512, 1024, 4096]
TAIL_POINTS = {rows: WORKING_POINTS[rows] for rows in (4736, 9184)}


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG, trace_region_size=TRACE_REGION_SIZE)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


@pytest.fixture(scope="module")
def results_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("sparse_ring_sdpa_perf") / "results.jsonl"
    logger.info(f"sparse ring SDPA perf results: {path}")
    return path


def traced_ms(mesh_device, fn):
    """Compile call, one trace capture, REPS timed replays: the replay times in ms. `fn` returns the tensor it
    allocates, or None when it writes only persistent buffers."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, trace_id)
    _free(out)
    return times


def _free(t):
    if t is not None:
        ttnn.deallocate(t)


class Bench:
    """Device inputs for one shape (the last one used; the previous is freed) and its dense / gather baselines."""

    def __init__(self, rt):
        self.rt, self.key, self.case, self.tensors, self.baselines = rt, None, None, (), {}
        # The MiniMax-H3 attention module's SDPA compute config, so timings match the model's.
        self.ckc = ttnn.init_device_compute_kernel_config(
            rt.mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
        )

    def shape(self, rows, logical_n=None, lofi=False):
        """The case for `rows` per device and `logical_n` real tokens (default: every row); lofi: bf16 Q and bf8 K / V
        rounded for LoFi matmuls (Case(lofi=True))."""
        logical_n = rows * MESH_CONFIG.sp_size if logical_n is None else logical_n
        if (rows, logical_n, lofi) != self.key:
            self._free()
            geo = Geometry(rows, logical_n, K_CHUNKS[0], dense_chunks=DENSE_CHUNKS[rows])
            case = Case(self.rt, geo, dtype=ttnn.bfloat8_b, lofi=True) if lofi else Case(self.rt, geo)
            shape = (1, case.H, case.S * rows, D)
            g = torch.Generator().manual_seed(rows)
            qkv = [torch.randn(shape, generator=g, dtype=torch.bfloat16) for _ in range(3)]
            q, k, v = case.prepare(*qkv)
            self.tensors = (
                case._upload(q, case.seq_dims, dtype=case.q_dtype),
                *(case._upload(x, case.seq_dims, dtype=case.dtype) for x in (k, v)),
            )
            pooled = (case.quantize(pool(x, geo, case.S), bits=case.kv_bits()) for x in (k, v))
            self.pooled = tuple(case._upload(x, case.dims(None, 1), dtype=case.dtype) for x in pooled)
            self.key, self.case = (rows, logical_n, lofi), case
        return self.case

    def _free(self):
        if self.case is not None:
            for t in (*self.tensors, *self.pooled, self.case.buf_k, self.case.buf_v):
                ttnn.deallocate(t)
        self.key, self.case, self.tensors, self.pooled = None, None, (), ()

    def dense(self):
        rt, case = self.rt, self.case
        q, k, v = self.tensors
        return lambda: call_sdpa(
            q,
            k,
            v,
            case.geo.logical_n,
            False,
            False,
            case.buf_k,
            case.buf_v,
            case.dense_config,
            self.ckc,
            rt.ccl_semaphore_handles,
            rt.num_links,
            rt.sp_axis,
            rt.mesh_device,
            rt.topology,
            rt.worker_sub_device_id,
            rt.ccl_column,
        )

    def gather(self):
        rt, case = self.rt, self.case
        _, k, v = self.tensors

        def fn():  # writes the persistent buffers only: nothing to free
            ttnn.experimental.ring_attention_all_gather_async(
                [k, v],
                persistent_output_buffer=[case.buf_k, case.buf_v],
                dim=2,
                multi_device_global_semaphore=rt.ccl_semaphore_handles,
                num_links=rt.num_links,
                cluster_axis=rt.sp_axis,
                mesh_device=rt.mesh_device,
                topology=rt.topology,
                subdevice_id=rt.worker_sub_device_id,
            )

        return fn

    def sparse(self, t_mask, k_chunk, pooled, dense_blocks=None, dense_q_buf=None, dense_partials=None):
        rt, case = self.rt, self.case
        q, k, v = self.tensors
        pk, pv = self.pooled if pooled else (None, None)
        config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=rt.sdpa_compute_grid,
            q_chunk_size=BLOCK,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
            matmul_math_fidelity=case.sparse_config.matmul_math_fidelity,
        )
        return lambda: ttnn.transformer.sparse_ring_sdpa(
            q,
            k,
            v,
            t_mask,
            pk,
            pv,
            persistent_output_buffer_k=case.buf_k,
            persistent_output_buffer_v=case.buf_v,
            logical_n=case.geo.logical_n,
            program_config=config,
            compute_kernel_config=self.ckc,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=rt.mesh_device,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            block_size=BLOCK,
            dense_blocks=dense_blocks,
            persistent_output_buffer_dense_q=dense_q_buf,
            dense_partials_debug=dense_partials,
        )

    def baseline(self, rows, logical_n=None):
        """(dense, gather) replay times for `rows` (and `logical_n`) with bf16 inputs, timed on first use."""
        self.shape(rows, logical_n)
        if self.key not in self.baselines:
            mesh = self.rt.mesh_device
            self.baselines[self.key] = (traced_ms(mesh, self.dense()), traced_ms(mesh, self.gather()))
        return self.baselines[self.key]

    def lofi_gather(self, rows, logical_n=None):
        """The gather's replay times with the LoFi inputs (bf8 K / V), timed on first use."""
        self.shape(rows, logical_n, lofi=True)
        if self.key not in self.baselines:
            self.baselines[self.key] = traced_ms(self.rt.mesh_device, self.gather())
        return self.baselines[self.key]


@pytest.fixture(scope="module")
def bench(runtime):
    b = Bench(runtime)
    yield b
    b._free()


def record(results_file, rows, mode, times, *, k_chunk=None, density=None, dense=None, gather=None, structure="random"):
    op_ms = statistics.median(times)
    rec = {
        "shape": SHAPE_NAMES[rows],
        "rows": rows,
        "mesh": f"{MESH_CONFIG.tp_size}x{MESH_CONFIG.sp_size}",
        "mode": mode,
        "structure": structure,
        "k_chunk": k_chunk,
        "density": density,
        "op_ms": op_ms,
        "all_ms": times,
        "speedup_vs_dense": None if dense is None else statistics.median(dense) / op_ms,
        "gap_to_gather_ms": None if gather is None else op_ms - statistics.median(gather),
    }
    extra = ""
    if rec["speedup_vs_dense"] is not None:
        extra += f", {rec['speedup_vs_dense']:.2f}x vs dense"
    if rec["gap_to_gather_ms"] is not None:
        extra += f", {rec['gap_to_gather_ms']:+.3f} ms vs gather"
    logger.info(
        f"[{rec['shape']} rows={rows} {rec['mesh']}] {mode:<6} {structure} k={k_chunk} d={density}: {op_ms:.3f} ms "
        f"(all {' '.join(f'{t:.3f}' for t in times)}){extra}"
    )
    with open(results_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def time_mask(
    bench,
    results_file,
    rows,
    make,
    k_chunk,
    pooled,
    structure="random",
    logical_n=None,
    lofi=False,
    dense_blocks=None,
    dense_local=False,
):
    """Time the sparse op on the mask `make(case)` builds and record it against the bf16 dense op and the gather of
    the same inputs; dense_blocks (with its dense-Q buffer) go to the op, and dense_local keeps every device's dense
    partials local (dense_partials_debug: no return hops, no merge), to time the dense units alone."""
    dense, gather = bench.baseline(rows, logical_n)
    if lofi:
        gather = bench.lofi_gather(rows, logical_n)
    case = bench.shape(rows, logical_n, lofi)
    mask = make(case)
    block_density = _density(mask, case.geo, case.S)
    t_mask = case._upload(mask, case.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    del mask
    dense_q_buf = None if dense_blocks is None else case.dense_q_buffer(dense_blocks)
    partials = case.dense_partials_buffer(dense_blocks) if dense_local else None
    try:
        times = traced_ms(
            case.rt.mesh_device, bench.sparse(t_mask, k_chunk, pooled, dense_blocks, dense_q_buf, partials)
        )
    finally:
        for t in (t_mask, dense_q_buf, partials):
            if t is not None:
                ttnn.deallocate(t)
    mode = ("sparse" if pooled else "kept") + ("-lofi" if lofi else "")
    return record(
        results_file,
        rows,
        mode,
        times,
        k_chunk=k_chunk,
        density=round(block_density, 4),
        dense=dense,
        gather=gather,
        structure=structure,
    )


@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_dense_and_gather_perf(bench, results_file, rows):
    """The two baselines: the dense op and the gather it overlaps."""
    dense, gather = bench.baseline(rows)
    record(results_file, rows, "dense", dense, k_chunk=DENSE_CHUNKS[rows][1], gather=gather)
    record(results_file, rows, "gather", gather)


@pytest.mark.parametrize("pooled", [True, False], ids=["sparse", "kept"])
@pytest.mark.parametrize("density", DENSITIES, ids=[f"d{d}" for d in DENSITIES])
@pytest.mark.parametrize("k_chunk", K_CHUNKS, ids=[f"k{k}" for k in K_CHUNKS])
@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_sparse_ring_sdpa_perf(bench, results_file, rows, k_chunk, density, pooled):
    """The sparse op on a synthetic mask, against the dense op and the gather floor of the same shape."""

    def make(case):
        return make_mask(case.geo, case.S, case.H, density=density, seed=rows)

    time_mask(bench, results_file, rows, make, k_chunk, pooled)


@pytest.mark.parametrize("density", FAMILY_DENSITIES, ids=[f"d{d}" for d in FAMILY_DENSITIES])
@pytest.mark.parametrize("structure", MASK_STRUCTURES)
@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_mask_family_perf(bench, results_file, rows, structure, density):
    """The sparse op at the production k_chunk over the mask family: how its time depends on mask structure."""

    def make(case):
        return make_structured_mask(case.geo, case.S, case.H, structure=structure, density=density, seed=rows)

    time_mask(bench, results_file, rows, make, FAMILY_K_CHUNK, True, structure=structure)


@pytest.mark.parametrize("pooled", [True, False], ids=["sparse", "kept"])
@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_lofi_perf(bench, results_file, rows, pooled):
    """LoFi matmuls with bf8 K / V and bf16 Q (inputs rounded as for LoFi) at density 0.2, against the bf16 dense op."""

    def make(case):
        return make_mask(case.geo, case.S, case.H, density=0.2, seed=rows)

    time_mask(bench, results_file, rows, make, FAMILY_K_CHUNK, pooled, lofi=True)


@pytest.mark.skipif(MESH_CONFIG.sp_size != 8, reason="the frame-sparse video shapes are per device on SP=8")
@pytest.mark.parametrize("lofi", [False, True], ids=["bf16", "lofi"])
@pytest.mark.parametrize("rows", list(FRAME_POINTS), ids=[name for name, _ in FRAME_POINTS.values()])
def test_frame_mask_perf(bench, results_file, rows, lofi):
    """Video frame shapes (window 5 and the last frame, kept blocks only) against the dense op."""
    tokens_per_frame = FRAME_POINTS[rows][1]
    allow = window_allow(FRAMES_REAL, FRAMES_PADDED, window=5, add_last_frame=True)

    def make(case):
        return frame_block_mask(allow, tokens_per_frame, case.geo, case.S, case.H)

    logical_n = FRAMES_REAL * tokens_per_frame
    time_mask(
        bench, results_file, rows, make, FAMILY_K_CHUNK, False, structure="frames", logical_n=logical_n, lofi=lofi
    )


def dense_tail_mask(geo, S, H, *, tail_tokens, position, variant, density=0.2, seed=0):
    """make_mask at `density` with a dense tail: the first (position "start", shard 0) or last ("end", shard S - 1)
    `tail_tokens` of the sequence. Every real Q block keeps the tail's K blocks; with variant "rows_cols" the tail's Q
    blocks also keep every K block ("cols": they keep the random density)."""
    nb, nb_pad = geo.nb(), geo.nb_pad()
    n = tail_tokens // BLOCK
    assert (
        tail_tokens % BLOCK == 0 and n <= nb and geo.logical_n == S * geo.rows
    ), "the tail must fill one shard's blocks"
    s, blocks = (0, torch.arange(n)) if position == "start" else (S - 1, torch.arange(nb - n, nb))
    mask = make_mask(geo, S, H, density=density, seed=seed)
    full = (_block_lengths(geo, S) == BLOCK).reshape(-1).to(mask.dtype)
    mask[:, :, :nb, s * nb_pad + blocks] = 1
    if variant == "rows_cols":
        mask[s, :, blocks] = 1
    return mask * full


def dense_tail_range(geo, S, *, tail_tokens, position):
    """The dense_blocks range of dense_tail_mask's tail."""
    nb, nb_pad, n = geo.nb(), geo.nb_pad(), tail_tokens // BLOCK
    return [(0, n)] if position == "start" else [((S - 1) * nb_pad + nb - n, (S - 1) * nb_pad + nb)]


@pytest.mark.parametrize("pooled", [True, False], ids=["sparse", "kept"])
@pytest.mark.parametrize("tail_tokens", TAIL_TOKENS, ids=[f"T{t}" for t in TAIL_TOKENS])
@pytest.mark.parametrize("variant", ["rows_cols", "cols", "dense_blocks", "dense_blocks_local"])
@pytest.mark.parametrize("position", ["start", "end"])
@pytest.mark.parametrize("rows", list(TAIL_POINTS), ids=list(TAIL_POINTS.values()))
def test_dense_tail_perf(bench, results_file, rows, position, variant, tail_tokens, pooled):
    """The sparse op at the production k_chunk on density-0.2 video blocks with a dense text / audio tail on one
    device: in the mask (dense_tail_mask: rows and columns, or columns only), or as dense_blocks (rows split across
    the devices, columns by dense_keys) on the plain density-0.2 mask; dense_blocks_local times the same without the
    partials' return to the owner (dense_partials_debug), so dense_blocks minus it is the return path's cost."""
    via_dense_blocks = variant.startswith("dense_blocks")

    def make(case):
        if via_dense_blocks:
            mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=rows)
            return mask * (_block_lengths(case.geo, case.S) == BLOCK).reshape(-1).to(mask.dtype)
        return dense_tail_mask(
            case.geo, case.S, case.H, tail_tokens=tail_tokens, position=position, variant=variant, seed=rows
        )

    case = bench.shape(rows)
    dense = dense_tail_range(case.geo, case.S, tail_tokens=tail_tokens, position=position) if via_dense_blocks else None
    structure = f"dense_tail_{position}_{variant}_T{tail_tokens}"
    time_mask(
        bench,
        results_file,
        rows,
        make,
        FAMILY_K_CHUNK,
        pooled,
        structure=structure,
        dense_blocks=dense,
        dense_local=variant == "dense_blocks_local",
    )


# Load balancing: rows of very different cost that nobody flags. heavy_rows: 2 % of the (head, Q block) rows keep 90 %
# of the blocks (within a device); causal: block-triangular (across devices). Only the kept route balances; the sparse
# (pooled) route runs the static split on the same masks.
BALANCE_STRUCTURES = ["heavy_rows", "causal"]


@pytest.mark.parametrize("pooled", [True, False], ids=["sparse", "kept"])
@pytest.mark.parametrize("structure", BALANCE_STRUCTURES)
@pytest.mark.parametrize("rows", list(TAIL_POINTS), ids=list(TAIL_POINTS.values()))
def test_balance_perf(bench, results_file, rows, structure, pooled):
    """The sparse op at the production k_chunk on masks with very uneven rows, on both routes."""

    def make(case):
        n_heavy = max(1, (case.S * case.H * case.geo.nb()) // 50)
        return balance_mask(case.geo, case.S, case.H, kind=structure, density=0.2, seed=rows, heavy_rows=n_heavy)

    time_mask(bench, results_file, rows, make, FAMILY_K_CHUNK, pooled, structure=f"balance_{structure}")


def test_balance_trace_masks(runtime):
    """Correctness under trace replay: one captured trace of the op, replayed after new mask contents are written into
    the same device tensor (prefix, causal, random): costs, ranges and swaps are recomputed on device every replay."""
    geo = ALIGNED["r1024k256"]
    case = Case(runtime, geo)
    mesh = runtime.mesh_device
    q, k, v = case.inputs(60)
    tq = case._upload(q, case.seq_dims, dtype=case.q_dtype)
    tk, tv = (case._upload(x, case.seq_dims, dtype=case.dtype) for x in (k, v))
    masks = [
        balance_mask(geo, case.S, case.H, kind="prefix", seed=60),
        balance_mask(geo, case.S, case.H, kind="causal", seed=61),
        make_mask(geo, case.S, case.H, density=0.2, seed=62),
    ]

    def host_mask(m):
        return ttnn.from_torch(
            m, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=case.mapper(case.dims(0, 1))
        )

    t_mask = case._upload(masks[0], case.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    _free(case.call(tq, tk, tv, t_mask))  # compile
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = case.call(tq, tk, tv, t_mask)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    try:
        for i, mask in enumerate(masks):
            ttnn.copy_host_to_device_tensor(host_mask(mask), t_mask)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            sparse = ttnn.to_torch(out, mesh_composer=case.composer).float()[:, :, : geo.logical_n]
            pk, pv = (pool(x, geo, case.S) for x in (k, v))  # unused by the kept-only reference, but required
            ref = sparse_reference(
                q,
                k,
                v,
                mask,
                pk,
                pv,
                geo,
                case.S,
                case.scale,
                pooled=False,
                heads=sorted({0, 1, case.H // 2, case.H - 1}),
            )[:, :, : geo.logical_n]
            r = {"sparse": sparse, "dense": None, "ref": ref}
            _check(f"trace replay {i}", r, _density(mask, geo, case.S), dense_min=None, ref_min=0.999)
    finally:
        ttnn.release_trace(mesh, trace_id)
        _free(out)
        ttnn.deallocate(t_mask)
