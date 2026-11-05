# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op-level split KV as production runs it: Kimi K3 / K2.7 MLA expanded K and V (ttMLA's attention) over the
TP-deduped KV cache, gathered over the whole mesh.

Chip (s, t) of the SP x TP mesh (tensor rank s * TP + t) stores one REGION-row stripe of every chunk, block-cyclically:
chunk region r lives on rank r % ranks at local row (r // ranks) * REGION. Q is sharded by sequence over SP and by head
over TP. K and V are expanded per head from the gathered latent inside the op (W_UK, W_UV^T), with the chunk schedule
ttMLA picks for the per-chip shape."""

import math
from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_N_CHUNKS,
    EXPANDED_K_CHUNKED_MODEL_CONFIGS,
    EXPANDED_K_MIN_HEAD_PCC,
    MESH_CONFIG,
    _make_ring_mla_metadata,
    _ring_mla_host_scalar_tensor,
    close_ring_joint_sdpa_runtime,
    compute_chunked_prefill_perf_check_utilization,
    expand_mla_k_reference,
    expanded_kv_expansion_utilization,
    get_matmul_peak_multiplier,
    expanded_k_production_chunks,
    model_fabric_payload_size,
    open_ring_joint_sdpa_runtime,
    profile_ring_joint_traced_duration_ns,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

TP = 4
# Rows each chip stores per chunk: ttMLA's stripe, 640 Q rows per SP row over TP.
REGION = 160
# Per-head output norm against the reference's (bf8 noise moves it ~2%): PCC alone would pass a uniformly mis-scaled
# (mis-normalized) output.
NORM_RATIO_TOLERANCE = 0.05
# Gather scratch poison: past each source's transferred slabs it must stay untouched, and a replay that read stale
# scratch instead of gathering would attend to it.
SCRATCH_POISON = 13.0
MODELS = ["kimi_k3_expanded_k", "kimi_k2_7_expanded_k"]
MODEL_IDS = ["k3", "k2_7"]


def as_device_reads(t):
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).to(t.dtype)


def open_split_kv_runtime(model, trace_region_size=0, kernel_config_headroom_guard=True):
    """The SP x TP mesh with a full-mesh gather and the model's production fabric packets."""
    if MESH_CONFIG.num_devices not in (8, 32):
        pytest.skip("Split KV runs on an eight-device LoudBox / QuietBox or a 32-device Galaxy")
    sp = MESH_CONFIG.num_devices // TP
    # The runtime helper opens (tp_size, sp_size); split KV wants (SP, TP).
    return open_ring_joint_sdpa_runtime(
        replace(MESH_CONFIG, tp_size=sp, sp_size=TP),
        full_mesh=True,
        trace_region_size=trace_region_size,
        fabric_payload_size=model_fabric_payload_size(model),
        kernel_config_headroom_guard=kernel_config_headroom_guard,
    )


class SplitKV:
    """Q, the TP-deduped latent cache (capacity_chunks chunks, num_users x num_layers slots), W_UK and W_UV^T on the
    mesh, the op's arguments, and an fp32 reference over the same quantized inputs."""

    def __init__(self, runtime, model_name, capacity_chunks, num_users=1, num_layers=1, check=True):
        mesh = runtime.mesh_device
        self.runtime, self.mesh = runtime, mesh
        self.sp, self.tp = tuple(mesh.shape)
        assert self.tp == TP
        model = EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name]
        self.model = model
        self.ranks = self.sp * self.tp
        self.chunk = self.ranks * REGION
        self.q_slab = self.tp * REGION
        self.capacity = capacity_chunks * self.chunk
        self.num_layers = num_layers
        heads = model.nhq  # per chip
        self.heads = heads
        latent = model.d_k - (model.d_q - model.k_expand_nope)
        self.latent = latent
        source_rows = capacity_chunks * REGION
        cache_batch = num_users * num_layers

        torch.manual_seed(20261104)
        q = torch.randn(1, heads * self.tp, self.chunk, model.d_q).bfloat16()
        latent_k = torch.randn(cache_batch, 1, self.capacity, model.d_k).bfloat16()
        w_uk = as_device_reads(torch.randn(1, heads * self.tp, model.k_expand_nope, latent) * latent**-0.5)
        w_uv_t = as_device_reads(torch.randn(1, heads * self.tp, model.d_v, latent) * latent**-0.5)
        sources = torch.empty(cache_batch, self.ranks, source_rows, model.d_k, dtype=torch.bfloat16)
        for region in range(self.capacity // REGION):
            rank, local = region % self.ranks, (region // self.ranks) * REGION
            sources[:, rank, local : local + REGION] = latent_k[:, 0, region * REGION : (region + 1) * REGION]

        self.tt_q = ttnn.from_torch(
            q,
            dtype=model.q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(self.sp, self.tp), dims=[2, 1]),
        )
        self.tt_k = ttnn.from_torch(
            sources.reshape(cache_batch, 1, self.ranks * source_rows, model.d_k),
            dtype=model.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        # W_UK and W_UV^T hold every head's rows: TP shards them by head like Q, SP replicates them.
        weight_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(self.sp, self.tp), dims=[None, 1])
        tt_w_uk, tt_w_uv = (
            ttnn.from_torch(
                w,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=weight_mapper,
            )
            for w in (w_uk, w_uv_t)
        )
        # Larger than the gathered capacity, so the op cannot rely on scratch and cache capacities matching.
        self.scratch_rows = self.capacity + self.chunk
        self.scratch_poison = ttnn.from_torch(
            torch.full((1, 1, self.scratch_rows, model.d_k), SCRATCH_POISON),
            dtype=model.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.scratch = ttnn.to_device(self.scratch_poison, mesh)

        q_chunk, k_chunk, k_splits = expanded_k_production_chunks(model_name)
        self.kwargs = dict(
            persistent_output_buffer_k=self.scratch,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
                k_splits=k_splits,
                matmul_math_fidelity=model.matmul_math_fidelity,
            ),
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=None,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            scale=model.scale,
            k_expand_weight=tt_w_uk,
            v_expand_weight=tt_w_uv,
            kv_cache_num_layers=num_layers,
            kv_cache_layer_idx=0,
        )
        self.metadata = None

        if check:
            # The reference reads what the device holds: the bf8 latent as stored, the bf8 weights.
            stored = torch.stack([ttnn.to_torch(s)[:, 0] for s in ttnn.get_device_tensors(self.tt_k)], dim=1)
            for region in range(self.capacity // REGION):
                rank, local = region % self.ranks, (region // self.ranks) * REGION
                latent_k[:, 0, region * REGION : (region + 1) * REGION] = stored[:, rank, local : local + REGION]
            self.q, self.latent_k, self.w_uk, self.w_uv = q, latent_k, w_uk, w_uv_t.transpose(-2, -1)

    def use_metadata(self, slot, prefix):
        """Read the slot and prefix on device (the trace-safe path) instead of host scalars."""
        self.metadata = _make_ring_mla_metadata(self.mesh, slot, prefix)
        self.kwargs.update(slot_id=self.metadata[0], kv_actual_isl_tensor=self.metadata[1])
        self.kwargs.pop("kv_actual_isl", None)
        self.kwargs.pop("kv_cache_batch_idx", None)

    def set_step(self, prefix, slot=0, layer=0, logical_n=None):
        """Point the op at Q rows [prefix, prefix + chunk) of user slot `slot`, layer `layer`; logical_n defaults to
        the chunk end (capped at capacity), or to capacity on the metadata path (the trace-safe host value)."""
        end = min(prefix + self.chunk, self.capacity)
        self.kwargs["kv_cache_layer_idx"] = layer
        if self.metadata is None:
            self.kwargs.update(kv_actual_isl=prefix, kv_cache_batch_idx=slot)
            self.kwargs["logical_n"] = end if logical_n is None else logical_n
        else:
            ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(self.mesh, slot), self.metadata[0])
            ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(self.mesh, prefix), self.metadata[1])
            self.kwargs["logical_n"] = self.capacity if logical_n is None else logical_n
        return end if logical_n is None else min(end, logical_n)

    def run(self):
        output, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            self.tt_q, self.tt_k, None, **self.kwargs
        )
        return output

    def check(self, output, prefix, end, slot=0, layer=0):
        """Per (chip, head) PCC against fp32 attention over K and V expanded from the stored latent."""
        model = self.model
        scale = model.scale if model.scale is not None else model.d_q**-0.5
        latent = self.latent_k[slot * self.num_layers + layer : slot * self.num_layers + layer + 1, :, :end]
        shards = ttnn.get_device_tensors(output)
        assert len(shards) == self.ranks
        for lane in range(self.tp):
            heads = slice(lane * self.heads, (lane + 1) * self.heads)
            k = expand_mla_k_reference(latent, self.w_uk[:, heads]).float()[0]
            v = torch.einsum("knl,hld->hnd", latent[0, ..., : self.latent].float(), self.w_uv[0, heads].float())
            for q_rank in range(self.sp):
                # The absolute rows this SP rank owns, independent of the kernel's rotation formulas.
                positions = torch.tensor([p for p in range(prefix, end) if (p // self.q_slab) % self.sp == q_rank])
                if not len(positions):
                    continue  # no promise for padded Q rows
                q = self.q[0, heads, q_rank * self.q_slab : q_rank * self.q_slab + len(positions)].float()
                scores = (q @ k.transpose(-1, -2)) * scale
                scores = scores.masked_fill(torch.arange(end)[None, :] > positions[:, None], -torch.inf)
                expected = scores.softmax(-1) @ v
                actual = ttnn.to_torch(shards[q_rank * self.tp + lane]).float()[0, :, : len(positions)]
                assert torch.isfinite(actual).all(), f"non-finite output on chip ({q_rank}, {lane})"
                for h in range(self.heads):
                    passed, pcc = comp_pcc(expected[h], actual[h], EXPANDED_K_MIN_HEAD_PCC)
                    assert passed, f"chip ({q_rank}, {lane}) head {h}: PCC {pcc} < {EXPANDED_K_MIN_HEAD_PCC}"
                    ratio = (actual[h].norm() / expected[h].norm()).item()
                    assert (
                        abs(ratio - 1) <= NORM_RATIO_TOLERANCE
                    ), f"chip ({q_rank}, {lane}) head {h}: output norm {ratio:.4f}x the reference's"

    def check_scratch_tail(self, end):
        """Every source keeps its stride in the scratch: the first tile past its transferred slabs stays poison."""
        source_rows = self.capacity // self.ranks
        live_slabs = math.ceil(end / self.chunk)
        if live_slabs * REGION >= source_rows:
            return
        for source in range(self.ranks):
            row = source * source_rows + live_slabs * REGION
            tail = ttnn.slice(self.scratch, (0, 0, row, 0), (1, 1, row + 32, self.model.d_k))
            for shard in ttnn.get_device_tensors(tail):
                assert torch.all(ttnn.to_torch(shard) == SCRATCH_POISON), f"source {source}: scratch tail overwritten"


# (prefix chunks, row offset, metadata, logical rows): first chunk, a deeper prefix on the metadata path, a prefix
# ending inside a chip's region (rotation within the stripe), one crossing a Q slab, and a short prompt that leaves most
# sources without valid KV, so their ring iterations go inactive (the op promises nothing for Q rows past logical_n,
# so only the rows below it are checked).
ACCURACY_CASES = [
    pytest.param(0, 0, False, None, id="first-scalar"),
    pytest.param(3, 0, True, None, id="deep-metadata"),
    pytest.param(1, 32, True, None, id="region-rotation-metadata"),
    pytest.param(1, 672, False, None, id="qslab-rotation-scalar"),
    pytest.param(0, 0, False, 3 * REGION, id="inactive-sources"),
]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("depth,offset,metadata,logical_rows", ACCURACY_CASES)
@pytest.mark.parametrize("model_name", MODELS, ids=MODEL_IDS)
def test_split_kv_accuracy(model_name, depth, offset, metadata, logical_rows):
    """Production attention on the TP-deduped cache against the fp32 reference, on two dispatches of one program."""
    runtime = open_split_kv_runtime(EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name])
    try:
        mesh = runtime.mesh_device
        mesh.enable_program_cache()
        case = SplitKV(runtime, model_name, capacity_chunks=4)
        prefix = depth * case.chunk + offset
        if metadata:
            case.use_metadata(0, prefix)
        for dispatch in range(2):
            ttnn.copy_host_to_device_tensor(case.scratch_poison, case.scratch)
            end = case.set_step(prefix, logical_n=logical_rows)
            # The checks below add their own programs: count only this dispatch's.
            before = mesh.num_program_cache_entries()
            output = case.run()
            ttnn.synchronize_device(mesh)
            assert dispatch == 0 or mesh.num_program_cache_entries() == before, "the second dispatch compiled a program"
            case.check(output, prefix, end)
            case.check_scratch_tail(end)
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("metadata", [False, True], ids=["scalar", "metadata"])
def test_split_kv_users_and_layers(metadata):
    """An indexed cache of 3 users x 2 layers: switching the user, and on the scalar path the layer, reuses the
    program; on the metadata path the layer is structural, so each is compiled once."""
    model_name = "kimi_k3_expanded_k"
    runtime = open_split_kv_runtime(EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name])
    try:
        mesh = runtime.mesh_device
        mesh.enable_program_cache()
        case = SplitKV(runtime, model_name, capacity_chunks=3, num_users=3, num_layers=2)
        prefix = case.chunk + 32
        if metadata:
            case.use_metadata(0, prefix)
        warm = {}
        for user, layer in ((2, 1), (1, 0), (0, 1), (2, 0), (1, 1)):
            ttnn.copy_host_to_device_tensor(case.scratch_poison, case.scratch)
            end = case.set_step(prefix, slot=user, layer=layer)
            before = mesh.num_program_cache_entries()
            output = case.run()
            ttnn.synchronize_device(mesh)
            key = layer if metadata else None
            if key in warm:
                assert mesh.num_program_cache_entries() == before, f"user {user} layer {layer} compiled a new program"
            warm[key] = True
            case.check(output, prefix, end, slot=user, layer=layer)
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


@pytest.mark.timeout(1800)
def test_split_kv_trace_replay():
    """One trace per layer, replayed with the users and prefixes (growing and shrinking) rewritten on device: every
    replay matches the reference, with the scratch re-poisoned so stale gathered rows cannot pass for new ones."""
    model_name = "kimi_k3_expanded_k"
    runtime = open_split_kv_runtime(EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name], trace_region_size=4 * 1024 * 1024)
    trace_ids = []
    try:
        mesh = runtime.mesh_device
        mesh.enable_program_cache()
        case = SplitKV(runtime, model_name, capacity_chunks=4, num_users=2, num_layers=2)
        case.use_metadata(0, 0)
        traced = []
        for layer in range(case.num_layers):
            case.set_step(0, layer=layer)
            case.run()  # compile outside the capture
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced.append(case.run())
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            trace_ids.append(trace_id)
        for user, depth, offset in ((1, 2, 0), (0, 0, 0), (1, 3, 0), (0, 1, 32), (1, 0, 0)):
            prefix = depth * case.chunk + offset
            end = case.set_step(prefix, slot=user)
            for layer, (trace_id, output) in enumerate(zip(trace_ids, traced, strict=True)):
                ttnn.copy_host_to_device_tensor(case.scratch_poison, case.scratch)
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                case.check(output, prefix, min(end, case.capacity), slot=user, layer=layer)
    finally:
        for trace_id in trace_ids:
            ttnn.release_trace(runtime.mesh_device, trace_id)
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("model_name", MODELS, ids=MODEL_IDS)
def test_split_kv_determinism(model_name):
    """Three dispatches on the same inputs give bit-identical outputs."""
    runtime = open_split_kv_runtime(EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name])
    try:
        mesh = runtime.mesh_device
        case = SplitKV(runtime, model_name, capacity_chunks=4, check=False)
        case.use_metadata(0, 0)
        case.set_step(3 * case.chunk)
        outputs = []
        for _ in range(3):
            output = case.run()
            ttnn.synchronize_device(mesh)
            outputs.append([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)])
        for run, shards in enumerate(outputs[1:], start=1):
            for chip, (first, again) in enumerate(zip(outputs[0], shards, strict=True)):
                assert torch.equal(first, again), f"dispatch {run} differs from dispatch 0 on chip {chip}"
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


@pytest.mark.parametrize(
    "invalid_case,error",
    [
        pytest.param("absorbed", "requires MLA expanded K and V", id="absorbed"),
        pytest.param("capacity", "must not exceed input KV capacity", id="capacity"),
        pytest.param("fp32", "requires streaming compute", id="fp32"),
    ],
)
def test_split_kv_rejects(invalid_case, error, expect_error):
    """Split KV refuses ring_mla's absorbed attention, a logical length past the cache, and fp32 accumulation."""
    model_name = "kimi_k3_expanded_k"
    runtime = open_split_kv_runtime(EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name])
    try:
        case = SplitKV(runtime, model_name, capacity_chunks=2, check=False)
        case.set_step(0)
        if invalid_case == "absorbed":
            # ring_mla: Q at the latent's width, V the latent's first columns, no expansion.
            model = case.model
            q = ttnn.from_torch(
                torch.zeros(1, model.nhq * TP, case.chunk, model.d_k),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=case.mesh,
                mesh_mapper=ttnn.ShardTensor2dMesh(case.mesh, mesh_shape=(case.sp, case.tp), dims=[2, 1]),
            )
            kwargs = {
                k: v
                for k, v in case.kwargs.items()
                if k not in ("persistent_output_buffer_k", "is_causal", "k_expand_weight", "v_expand_weight")
            }
            with expect_error(RuntimeError, error):
                ttnn.transformer.ring_mla(
                    q, case.tt_k, persistent_output_buffer_kv=case.scratch, head_dim_v=case.latent, **kwargs
                )
            return
        if invalid_case == "capacity":
            case.kwargs["logical_n"] = case.capacity + 32
        else:
            case.kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
                case.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
            )
        with expect_error(RuntimeError, error):
            case.run()
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


# Final 50k+5k chunk of the production prefill on the 32-device Galaxy (110 SDPA cores per chip, TorusXY, production
# fabric packets): math util as test_ring_joint_attention_chunked_expanded_k_perf_check counts it (the per-chip work
# matches the SP ring's), +/- margin.
# Measured 2026-11-05 on bh-glx-120-c02u02: K3 5.04 ms, K2.7 3.56 ms.
SPLIT_KV_PERF_CHECK_CONFIGS = [
    # (model_name, expected_util, margin)
    ("kimi_k3_expanded_k", 57.61, 0.05),
    ("kimi_k2_7_expanded_k", 54.43, 0.05),
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "model_name,expected_util,margin", SPLIT_KV_PERF_CHECK_CONFIGS, ids=[c[0] for c in SPLIT_KV_PERF_CHECK_CONFIGS]
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.skipif(not MESH_CONFIG.is_galaxy, reason="split-KV production perf is calibrated on the Galaxy")
@pytest.mark.skipif(
    MESH_CONFIG.is_galaxy and not is_high_power(),
    reason="galaxy perf job requires a high-power (>=130W TDP) host",
)
def test_split_kv_perf_check(model_name, expected_util, margin):
    """Traced final chunk (prefix of CHUNKED_PREFILL_N_CHUNKS - 1 chunks) of production attention on the TP-deduped
    cache."""
    model = EXPANDED_K_CHUNKED_MODEL_CONFIGS[model_name]
    runtime = open_split_kv_runtime(model, trace_region_size=4 * 1024 * 1024, kernel_config_headroom_guard=False)
    trace_id = None
    try:
        mesh = runtime.mesh_device
        sdpa_cores = runtime.sdpa_compute_grid[0] * runtime.sdpa_compute_grid[1]
        case = SplitKV(runtime, model_name, capacity_chunks=CHUNKED_PREFILL_N_CHUNKS, check=False)
        perf_chunk = CHUNKED_PREFILL_N_CHUNKS - 1
        case.set_step(perf_chunk * case.chunk)
        case.run()  # compile outside the capture
        ttnn.synchronize_device(mesh)

        def run_with_replay(measure):
            nonlocal trace_id
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            case.run()
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)

            def replay():
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

            measure(replay)

        duration_ns, _ = profile_ring_joint_traced_duration_ns(mesh, run_with_replay)
    finally:
        if trace_id is not None:
            ttnn.release_trace(runtime.mesh_device, trace_id)
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)
    utilization, _ = compute_chunked_prefill_perf_check_utilization(
        MESH_CONFIG, model, case.chunk, perf_chunk, duration_ns, sdpa_cores
    )
    utilization += expanded_kv_expansion_utilization(model, case.chunk, perf_chunk, duration_ns, sdpa_cores)
    utilization /= get_matmul_peak_multiplier(model)
    logger.info(f"split-KV {model_name} final chunk: {duration_ns / 1e6:.3f} ms, math util {utilization:.2f}%")
    low, high = expected_util * (1 - margin), expected_util * (1 + margin)
    assert low <= utilization <= high, (
        f"split-KV {model_name}: math utilization {utilization:.2f}% outside band [{low:.2f}, {high:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {margin * 100:.1f}%)"
    )
