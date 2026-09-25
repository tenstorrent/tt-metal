# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Mesh test for hybrid_routed_expert_moe overlapped with combine_fabric2d in one program.

Graded against the same two ops run back to back: the solo hybrid routed expert, then the standalone
combine_fabric2d on its output. Combine only moves tokens, and both runs compute the routed expert
with the same binaries, so the two outputs must be bit-identical. A mismatch is a handoff bug: combine
read an expert before the routed expert finished writing it.

seq 640, because shorter sequences finish every expert before combine reaches it and never exercise
the wait. Threshold 0 runs the unified half alone; the median count splits the experts across both.

The op is not wired into any model; nothing here should run in CI.
"""

from types import SimpleNamespace

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.utils_for_testing import comp_pcc

pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_FULL_MESH = (8, 4)
_MESHES = {
    (8, 4): ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    (8, 1): ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
}
_SEQ_LEN_PER_CHIP = 640
_CAPACITY_FACTOR = 8
_HIDDEN_DIM = 2048
_PERF_RUNS = 3
# Measured programs per configuration in the speedup test; the median is reported.
_SPEEDUP_ITERS = 5
# Kernel directories that tell the three programs apart in the real-time profiler's records.
_OVERLAP_KERNELS = "/hybrid_routed_expert_ffn/device/kernels/combine/"
_SOLO_RE_KERNELS = "/hybrid_routed_expert_ffn/device/kernels/"
_COMBINE_KERNELS = "/deepseek_prefill/combine_fabric2d/"


def _scaled_model(mesh):
    """DeepSeek V3 at the full mesh's experts-per-chip and per-group expert activation."""
    model = DeepSeekV3Config()
    chips = mesh[0] * mesh[1]
    model.NUM_ROUTED_EXPERTS = (model.NUM_ROUTED_EXPERTS // (_FULL_MESH[0] * _FULL_MESH[1])) * chips
    if mesh[1] != _FULL_MESH[1]:
        model.NUM_EXPERTS_PER_TOKEN = max(1, (model.NUM_EXPERTS_PER_TOKEN // _FULL_MESH[1]) * mesh[1])
    return model


def _mesh_params():
    params = []
    for mesh, fabric_cfg in _MESHES.items():
        topo = "ring" if fabric_cfg == ttnn.FabricConfig.FABRIC_2D_TORUS_Y else f"mesh-{mesh[0]}x{mesh[1]}"
        for threshold_id in ("t0", "tmedian"):
            params.append(
                pytest.param(
                    mesh,
                    fabric_to_device_params(fabric_cfg),
                    threshold_id,
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=topo),
                    id=f"{mesh[0]}x{mesh[1]}-{threshold_id}",
                )
            )
    return params


def _int_tensor(torch_tensor, mesh_device, mesh_mapper, dtype=ttnn.int32):
    return ttnn.from_torch(
        torch_tensor, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=dtype
    )


def _build_case(mesh_device, device_params, threshold_id):
    """One seq-640 layer on `mesh_device`: its inputs, and the three ways to run it."""
    torch.manual_seed(42)
    model = _scaled_model(tuple(mesh_device.shape))
    emb_dim = model.EMB_SIZE
    num_routed_experts = model.NUM_ROUTED_EXPERTS
    num_experts_per_tok = model.NUM_EXPERTS_PER_TOKEN
    num_links = 2

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    num_devices = mesh_device.get_num_devices()

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        _SEQ_LEN_PER_CHIP,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        _CAPACITY_FACTOR,
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size,
        _SEQ_LEN_PER_CHIP,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        _SEQ_LEN_PER_CHIP,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    dispatched_buffer, dispatched_metadata = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=_SEQ_LEN_PER_CHIP,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )(x, weights, indices, expert_offsets)

    counts = expert_token_counts.flatten()
    if threshold_id == "t0":
        threshold = 0
    else:
        threshold = max(1, int(counts[counts > 0].median().item()))
    assert threshold < max_dispatched_tokens_per_expert
    logger.info(
        f"{num_routed_experts=} {num_experts_per_tok=} {experts_per_chip=} {max_dispatched_tokens_per_expert=} "
        f"{threshold=} fused experts={(counts <= threshold).sum().item()}/{counts.numel()}"
    )

    ep_mapper = get_ep_mesh_mapper(mesh_device)
    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    # ROW_MAJOR bf16: the tilized path is the one that writes the bf16 tiles combine reads.
    tt_x = ttnn.from_torch(
        dispatched_buffer, mesh_mapper=ep_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_metadata = _int_tensor(dispatched_metadata, mesh_device, ep_mapper)
    # UINT32, which the routed expert requires and combine also takes.
    # (1, experts) per device: the routed expert takes 1D or 2D, combine reads the last two dims.
    tt_counts = ttnn.squeeze(_int_tensor(expert_token_counts, mesh_device, counts_mapper, dtype=ttnn.uint32), 0)
    tt_region_offsets = ttnn.squeeze(
        _int_tensor(expert_region_offsets, mesh_device, counts_mapper, dtype=ttnn.uint32), 0
    )
    # Replicated along the ring axis: every chip needs every origin chip's run boundaries.
    tt_expert_offsets = _int_tensor(
        expert_offsets,
        mesh_device,
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
        dtype=ttnn.uint32,
    )
    idx_table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    tt_idx_slice = _int_tensor(idx_table, mesh_device, ep_mapper, dtype=ttnn.uint32)
    tt_idx_slice = ttnn.squeeze(ttnn.squeeze(tt_idx_slice, 0), 0)
    tt_idx_full = _int_tensor(idx_table, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device), dtype=ttnn.uint32)

    # One weight set shared by every expert keeps host conversion bounded; the handoff does not care
    # which weights an expert has, only when its rows are written.
    expert_weights = {
        "gate_proj": torch.randn(_HIDDEN_DIM, emb_dim) * 0.02,
        "up_proj": torch.randn(_HIDDEN_DIM, emb_dim) * 0.02,
        "down_proj": torch.randn(emb_dim, _HIDDEN_DIM) * 0.02,
    }
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=tt_idx_slice,
        emb_dim=emb_dim,
        hidden_dim=_HIDDEN_DIM,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=[expert_weights] * (num_devices * experts_per_chip),
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    def routed_expert(**overlap):
        return ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
            tt_x,
            tt_region_offsets,
            tt_counts,
            tt_idx_slice,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            hybrid_token_threshold=threshold,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=ttnn.RoutedExpertActivation.Silu,
            **overlap,
        )

    def solo_routed_expert():
        return routed_expert(output_dtype=ttnn.bfloat16)

    def combine(re_output):
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            re_output,
            tt_metadata,
            tt_counts,
            tt_region_offsets,
            tt_expert_offsets,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=per_axis_topology(device_params["fabric_config"])[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def overlapped():
        return routed_expert(
            dispatched_metadata=tt_metadata,
            expert_offsets=tt_expert_offsets,
            replicated_global_expert_idx_table=tt_idx_full,
            combine_axis=sp_axis,
            combine_num_links=num_links,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=_SEQ_LEN_PER_CHIP,
        )

    return SimpleNamespace(
        emb_dim=emb_dim,
        composer=get_ep_mesh_composer(mesh_device),
        solo_routed_expert=solo_routed_expert,
        combine=combine,
        overlapped=overlapped,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id", _mesh_params(), indirect=["mesh_device", "device_params"]
)
def test_hybrid_routed_expert_combine_overlap(mesh_device, device_params, threshold_id):
    case = _build_case(mesh_device, device_params, threshold_id)
    emb_dim = case.emb_dim
    expected = ttnn.to_torch(case.combine(case.solo_routed_expert()), mesh_composer=case.composer)

    # Twice: the second run is a program-cache hit, which reuses the cached arena and fwd_arrived.
    for run in range(2):
        actual = ttnn.to_torch(case.overlapped(), mesh_composer=case.composer)
        assert actual.shape == expected.shape, f"run {run}: shape {actual.shape} != {expected.shape}"
        if not torch.equal(actual, expected):
            tokens_expected = expected.reshape(-1, emb_dim).float()
            tokens_actual = actual.reshape(-1, emb_dim).float()
            mismatched = (tokens_actual != tokens_expected).any(dim=-1)
            bad_expected = tokens_expected[mismatched]
            bad_actual = tokens_actual[mismatched]
            _, pcc = comp_pcc(expected, actual)
            _, bad_pcc = comp_pcc(bad_expected, bad_actual)
            # Rounding differences keep a mismatched token close to its reference; a token read before the
            # routed expert wrote it is unrelated to it.
            pytest.fail(
                f"run {run}: {mismatched.sum().item()}/{mismatched.numel()} output tokens differ from solo "
                f"routed expert + combine_fabric2d (PCC {pcc:.6f}; over those tokens PCC {bad_pcc:.6f}, "
                f"elements differing {(bad_actual != bad_expected).float().mean().item():.2%}, "
                f"max |diff| {(bad_actual - bad_expected).abs().max().item():.4g} vs max |ref| "
                f"{bad_expected.abs().max().item():.4g})"
            )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id", _mesh_params(), indirect=["mesh_device", "device_params"]
)
def test_hybrid_routed_expert_combine_overlap_perf(mesh_device, device_params, threshold_id):
    """The three programs back to back for the device profiler: solo routed expert, standalone combine, and
    the overlap, each run _PERF_RUNS times with the first a warm-up.

    Every solo routed-expert run comes first: its per-call arena takes all free L1, which combine's
    fwd_arrived holds a piece of from its first call on.
    """
    case = _build_case(mesh_device, device_params, threshold_id)
    re_outputs = [case.solo_routed_expert() for _ in range(_PERF_RUNS)]
    ttnn.synchronize_device(mesh_device)
    for re_output in re_outputs:
        case.combine(re_output)
    ttnn.synchronize_device(mesh_device)
    for _ in range(_PERF_RUNS):
        case.overlapped()
    ttnn.synchronize_device(mesh_device)


def _median_program_ns(mesh_device, run_fn, iters, is_target, label):
    """Median device time, slowest chip, of the one program per run that `is_target` picks out."""

    def run_all():
        return [run_fn() for _ in range(iters)]

    outputs, per_program = profile_realtime_program_merged(mesh_device, run_all)
    matched = [e["duration_ns"] for e in per_program.values() if is_target(e["kernel_sources"])]
    # The warm-up run's record can be delivered after the window opens; records arrive in dispatch order,
    # so the measured runs are the last `iters`.
    assert len(matched) >= iters, f"{label}: expected {iters} programs, the profiler matched {len(matched)}"
    return outputs, statistics.median(matched[-iters:])


def _has(sources, path):
    return any(path in source.replace("\\", "/") for source in sources)


@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, threshold_id", _mesh_params(), indirect=["mesh_device", "device_params"]
)
def test_hybrid_routed_expert_combine_overlap_speedup(mesh_device, device_params, threshold_id):
    """The overlapped program against the same two ops back to back: hybrid routed expert, then
    combine_fabric2d. Sequential is the sum of the two programs' device times, so it assumes no gap
    between them and is the best a sequential dispatch can do; the overlap must beat it.

    Every solo routed-expert run comes first: its per-call arena takes all free L1, which combine's
    fwd_arrived holds a piece of from its first call on. One warm-up run of each program fills the
    program cache outside the measured window.
    """
    require_realtime_profiler("the RE + combine overlap speedup test")
    case = _build_case(mesh_device, device_params, threshold_id)

    warm_re = case.solo_routed_expert()
    re_outputs, re_ns = _median_program_ns(
        mesh_device,
        case.solo_routed_expert,
        _SPEEDUP_ITERS,
        lambda k: _has(k, _SOLO_RE_KERNELS) and not _has(k, _OVERLAP_KERNELS),
        "solo routed expert",
    )

    case.combine(warm_re)
    re_iter = iter(re_outputs)
    _, combine_ns = _median_program_ns(
        mesh_device,
        lambda: case.combine(next(re_iter)),
        _SPEEDUP_ITERS,
        lambda k: _has(k, _COMBINE_KERNELS),
        "combine_fabric2d",
    )

    case.overlapped()
    _, overlap_ns = _median_program_ns(
        mesh_device, case.overlapped, _SPEEDUP_ITERS, lambda k: _has(k, _OVERLAP_KERNELS), "overlap"
    )

    sequential_ns = re_ns + combine_ns
    saved_ns = sequential_ns - overlap_ns
    logger.info(
        f"RE+combine {threshold_id}: sequential {sequential_ns / 1e3:.1f} us (RE {re_ns / 1e3:.1f} + combine "
        f"{combine_ns / 1e3:.1f}), overlap {overlap_ns / 1e3:.1f} us -> {sequential_ns / overlap_ns:.3f}x, "
        f"saved {saved_ns / 1e3:.1f} us ({saved_ns / sequential_ns:.1%}); combine hidden "
        f"{min(saved_ns, combine_ns) / combine_ns:.0%}"
    )
    assert overlap_ns < sequential_ns, (
        f"overlap {overlap_ns / 1e3:.1f} us is not faster than RE then combine "
        f"({re_ns / 1e3:.1f} + {combine_ns / 1e3:.1f} = {sequential_ns / 1e3:.1f} us)"
    )
