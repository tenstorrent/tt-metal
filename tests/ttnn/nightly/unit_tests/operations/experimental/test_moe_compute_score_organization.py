# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for moe_compute routing-score physical organizations."""

import json
import os
import random
import statistics
import time

import pytest
import torch
import ttnn

from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
)
from ttnn.operations.ccl import MoEActivationFunction

from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    create_sharded_memory_config,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    prepare_output_tensor_from_combine_writer,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _build_quantized_weight_tensors_cpu_prepare,
)


_TOKEN_VALUES = (1, 2, 31, 32, 33)
_K_VALUES = (2, 4, 6, 8)
_SCORE_CASES = tuple(pytest.param(tokens, k, id=f"t{tokens}-k{k}") for tokens in _TOKEN_VALUES for k in _K_VALUES)


def _make_scores(tokens, selected_experts_k, score_case):
    lane = torch.arange(1, selected_experts_k + 1, dtype=torch.float32)
    token = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)

    if score_case == "distinct":
        scores = lane.unsqueeze(0) / 32.0 + token / 256.0
    elif score_case == "signed":
        # Token zero keeps every lane distinct and nonzero. Token one also covers
        # a zero and negative, finite, non-softmax score without losing the first guard.
        scores = lane.unsqueeze(0) / 8.0 + token / 64.0
        scores[0, 0] = -0.5
        scores[1, 0] = -0.75
        scores[1, 1] = 0.0
    elif score_case == "non_normalized":
        scores = lane.unsqueeze(0) * 0.5 + token / 128.0
    elif score_case == "normalized":
        scores = lane.unsqueeze(0).repeat(tokens, 1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
    elif score_case == "one_hot":
        scores = torch.zeros(tokens, selected_experts_k, dtype=torch.float32)
        scores[torch.arange(tokens), torch.arange(tokens) % selected_experts_k] = 1.0
        assert set(torch.argmax(scores, dim=-1).tolist()) == set(range(selected_experts_k))
    else:
        raise AssertionError(f"unknown score case: {score_case}")

    assert torch.isfinite(scores).all()
    return scores.to(torch.bfloat16).unsqueeze(0)


def _single_core_range(core):
    coord = ttnn.CoreCoord(core.x, core.y)
    return ttnn.CoreRangeSet({ttnn.CoreRange(coord, coord)})


def _upload_tensor(
    mesh_device,
    host_tensor,
    dtype,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    layout=ttnn.ROW_MAJOR_LAYOUT,
):
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _upload_mapping(mesh_device, experts_per_device):
    expert_mapping = gen_expert_mapping(1, 1, None, experts_per_device, experts_per_device, experts_per_device)
    return ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _moe_kwargs(output_height_shard_dim, intermediate_size, compute_only):
    return {
        "layer_id": 0,
        "output_height_shard_dim": output_height_shard_dim,
        "intermediate_size": intermediate_size,
        "has_bias": False,
        "activation_type": MoEActivationFunction.SILU,
        "compute_only": compute_only,
    }


def _upload_sparse_inputs(mesh_device, sparse_buffer, expert_indices, logical_scores, drain_core):
    tokens = expert_indices.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    indices_mem_config = create_sharded_memory_config(drain_core, [tokens, selected_experts_k], ttnn.uint16)
    contiguous_scores_mem_config = create_sharded_memory_config(drain_core, [tokens, selected_experts_k], ttnn.bfloat16)
    scalar_page_scores_mem_config = create_sharded_memory_config(
        drain_core, [tokens * selected_experts_k, 1], ttnn.bfloat16
    )

    tt_sparse = _upload_tensor(mesh_device, sparse_buffer, ttnn.bfloat16)
    tt_indices = _upload_tensor(mesh_device, expert_indices, ttnn.uint16, indices_mem_config)
    tt_scores_contiguous = _upload_tensor(mesh_device, logical_scores, ttnn.bfloat16, contiguous_scores_mem_config)
    # Construct the scalar-page tensor directly at upload. No TTNN reshape,
    # permute, slice, or concat operation is part of the test or feature path.
    tt_scores_scalar_page = _upload_tensor(
        mesh_device,
        logical_scores.unsqueeze(-1),
        ttnn.bfloat16,
        scalar_page_scores_mem_config,
    )
    return tt_sparse, tt_indices, tt_scores_contiguous, tt_scores_scalar_page


def _make_optional_output(mesh_device, selected_experts_k, tokens, hidden_size):
    return ttnn.from_torch(
        torch.zeros([selected_experts_k, tokens, hidden_size], dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )


def _capture_outputs(mesh_device, outputs, compute_only):
    metadata_dram = ttnn.to_memory_config(outputs[1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    result_index = 4 if compute_only else 5
    result_dram = ttnn.to_memory_config(outputs[result_index], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    metadata_host = ttnn.to_torch(metadata_dram, mesh_composer=composer)
    result_host = ttnn.to_torch(result_dram, mesh_composer=composer)

    if metadata_dram.buffer_address() != outputs[1].buffer_address():
        ttnn.deallocate(metadata_dram)
    if result_dram.buffer_address() != outputs[result_index].buffer_address():
        ttnn.deallocate(result_dram)
    return metadata_host, result_host


def _deallocate_outputs(outputs, keep_optional_output=False):
    # Slots 3 and 4 alias the same L1 buffer; deallocating slot 4 releases it.
    for index in (0, 1, 2, 4):
        ttnn.deallocate(outputs[index])
    if len(outputs) == 6 and not keep_optional_output:
        ttnn.deallocate(outputs[5])


def _assert_every_score_bit(metadata, expert_indices, logical_scores, experts_per_device):
    selected_experts_k = expert_indices.shape[-1]
    tokens = expert_indices.shape[-2]
    aligned_row_elements = ((2 * experts_per_device + 1) * 4 + 15) // 16 * 4
    flat = metadata[0].flatten().to(torch.int64)
    expected_score_bits = logical_scores.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF

    for token in range(tokens):
        row = token * aligned_row_elements
        assert flat[row].item() == token
        observed_lanes = set()
        for lane in range(selected_experts_k):
            expert = expert_indices[0, token, lane].item()
            actual_lane = flat[row + 1 + expert].item()
            actual_score_bits = flat[row + 1 + experts_per_device + expert].item() & 0xFFFF
            assert actual_lane == lane, f"token={token} expert={expert}: expected K lane {lane}, got {actual_lane}"
            assert actual_score_bits == expected_score_bits[0, token, lane].item(), (
                f"token={token} lane={lane} expert={expert}: expected score bits "
                f"0x{expected_score_bits[0, token, lane].item():04x}, got 0x{actual_score_bits:04x}"
            )
            observed_lanes.add(actual_lane)
        assert observed_lanes == set(range(selected_experts_k))


def _assert_organizations_match(
    contiguous_metadata,
    scalar_page_metadata,
    contiguous_result,
    scalar_page_result,
    mesh_device,
    expert_indices,
    compute_only,
    tokens,
    experts_per_device,
    output_height_shard_dim,
    output_width_shard_dim,
    hidden_size,
):
    aligned_row_elements = ((2 * experts_per_device + 1) * 4 + 15) // 16 * 4
    relevant_metadata_elements = tokens * aligned_row_elements
    torch.testing.assert_close(
        contiguous_metadata[0].flatten()[:relevant_metadata_elements],
        scalar_page_metadata[0].flatten()[:relevant_metadata_elements],
        rtol=0,
        atol=0,
    )

    if not compute_only:
        torch.testing.assert_close(contiguous_result, scalar_page_result, rtol=0, atol=0, equal_nan=True)
        return

    # ComputeOnly slot 4 is a two-expert double buffer; compare only initialized rows.
    worker_grid = mesh_device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(worker_grid.x - 1, worker_grid.y - 1),
            )
        }
    )
    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size
    )
    expert_counts = torch.bincount(expert_indices.flatten().to(torch.int64), minlength=experts_per_device)
    active_buffer_counts = expert_counts[-2:]

    def meaningful_compute_output(raw_result):
        return prepare_output_tensor_from_combine_writer(
            raw_result,
            active_buffer_counts,
            0,
            all_core_range_set,
            output_shard_cores,
            output_height_shard_dim,
            output_width_shard_dim,
            2,
            hidden_size,
        )

    torch.testing.assert_close(
        meaningful_compute_output(contiguous_result),
        meaningful_compute_output(scalar_page_result),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("tokens,selected_experts_k", _SCORE_CASES)
@pytest.mark.parametrize("mesh_shape,mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@torch.no_grad()
def test_moe_compute_score_organizations(
    mesh_device,
    mesh_shape,
    tokens,
    selected_experts_k,
):
    """The two physical organizations must preserve every score bit and output."""
    arch = mesh_device.arch()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip(f"moe_compute score organization is supported only on Wormhole and Blackhole, got {arch}")

    torch.manual_seed(20260817)
    random.seed(20260817)
    assert mesh_shape == (1, 1)
    key = (tokens, selected_experts_k)
    score_case = {(2, 4): "signed", (32, 8): "one_hot"}.get(
        key, {31: "non_normalized", 33: "normalized"}.get(tokens, "distinct")
    )
    compute_only = key not in ((2, 4), (32, 8))
    exercise_cache_hit = key == (2, 4)

    experts_per_device = 8
    hidden_size = 320
    ring_n = effective_matmul_ring_size(mesh_device)
    intermediate_size = max(256, 32 * ring_n)
    output_height_shard_dim = 4
    output_width_shard_dim = auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n)

    drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
    )
    drain_core = _single_core_range(drain_core_coord)

    sparse_buffer, generated_indices, _, _ = gen_sparse_buffer_and_indices(
        tokens,
        hidden_size,
        experts_per_device,
        selected_experts_k,
        mesh_shape,
        None,
        dtype=torch.bfloat16,
    )
    expert_indices = generated_indices.reshape(1, tokens, selected_experts_k)
    logical_scores = _make_scores(tokens, selected_experts_k, score_case)

    tt_mapping = _upload_mapping(mesh_device, experts_per_device)

    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
        mesh_device, hidden_size, intermediate_size
    )
    torch_w0 = create_torch_w0(1, experts_per_device, hidden_size, intermediate_size)
    torch_w1 = create_torch_w1(1, experts_per_device, hidden_size, intermediate_size)
    torch_w2 = create_torch_w2(1, experts_per_device, intermediate_size, hidden_size)
    w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
        1,
        experts_per_device,
        hidden_size,
        intermediate_size,
        w0_w1_shard_map,
        w2_shard_map,
        dram_core_range_set,
        has_bias=False,
    )
    tt_w0_w1, tt_w2 = _build_quantized_weight_tensors_cpu_prepare(
        mesh_device,
        torch_w0,
        torch_w1,
        torch_w2,
        None,
        None,
        None,
        1,
        experts_per_device,
        hidden_size,
        intermediate_size,
        False,
        w0_w1_shard_map,
        w2_shard_map,
        w0_w1_mem_config,
        w2_mem_config,
    )

    tt_sparse, tt_indices, tt_scores_contiguous, tt_scores_scalar_page = _upload_sparse_inputs(
        mesh_device, sparse_buffer, expert_indices, logical_scores, drain_core
    )

    def run(sparse_tensor, score_tensor, indices_tensor, optional_output_tensor):
        return ttnn.experimental.moe_compute(
            sparse_tensor,
            indices_tensor,
            score_tensor,
            tt_mapping,
            tt_w0_w1,
            tt_w2,
            optional_output_tensor=optional_output_tensor,
            **_moe_kwargs(output_height_shard_dim, intermediate_size, compute_only),
        )

    def execute_pair(
        sparse_tensor,
        indices_tensor,
        contiguous_scores,
        scalar_scores,
        optional_outputs,
        *,
        expect_cache_hit,
        capture_scalar_graph=False,
        keep_optional_outputs=False,
    ):
        captured = []
        output_handles = []
        for organization, scores, optional_output in zip(
            ("ContiguousK", "ScalarPageK"),
            (contiguous_scores, scalar_scores),
            optional_outputs,
        ):
            cache_entries_before = mesh_device.num_program_cache_entries()
            if capture_scalar_graph and organization == "ScalarPageK":
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    outputs = run(sparse_tensor, scores, indices_tensor, optional_output)
                finally:
                    graph = ttnn.graph.end_graph_capture()
                calltrace = ttnn.graph.extract_calltrace(graph)
                calltrace_text = "\n".join(calltrace).lower()
                assert "moe" in calltrace_text
                for forbidden_op in ("reshape", "slice", "concat", "permute"):
                    assert (
                        forbidden_op not in calltrace_text
                    ), f"ScalarPageK graph unexpectedly dispatched {forbidden_op}: {calltrace}"
                print(f"MOE_SCORE_SCALAR_PAGE_GRAPH_CALLTRACE={calltrace}")
            else:
                outputs = run(sparse_tensor, scores, indices_tensor, optional_output)

            cache_entries_after = mesh_device.num_program_cache_entries()
            if expect_cache_hit:
                assert cache_entries_after == cache_entries_before
            else:
                assert cache_entries_after > cache_entries_before
            metadata, result = _capture_outputs(mesh_device, outputs, compute_only)
            _assert_every_score_bit(metadata, expert_indices, logical_scores, experts_per_device)
            captured.append((metadata, result))
            output_handles.append(outputs)
            _deallocate_outputs(outputs, keep_optional_output=keep_optional_outputs)

        _assert_organizations_match(
            captured[0][0],
            captured[1][0],
            captured[0][1],
            captured[1][1],
            mesh_device,
            expert_indices,
            compute_only,
            tokens,
            experts_per_device,
            output_height_shard_dim,
            output_width_shard_dim,
            hidden_size,
        )
        return tuple(output_handles)

    cold_optional_outputs = tuple(
        None if compute_only else _make_optional_output(mesh_device, selected_experts_k, tokens, hidden_size)
        for _ in range(2)
    )
    contiguous_outputs, scalar_outputs = execute_pair(
        tt_sparse,
        tt_indices,
        tt_scores_contiguous,
        tt_scores_scalar_page,
        cold_optional_outputs,
        expect_cache_hit=False,
        capture_scalar_graph=tokens == 32 and selected_experts_k == 8 and score_case == "one_hot",
        keep_optional_outputs=exercise_cache_hit,
    )

    perf_repeats = int(os.environ.get("MOE_SCORE_PERF_REPEATS", "0"))
    assert 0 <= perf_repeats <= 100
    if perf_repeats:
        assert compute_only
        host_samples_ns = {}
        for organization, scores in (("ContiguousK", tt_scores_contiguous), ("ScalarPageK", tt_scores_scalar_page)):
            samples = []
            for _ in range(perf_repeats):
                start_ns = time.perf_counter_ns()
                outputs = run(tt_sparse, scores, tt_indices, None)
                ttnn.synchronize_device(mesh_device)
                samples.append(time.perf_counter_ns() - start_ns)
                _deallocate_outputs(outputs)
            host_samples_ns[organization] = samples
        perf_result = {
            "tokens": tokens,
            "K": selected_experts_k,
            "repeats": perf_repeats,
            "execution_order": list(host_samples_ns),
            "host_sync_ns": host_samples_ns,
            "host_sync_median_ns": {name: statistics.median(samples) for name, samples in host_samples_ns.items()},
        }
        print("MOE_SCORE_MICROBENCH=" + json.dumps(perf_result, sort_keys=True))

    if exercise_cache_hit:
        # Allocate replacements while the originals are live so address changes are guaranteed.
        new_sparse, new_indices, new_scores_contiguous, new_scores_scalar_page = _upload_sparse_inputs(
            mesh_device, sparse_buffer, expert_indices, logical_scores, drain_core
        )
        new_optional_outputs = tuple(
            _make_optional_output(mesh_device, selected_experts_k, tokens, hidden_size) for _ in range(2)
        )
        for replacement, original in zip(
            (new_sparse, new_indices, new_scores_contiguous, new_scores_scalar_page, *new_optional_outputs),
            (
                tt_sparse,
                tt_indices,
                tt_scores_contiguous,
                tt_scores_scalar_page,
                contiguous_outputs[5],
                scalar_outputs[5],
            ),
        ):
            assert replacement.buffer_address() != original.buffer_address()

        for tensor in (
            tt_indices,
            tt_scores_contiguous,
            tt_scores_scalar_page,
            contiguous_outputs[5],
            scalar_outputs[5],
        ):
            ttnn.deallocate(tensor)
        ttnn.synchronize_device(mesh_device)

        execute_pair(
            new_sparse,
            new_indices,
            new_scores_contiguous,
            new_scores_scalar_page,
            new_optional_outputs,
            expect_cache_hit=True,
        )
        for tensor in (new_sparse, new_indices, new_scores_contiguous, new_scores_scalar_page):
            ttnn.deallocate(tensor)
    else:
        for tensor in (tt_indices, tt_scores_contiguous, tt_scores_scalar_page):
            ttnn.deallocate(tensor)

    for tensor in (tt_sparse, tt_mapping, tt_w0_w1, tt_w2):
        ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape,mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@torch.no_grad()
def test_moe_compute_score_validation(mesh_device, mesh_shape, expect_error):
    """Malformed score contracts fail on the host without populating the program cache."""
    arch = mesh_device.arch()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip(f"moe_compute score validation is supported only on Wormhole and Blackhole, got {arch}")

    assert mesh_shape == (1, 1)
    tokens = 2
    k = 4
    experts_per_device = 8
    hidden_size = 320
    ring_n = effective_matmul_ring_size(mesh_device)
    intermediate_size = max(256, 32 * ring_n)
    output_height_shard_dim = 4
    output_width_shard_dim = auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n)

    drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
    )
    drain_core = _single_core_range(drain_core_coord)
    worker_grid = mesh_device.compute_with_storage_grid_size()
    if worker_grid.x > 1:
        other_core_coord = ttnn.CoreCoord((drain_core_coord.x + 1) % worker_grid.x, drain_core_coord.y)
    else:
        assert worker_grid.y > 1
        other_core_coord = ttnn.CoreCoord(drain_core_coord.x, (drain_core_coord.y + 1) % worker_grid.y)
    other_core = _single_core_range(other_core_coord)

    sparse_buffer = torch.zeros([1, tokens, hidden_size], dtype=torch.bfloat16)
    expert_indices = torch.arange(k, dtype=torch.int32).reshape(1, 1, -1).repeat(1, tokens, 1)
    logical_scores = _make_scores(tokens, k, "distinct")
    tt_sparse, tt_indices, tt_scores, tt_scalar_scores = _upload_sparse_inputs(
        mesh_device, sparse_buffer, expert_indices, logical_scores, drain_core
    )

    tt_mapping = _upload_mapping(mesh_device, experts_per_device)

    # Validation rejects every malformed sparse input before program creation, so only
    # rank and the expert-count dimension of these small device-resident weights matter.
    dummy_weight = torch.zeros([1, 1, experts_per_device, 1, 32, 128], dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(
        dummy_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    created_tensors = [tt_sparse, tt_indices, tt_scores, tt_scalar_scores, tt_mapping, tt_weight]

    def upload(host_tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG):
        tensor = _upload_tensor(mesh_device, host_tensor, dtype, memory_config, layout)
        created_tensors.append(tensor)
        return tensor

    def sharded(host_tensor, dtype, shard_shape, core=drain_core):
        return upload(
            host_tensor,
            dtype,
            memory_config=create_sharded_memory_config(core, shard_shape, dtype),
        )

    def invoke(indices_tensor, scores_tensor):
        return ttnn.experimental.moe_compute(
            tt_sparse,
            indices_tensor,
            scores_tensor,
            tt_mapping,
            tt_weight,
            tt_weight,
            **_moe_kwargs(output_height_shard_dim, intermediate_size, True),
        )

    def assert_rejected(indices_tensor, scores_tensor, message):
        cache_entries_before = mesh_device.num_program_cache_entries()
        with expect_error(RuntimeError, message):
            invoke(indices_tensor, scores_tensor)
        assert mesh_device.num_program_cache_entries() == cache_entries_before

    def zeros(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def score_case(tensor, message):
        return tt_indices, tensor, message

    def indices_case(tensor, message):
        return tensor, tt_scores, message

    malformed_cases = [
        score_case(upload(zeros([1, tokens, k, 2]), ttnn.bfloat16), r"trailing singleton"),
        score_case(upload(zeros([1, tokens + 1, k]), ttnn.bfloat16), r"routing-score token dimension"),
        score_case(upload(zeros([1, tokens, k + 1]), ttnn.bfloat16), r"routing-score K dimension"),
        (
            upload(zeros([1, tokens + 1, k], torch.int32), ttnn.uint16),
            upload(zeros([1, tokens + 1, k]), ttnn.bfloat16),
            r"trailing token dimension must match the activation token count",
        ),
        # The volume still equals tokens*K, but trailing [1,K] is not [tokens,K].
        (
            upload(expert_indices.reshape(2, 1, k), ttnn.uint16),
            upload(logical_scores.reshape(2, 1, k), ttnn.bfloat16),
            r"trailing token dimension must match the activation token count",
        ),
        score_case(upload(logical_scores.float(), ttnn.float32), r"tilize_expert_scores_tensor must be BFLOAT16"),
        indices_case(upload(expert_indices, ttnn.uint32), r"tilize_expert_indices_tensor must be UINT16"),
        score_case(upload(logical_scores, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), r"scores_tensor must be ROW_MAJOR"),
        indices_case(upload(expert_indices, ttnn.uint16, layout=ttnn.TILE_LAYOUT), r"indices_tensor must be ROW_MAJOR"),
        score_case(upload(logical_scores, ttnn.bfloat16), r"must use HEIGHT_SHARDED memory layout"),
        score_case(sharded(logical_scores, ttnn.bfloat16, [tokens + 1, k]), r"scores_tensor shard shape must be"),
        score_case(
            sharded(logical_scores.unsqueeze(-1), ttnn.bfloat16, [tokens * k + 1, 1]),
            r"scores_tensor shard shape must be",
        ),
        indices_case(sharded(expert_indices, ttnn.uint16, [tokens + 1, k]), r"indices_tensor shard shape must be"),
        score_case(
            sharded(logical_scores, ttnn.bfloat16, [tokens, k], other_core),
            r"must be placed on the selected tilize drain core",
        ),
        indices_case(
            sharded(expert_indices, ttnn.uint16, [tokens, k], other_core),
            r"must be placed on the selected tilize drain core",
        ),
        indices_case(upload(expert_indices.flatten(), ttnn.uint16), r"indices_tensor must be rank 2, 3, or 4"),
        score_case(upload(zeros([1, 1, 1, tokens, k, 1]), ttnn.bfloat16), r"scores_tensor must be rank 2 through 5"),
        score_case(upload(logical_scores.unsqueeze(-2), ttnn.bfloat16), r"trailing singleton"),
    ]
    for indices_tensor, scores_tensor, message in malformed_cases:
        assert_rejected(indices_tensor, scores_tensor, message)

    for tensor in created_tensors:
        ttnn.deallocate(tensor)
