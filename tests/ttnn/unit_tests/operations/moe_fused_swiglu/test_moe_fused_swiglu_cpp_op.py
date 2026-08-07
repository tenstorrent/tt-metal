# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the standard C++ device operation.

Most numerical tests call the compatibility wrapper in
``ttnn.operations.moe_fused_swiglu``.  That wrapper now delegates to the C++
operation, but these tests deliberately call the nanobound entry point so that
factory construction, runtime-argument rebinding, program-cache keys, and
cache-hit validation cannot accidentally fall back to the former generic-op
implementation.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu as python_op
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs


TILE = 32
EMB = 6144
HIDDEN = 1024
CAPACITY = 1024
GRID = ttnn.CoreCoord(11, 8)
NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 8
LOCAL_EXPERT_ID = 3
GLOBAL_EXPERT_ID = 137
PCC_GATE = 0.975


def _cpp_op():
    return ttnn.experimental.deepseek_prefill.moe_fused_swiglu


def _to_device(tensor, dtype, layout, device):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(device, *, seed, count, input_dtype=ttnn.bfloat16, input_layout=ttnn.ROW_MAJOR_LAYOUT):
    torch.manual_seed(seed)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.bfloat16)
    if count < CAPACITY:
        x[:, :, count:, :] = 100.0
    weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor(
        [(11 + 37 * local) % NUM_GLOBAL_EXPERTS for local in range(NUM_LOCAL_EXPERTS)],
        dtype=torch.int32,
    )
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID

    return {
        "host_x": x,
        "host_weights": weights,
        "x": _to_device(x, input_dtype, input_layout, device),
        "weights": [_to_device(w, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for w in weights],
        "counts": _to_device(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device),
        "idx": _to_device(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device),
    }


def _call_cpp(case, **kwargs):
    return _cpp_op()(
        case["x"],
        *case["weights"],
        case["counts"],
        case["idx"],
        LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
        core_grid=GRID,
        **kwargs,
    )


def _reference(case, count):
    x = case["host_x"][0, 0, :count].float()
    w_gate, w_up, w_down = (weight.float() for weight in case["host_weights"])
    return (torch.nn.functional.silu(x @ w_gate) * (x @ w_up)) @ w_down


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def test_cpp_binding_matches_python_wrapper_bitwise(device):
    """The public compatibility name and direct nanobinding execute one implementation."""
    count = 64
    case = _build(device, seed=11, count=count)

    direct = _call_cpp(case)
    wrapped = python_op(
        case["x"],
        *case["weights"],
        case["counts"],
        case["idx"],
        LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
        core_grid=(11, 8),
    )

    direct_host = ttnn.to_torch(direct)[0, 0, :count]
    wrapped_host = ttnn.to_torch(wrapped)[0, 0, :count]
    assert torch.equal(direct_host, wrapped_host)
    assert_with_pcc(_reference(case, count), direct_host.float(), PCC_GATE)


def test_standard_cpp_port_matches_current_legacy_factory_bitwise(device):
    """Migration diagnostic: the C++ factory must preserve the current kernel contract exactly."""
    from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
        create_program_descriptor,
        make_mailbox,
    )

    count = 64
    case = _build(device, seed=12, count=count)
    standard = _call_cpp(case)

    legacy_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, CAPACITY, EMB]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    device_grid = device.compute_with_storage_grid_size()
    mailbox = make_mailbox(device, int(device_grid.x) * int(device_grid.y))
    descriptor = create_program_descriptor(
        case["x"],
        *case["weights"],
        case["counts"],
        case["idx"],
        legacy_output,
        mailbox,
        local_expert_id=LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=True,
        ),
        core_grid=(11, 8),
    )
    ttnn.generic_op(
        [
            case["x"],
            *case["weights"],
            case["counts"],
            case["idx"],
            mailbox,
            legacy_output,
        ],
        descriptor,
    )

    standard_host = ttnn.to_torch(standard)[0, 0, :count]
    legacy_host = ttnn.to_torch(legacy_output)[0, 0, :count]
    assert torch.equal(
        standard_host, legacy_host
    ), f"standard/legacy mismatch: max abs {(standard_host - legacy_host).abs().max().item()}"


def test_program_cache_rebinds_every_tensor_address(device, isolate_program_cache):
    """A cache hit must patch x, all weights, aux tensors, and the output address."""
    count = 32
    first = _build(device, seed=21, count=count)
    second = _build(device, seed=22, count=count)
    first_output = _to_device(
        torch.full((1, 1, CAPACITY, EMB), -7.5, dtype=torch.bfloat16),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
    )
    second_output = _to_device(
        torch.full((1, 1, CAPACITY, EMB), -7.5, dtype=torch.bfloat16),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
    )

    returned_first = _call_cpp(first, output=first_output)
    returned_second = _call_cpp(second, output=second_output)

    assert device.num_program_cache_entries() == 1
    assert returned_first.buffer_address() == first_output.buffer_address()
    assert returned_second.buffer_address() == second_output.buffer_address()
    first_host = ttnn.to_torch(returned_first)[0, 0, :count]
    second_host = ttnn.to_torch(returned_second)[0, 0, :count]
    assert not torch.equal(first_host, second_host), "cache hit reused stale tensor addresses"
    assert_with_pcc(_reference(second, count), second_host.float(), PCC_GATE)


def test_program_cache_rebinds_region_offsets(device, isolate_program_cache):
    """The optional direct-write offset is an address binding, including on cache hits."""
    count = 32
    case = _build(device, seed=23, count=count)
    outputs = [
        _to_device(
            torch.zeros((1, 1, CAPACITY, EMB), dtype=torch.bfloat16),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            device,
        )
        for _ in range(2)
    ]
    offsets = []
    for start in (32, 64):
        host_offsets = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
        host_offsets[GLOBAL_EXPERT_ID] = start
        offsets.append(_to_device(host_offsets, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device))

    _call_cpp(case, output=outputs[0], expert_region_offsets=offsets[0])
    _call_cpp(case, output=outputs[1], expert_region_offsets=offsets[1])

    assert device.num_program_cache_entries() == 1
    first = ttnn.to_torch(outputs[0])[0, 0]
    second = ttnn.to_torch(outputs[1])[0, 0]
    assert torch.equal(first[32:64], second[64:96])
    assert torch.count_nonzero(second[32:64]) == 0, "cache hit reused the first offsets tensor address"


def test_compute_configuration_participates_in_program_cache_key(device, isolate_program_cache):
    """Math configuration is a compile-time kernel property and must not share a program."""
    case = _build(device, seed=31, count=0)
    approximate = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    precise = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    _call_cpp(case, compute_kernel_config=approximate)
    _call_cpp(case, compute_kernel_config=precise)

    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"fp32_dest_acc_en": True}, "fp32_dest_acc_en"),
        ({"packer_l1_acc": True}, "packer_l1_acc"),
        ({"dst_full_sync_en": True}, "dst_full_sync_en"),
    ],
)
def test_unsupported_compute_configurations_are_rejected(device, overrides, message, expect_error):
    """These knobs contradict fixed DEST/tilize assumptions and must not be silently ignored."""
    case = _build(device, seed=32, count=0)
    values = {
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "math_approx_mode": True,
        "fp32_dest_acc_en": False,
        "packer_l1_acc": False,
        "dst_full_sync_en": False,
    }
    values.update(overrides)
    config = ttnn.WormholeComputeKernelConfig(**values)

    with expect_error(RuntimeError, message):
        _call_cpp(case, compute_kernel_config=config)


def test_cache_hit_rejects_output_aliasing_activations(device, isolate_program_cache, expect_error):
    """Address-only changes are revalidated even when tensor specs hit the cache."""
    case = _build(
        device,
        seed=41,
        count=32,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
    )
    valid_output = _to_device(
        torch.zeros((1, 1, CAPACITY, EMB), dtype=torch.bfloat16),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
    )

    _call_cpp(case, output=valid_output)
    with expect_error(RuntimeError, r"(?i)must not alias activations"):
        _call_cpp(case, output=case["x"])

    assert device.num_program_cache_entries() == 1


def test_aux_shape_validation_is_consistent_in_cpp_and_wrapper(device, expect_error):
    case = _build(device, seed=51, count=32)
    bad_counts = _to_device(
        torch.zeros((2, NUM_GLOBAL_EXPERTS // 2), dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )

    with expect_error(RuntimeError, r"(?i)counts.*1D|counts.*\(1, N\)"):
        _cpp_op()(
            case["x"],
            *case["weights"],
            bad_counts,
            case["idx"],
            LOCAL_EXPERT_ID,
            core_grid=GRID,
        )
    with expect_error(ValueError, r"(?i)counts.*1-D|counts.*\(1, N\)"):
        python_op(
            case["x"],
            *case["weights"],
            bad_counts,
            case["idx"],
            LOCAL_EXPERT_ID,
            core_grid=(11, 8),
        )


def test_default_grid_is_the_full_compute_with_storage_grid(device):
    """Omitting core_grid and spelling out the full device grid select one geometry."""
    count = 64
    case = _build(device, seed=61, count=count)
    device_grid = device.compute_with_storage_grid_size()

    default_grid = _cpp_op()(
        case["x"],
        *case["weights"],
        case["counts"],
        case["idx"],
        LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
    )
    explicit_full_grid = _cpp_op()(
        case["x"],
        *case["weights"],
        case["counts"],
        case["idx"],
        LOCAL_EXPERT_ID,
        input_m_tiles=CAPACITY // TILE,
        core_grid=ttnn.CoreCoord(int(device_grid.x), int(device_grid.y)),
    )

    default_host = ttnn.to_torch(default_grid)[0, 0, :count]
    explicit_host = ttnn.to_torch(explicit_full_grid)[0, 0, :count]
    assert torch.equal(default_host, explicit_host)
    assert_with_pcc(_reference(case, count), default_host.float(), PCC_GATE)


def test_grid_larger_than_the_device_is_rejected_not_clamped(device, expect_error):
    case = _build(device, seed=62, count=0)
    device_grid = device.compute_with_storage_grid_size()
    oversized_grid = ttnn.CoreCoord(int(device_grid.x) + 1, int(device_grid.y))

    with expect_error(RuntimeError, r"(?i)requested grid.*exceeds device grid"):
        _cpp_op()(
            case["x"],
            *case["weights"],
            case["counts"],
            case["idx"],
            LOCAL_EXPERT_ID,
            core_grid=oversized_grid,
        )
    with expect_error(ValueError, r"(?i)requested grid.*exceeds device grid"):
        weight_memory_configs(device, EMB, HIDDEN, core_grid=(int(device_grid.x) + 1, int(device_grid.y)))


def test_n3072_bf16_rm_nd_sharded_crosses_multiple_m_blocks(device):
    """Exercise the L1-pressure fallback at N=3072 on an explicit 11x8 grid.

    Capacity 5120 selects the one-slot x/depth-2 h/non-resident-W_down geometry.  The input repeats
    every 256 rows, so every complete M block must produce exactly the same bytes.  Count 513 also
    enters a third, one-row tail block; comparing it to row zero catches both stale streamed weights
    and an incorrect runtime-M transition without requiring a 513x7168 fp32 host golden matmul.
    """
    emb, hidden, capacity, count = 7168, 3072, 5120, 513
    torch.manual_seed(3072)
    input_block = torch.randn((1, 1, 256, emb), dtype=torch.bfloat16)
    host_x = input_block.repeat(1, 1, capacity // 256, 1)
    host_weights = [torch.randn(shape, dtype=torch.bfloat16) for shape in ((emb, hidden), (emb, hidden), (hidden, emb))]
    gate_up_memory_config, down_memory_config = weight_memory_configs(device, emb, hidden, core_grid=(11, 8))
    tt_x = _to_device(host_x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    tt_weights = [
        ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        for weight, memory_config in zip(
            host_weights,
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor(
        [(11 + 37 * local) % NUM_GLOBAL_EXPERTS for local in range(NUM_LOCAL_EXPERTS)],
        dtype=torch.int32,
    )
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    tt_counts = _to_device(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    tt_idx = _to_device(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    output = _cpp_op()(
        tt_x,
        *tt_weights,
        tt_counts,
        tt_idx,
        LOCAL_EXPERT_ID,
        input_m_tiles=capacity // TILE,
        dtype=ttnn.bfloat8_b,
        core_grid=GRID,
    )
    host_output = ttnn.to_torch(output)[0, 0, :count].clone()
    assert torch.isfinite(host_output).all()
    assert torch.equal(
        host_output[:256], host_output[256:512]
    ), "N=3072: second M block differs for byte-identical input rows"
    assert torch.equal(
        host_output[0], host_output[512]
    ), "N=3072: one-row tail block differs from its identical row-zero input"
