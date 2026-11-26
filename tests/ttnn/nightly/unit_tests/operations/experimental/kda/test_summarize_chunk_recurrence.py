# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.summarize_chunk_recurrence: each group-head's chunk recurrence as one affine transform
S_after = A @ S_before + B, stored in complement form as K rows of [E | B], E = A - I.

Reached by: K3 and K3.1 prefill (sequence-parallel grouped summary).
Follows the KDA single-op test standard (README.md).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_op_checks as checks
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    CHUNK_SIZE,
    PROTOCOL_NAMES,
    assert_summary_reconstructs_state,
    device_protocol,
    group_summary_nd_sharded,
    host_protocol,
    initial_state,
    pack_summary,
    recurrent_oracle,
    summary_oracle,
    to_device,
    unpack_summary,
)
from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device({"l1_small_size": 24576})]

# Every enforced checklist item has a test here, so nothing is declared not applicable.
NOT_APPLICABLE: dict[str, str] = {}


# --- Cases ---


@dataclass(frozen=True)
class Case:
    id: str
    batch_heads: int
    num_chunks: int
    dim: int
    seed: int
    # Gate of the packed summaries, measured for this case and seed.
    tolerance: checks.Tolerance
    bf16: frozenset[str] = frozenset()
    production: bool = False
    # Device-time reference by `init`; performance cases only.
    reference_ns: Mapping[bool, int] | None = None


# The sequence-parallel grouped summary keeps these protocol tensors in BF16.
PRODUCTION_BF16 = frozenset({"kd", "q_decay", "final_decay"})

# Development shape. Perf references: complement-form [E | B] summary, measured on a P150b (12x10 grid) 2026-11-02.
UNIT = Case(
    "unit-bh8-n4-d32",
    batch_heads=8,
    num_chunks=4,
    dim=32,
    seed=1911,
    tolerance=checks.Tolerance(nrmse=0.005),
    reference_ns={True: 15_900, False: 15_200},
)
# Kimi-K3 on Galaxy SP8xTP4: one device's 640 tokens of 24 heads.
# The tolerance comes from the former conditioning test, which gated A and B (not the packed E) at this shape:
# re-measure it for the packed output.
K3_GALAXY = Case(
    "k3-galaxy-sp8tp4",
    batch_heads=24,
    num_chunks=20,
    dim=128,
    seed=117,
    tolerance=checks.Tolerance(nrmse=0.02),
    bf16=PRODUCTION_BF16,
    production=True,
    reference_ns={True: 132_317, False: 123_524},
)
SHAPES = (
    Case("single-chunk-fp32", 2, 1, 32, seed=811, tolerance=checks.Tolerance(nrmse=0.004)),
    Case("three-chunk-all-bf16", 4, 3, 64, seed=811, tolerance=checks.Tolerance(nrmse=0.004), bf16=BF16_ALLOWED),
    Case(
        "grouped-four-chunk",
        8,
        4,
        32,
        seed=811,
        tolerance=checks.Tolerance(nrmse=0.004),
        bf16=frozenset({"v_beta", "kd", "final_decay"}),
    ),
)


@dataclass(frozen=True)
class Placement:
    id: str
    memory_config: Callable[[ttnn.MeshDevice, Case, int], ttnn.MemoryConfig]
    tail: bool


def _height_sharded(device: ttnn.MeshDevice, case: Case, slots: int) -> ttnn.MemoryConfig:
    """One [K, 2K] transform per core, slot-major."""
    cores = ttnn.num_cores_to_corerangeset(
        slots * case.batch_heads, device.compute_with_storage_grid_size(), row_wise=True
    )
    return ttnn.create_sharded_memory_config(
        (case.dim, 2 * case.dim),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _nd_sharded(device: ttnn.MeshDevice, case: Case, slots: int) -> ttnn.MemoryConfig:
    """The grouped placement: each group-head's head and tail rows form one L1 shard on its own core."""
    return group_summary_nd_sharded(device, slots, case.batch_heads, case.dim, case.dim)


PLACEMENTS = (
    Placement("dram-interleaved", lambda device, case, slots: ttnn.DRAM_MEMORY_CONFIG, tail=False),
    Placement("nd-sharded-l1", _nd_sharded, tail=True),
    Placement("height-sharded-l1", _height_sharded, tail=True),
)
PLACEMENT_CASES = (
    Case("bh4-n2-d32", 4, 2, 32, seed=812, tolerance=checks.Tolerance(nrmse=0.004)),
    Case("bh4-n2-d128", 4, 2, 128, seed=812, tolerance=checks.Tolerance(nrmse=0.004)),
)

PERF_MARGIN = 0.05


# --- Harness: the only code that knows the op's protocol ---


def host_inputs(case: Case, seed: int | None = None) -> tuple[torch.Tensor, ...]:
    return host_protocol(
        case.batch_heads,
        case.num_chunks,
        case.dim,
        case.dim,
        bf16_names=case.bf16,
        seed=case.seed if seed is None else seed,
    )


def reference(host: Sequence[torch.Tensor]) -> torch.Tensor:
    """The head slot [1, N, K, 2K] of the packed summaries."""
    return pack_summary(*summary_oracle(host)).unsqueeze(0)


def identity_tail(case: Case) -> torch.Tensor:
    """What an unsplit rank's tail slots hold with init: the identity transform, (E, B) = (0, 0)."""
    return pack_summary(torch.eye(case.dim), torch.zeros(case.dim, case.dim)).expand(case.batch_heads, -1, -1)


def device_inputs(host: Sequence[torch.Tensor], device: ttnn.MeshDevice) -> tuple[ttnn.Tensor, ...]:
    return device_protocol(host, device)


def run(inputs: Sequence[ttnn.Tensor], *, actual_start: ttnn.Tensor, **options):
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, actual_start=actual_start, **options)


def production_options(device: ttnn.MeshDevice, case: Case) -> dict:
    """How the model calls the op: grouped ND-sharded head and tail summaries, HiFi4 with FP32 accumulation."""
    return {
        "tail": True,
        "memory_config": _nd_sharded(device, case, slots=2),
        "compute_kernel_config": explicit_default_compute_config(device),
    }


def explicit_default_compute_config(device: ttnn.MeshDevice) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
        throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
    )


def work(inputs: Sequence[torch.Tensor | ttnn.Tensor], outputs: Sequence[torch.Tensor | ttnn.Tensor]):
    """FPU and SFPU work of one call: per (head, chunk), the protocol's matmuls on F(0) and F(I)."""
    if len(inputs) != 7 or len(outputs) != 1:
        raise ValueError("chunk-recurrence summary requires seven inputs and one packed output")
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in (*inputs, *outputs)):
        raise ValueError("chunk-recurrence summary tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")
    batch_heads, num_chunks, chunk_size, value_dim = inputs[0].shape
    key_dim = inputs[1].shape[-1]
    expected_input_shapes = (
        (batch_heads, num_chunks, CHUNK_SIZE, value_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, 1),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if (
        chunk_size != CHUNK_SIZE
        or key_dim != value_dim
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or len(outputs[0].shape) != 4
        or tuple(outputs[0].shape)[0] not in (1, 2)
        or tuple(outputs[0].shape)[1:] != (batch_heads, key_dim, key_dim + value_dim)
    ):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")
    instances = batch_heads * num_chunks
    return (
        checks.perf_model.FpuOps(
            matrix_flops=instances * (8 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
            multiply_ops=instances * 2 * key_dim * value_dim,
            add_ops=instances * 2 * CHUNK_SIZE * value_dim,
        ),
        checks.perf_model.SfpuOps(),
    )


# --- 1. Correct values ---


@pytest.mark.parametrize("case", checks.case_params(*SHAPES, UNIT, K3_GALAXY))
def test_summarize_chunk_recurrence_accuracy(device, zero_actual_start, case: Case) -> None:
    host = host_inputs(case)
    options = production_options(device, case) if case.production else {}
    summaries = run(device_inputs(host, device), actual_start=zero_actual_start, **options)
    checks.assert_close(reference(host), ttnn.to_torch(summaries)[:1], case.tolerance, name=f"{case.id} summaries")


@pytest.mark.parametrize("case", checks.case_params(*SHAPES, UNIT))
def test_summarize_chunk_recurrence_matches_scanned_state(device, zero_actual_start, case: Case) -> None:
    """Applied to a random state, the emitted transform gives the state the recurrence scans to."""
    host = host_inputs(case)
    summaries = run(device_inputs(host, device), actual_start=zero_actual_start)
    affine_a, affine_b = unpack_summary(ttnn.to_torch(summaries)[0])
    assert_summary_reconstructs_state(host, affine_a, affine_b, nrmse_threshold=case.tolerance.nrmse)


@pytest.mark.parametrize(
    "case",
    checks.case_params(
        Case("three-chunk-fp32", 2, 3, 32, seed=813, tolerance=checks.Tolerance(nrmse=0.004)),
        Case(
            "four-chunk-production-bf16",
            4,
            4,
            128,
            seed=813,
            tolerance=checks.Tolerance(nrmse=0.004),
            bf16=PRODUCTION_BF16,
        ),
    ),
)
def test_summarize_chunk_recurrence_matches_recurrent_scan(device, zero_actual_start, case: Case) -> None:
    """Unsplit from the sequence start, every chunk enters at the entry state E: the emitted F(0) and F(I) give the
    recurrent scan's outputs as F(0) + (F(I) - F(0)) @ E, and the emitted final transform, in complement form
    (E_A = A - I, B), its final state E + (E_A @ E + B)."""
    host = host_inputs(case)
    entry = initial_state(case.batch_heads, case.dim, case.dim, seed=3142)
    expected_output, expected_final = recurrent_oracle(host, entry)
    expected_a, expected_b = summary_oracle(host)

    packed, terms, final_e, final_b = run(
        device_inputs(host, device), actual_start=zero_actual_start, emit_outputs=True
    )

    # Interleaved, a terms block is a head's chunks: its F(0) rows, then its F(I) rows.
    terms_shape = (case.batch_heads, 2, case.num_chunks * CHUNK_SIZE, case.dim)
    for tensor, shape in (
        (terms, terms_shape),
        (final_e, (case.batch_heads, case.dim, case.dim)),
        (final_b, (case.batch_heads, case.dim, case.dim)),
    ):
        assert tensor.dtype == ttnn.float32
        assert tuple(tensor.shape) == shape
    zero, identity = ttnn.to_torch(terms).reshape(case.batch_heads, 2, case.num_chunks, CHUNK_SIZE, case.dim).unbind(1)
    e, b = ttnn.to_torch(final_e), ttnn.to_torch(final_b)
    tolerance = case.tolerance
    checks.assert_close(
        expected_output,
        (zero + torch.matmul(identity - zero, entry.unsqueeze(1))).flatten(1, 2),
        tolerance,
        name="chunk outputs from the emitted terms",
    )
    checks.assert_close(
        expected_final, entry + (torch.matmul(e, entry) + b), tolerance, name="final state from the emitted transform"
    )
    checks.assert_close(expected_a - torch.eye(case.dim), e, tolerance, name="emitted final_e")
    checks.assert_close(expected_b, b, tolerance, name="emitted final_b")
    # The packed head is the same complement-form transform rounded to BFLOAT16.
    packed_host = ttnn.to_torch(packed).float()[0]
    checks.assert_close(e, packed_host[:, :, : case.dim], tolerance, name="packed head E")
    checks.assert_close(b, packed_host[:, :, case.dim :], tolerance, name="packed head B")


@pytest.mark.parametrize("case", checks.case_params(UNIT))
def test_summarize_chunk_recurrence_compute_config_approximate_math_stays_accurate(
    device, zero_actual_start, isolated_program_cache, case: Case
) -> None:
    """Approximate math gets its own program and stays as accurate as exact math."""
    host = host_inputs(case, seed=818)
    inputs = device_inputs(host, device)
    approximate = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    exact, approximated = checks.assert_programs_added(
        device,
        [
            partial(run, inputs, actual_start=zero_actual_start),
            partial(run, inputs, actual_start=zero_actual_start, compute_kernel_config=approximate),
        ],
        [None, 1],
    )
    for label, summaries in (("exact", exact), ("approximate", approximated)):
        checks.assert_close(reference(host), ttnn.to_torch(summaries), case.tolerance, name=f"{label} math summaries")


# --- 3. Repeatability ---


@pytest.mark.parametrize("case", checks.case_params(UNIT))
def test_summarize_chunk_recurrence_is_deterministic(device, zero_actual_start, case: Case) -> None:
    inputs = device_inputs(host_inputs(case, seed=1441), device)
    checks.assert_deterministic(device, partial(run, inputs, actual_start=zero_actual_start))


# --- 4. Interface contract ---


@pytest.mark.parametrize("case", PLACEMENT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("placement", PLACEMENTS, ids=lambda placement: placement.id)
def test_summarize_chunk_recurrence_output_contract(
    device, zero_actual_start, placement: Placement, case: Case
) -> None:
    """BF16 [S, N, K, 2K] in the requested memory; with tail, an unsplit rank's tail slots hold the identity.
    The placement changes where values land, not what they are: the head slot equals the interleaved result."""
    slots = 1 + placement.tail
    inputs = device_inputs(host_inputs(case), device)
    memory_config = placement.memory_config(device, case, slots)
    (summaries,) = checks.assert_output_contract(
        inputs,
        partial(run, inputs, actual_start=zero_actual_start, memory_config=memory_config, tail=placement.tail),
        dtypes=(ttnn.bfloat16,),
        shapes=((slots, case.batch_heads, case.dim, 2 * case.dim),),
        memory_config=memory_config,
    )
    placed = ttnn.to_torch(summaries)
    interleaved = ttnn.to_torch(run(inputs, actual_start=zero_actual_start))
    checks.assert_bit_identical(interleaved[0], placed[0], name=f"{placement.id} head slot vs interleaved")
    if placement.tail:
        checks.assert_bit_identical(identity_tail(case).bfloat16(), placed[1], name=f"{placement.id} tail identity")


def _host_tensor(index: int):
    def mutate(device, host, inputs):
        tensor = host[index]
        dtype = ttnn.bfloat16 if tensor.dtype == torch.bfloat16 else ttnn.float32
        inputs[index] = ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return inputs, {}

    return mutate


def _dtype(index: int, dtype: ttnn.DataType):
    def mutate(device, host, inputs):
        inputs[index] = to_device(host[index], device, dtype=dtype)
        return inputs, {}

    return mutate


def _key_value_mismatch(device, host, inputs):
    return list(device_protocol(host_protocol(2, 2, 32, 64), device)), {}


def _value_tiles_exceed_dest(device, host, inputs):
    # One head per core leaves no room to split the 8 value tiles, and FP32 half-sync dest holds 4.
    grid = device.compute_with_storage_grid_size()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), fp32_dest_acc_en=True, dst_full_sync_en=False, packer_l1_acc=False
    )
    return list(device_protocol(host_protocol(grid.x * grid.y, 1, 256, 256), device)), {"compute_kernel_config": config}


def _output_sharding(memory_layout: ttnn.TensorMemoryLayout):
    def mutate(device, host, inputs):
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [64, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return inputs, {"memory_config": ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)}

    return mutate


def _compute_config(**fields):
    def mutate(device, host, inputs):
        config = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, **fields)
        return inputs, {"compute_kernel_config": config}

    return mutate


# id -> (mutation of the valid call, the guard's message). A mutation gets (device, host, inputs) and returns the
# mutated (inputs, options).
INVALID_INPUTS = {
    **{
        f"host-{name}": (_host_tensor(index), f"{name} must be an allocated device tensor")
        for index, name in enumerate(PROTOCOL_NAMES)
    },
    "key-value-mismatch": (_key_value_mismatch, "K must equal V"),
    "q-decay-bfloat8-b": (_dtype(2, ttnn.bfloat8_b), "q_decay must be FLOAT32 or BFLOAT16"),
    "intra-bfloat16": (_dtype(3, ttnn.bfloat16), "intra must be FLOAT32"),
    "value-tiles-exceed-dest": (_value_tiles_exceed_dest, "value tiles exceed the 4 destination-register tiles"),
}
INVALID_OPTIONS = {
    "width-sharded-output": (
        _output_sharding(ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        "memory_config must use interleaved, height-sharded or ND-sharded memory",
    ),
    "block-sharded-output": (
        _output_sharding(ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        "memory_config must use interleaved, height-sharded or ND-sharded memory",
    ),
}
UNSUPPORTED_COMPUTE_CONFIGS = {
    "packer-l1-acc": (_compute_config(packer_l1_acc=True, fp32_dest_acc_en=True), "packer_l1_acc=true is unsupported"),
    "bf16-dest": (
        _compute_config(packer_l1_acc=False, fp32_dest_acc_en=False),
        "fp32_dest_acc_en=false is unsupported",
    ),
}
REFUSAL_CASE = Case("bh2-n2-d32", 2, 2, 32, seed=1731, tolerance=checks.Tolerance(nrmse=0.004))


def _assert_refused(device, actual_start, expect_error, mutation, message) -> None:
    host = host_inputs(REFUSAL_CASE)
    inputs, options = mutation(device, host, list(device_inputs(host, device)))
    with expect_error(RuntimeError, message):
        run(inputs, actual_start=actual_start, **options)


@pytest.mark.parametrize(("mutation", "message"), list(INVALID_INPUTS.values()), ids=list(INVALID_INPUTS))
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    device, zero_actual_start, expect_error, mutation, message
) -> None:
    _assert_refused(device, zero_actual_start, expect_error, mutation, message)


@pytest.mark.parametrize(("mutation", "message"), list(INVALID_OPTIONS.values()), ids=list(INVALID_OPTIONS))
def test_summarize_chunk_recurrence_rejects_invalid_options(
    device, zero_actual_start, expect_error, mutation, message
) -> None:
    _assert_refused(device, zero_actual_start, expect_error, mutation, message)


@pytest.mark.parametrize(
    ("mutation", "message"), list(UNSUPPORTED_COMPUTE_CONFIGS.values()), ids=list(UNSUPPORTED_COMPUTE_CONFIGS)
)
def test_summarize_chunk_recurrence_rejects_unsupported_compute_config(
    device, zero_actual_start, expect_error, mutation, message
) -> None:
    _assert_refused(device, zero_actual_start, expect_error, mutation, message)


@pytest.mark.parametrize("case", checks.case_params(UNIT))
def test_summarize_chunk_recurrence_default_compute_config_matches_explicit(
    device, zero_actual_start, isolated_program_cache, case: Case
) -> None:
    inputs = device_inputs(host_inputs(case, seed=817), device)
    checks.assert_default_config_matches(
        device,
        partial(run, inputs, actual_start=zero_actual_start),
        partial(
            run, inputs, actual_start=zero_actual_start, compute_kernel_config=explicit_default_compute_config(device)
        ),
    )


@pytest.mark.parametrize(
    "removed_keyword",
    [
        "chunk_size",
        "initial_state",
        "state_only",
        "identity_tile",
        "summary_pair",
        "output_bf16",
        "raw_seed",
        "packed_head",
    ],
)
def test_summarize_chunk_recurrence_does_not_expose_removed_modes(
    device, zero_actual_start, expect_error, removed_keyword: str
) -> None:
    inputs = device_inputs(host_inputs(REFUSAL_CASE), device)
    with expect_error(TypeError, "incompatible function arguments"):
        run(inputs, actual_start=zero_actual_start, **{removed_keyword: True})


# --- 5. Reuse ---


@pytest.mark.parametrize("case", checks.case_params(UNIT))
def test_summarize_chunk_recurrence_cache_hit_rebinds_fresh_tensors(device, zero_actual_start, case: Case) -> None:
    hosts = (host_inputs(case, seed=1911), host_inputs(case, seed=1912))
    checks.assert_cache_hit_rebinds(
        device,
        partial(run, actual_start=zero_actual_start),
        tuple(device_inputs(host, device) for host in hosts),
        tuple((reference(host),) for host in hosts),
        case.tolerance,
    )


@pytest.mark.parametrize("case", checks.case_params(UNIT))
def test_summarize_chunk_recurrence_program_key_includes_tail_and_init(
    device, zero_actual_start, isolated_program_cache, case: Case
) -> None:
    """tail and init change what the writer emits, so each combination owns a program; an unsplit rank's tail slots
    are undefined, so init decides whether they hold the identity."""
    host = host_inputs(case, seed=1913)
    inputs = device_inputs(host, device)
    combinations = ((False, True), (True, True), (True, False), (False, False))
    outputs = checks.assert_programs_added(
        device,
        [partial(run, inputs, actual_start=zero_actual_start, tail=tail, init=init) for tail, init in combinations],
        [1] * len(combinations),
    )
    for (tail, init), summaries in zip(combinations, outputs, strict=True):
        assert tuple(summaries.shape) == (1 + tail, case.batch_heads, case.dim, 2 * case.dim)
        host_summaries = ttnn.to_torch(summaries).float()
        checks.assert_close(
            reference(host)[0], host_summaries[0], case.tolerance, name=f"head slot tail={tail} init={init}"
        )
        if tail and init:
            checks.assert_bit_identical(
                identity_tail(case).bfloat16(), host_summaries[1].bfloat16(), name="unsplit tail identity"
            )


# --- 7. Performance ---


def test_summarize_chunk_recurrence_work_golden() -> None:
    inputs = (
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 2)),
        torch.empty((1, 1, 32, 32)),
        torch.empty((1, 1, 2, 32)),
        torch.empty((1, 1, 2, 1)),
        torch.empty((1, 1, 32, 32)),
    )
    fpu, sfpu = work(inputs, (torch.empty((2, 1, 2, 4)),))
    assert fpu == checks.perf_model.FpuOps(matrix_flops=9216, multiply_ops=8, add_ops=128)
    assert sfpu == checks.perf_model.SfpuOps()


@checks.performance_test
@pytest.mark.parametrize("init", [True, False], ids=["identity-fill", "no-fill"])
@pytest.mark.parametrize("case", checks.case_params(UNIT, K3_GALAXY))
def test_summarize_chunk_recurrence_performance(device, zero_actual_start, case: Case, init: bool) -> None:
    """With tail, an unsplit rank's tail slots are all undefined, so identity-fill measures the worst-case fill."""
    inputs = device_inputs(host_inputs(case, seed=117), device)
    options = production_options(device, case) if case.production else {"tail": True}

    def utilization(summaries, duration_ns):
        fpu, sfpu = work(inputs, (summaries,))
        return checks.perf_model.performance(
            fpu=fpu,
            sfpu=sfpu,
            inputs=inputs,
            outputs=(summaries,),
            measured_ns=duration_ns,
            math_fidelity=ttnn.MathFidelity.HiFi4,
        )

    summaries = checks.assert_device_time(
        device,
        partial(run, inputs, actual_start=zero_actual_start, init=init, **options),
        reference_ns=case.reference_ns[init],
        margin=PERF_MARGIN,
        label=f"summarize_chunk_recurrence {case.id} init={init}",
        utilization=utilization,
    )
    assert tuple(summaries.shape) == (2, case.batch_heads, case.dim, 2 * case.dim)
