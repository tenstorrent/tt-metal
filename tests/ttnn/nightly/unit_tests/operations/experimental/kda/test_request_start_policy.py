# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""External seeds are selected from device bounds, including under trace replay."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import (
    test_affine_exclusive_scan as affine,
    test_chain_affine_transforms as chain,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    make_actual_start,
    qkv_device_inputs,
    qkv_reference,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device({"l1_small_size": 24576})]


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_affine_request_policy_rejects_computed_sp_entries(mesh_device, expect_error):
    a, b, seed = affine._host_inputs(2, 4, 32, 32)
    a_tt, b_tt, seed_tt = (affine._to_device(t, mesh_device) for t in (a, b, seed))
    start = make_actual_start(mesh_device)
    with expect_error(RuntimeError, "requires SP1"):
        ttnn.experimental.kda.affine_exclusive_scan(
            a_tt,
            b_tt,
            seed_tt,
            4,
            actual_start=start,
            local_rows=128,
            tail_a=a_tt,
            tail_b=b_tt,
            tail_entry_states=seed_tt,
            zero_initial_state_on_start=True,
        )


@pytest.mark.parametrize("operation", ["chain", "affine", "convolution"])
@pytest.mark.parametrize("width", [32, 128])
def test_request_start_policy(device, operation, width):
    """Both static policies use one program each across absolute start changes.

    The enabled trace is captured at a positive offset, then reused for two requests.
    A NaN seed on the last restart makes accidental external reads visible.
    """
    start_tt = make_actual_start(device, 32)
    if operation == "chain":
        a, b, seed = chain._host_inputs(2, width, width)
        transforms, seed_tt = chain._device_inputs(a, b, seed, device)

        def run(enabled):
            return ttnn.experimental.kda.chain_affine_transforms(
                transforms,
                seed_tt,
                actual_start=start_tt,
                local_rows=32,
                zero_initial_state_on_start=enabled,
            )

        def oracle(initial):
            return chain._oracle(a, b, initial)

    elif operation == "affine":
        groups = 4
        a, b, seed = affine._host_inputs(2, groups, width, width)
        a_tt, b_tt, seed_tt = (affine._to_device(t, device) for t in (a, b, seed))

        def run(enabled):
            return (
                ttnn.experimental.kda.affine_exclusive_scan(
                    a_tt,
                    b_tt,
                    seed_tt,
                    groups,
                    actual_start=start_tt,
                    local_rows=32 * groups,
                    tail_a=a_tt,
                    tail_b=b_tt,
                    tail_entry_states=seed_tt,
                    zero_initial_state_on_start=enabled,
                ),
            )

        def oracle(initial):
            return (affine._oracle(a, b, initial, 2, groups),)

    else:
        widths = (width, width, width)
        (inputs, seed, taps), (input_tt, seed_tt, taps_tt) = qkv_device_inputs(device, widths=widths)

        def run(enabled):
            return ttnn.experimental.kda.qkv_causal_conv1d_silu(
                input_tt,
                seed_tt,
                *taps_tt,
                *widths,
                actual_start=start_tt,
                predecessor_carry=seed_tt,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=width),
                zero_initial_state_on_start=enabled,
            )

        def oracle(initial):
            return qkv_reference(inputs, initial, taps, widths)

    def set_start(value):
        source = make_actual_start(device, value)
        ttnn.copy(source, start_tt)
        ttnn.deallocate(source)

    def check(outputs, initial):
        for expected, output in zip(oracle(initial), outputs, strict=True):
            actual = ttnn.to_torch(output).float()
            if torch.count_nonzero(expected) == 0:
                torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
            else:
                assert_accurate(expected.float(), actual, name=operation, pcc_threshold=0.999)

    device.enable_program_cache()
    for enabled in (False, True):
        for start in (0, 32, 128, 0):
            set_start(start)
            before = device.num_program_cache_entries()
            outputs = run(enabled)
            after = device.num_program_cache_entries()
            # First execution of each static policy may compile; changing metadata must not.
            if start != 0:
                assert after == before
            check(outputs, torch.zeros_like(seed) if enabled and start == 0 else seed)
            for output in outputs:
                ttnn.deallocate(output)

    set_start(32)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = run(True)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for start in (0, 32, 0, 128):
            set_start(start)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            check(outputs, torch.zeros_like(seed) if start == 0 else seed)
        poison = ttnn.from_torch(
            torch.full_like(seed, float("nan")),
            dtype=seed_tt.dtype,
            layout=seed_tt.layout,
            device=device,
        )
        ttnn.copy(poison, seed_tt)
        ttnn.deallocate(poison)
        set_start(0)
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        check(outputs, torch.zeros_like(seed))
    finally:
        ttnn.release_trace(device, trace)
        for output in outputs:
            ttnn.deallocate(output)
