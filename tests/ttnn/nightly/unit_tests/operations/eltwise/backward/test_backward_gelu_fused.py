# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc

INPUT_SHAPES = (
    (torch.Size([32])),
    (torch.Size([25, 34])),  # not aligned by tile size
    (torch.Size([32, 32])),
    (torch.Size([1, 32, 32])),
    (torch.Size([1, 1, 32, 32])),
    (torch.Size([1, 1, 320, 384])),
    (torch.Size([1, 3, 320, 384])),
    (torch.Size([1, 3, 323, 389])),  # not aligned by tile size
)


def gen_data(input_shapes, low, high, device, required_grad=False, is_row_major=False, seed=213919):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(seed)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.TILE_LAYOUT).to(device)

    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate)

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.experimental.gelu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default_opt_output(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.experimental.gelu_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "grad_dtype, input_dtype",
    (
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
    ),
)
def test_bw_gelu_grad_dtype_must_match_input(grad_dtype, input_dtype, device, expect_error):
    shape = torch.Size([1, 1, 320, 384])
    torch.manual_seed(213919)

    in_data = (torch.rand(shape, dtype=torch.bfloat16) * 200 - 100).requires_grad_(True)
    grad_data = torch.rand(shape, dtype=torch.bfloat16) * 10 - 5

    input_tensor = ttnn.from_torch(in_data.detach(), input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if grad_dtype != input_dtype:
        with expect_error(RuntimeError, "grad_output and input data types to match"):
            ttnn.experimental.gelu_bw(grad_tensor, input_tensor)
        return

    result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor)
    assert result.dtype == input_dtype

    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)
    assert compare_pcc([result], golden_function(grad_data, in_data))


def test_bw_gelu_program_cache_regression(device):
    """Program-cache regression guard for the experimental gelu_bw Metal 2.0 port.

    The Metal 2.0 factory binds tensors as named TensorParameters instead of passing buffer
    addresses through runtime args, so on a program-cache HIT the framework re-resolves each
    TensorArgument into the cached program rather than the op re-emitting addresses. A stale
    binding would keep pointing at the first call's buffers and silently produce wrong results
    (or read freed memory) on every later call.

    Every call below allocates fresh inputs/outputs (distinct seeds -> different data AND
    different buffer addresses), asserts correctness, and asserts the cache reused the program
    instead of compiling a new one. Both compute kernels ("none" -> poly, "tanh") and the
    preallocated-output path are covered, since each binds a different set of tensors.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = torch.Size([1, 1, 320, 384])  # multi-tile, spans several cores
    golden_function = ttnn.get_golden_function(ttnn.experimental.gelu_bw)

    def fresh_inputs(seed):
        in_data, input_tensor = gen_data(shape, -100, 100, device, True, seed=seed)
        grad_data, grad_tensor = gen_data(shape, -5, 5, device, seed=seed + 1000)
        return in_data, input_tensor, grad_data, grad_tensor

    def check(result, grad_data, in_data, label):
        assert compare_pcc([result], golden_function(grad_data, in_data)), f"{label}: output mismatch"

    # --- Cache misses: the two approximations must not share an entry ---
    in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=0)
    result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="none")
    check(result, grad_data, in_data, "approximate='none' (first call)")
    assert device.num_program_cache_entries() == 1, "first gelu_bw(approximate='none') must create exactly one entry"

    in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=1)
    result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate="tanh")
    check(result, grad_data, in_data, "approximate='tanh' (first call)")
    assert device.num_program_cache_entries() == 2, (
        "gelu_bw(approximate='tanh') must create a SEPARATE cache entry from 'none' -- the "
        "approximation selects a different compute kernel, so it must be part of the program hash."
    )

    # --- Cache hits: re-run both modes with new buffers. This is the rebinding check. ---
    for seed, approximate in ((42, "none"), (99, "tanh"), (7, "none"), (13, "tanh")):
        in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=seed)
        result = ttnn.experimental.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
        check(result, grad_data, in_data, f"approximate={approximate!r} cache hit (seed={seed})")
        assert device.num_program_cache_entries() == 2, (
            f"re-running approximate={approximate!r} with freshly allocated tensors must reuse the "
            "cached program (no new entry). A new entry means buffers were wrongly folded into the hash."
        )

    # --- Cache hits on the preallocated-output path (binds OUTPUT to a caller-owned tensor) ---
    for approximate in ("none", "tanh"):
        entries_before = None
        for seed in (100, 200):
            in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=seed)
            input_grad = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.experimental.gelu_bw(
                grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=0
            )
            check(input_grad, grad_data, in_data, f"approximate={approximate!r} preallocated output (seed={seed})")

            if entries_before is None:
                entries_before = device.num_program_cache_entries()
            else:
                assert device.num_program_cache_entries() == entries_before, (
                    f"re-running approximate={approximate!r} with a fresh preallocated output must reuse "
                    "the cached program; the caller-owned output tensor must be rebound, not re-hashed."
                )

    device.disable_and_clear_program_cache()
