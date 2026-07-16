# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.gelu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    assert torch.allclose(golden_tensor[0].to(torch.bfloat16), ttnn.to_torch(tt_output_tensor_on_device[0]), atol=0.01)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "approximate, atol_value",
    (
        ("none", 0.01),
        ("tanh", 0.01),
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, atol_value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)

    assert torch.allclose(
        golden_tensor[0].to(torch.bfloat16), ttnn.to_torch(tt_output_tensor_on_device[0]), atol=atol_value
    )


def test_bw_gelu_program_cache_regression(device):
    """Program-cache regression guard for the gelu_bw descriptor migration.

    The op relies on two contracts that only ever exercise on a cache HIT (i.e. the second and
    later invocations):
      * the input/grad/output ``Buffer*`` bindings must be re-patched into the cached program on
        every dispatch. If they were frozen at compile time, a cache hit with freshly allocated
        tensors would read/write stale addresses and silently return the *first* call's gradients.
      * the ``approximate`` flag must select a distinct cached kernel (polynomial vs tanh). If it
        were not part of the program hash, the "tanh" call would reuse the "none" program (or vice
        versa) and compute the wrong gradient.

    This test enables the program cache and invokes each mode repeatedly with newly allocated
    inputs, outputs, and non-unit gradients (distinct seeds -> different data + different buffer
    addresses), checks correctness on every call, and asserts the expected cache reuse/separation.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = torch.Size([1, 1, 320, 384])
    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)

    def run_and_check(approximate, seed):
        # Fresh allocations (distinct seed -> distinct data and buffer addresses) each call.
        in_data, input_tensor = data_gen_with_range(shape, -100, 100, device, True, seed=seed)
        grad_data, grad_tensor = data_gen_with_range(shape, -5, 5, device, seed=seed + 1000)

        tt_out = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)

        golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
        assert compare_pcc(
            tt_out, golden_tensor, pcc=0.999
        ), f"gelu_bw(approximate={approximate!r}) mismatch on cache-enabled run (seed={seed})"

    # First call for each mode compiles a program; the two modes must NOT share a cache entry.
    run_and_check("none", seed=0)
    assert (
        device.num_program_cache_entries() == 1
    ), "first gelu_bw(approximate='none') must create exactly one cache entry"

    run_and_check("tanh", seed=1)
    assert device.num_program_cache_entries() == 2, (
        "gelu_bw(approximate='tanh') must create a SEPARATE cache entry from 'none' -- the "
        "approximation flag must be part of the program hash / select a distinct kernel."
    )

    # Re-run both modes with new buffers/data: must be cache HITS (no new entries) and still correct,
    # which only holds if the Buffer* bindings are re-patched on the fast path.
    run_and_check("none", seed=2)
    run_and_check("tanh", seed=3)
    assert device.num_program_cache_entries() == 2, (
        "re-running each mode with freshly allocated tensors must reuse the cached programs "
        "(no new entries). A new entry means the buffers/mode were wrongly folded into the hash."
    )

    logger.debug("gelu_bw program-cache regression: 2 entries, both modes correct across cache hits")

    device.disable_and_clear_program_cache()
