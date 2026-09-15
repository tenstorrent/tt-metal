# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics, assert_with_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import ttnn_softmax
from models.common.utility_functions import torch_random

# Module-scoped device: every test here shares one device configuration
pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("shape", [[2, 10, 512, 8192]])
def test_ttnn_softmax_sdxl_attention(device, shape):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_output = F.softmax(torch_input, dim=-1, dtype=torch.bfloat16)
    tt_output = ttnn_softmax(tt_input, dim=-1, numeric_stable=True)
    tt_output_torch = ttnn.to_torch(tt_output)
    # PCC at bf16 on Wt=256 numeric-stable softmax is sensitive to the small/large-kernel
    # choice in softmax_program_factory_attention_optimized.cpp; 0.998 is within bf16 noise.
    assert_numeric_metrics(
        torch_output,
        tt_output_torch,
        pcc_threshold=0.998,
        rtol=0.136,
        atol=0.001,
        frobenius_threshold=0.068,
    )

    # test program cache
    torch_output2 = F.softmax(torch_output, dim=-1, dtype=torch.bfloat16)
    tt_output2 = ttnn_softmax(tt_output, dim=-1, numeric_stable=True)
    tt_output_torch2 = ttnn.to_torch(tt_output2)
    assert_numeric_metrics(
        torch_output2,
        tt_output_torch2,
        pcc_threshold=0.998,
        rtol=0.136,
        atol=0.001,
        frobenius_threshold=0.068,
    )


@pytest.mark.parametrize("target_sequence_size", [8192])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("iterations", [50])
def test_transformer_attention_softmax_inplace_large_kernel_stress(
    device, target_sequence_size, sequence_size, iterations
):
    """Regression test for #45200: in-place attention_softmax_ on a large-kernel tensor"""
    torch.manual_seed(0)

    input_shape = (1, 1, sequence_size, target_sequence_size)
    torch_input_tensor = torch_random(input_shape, -5.0, 5.0, dtype=torch.bfloat16)
    torch_attention_mask = torch_random(input_shape, 0, 1.0, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    torch_output_tensor = golden_function(torch_input_tensor, head_size=None, attention_mask=torch_attention_mask)

    # Mask is read-only across iterations, so upload once.
    attention_mask = ttnn.from_torch(torch_attention_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    first_output = None
    for i in range(iterations):
        # Re-upload the input every iteration because attention_softmax_ mutates it in place.
        input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        input_addr = input_tensor.buffer_address()
        output_tensor = ttnn.transformer.attention_softmax_(
            input_tensor, head_size=None, attention_mask=attention_mask, causal_mask=True
        )
        # The op must remain truly in-place (output aliases the input buffer).
        assert output_tensor.buffer_address() == input_addr
        output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))
        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)

        if first_output is None:
            first_output = output_tensor
        else:
            assert torch.equal(first_output, output_tensor), (
                f"In-place large-kernel softmax is non-deterministic at iteration {i}: "
                f"output differs from iteration 0 (max abs delta "
                f"{(first_output - output_tensor).abs().max().item()})"
            )


# A softmax row must sum to 1. The metrics the other tests here use cannot see a violation of it:
# PCC is invariant under a uniform per-row scale, and atol=0.04 is orders of magnitude above values
# of order 1/w. With the SUM reduce accumulating in a 16-bit Dest this measured 0.865 at w=8192
# while test_large_softmax passed at the same shapes.
#
# The shapes span both kernels and both ranks. The estimator in
# softmax_program_factory_attention_optimized.cpp switches to the large kernel once the circular
# buffers reach 90% of L1, which on these rows sits above w=8192, so w=32768 exercises the
# accumulator that round-trips through L1 once per pass and the shorter rows do not.
@pytest.mark.parametrize(
    "shape",
    [(1, 32, 1024), (1, 32, 8192), (1, 1, 32, 2048), (1, 1, 32, 8192), (1, 1, 32, 32768), (1, 1, 32, 128000)],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_softmax_rows_are_normalised(device, shape, dtype):
    torch.manual_seed(0)
    torch_input_tensor = torch_random(shape, -1, 1, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.to_torch(ttnn.from_device(ttnn.softmax(input_tensor, dim=-1, numeric_stable=True)))

    row_sums = output_tensor.to(torch.float32).sum(-1)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=0.02
    ), f"row sums range [{float(row_sums.min()):.5f}, {float(row_sums.max()):.5f}], expected 1.0"
