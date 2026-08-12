# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import csv

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import skip_for_wormhole_b0
from models.common.utility_functions import torch_random


NON_COMPARABLE_STABLE_OPERATIONS = {
    # Host/device lifecycle, transfers, serialization, and control.
    "ttnn.allocate_tensor_on_device",
    "ttnn.allocate_tensor_on_host",
    "ttnn.as_tensor",
    "ttnn.composite_example",
    "ttnn.composite_example_multiple_return",
    "ttnn.copy_device_to_host_tensor",
    "ttnn.copy_host_to_device_tensor",
    "ttnn.copy_host_to_device_tensor_partial",
    "ttnn.deallocate",
    "ttnn.dram_prefetcher",
    "ttnn.dump_tensor",
    "ttnn.from_buffer",
    "ttnn.generic_op",
    "ttnn.load_tensor",
    "ttnn.manual_seed",
    "ttnn.move",
    "ttnn.test_hang_device_operation",
    "ttnn.unary_chain",
    # Stochastic operations cannot reproduce the device RNG stream in Torch.
    "ttnn.bernoulli",
    "ttnn.rand",
    "ttnn.randn",
    "ttnn.sampling",
    "ttnn.uniform",
    # Cache and padding-buffer operations only mutate existing device state.
    "ttnn.fill_cache",
    "ttnn.fill_implicit_tile_padding",
    "ttnn.fill_ones_rm",
    "ttnn.fill_rm",
    "ttnn.kv_cache.fill_cache_for_user_",
    "ttnn.kv_cache.update_cache_for_token_",
    "ttnn.update_cache",
    # Model-oriented fused operations are outside stable golden coverage.
    "ttnn.moe",
    "ttnn.moe_expert_token_remap",
    "ttnn.moe_routing_remap",
    "ttnn.plus_one",
    "ttnn.fused_rms_minimal",
    "ttnn.transformer.chunk_gated_delta_rule",
    "ttnn.transformer.chunked_flash_mla_prefill",
    "ttnn.transformer.exp_ring_joint_scaled_dot_product_attention",
    "ttnn.transformer.flash_mla_prefill",
    "ttnn.transformer.flash_multi_latent_attention_decode",
    "ttnn.transformer.gated_delta_attn_seq",
    "ttnn.transformer.paged_flash_multi_latent_attention_decode",
    "ttnn.transformer.ring_distributed_scaled_dot_product_attention",
    "ttnn.transformer.ring_joint_scaled_dot_product_attention",
    "ttnn.transformer.ring_mla",
    # These collectives expose device-local or partially unspecified values that
    # cannot be represented by comparison mode's single composed CPU tensor.
    "ttnn.all_broadcast",
    "ttnn.all_reduce",
    "ttnn.all_to_all_combine",
    "ttnn.all_to_all_dispatch",
    "ttnn.point_to_point",
    "ttnn.reduce_scatter",
    "ttnn.reduce_to_root",
    # This operation returns a Python float rather than a comparable tensor.
    "ttnn.pearson_correlation_coefficient",
}


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("dim", [-1])
def test_softmax(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.softmax(torch_input_tensor, dim=dim, dtype=torch.bfloat16)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config("comparison_mode_pcc", 0.99):
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor = ttnn.to_device(input_tensor, device)
        output_tensor = ttnn.softmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_exp(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.exp(torch_input_tensor)

    with ttnn.manage_config("enable_comparison_mode", True):
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.exp(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1])
def test_failed_comparison(device, batch_size, h, w, dim, expect_error):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)

    ttnn.softmax.golden_function = lambda x, **_: x  # override the proper golden function implementation

    def run():
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor = ttnn.to_device(input_tensor, device)
        ttnn.softmax(input_tensor, dim=dim)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config("comparison_mode_pcc", 0.99):
        with ttnn.manage_config("comparison_mode_should_raise_exception", False):
            run()

        with ttnn.manage_config("comparison_mode_should_raise_exception", True):
            with expect_error(RuntimeError):
                run()


def test_dump_all_operations(tmp_path):
    csv_path = tmp_path / "all_ops.csv"
    ttnn.dump_operations(csv_path, include_experimental=True)
    # for local inspection
    ttnn.dump_operations("all_ops.csv", include_experimental=True)

    operations = {
        operation.python_fully_qualified_name: operation
        for operation in ttnn.query_registered_operations(include_experimental=True)
    }
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert {row["python_fully_qualified_name"] for row in rows} == set(operations)
    for row in rows:
        operation = operations[row["python_fully_qualified_name"]]
        try:
            ttnn.get_golden_function(operation)
            expected_has_golden_function = True
        except RuntimeError:
            expected_has_golden_function = False

        assert row["has_golden_function"] == str(expected_has_golden_function)


def test_stable_tensor_operations_have_golden_functions():
    operations = {
        operation.python_fully_qualified_name: operation
        for operation in ttnn.query_registered_operations(include_experimental=True)
    }

    assert NON_COMPARABLE_STABLE_OPERATIONS <= operations.keys()

    missing_golden_functions = sorted(
        name
        for name, operation in operations.items()
        if not name.startswith("ttnn.experimental.")
        and name not in NON_COMPARABLE_STABLE_OPERATIONS
        and operation.golden_function is None
    )

    assert (
        not missing_golden_functions
    ), "Stable tensor operations must provide golden functions. Missing: " + ", ".join(missing_golden_functions)
