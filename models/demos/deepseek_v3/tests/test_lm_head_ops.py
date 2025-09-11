# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math
from textwrap import dedent

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import LinearConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    MAX_BATCH_SIZE,
    SEQ_LEN_CHUNK_SIZE,
    dram_sharded_weight_config,
    even_int_div,
    find_largest_divisor,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from models.demos.deepseek_v3.utils.run_config import _convert_run_config_to_pretty_print
from models.demos.deepseek_v3.utils.test_utils import assert_hidden_dim_pcc


def _roundup(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _build_decode_activation_memcfg(per_device_width: int, activation_sharding_num_cores: int, mesh_device: ttnn.Device):
    return ttnn.create_sharded_memory_config_(
        shape=(
            ttnn.core.roundup(MAX_BATCH_SIZE, ttnn.TILE_SIZE),
            ttnn.core.roundup(even_int_div(per_device_width, activation_sharding_num_cores), ttnn.TILE_SIZE),
        ),
        core_grid=ttnn.num_cores_to_corerangeset(
            activation_sharding_num_cores,
            ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y),
            row_wise=True,
        ),
        strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def _get_prefill_pc(seq_len: int, hidden_dim: int, vocab_size: int, num_devices: int, core_grid_size: ttnn.CoreCoord):
    per_device_in_features, per_device_out_features = hidden_dim, even_int_div(vocab_size, num_devices)

    per_core_M_tiles = ttnn.core.divup(seq_len, ttnn.TILE_SIZE * core_grid_size.y)
    K_tiles = ttnn.core.divup(per_device_in_features, ttnn.TILE_SIZE)
    per_core_N_tiles = ttnn.core.divup(per_device_out_features, ttnn.TILE_SIZE * core_grid_size.x)

    out_subblock_h = 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid_size,
        in0_block_w=find_largest_divisor(K_tiles),
        out_subblock_h=out_subblock_h,
        out_subblock_w=find_largest_divisor(per_core_N_tiles, out_subblock_h),
        per_core_M=per_core_M_tiles,
        per_core_N=per_core_N_tiles,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _make_weight_tensor(h: int, v: int, mesh_device: ttnn.Device):
    # Torch weight in standard PyTorch layout: [V, H]
    torch_weight = torch.randn(v, h, dtype=torch.bfloat16)
    # Convert to TTNN tensor in [H, V/D] sharded across mesh columns
    d = mesh_device.get_num_devices()
    per_device_v = even_int_div(v, d)
    weight_ttnn = ttnn.from_torch(
        torch_weight.T.contiguous(),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram_sharded_weight_config(h, per_device_v, mesh_device.dram_grid_size()),
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, -1),
    )
    return torch_weight, weight_ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_lm_head_linear_decode_ops(hf_config, mesh_device, set_deterministic_env):
    # Require Galaxy (32 devices) to match program/memory config derivations
    assert mesh_device.get_num_devices() == 32, "This test requires a 4x8 Galaxy (32 devices)."

    H = hf_config.hidden_size
    V = hf_config.vocab_size
    D = mesh_device.get_num_devices()
    seq_len = 32

    # Build weights (Torch + TTNN) and input
    torch_weight, weight_ttnn = _make_weight_tensor(H, V, mesh_device)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.bfloat16)

    max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
    input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(H, max_num_cores))
    output_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(even_int_div(V, D), max_num_cores))

    # Build activation memory configs (decode)
    input_memcfg = _build_decode_activation_memcfg(H, input_num_cores, mesh_device)
    output_memcfg = _build_decode_activation_memcfg(even_int_div(V, D), output_num_cores, mesh_device)

    # Program + compute configs
    program_cfg = get_dram_sharded_matmul_config(
        MAX_BATCH_SIZE, H, even_int_div(V, D), input_num_cores, output_num_cores
    )

    # Assemble op config for pretty-print validation
    run_cfg = {
        "linear": LinearConfig(
            input_tensor_b=weight_ttnn,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_cfg,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        )
    }

    # Pretty-print subset and compare to expected
    pretty = _convert_run_config_to_pretty_print(run_cfg)
    # Expected fields
    per_device_v = even_int_div(V, D)
    dram_cores = mesh_device.dram_grid_size().x
    padded_per_dram_core = _roundup(math.ceil(per_device_v / dram_cores), ttnn.TILE_SIZE)
    k_tiles = H // ttnn.TILE_SIZE
    in0_block_w = find_largest_divisor(k_tiles // input_num_cores)
    expected = dedent(
        f"""
        {{
          'linear': LinearConfig(
            input_tensor_b=ttnn.Tensor(shape=Shape([{H}, {per_device_v}]), dtype=DataType.BFLOAT4_B, memory=WIDTH_SHARDED_DRAM, sharded[{H}, {padded_per_dram_core}]),
            memory_config=MemoryConfig(layout=WIDTH_SHARDED, buffer=L1),
            compute_kernel_config=ComputeKernelConfig(math_fidelity=LoFi),
            program_config=MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w={in0_block_w}, ...)
          ),
        }}
        """
    ).strip()
    assert pretty.strip() == expected, f"Pretty config mismatch.\nGot:\n{pretty}\nExpected:\n{expected}"

    # Build TTNN input with the same memory layout as decode
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=input_memcfg,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run the op
    tt_output = ttnn.linear(tt_input, **run_cfg["linear"])  # type: ignore[arg-type]
    ttnn.deallocate(tt_input)

    # Validate output memory config
    assert tt_output.memory_config() == output_memcfg

    # Compare to Torch baseline
    torch_output_ref = torch_input @ torch_weight
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(tt_output)
    assert_hidden_dim_pcc(tt_output_torch, torch_output_ref, pcc_required=0.98)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("seq_len", [1024, 2048])  # includes chunking case
def test_lm_head_linear_prefill_ops(hf_config, mesh_device, set_deterministic_env, seq_len: int):
    # Require Galaxy (32 devices) to match program/memory config derivations
    assert mesh_device.get_num_devices() == 32, "This test requires a 4x8 Galaxy (32 devices)."

    H = hf_config.hidden_size
    V = hf_config.vocab_size
    D = mesh_device.get_num_devices()

    # Build weights (Torch + TTNN) and input
    torch_weight, weight_ttnn = _make_weight_tensor(H, V, mesh_device)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.bfloat16)

    # Prefill program + compute configs
    pc = _get_prefill_pc(seq_len, H, V, D, ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y))
    run_cfg = {
        "linear": LinearConfig(
            input_tensor_b=weight_ttnn,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            program_config=pc,
        )
    }

    # Pretty-print subset and compare to expected
    per_device_v = even_int_div(V, D)
    dram_cores = mesh_device.dram_grid_size().x
    padded_per_dram_core = _roundup(math.ceil(per_device_v / dram_cores), ttnn.TILE_SIZE)
    K_tiles = ttnn.core.divup(H, ttnn.TILE_SIZE)
    in0_block_w = find_largest_divisor(K_tiles)
    pretty = _convert_run_config_to_pretty_print(run_cfg)
    expected = dedent(
        f"""
        {{
          'linear': LinearConfig(
            input_tensor_b=ttnn.Tensor(shape=Shape([{H}, {per_device_v}]), dtype=DataType.BFLOAT4_B, memory=WIDTH_SHARDED_DRAM, sharded[{H}, {padded_per_dram_core}]),
            memory_config=MemoryConfig(layout=INTERLEAVED, buffer=DRAM),
            compute_kernel_config=ComputeKernelConfig(math_fidelity=LoFi),
            program_config=MatmulMultiCoreReuseMultiCastProgramConfig(in0_block_w={in0_block_w}, ...)
          ),
        }}
        """
    ).strip()
    assert pretty.strip() == expected, f"Pretty config mismatch.\nGot:\n{pretty}\nExpected:\n{expected}"

    # Build TTNN input for prefill
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Apply module chunking semantics if needed
    if seq_len > SEQ_LEN_CHUNK_SIZE:
        tt_input = ttnn.reshape(tt_input, [1, even_int_div(seq_len, SEQ_LEN_CHUNK_SIZE), SEQ_LEN_CHUNK_SIZE, -1])

    # Run the op
    tt_output = ttnn.linear(tt_input, **run_cfg["linear"])  # type: ignore[arg-type]
    ttnn.deallocate(tt_input)

    # De-chunk output if chunked
    _, num_chunks, _, out_dim = tt_output.shape
    if num_chunks > 1:
        tt_output = ttnn.reshape(tt_output, [1, 1, -1, out_dim])

    # Compare to Torch baseline
    torch_output_ref = torch_input @ torch_weight
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(tt_output)
    assert_hidden_dim_pcc(tt_output_torch, torch_output_ref, pcc_required=0.98)


if __name__ == "__main__":
    pytest.main([__file__])

