# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.config_helpers import (
    MAX_BATCH_SIZE,
    even_int_div,
    get_activation_sharding_core_counts_for_dram_matmul,
)
from models.demos.deepseek_v3.utils.run_config import _convert_run_config_to_pretty_print
from models.demos.deepseek_v3.utils.composite_ops import mesh_scatter


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


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mesh_scatter_primitives_match_composite(hf_config, mesh_device, set_deterministic_env):
    # Ensure a Galaxy mesh for this test so row/col loops are meaningful
    assert mesh_device.get_num_devices() == 32, "This test requires a 4x8 Galaxy (32 devices)."

    rows, cols = tuple(mesh_device.shape)
    from_row = 0

    # Build a width-sharded L1 activation tensor similar to LMHead decode input
    H = hf_config.hidden_size
    seq_len = 64
    max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
    input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(H, max_num_cores))
    input_memcfg = _build_decode_activation_memcfg(H, input_num_cores, mesh_device)

    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.bfloat16)

    # Create two identical inputs to compare primitive vs composite behavior
    x_prim = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=input_memcfg,
        layout=ttnn.TILE_LAYOUT,
    )
    x_comp = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=input_memcfg,
        layout=ttnn.TILE_LAYOUT,
    )

    # 1) Primitive sequence mirroring composite behavior
    old_memcfg = x_prim.memory_config()
    if not old_memcfg.interleaved:
        inter_cfg = ttnn.L1_MEMORY_CONFIG if old_memcfg.buffer_type == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG
        tmp = ttnn.to_memory_config(x_prim, inter_cfg)
    else:
        tmp = x_prim
        old_memcfg = None

    # Scatter across rows from `from_row`, over all columns
    for to_row in range(rows):
        if to_row == from_row:
            continue
        for col in range(cols):
            ttnn.point_to_point(
                tmp,
                ttnn.MeshCoordinate(to_row, col),
                ttnn.MeshCoordinate(from_row, col),
                ttnn.Topology.Linear,
                optional_output_tensor=tmp,
            )

    # Restore original layout and copy back
    if old_memcfg is not None:
        reconfig = ttnn.to_memory_config(tmp, x_prim.memory_config())
        ttnn.deallocate(tmp)
        ttnn.copy(reconfig, x_prim)
        ttnn.deallocate(reconfig)

    # 2) Composite op
    mesh_scatter(x_comp, mesh_shape=tuple(mesh_device.shape), scatter_idx=(from_row, None))

    # Compare results by composing to torch; both should equal original torch_input
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    prim_out = ttnn.to_torch(x_prim, mesh_composer=composer)
    comp_out = ttnn.to_torch(x_comp, mesh_composer=composer)

    # Cleanup device tensors
    ttnn.deallocate(x_prim)
    ttnn.deallocate(x_comp)

    assert torch.allclose(prim_out, torch_input, atol=1e-2, rtol=1e-2)
    assert torch.allclose(comp_out, torch_input, atol=1e-2, rtol=1e-2)

    # 3) Validate pretty print for the mesh_scatter config dict
    cfg = {"mesh_scatter": {"mesh_shape": (rows, cols), "scatter_idx": (from_row, None)}}
    pretty = _convert_run_config_to_pretty_print(cfg)
    expected = dedent(
        f"""
        {{
          'mesh_scatter': {{
            'mesh_shape': ({rows}, {cols}),
            'scatter_idx': (
              {from_row},
              None,
            ),
          }},
        }}
        """
    ).strip()
    assert pretty.strip() == expected, f"Pretty config mismatch.\nGot:\n{pretty}\nExpected:\n{expected}"


if __name__ == "__main__":
    pytest.main([__file__])

