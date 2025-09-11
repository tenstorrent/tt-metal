# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math
from textwrap import dedent

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    EmbeddingConfig,
    TypecastConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import even_int_div
from models.demos.deepseek_v3.utils.run_config import _convert_run_config_to_pretty_print


def _roundup(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _make_embedding_weight_tensors(v: int, h: int, mesh_device: ttnn.Device):
    # Torch weight in standard PyTorch layout: [V, H]
    torch_weight = torch.randn(v, h, dtype=torch.bfloat16)

    # Shard along last dim across the mesh (columns) → per-device [V, H/D]
    d = mesh_device.get_num_devices()
    weight_ttnn = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, -1),
    )
    return torch_weight, weight_ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_embedding_prefill_ops(hf_config, mesh_device, set_deterministic_env):
    # Require Galaxy mesh for parity with demo configs
    assert mesh_device.get_num_devices() == 32, "This test requires a 4x8 Galaxy (32 devices)."

    H = hf_config.hidden_size
    V = hf_config.vocab_size
    D = mesh_device.get_num_devices()
    per_device_h = even_int_div(H, D)

    # Input ids with seq_len divisible by tile size (no padding path)
    seq_len = 64
    torch_ids = torch.randint(low=0, high=V, size=(1, 1, seq_len), dtype=torch.int32)

    # Build weights and ids on device
    torch_weight, weight_ttnn = _make_embedding_weight_tensors(V, H, mesh_device)
    ids_ttnn = ttnn.from_torch(
        torch_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Assemble op configs for pretty-print validation (embedding + typecast)
    run_cfg = {
        "embedding": EmbeddingConfig(
            weight=weight_ttnn,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        ),
        "typecast": TypecastConfig(dtype=ttnn.float32),
    }

    # Validate pretty string for the subset under test
    pretty = _convert_run_config_to_pretty_print(run_cfg)
    # Weight per-device shape [V, H/D]; memory is DRAM interleaved
    expected = dedent(
        f"""
        {{
          'embedding': EmbeddingConfig(
            weight=ttnn.Tensor(shape=Shape([{V}, {per_device_h}]), dtype=DataType.BFLOAT16, memory=INTERLEAVED_DRAM),
            memory_config=MemoryConfig(layout=INTERLEAVED, buffer=DRAM),
            layout=<Layout.TILE: 1>
          ),
          'typecast': TypecastConfig(
            dtype=<DataType.FLOAT32: 1>,
            memory_config=None,
            sub_core_grids=None
          ),
        }}
        """
    ).strip()
    assert pretty.strip() == expected, f"Pretty config mismatch.\nGot:\n{pretty}\nExpected:\n{expected}"

    # Execute ops: embedding → unsqueeze → typecast
    emb = ttnn.embedding(ids_ttnn, **run_cfg["embedding"])  # type: ignore[arg-type]
    emb = ttnn.unsqueeze(emb, 0)  # [1, 1, 1, S, H/D]
    emb_fp32 = ttnn.typecast(emb, **run_cfg["typecast"])  # type: ignore[arg-type]
    ttnn.deallocate(emb)

    # Compose pre-gather shards across last dim to compare to Torch
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=4)
    emb_host = ttnn.to_torch(emb_fp32, mesh_composer=composer)
    ttnn.deallocate(emb_fp32)

    # Torch baseline (FP32 to match typecast)
    torch_ref = torch.nn.functional.embedding(torch_ids, torch_weight).to(torch.float32)
    torch_ref = torch_ref.unsqueeze(0)  # [1, 1, 1, S, H]

    # Squeeze the extra dim to get 4D tensors for PCC helper or direct compare
    emb_host_4d = emb_host.squeeze(2)
    torch_ref_4d = torch_ref.squeeze(2)

    assert torch.allclose(emb_host_4d, torch_ref_4d, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_embedding_decode_all_gather_shapes_and_cfg(hf_config, mesh_device, ccl, set_deterministic_env):
    # Require Galaxy mesh
    assert mesh_device.get_num_devices() == 32, "This test requires a 4x8 Galaxy (32 devices)."

    H = hf_config.hidden_size
    V = hf_config.vocab_size
    D = mesh_device.get_num_devices()
    per_device_h = even_int_div(H, D)

    # Choose a seq_len that is not tile-aligned to exercise padding and slicing
    seq_len = 47
    pad = (ttnn.TILE_SIZE - (seq_len % ttnn.TILE_SIZE)) % ttnn.TILE_SIZE

    torch_ids = torch.randint(low=0, high=V, size=(1, 1, seq_len), dtype=torch.int32)
    torch_weight, weight_ttnn = _make_embedding_weight_tensors(V, H, mesh_device)
    ids_ttnn = ttnn.from_torch(
        torch_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    run_cfg = {
        "embedding": EmbeddingConfig(weight=weight_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT),
        "typecast": TypecastConfig(dtype=ttnn.float32),
    }

    # Minimal pretty-print check for all_gather parameters (avoid dynamic semaphore printing)
    ag_cfg_min = {
        "all_gather": {
            "dim": -1,
            "cluster_axis": 0,
            "topology": ttnn.Topology.Linear,
        }
    }
    pretty_ag = _convert_run_config_to_pretty_print(ag_cfg_min)
    expected_ag = dedent(
        """
        {
          'all_gather': {
            'dim': -1,
            'cluster_axis': 0,
            'topology': <Topology.Linear: 1>,
          },
        }
        """
    ).strip()
    assert pretty_ag.strip() == expected_ag, f"Pretty config mismatch.\nGot:\n{pretty_ag}\nExpected:\n{expected_ag}"

    # Execute decode path semantics: optional pad → embedding → unsqueeze → typecast → all_gather_async → slice back
    if pad:
        ids_padded = ttnn.pad(ids_ttnn, [(0, 0), (0, 0), (0, pad)], 0)
        ttnn.deallocate(ids_ttnn)
        ids_ttnn = ids_padded

    emb = ttnn.embedding(ids_ttnn, **run_cfg["embedding"])  # type: ignore[arg-type]
    emb = ttnn.unsqueeze(emb, 0)
    ttnn.deallocate(ids_ttnn)
    emb_fp32 = ttnn.typecast(emb, **run_cfg["typecast"])  # type: ignore[arg-type]
    ttnn.deallocate(emb)

    # Build full all_gather config and run
    ag_cfg = AllGatherAsyncConfig(
        mesh_device=mesh_device,
        cluster_axis=0,
        dim=-1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=ccl.get_gather_sem(0),
        barrier_semaphore=ccl.get_barrier_sem(0),
        num_links=ccl.get_max_links(0),
        memory_config=None,
    )
    gathered = ttnn.experimental.all_gather_async(emb_fp32, **ag_cfg)
    ttnn.deallocate(emb_fp32)

    # Expect shape [1, 1, 1, padded_seq_len, H]
    assert len(gathered.shape) == 5
    assert gathered.shape[-1] == H
    assert gathered.shape[-2] == _roundup(seq_len, ttnn.TILE_SIZE)

    # Slice back to original seq_len
    if pad:
        out = gathered[:, :, :, :seq_len, :]
        ttnn.deallocate(gathered)
    else:
        out = gathered

    # Validate host composition across width shards pre-gather vs reference
    # Compose to host along last dimension; squeeze the extra dim for 4D compare
    out_host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=4)).squeeze(2)
    ttnn.deallocate(out)

    torch_ref = torch.nn.functional.embedding(torch_ids, torch_weight).to(torch.float32)

    assert torch.allclose(out_host, torch_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])

