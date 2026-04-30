# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for column-parallel (TP) MLP vs replicated MLP.

Tests:
  1. PCC ≥ 0.99: column-parallel MLP output matches reference PyTorch for S=256
  2. No OOM: column-parallel MLP runs S=16384 without crashing (replicated would OOM)

Run:
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_tp_mlp.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode

HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)
WEIGHT_CACHE = Path("/tmp/molmo2_weight_cache")
_MESH_SHAPE = {"T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))


# ---------------------------------------------------------------------------
# Column-parallel MLP (proposed Phase 2)
# ---------------------------------------------------------------------------


class TtMolmo2TextMLP_TP(LightweightModule):
    """Column-parallel SwiGLU MLP for Molmo2-8B.

    w1/w3: column-parallel (ShardTensor2dMesh dim=3) — each device holds
           [4096, intermediate/num_devices] columns.
    w2:    row-parallel  (ShardTensor2dMesh dim=2) — each device holds
           [intermediate/num_devices, 4096] rows.
    After w2: ttnn.all_reduce combines partial sums across T3K devices.

    Peak DRAM for S=34560:
      w1/w3 output: [1,1,S,1536] × 2 = 2 × 106 MB/device  (vs 2×850 MB replicated)
      w2 input:     [1,1,S,1536]      = 106 MB/device
      AllReduce: small
    """

    def __init__(self, mesh_device, state_dict, layer_num, configuration, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.intermediate_size = configuration.intermediate_size  # 12288
        self.compute_kernel_config = configuration.compute_kernel_config_hifi2

        layer_name = f"model.transformer.blocks.{layer_num}.mlp"
        cache_name = (
            (lambda n: weight_cache_path / f"{layer_name}.tp8.{n}")
            if weight_cache_path and not configuration.dummy_weights
            else (lambda _: None)
        )

        ff_proj = state_dict[f"{layer_name}.ff_proj.weight"]  # [24576, 4096]
        w_value = ff_proj[: self.intermediate_size]  # [12288, 4096]  (value/up)
        w_gate = ff_proj[self.intermediate_size :]  # [12288, 4096]  (gate)

        col_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=configuration.cluster_shape)
        row_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=configuration.cluster_shape)

        def _col(w, name):
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1,1,4096,12288] → shard last dim
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mapper,
                cache_file_name=cache_name(name),
            )

        def _row(w, name):
            return ttnn.as_tensor(
                w.T.unsqueeze(0).unsqueeze(0),  # [1,1,12288,4096] → shard input dim
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=row_mapper,
                cache_file_name=cache_name(name),
            )

        self.w1 = _col(w_gate, "w1_gate_tp8")
        self.w3 = _col(w_value, "w3_value_tp8")

        ff_out = state_dict[f"{layer_name}.ff_out.weight"]  # [4096, 12288]
        self.w2 = _row(ff_out, "w2_down_tp8")

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # silu(gate) * value  (per-device partial result)
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=w1_out.memory_config(),
        )
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Row-parallel w2: each device contributes [S, 4096] partial sum
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        # AllReduce partial sums → full [S, 4096] output replicated on all devices
        out = ttnn.all_reduce(
            w2_out,
            cluster_axis=1,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_out)
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict_and_cfg(mesh_device):
    from transformers import AutoModelForImageTextToText

    from models.demos.molmo2.tt.model_config import Molmo2Config

    sys.path.insert(0, str(HF_PATH))
    hf = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    sd = hf.state_dict()
    del hf
    cfg = Molmo2Config(mesh_device=mesh_device)
    WEIGHT_CACHE.mkdir(parents=True, exist_ok=True)
    return sd, cfg


# ---------------------------------------------------------------------------
# Reference: pure PyTorch SwiGLU MLP
# ---------------------------------------------------------------------------


def ref_mlp(x_bfloat16, state_dict, layer_num):
    ln = f"model.transformer.blocks.{layer_num}.mlp"
    ff_proj = state_dict[f"{ln}.ff_proj.weight"].float()  # [24576, 4096]
    ff_out = state_dict[f"{ln}.ff_out.weight"].float()  # [4096, 12288]
    intermediate_size = 12288
    w_value = ff_proj[:intermediate_size]  # [12288, 4096]
    w_gate = ff_proj[intermediate_size:]  # [12288, 4096]
    x = x_bfloat16.float()
    gate = F.silu(x @ w_gate.T)
    value = x @ w_value.T
    return (gate * value) @ ff_out.T


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_tp_mlp_pcc(mesh_device, state_dict_and_cfg):
    """Column-parallel MLP PCC ≥ 0.99 vs float32 reference for S=256."""
    sd, cfg = state_dict_and_cfg
    layer_num = 0
    S = 256

    mlp = TtMolmo2TextMLP_TP(
        mesh_device=mesh_device,
        state_dict=sd,
        layer_num=layer_num,
        configuration=cfg,
        weight_cache_path=WEIGHT_CACHE,
    )

    x_cpu = torch.randn(1, 1, S, 4096, dtype=torch.bfloat16)
    ref = ref_mlp(x_cpu, sd, layer_num)

    x_tt = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = mlp.forward(x_tt, mode=Mode.PREFILL)
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    ttnn.deallocate(out_tt)

    pcc = _pcc(out_cpu, ref)
    print(f"\nTP MLP PCC (S={S}): {pcc:.6f}")
    assert pcc >= 0.99, f"PCC {pcc:.4f} < 0.99"


def test_tp_mlp_no_oom_large_s(mesh_device, state_dict_and_cfg):
    """Column-parallel MLP runs S=16384 without DRAM OOM (replicated would OOM)."""
    sd, cfg = state_dict_and_cfg
    layer_num = 0
    S = 16384  # replicated MLP OOMs here; column-parallel should not

    mlp = TtMolmo2TextMLP_TP(
        mesh_device=mesh_device,
        state_dict=sd,
        layer_num=layer_num,
        configuration=cfg,
        weight_cache_path=WEIGHT_CACHE,
    )

    x_cpu = torch.zeros(1, 1, S, 4096, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = mlp.forward(x_tt, mode=Mode.PREFILL)
    out_shape = out_tt.shape
    ttnn.deallocate(out_tt)

    print(f"\nTP MLP S={S}: output shape {out_shape}  — no OOM ✓")
    assert out_shape[-2] == S
    assert out_shape[-1] == cfg.dim
