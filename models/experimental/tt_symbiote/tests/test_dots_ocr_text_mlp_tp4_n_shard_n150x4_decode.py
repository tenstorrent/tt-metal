# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Text decoder MLP decode (seq=1) on Wormhole 4×N150 with TP=4.

MLP I/O: hidden is ``ShardTensorToMesh(dim=-1)`` (K-sharded, 384/device).

First matmul (fused gate-up): column-parallel on **N=17920**, not K — full
hidden is all-gathered, weights are N-sharded (4480/device), no reduce_scatter.
Down-proj: K-sharded intermediate (2240/device), ``32×2240×1536`` matmul +
reduce_scatter on hidden. LoFi + BFP4, L1 at MLP boundaries.

Single-device decode uses the DRAM-width-sharded gate-up/down kernels (16c 8×2 /
8c 8×1); those are disabled when ``_tp_requires_ccl`` is true — this test covers
the TP4 CCL path only.

Run on the N150 lab (4 devices, ``MESH_DEVICE=N150x4``)::

    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    export ARCH_NAME=wormhole_b0
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    export MESH_DEVICE=N150x4

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_text_mlp_tp4_n_shard_n150x4_decode.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
from models.experimental.tt_symbiote.modules.linear import (
    _decode_down_proj_dram_sharded_program_config,
    _decode_gate_up_dram_sharded_program_config,
    _tp_requires_ccl,
)
from models.experimental.tt_symbiote.utils.device_management import set_device

HIDDEN = 1536
INTERMEDIATE = 8960  # dots.ocr text MLP intermediate
FUSED_GATE_UP_N = INTERMEDIATE * 2  # 17920
TP = 4
TP_MESH_SHAPE = (1, TP)
DECODE_SEQ_LEN = 1
HIDDEN_DIM = -1
PCC_THRESHOLD = 0.95


def _mesh_env() -> str:
    return os.environ.get("MESH_DEVICE", "N150x4")


def _device_params():
    dp: dict = {"trace_region_size": 300000000, "num_command_queues": 1}
    mesh_env = _mesh_env()
    if mesh_env not in ("", "N150x4"):
        pytest.skip(f"N150x4 lab test requires MESH_DEVICE=N150x4, got {mesh_env!r}")
    dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
    return dp


def _open_n150x4_tp_mesh(updated_device_params):
    if len(ttnn.get_pcie_device_ids()) < TP:
        pytest.skip(f"N150x4 requires at least {TP} PCIe devices")

    req_shape = TP_MESH_SHAPE
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_num = sys_shape[0] * sys_shape[1]

    if sys_num == TP:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*req_shape), **updated_device_params)
        if sys_shape == (2, 2):
            mesh_device.reshape(ttnn.MeshShape(*req_shape))
        return mesh_device, None

    parent_shape = sys_shape
    if not (req_shape[0] <= parent_shape[0] and req_shape[1] <= parent_shape[1]):
        rotated = (parent_shape[1], parent_shape[0])
        if req_shape[0] <= rotated[0] and req_shape[1] <= rotated[1]:
            parent_shape = rotated
        else:
            pytest.skip(f"N150x4 {req_shape} does not fit system mesh {sys_shape}")

    parent_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*parent_shape), **updated_device_params)
    tp_mesh = parent_mesh.create_submesh(ttnn.MeshShape(*req_shape))
    tp_mesh.reshape(ttnn.MeshShape(*req_shape))
    return tp_mesh, parent_mesh


def _assert_l1_resident(tensor, name: str) -> None:
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"
    assert (
        tensor.memory_config().buffer_type == ttnn.BufferType.L1
    ), f"{name} should reside in L1, got {tensor.memory_config()}"


@pytest.fixture(scope="function")
def n150x4_n_shard_mesh_device(device_params, silicon_arch_name, silicon_arch_wormhole_b0):
    """1×4 Wormhole mesh for text MLP N-shard decode (N150x4 lab)."""
    from tests.scripts.common import get_updated_device_params

    _ = silicon_arch_name, silicon_arch_wormhole_b0
    updated_device_params = get_updated_device_params(dict(device_params))
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    if fabric_config:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode or ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            fabric_tensix_config or ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    mesh_device, parent_mesh = _open_n150x4_tp_mesh(updated_device_params)
    if mesh_device.get_num_devices() != TP:
        pytest.skip(f"expected {TP} devices, got {mesh_device.get_num_devices()}")

    yield mesh_device

    if parent_mesh is not None:
        ttnn.close_mesh_device(mesh_device)
        ttnn.close_mesh_device(parent_mesh)
    else:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


class _TorchTextSwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _raw_ttnn(t):
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


def _gather_hidden_sharded(out_tt, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(
        _raw_ttnn(out_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, HIDDEN_DIM),
    ).to(torch.float32)


def _assert_decode_program_configs_disabled_on_tp(mesh_device, tt_mlp) -> None:
    """DRAM-sharded decode kernels are single-device only; TP4 must use CCL matmul path."""
    assert _tp_requires_ccl(mesh_device), "N150x4 TP mesh must use fabric CCL"
    gate_up = tt_mlp.fused_gate_up_proj
    down = tt_mlp.down_proj
    assert getattr(gate_up, "_gate_up_dram_input_shard_cfg", None) is None
    assert getattr(gate_up, "_gate_up_dram_weight", None) is None
    assert getattr(down, "_down_proj_dram_input_shard_cfg", None) is None
    decode_in = [1, 1, ttnn.TILE_SIZE, HIDDEN]
    per_dev_n = FUSED_GATE_UP_N // TP
    assert _decode_gate_up_dram_sharded_program_config(decode_in, [1, 1, HIDDEN, per_dev_n]) is None
    assert (
        _decode_down_proj_dram_sharded_program_config(
            [1, 1, ttnn.TILE_SIZE, INTERMEDIATE // TP], [1, 1, INTERMEDIATE // TP, HIDDEN // TP]
        )
        is None
    )


@pytest.mark.parametrize(
    "device_params",
    [_device_params()],
    indirect=True,
)
def test_dots_ocr_text_mlp_tp4_n_shard_n150x4_decode(n150x4_n_shard_mesh_device):
    """Decode SwiGLU on N150x4: gate-up N-sharded (17920), down K-sharded hidden I/O."""
    mesh_device = n150x4_n_shard_mesh_device
    assert HIDDEN % TP == 0 and FUSED_GATE_UP_N % TP == 0
    hidden_per_dev = HIDDEN // TP
    fused_n_per_dev = FUSED_GATE_UP_N // TP

    torch.manual_seed(0xDEC0DE)
    torch_mlp = _TorchTextSwiGLU(HIDDEN, INTERMEDIATE).to(torch.bfloat16).eval()
    x = torch.randn(1, DECODE_SEQ_LEN, HIDDEN, dtype=torch.bfloat16) * 0.1
    ref = torch_mlp.float()(x.float()).to(torch.float32)

    tt_mlp = TTNNDotsOCRMLP.from_torch(torch_mlp)
    # Match production decode dtypes: LoFi BF16 × BFP4 → BFP8 (profiler gate-up/down).
    tt_mlp.set_weight_dtype(ttnn.bfloat4_b)
    set_device(tt_mlp, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_mlp.preprocess_weights()
    tt_mlp.move_weights_to_device()
    _assert_decode_program_configs_disabled_on_tp(mesh_device, tt_mlp)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=HIDDEN_DIM),
    )
    assert int(x_tt.shape[HIDDEN_DIM]) == hidden_per_dev

    seen = {"in": False, "gate_up_n_shard": False, "out": False}
    original_forward = tt_mlp.forward
    original_gate_up = tt_mlp.fused_gate_up_proj.forward

    def checked_gate_up(hidden_states, output_memory_config=None):
        gate_up = original_gate_up(hidden_states, output_memory_config=output_memory_config)
        assert (
            int(_raw_ttnn(gate_up).shape[-1]) == fused_n_per_dev
        ), f"gate-up must be N-sharded ({fused_n_per_dev}), got {list(gate_up.shape)}"
        seen["gate_up_n_shard"] = True
        return gate_up

    def checked_forward(hidden_states):
        _assert_l1_resident(_raw_ttnn(hidden_states), "MLP input")
        assert int(_raw_ttnn(hidden_states).shape[-1]) == hidden_per_dev
        out = original_forward(hidden_states)
        _assert_l1_resident(_raw_ttnn(out), "MLP output")
        seen["in"] = seen["out"] = True
        return out

    tt_mlp.fused_gate_up_proj.forward = checked_gate_up
    tt_mlp.forward = checked_forward
    out_tt = tt_mlp(x_tt)
    ttnn.synchronize_device(mesh_device)
    assert seen == {"in": True, "gate_up_n_shard": True, "out": True}

    out = _gather_hidden_sharded(out_tt, mesh_device)
    out = out.reshape(ref.shape)

    passed, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(
        f"[text_mlp_decode N150x4 H={HIDDEN} H/dev={hidden_per_dev} fused_N/dev={fused_n_per_dev} "
        f"mesh={list(mesh_device.shape)}] pcc={float(pcc):.6f} (threshold {PCC_THRESHOLD})"
    )
    assert passed, f"Text MLP decode N-shard distorted: pcc={float(pcc):.6f} < {PCC_THRESHOLD}"
