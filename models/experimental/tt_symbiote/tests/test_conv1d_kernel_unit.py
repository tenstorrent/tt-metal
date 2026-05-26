# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the fused causal-conv1d-update tt-lang kernel.

Compares `fused_conv1d_update_step` to `torch_causal_conv1d_update` across
multiple decode steps, mirroring how the model uses it inside the
linear-attention forward.

Run with:
  export TTNN_GDN_KERNEL=1
  export MESH_DEVICE=QB2
  pytest models/experimental/tt_symbiote/tests/test_conv1d_kernel_unit.py -s
"""

import os

import pytest
import torch

import ttnn

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_causal_conv1d_update


MESH_DEVICE_MAP = {
    "T3K": (1, 8),
    "QB2": (1, 4),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_fused_conv1d_update_matches_torch(mesh_device):
    """N decode steps, fused kernel vs torch_causal_conv1d_update."""
    from models.experimental.tt_symbiote.modules.conv1d_kernel import (
        fused_conv1d_update_step,
    )
    from models.experimental.tt_symbiote.modules.ttnn_decode_ops import (
        init_conv_slots_from_torch_state,
        upload_conv1d_weights,
        upload_replicated,
    )

    torch.manual_seed(0)
    D = 8192  # conv_dim for Qwen3.6-35B-A3B; divisible by 64 cores * 32 tile = 2048
    K = 4
    NUM_STEPS = 4

    tokens = [torch.randn(1, D, 1, dtype=torch.bfloat16) for _ in range(NUM_STEPS)]
    init_conv_state = torch.randn(1, D, K, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(D, K, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(D, dtype=torch.bfloat16) * 0.05

    # --- Torch reference, multi-step.
    ref_state = init_conv_state.clone()
    ref_outs = []
    for x in tokens:
        y = torch_causal_conv1d_update(
            hidden_states=x,
            conv_state=ref_state,
            weight=weight,
            bias=bias,
            activation="silu",
        )
        ref_outs.append(y.clone())

    # --- Fused tt-lang kernel.
    weights_per_k = upload_conv1d_weights(weight.unsqueeze(1), mesh_device)
    bias_tt = upload_replicated(bias.unsqueeze(0), mesh_device)
    slots = init_conv_slots_from_torch_state(init_conv_state, mesh_device)
    y_out_buf = upload_replicated(torch.zeros(1, D, dtype=torch.bfloat16), mesh_device)

    def _replicated(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
        )

    def _to_torch_replicated(t_ttnn):
        return ttnn.to_torch(t_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]

    kernel_outs = []
    for x in tokens:
        x_2d = x.squeeze(-1)  # [1, D]
        x_tt = _replicated(x_2d)
        fused_conv1d_update_step(x_tt, slots, weights_per_k, bias_tt, y_out_buf)
        y_torch = _to_torch_replicated(y_out_buf)  # [1, D]
        kernel_outs.append(y_torch.unsqueeze(-1).clone())  # [1, D, 1]

    print(f"\nFused conv1d update — per-step PCC:")
    print(f"{'step':>4}  {'pcc':>10}  {'max_abs_err':>12}")
    for i, (ko, ro) in enumerate(zip(kernel_outs, ref_outs)):
        pcc = _pcc(ko, ro)
        mae = (ko.float() - ro.float()).abs().max().item()
        print(f"{i:>4}  {pcc:>10.6f}  {mae:>12.4e}")

    print(f"\nstep0 ref[0,:6,0]   = {ref_outs[0][0,:6,0].tolist()}")
    print(f"step0 kernel[0,:6,0]= {kernel_outs[0][0,:6,0].tolist()}")
    print(f"\nstep3 ref[0,:6,0]   = {ref_outs[-1][0,:6,0].tolist()}")
    print(f"step3 kernel[0,:6,0]= {kernel_outs[-1][0,:6,0].tolist()}")

    for i, (ko, ro) in enumerate(zip(kernel_outs, ref_outs)):
        pcc = _pcc(ko, ro)
        assert pcc > 0.99, f"step {i}: pcc too low: {pcc}"
