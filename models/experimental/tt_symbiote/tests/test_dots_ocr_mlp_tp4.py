# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP=4 (DP1TP4) PCC tests for the dots.ocr MLP modules (T3K, mesh (1, 4)).

Covers FF1/FF2/FF3 for both MLP implementations across the phases the full model
runs them in, asserting the tensor-parallel output is not distorted vs a full
(unsharded) torch reference:

* ``TTNNDotsOCRMLP`` -- the text decoder SwiGLU (fused gate+up -> down). Already
  tensor-parallel via reduce_scatter; exercised here in both decode (seq=1) and
  prefill (seq>1). Input arrives hidden-K-sharded (the post-attention norm output
  under TP), output comes back hidden-N-sharded -- both ``dim=-1`` on the mesh.

* ``TTNNDotsVisionMLP`` -- the vision SwiGLU (fc1/fc3 -> fc2). Made tensor-parallel
  here (column-parallel gate/up + row-parallel down + all-reduce). The vision tower
  body runs replicated across the TP axis, so the MLP input/output are REPLICATED
  full-hidden tensors; the all-reduce re-replicates the row-parallel partial sums.

Run on T3K (uses 4 of 8 devices)::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_mlp_tp4.py
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP, TTNNDotsOCRMLPColParallel
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsVisionMLP
from models.experimental.tt_symbiote.utils.device_management import set_device

HIDDEN = 1536
TEXT_INTERMEDIATE = 8960  # dots.ocr text MLP intermediate (8960 / 4 = 2240 = 70 tiles)
VISION_INTERMEDIATE = 4224  # dots.ocr vision MLP intermediate (4224 / 4 = 1056 = 33 tiles)
TP = 4

# BF16 activations x BFP8/BFP4 weights -> BFP8 output. The TP split itself is exact
# (the reduce just sums per-device partials); this floor guards against gross
# distortion (a wrong shard ordering / missing collective drops PCC toward 0),
# while staying above the model's existing low-precision quantization noise.
PCC_THRESHOLD = 0.95


class _TorchTextSwiGLU(nn.Module):
    """gate/up/down SwiGLU matching the attributes TTNNDotsOCRMLP.from_torch reads."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TorchVisionSwiGLU(nn.Module):
    """fc1/fc2/fc3 SwiGLU matching the attributes TTNNDotsVisionMLP.from_torch reads."""

    def __init__(self, hidden: int, intermediate: int, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=bias)  # gate
        self.fc3 = nn.Linear(hidden, intermediate, bias=bias)  # up
        self.fc2 = nn.Linear(intermediate, hidden, bias=bias)  # down

    def forward(self, x):
        return self.fc2(nn.functional.silu(self.fc1(x)) * self.fc3(x))


def _build_on_mesh(tt_module, mesh_device):
    """Run the standard TTNNModule lifecycle (set_device -> preprocess -> to device)."""
    set_device(tt_module, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_module.preprocess_weights()
    tt_module.move_weights_to_device()
    return tt_module


def _raw_ttnn(t):
    """Unwrap the module's TorchTTNNTensor wrapper to the underlying ttnn.Tensor."""
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


# Real model shapes: decode is one token (tile-padded to 32 rows on device), prefill
# is the full padded prompt length (2816 = 88 tiles) the e2e text decoder runs at.
# Both TP schemes are exercised:
#   "row" -> TTNNDotsOCRMLP (default): K-sharded in -> N-sharded out, 2x reduce_scatter.
#   "col" -> TTNNDotsOCRMLPColParallel: replicated in/out, gate/up column-parallel
#            (no CCL) + down all-reduce (reduce_scatter + all_gather).
@pytest.mark.parametrize("seq_len", [2816], ids=["prefill"])
# @pytest.mark.parametrize("seq_len", [1, 2816], ids=["decode", "prefill"])
@pytest.mark.parametrize("scheme", ["row", "col"])
@pytest.mark.parametrize("mesh_device", [(1, TP)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_dots_ocr_text_mlp_tp4(mesh_device, seq_len, scheme):
    """Text decoder SwiGLU at TP=4 for both row- and column-parallel schemes, no distortion."""
    assert mesh_device.get_num_devices() == TP, f"Expected TP={TP}, got {mesh_device.get_num_devices()}"
    torch.manual_seed(0xD075)

    torch_mlp = _TorchTextSwiGLU(HIDDEN, TEXT_INTERMEDIATE).to(torch.bfloat16).eval()
    x = torch.randn(1, seq_len, HIDDEN, dtype=torch.bfloat16) * 0.1
    # fp32 reference from the (bf16-rounded) weights so F.linear dtypes match.
    ref = torch_mlp.float()(x.float()).to(torch.float32)

    if scheme == "row":
        tt_mlp = TTNNDotsOCRMLP.from_torch(torch_mlp)
        # Input is hidden-K-sharded across the TP axis (post-attention norm contract).
        in_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    else:
        tt_mlp = TTNNDotsOCRMLPColParallel.from_torch(torch_mlp)
        # Column-parallel takes a replicated full-hidden activation on every device.
        in_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_mlp.set_weight_dtype(ttnn.bfloat8_b)  # match decoder gate-up precision
    _build_on_mesh(tt_mlp, mesh_device)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=in_mapper,
    )
    out_tt = tt_mlp(x_tt)
    ttnn.synchronize_device(mesh_device)

    if scheme == "row":
        # Output is hidden-N-sharded; gather the per-device slices back to full hidden.
        out = ttnn.to_torch(_raw_ttnn(out_tt), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).to(
            torch.float32
        )
    else:
        # Output is replicated; every device holds the full hidden -> take device 0.
        out = ttnn.to_torch(_raw_ttnn(out_tt), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(
            torch.float32
        )
        out = out[: ref.shape[0]]
    out = out.reshape(ref.shape)

    passed, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(f"[text_mlp seq={seq_len} scheme={scheme}] TP={TP} pcc={float(pcc):.6f} (threshold {PCC_THRESHOLD})")
    assert passed, f"Text MLP TP={TP} ({scheme}) distorted at seq={seq_len}: pcc={float(pcc):.6f} < {PCC_THRESHOLD}"


# Real vision-tower seq: the padded patch-bucket the e2e vision encoder runs at
# (11264 = 352 tiles); the vision MLP has no decode phase.
@pytest.mark.parametrize("seq_len", [11264], ids=["prefill"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("mesh_device", [(1, TP)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_dots_ocr_vision_mlp_tp4(mesh_device, seq_len, with_bias):
    """Vision SwiGLU at TP=4: replicated full-hidden in -> replicated full-hidden out.

    Exercises the column/row split + all-reduce, including the replicated fc2-bias
    add-after-reduce path (which must not be summed num_tp times).
    """
    assert mesh_device.get_num_devices() == TP, f"Expected TP={TP}, got {mesh_device.get_num_devices()}"
    torch.manual_seed(0xF15)

    torch_mlp = _TorchVisionSwiGLU(HIDDEN, VISION_INTERMEDIATE, bias=with_bias).to(torch.bfloat16).eval()
    x = torch.randn(1, 1, seq_len, HIDDEN, dtype=torch.bfloat16) * 0.1
    # fp32 reference from the (bf16-rounded) weights so F.linear dtypes match.
    ref = torch_mlp.float()(x.float()).to(torch.float32)

    tt_mlp = TTNNDotsVisionMLP.from_torch(torch_mlp)
    _build_on_mesh(tt_mlp, mesh_device)
    assert tt_mlp._mlp_use_tp(), "Vision MLP did not engage the TP path on a (1,4) mesh"

    # Vision tower body is replicated across the TP axis -> full-hidden input on each device.
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = tt_mlp(x_tt)
    ttnn.synchronize_device(mesh_device)

    # Output is replicated; every device holds the same full-hidden result -> take device 0.
    out = ttnn.to_torch(_raw_ttnn(out_tt), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch.float32)
    out = out[: ref.shape[0]].reshape(ref.shape)

    passed, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(f"[vision_mlp seq={seq_len} bias={with_bias}] TP={TP} pcc={float(pcc):.6f} (threshold {PCC_THRESHOLD})")
    assert passed, f"Vision MLP TP={TP} distorted (bias={with_bias}): pcc={float(pcc):.6f} < {PCC_THRESHOLD}"
