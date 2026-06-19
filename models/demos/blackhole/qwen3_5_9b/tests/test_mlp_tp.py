# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 validation for the Qwen3.5 SwiGLU MLP on a (1,4) Blackhole mesh.

Loads just one layer's gate/up/down weights from the FP8 checkpoint (fast,
RAM-light), runs the tensor-parallel Qwen35MLP forward, and compares against a
torch SwiGLU reference. Output is fractured along the hidden dim (reduce-scatter)
so it is gathered with ConcatMeshToTensor(dim=3).

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
        pytest models/demos/blackhole/qwen3_5_9b/tests/test_mlp_tp.py -v -s
"""
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.mlp import Qwen35MLP
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.tp_common import dequant_fp8_block


def _model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B"))


def _load_one_layer_mlp(model_path, layer_idx):
    """Dequantize just layers.<i>.mlp.{gate,up,down}_proj into a substate dict."""
    from safetensors import safe_open

    model_path = Path(model_path)
    weight_map = json.load(open(model_path / "model.safetensors.index.json"))["weight_map"]
    want = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        base = None
        for k in weight_map:
            if k.endswith(f"layers.{layer_idx}.mlp.{name}.weight"):
                base = k
                break
        assert base is not None, f"missing {name} for layer {layer_idx}"
        fn = weight_map[base]
        with safe_open(str(model_path / fn), framework="pt") as sf:
            w = sf.get_tensor(base)
            scale_key = base + "_scale_inv"
            sfn = weight_map.get(scale_key)
            if sfn is not None:
                with safe_open(str(model_path / sfn), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(scale_key))
            else:
                w = w.to(torch.bfloat16)
        want[f"{name}.weight"] = w
    return want


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_mlp_tp_pcc(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", _model_path())
    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    logger.info(f"devices={nd} dim={args.dim} hidden_dim={args.hidden_dim}")

    # args.CKPT_DIR is the resolved local snapshot dir (Qwen35ModelArgs downloads the hub id).
    mlp_state = _load_one_layer_mlp(args.CKPT_DIR, 0)

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    mlp = Qwen35MLP(mesh_device, mlp_state, None, args=args, tt_ccl=tt_ccl)

    # Torch reference: down(silu(gate(x)) * up(x))
    T = 32
    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    g = mlp_state["gate_proj.weight"].to(torch.float32)
    u = mlp_state["up_proj.weight"].to(torch.float32)
    d = mlp_state["down_proj.weight"].to(torch.float32)
    xf = x.to(torch.float32)[0, 0]  # [T, dim]
    ref = (torch.nn.functional.silu(xf @ g.T) * (xf @ u.T)) @ d.T  # [T, dim]

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = mlp.forward(x_tt)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)
    out_torch = ttnn.to_torch(out, mesh_composer=composer)[0, 0].to(torch.float32)  # [T, dim]

    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(ref, out_torch, 0.97)
    logger.info(f"MLP TP PCC = {pcc}")
    assert passing, f"MLP TP PCC too low: {pcc}"
