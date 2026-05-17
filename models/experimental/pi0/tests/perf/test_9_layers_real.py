"""SigLIP encoder layers 0–8 on REAL π0.5 weights — Vision_0 chip workload.

Maps to the spec's Vision_0 chip (mapping doc §5: SigLIP layers 0-8).
Runs the full encoder block 9 times in sequence with real per-layer weights,
validates against torch fp32 reference of the same 9-layer stack.

This is the intermediate K1.2 step — full per-layer compute on device,
real weights, but with host round-trips between stages within each layer
(not yet L1-to-L1 chained, that's the true persistent kernel).
"""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc, make_real_activation  # noqa: E402
from test_encoder_block_real import (  # noqa: E402
    torch_encoder_block_real,
    device_encoder_block_real,
    VP as VP_LAYER0,
)

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VP_BASE = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers."

NUM_LAYERS = 9  # Vision_0 chip per the mapping spec
M = 256
D = 1152
NUM_HEADS = 16
HEAD_DIM = 72
INTERMEDIATE = 4304


def load_layer_weights(sd: dict, layer_idx: int) -> dict:
    """Load all weights for one encoder layer from a pre-loaded safetensors dict."""
    prefix = f"{VP_BASE}{layer_idx}."

    def _get(name: str) -> torch.Tensor:
        return sd[f"{prefix}{name}"].to(torch.bfloat16)

    wq = _get("self_attn.q_proj.weight").T.contiguous()
    wk = _get("self_attn.k_proj.weight").T.contiguous()
    wv = _get("self_attn.v_proj.weight").T.contiguous()
    bq = _get("self_attn.q_proj.bias")
    bk = _get("self_attn.k_proj.bias")
    bv = _get("self_attn.v_proj.bias")
    qkv_w = torch.cat([wq, wk, wv], dim=1).contiguous()
    qkv_b = torch.cat([bq, bk, bv], dim=0).contiguous()

    return dict(
        ln1_w=_get("layer_norm1.weight"),
        ln1_b=_get("layer_norm1.bias"),
        qkv_w=qkv_w,
        qkv_b=qkv_b,
        o_w=_get("self_attn.out_proj.weight").T.contiguous(),
        o_b=_get("self_attn.out_proj.bias"),
        ln2_w=_get("layer_norm2.weight"),
        ln2_b=_get("layer_norm2.bias"),
        fc1_w_logical=_get("mlp.fc1.weight").T.contiguous(),
        fc1_b_logical=_get("mlp.fc1.bias"),
        fc2_w_logical=_get("mlp.fc2.weight").T.contiguous(),
        fc2_b=_get("mlp.fc2.bias"),
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_9_layers_real_weights(device):
    """Run 9 sequential SigLIP encoder layers with real π0.5 weights.

    Reference: torch fp32 of the same 9 layers stacked.
    """
    print(f"\nLoading π0.5 weights for layers 0..{NUM_LAYERS - 1} …")
    sd = load_file(PI05_WEIGHTS)
    all_weights = [load_layer_weights(sd, i) for i in range(NUM_LAYERS)]
    del sd

    x = make_real_activation(seed=42)  # (M, D) bf16

    # Cumulative parity: track torch and device side-by-side per layer.
    # Each layer's PCC reflects the COMPOUNDING error across all prior layers
    # (device output drifts slightly from torch per layer; we want to confirm
    # it stays well above the 0.99 gate even after 9 stacked layers).
    y_golden = x
    y_device = x
    pccs_per_layer = []
    for i, w in enumerate(all_weights):
        y_golden = torch_encoder_block_real(y_golden, w)
        y_device = device_encoder_block_real(device, y_device, w)
        p = pcc(y_golden, y_device)
        pccs_per_layer.append(p)
        print(f"  After layer {i}: cumulative PCC = {p:.6f}")

    print(f"\nFinal PCC (9 layers, real π0.5 weights) = {pccs_per_layer[-1]:.6f}")
    print(f"  PCC trajectory: " + " → ".join(f"{p:.4f}" for p in pccs_per_layer))

    assert pccs_per_layer[-1] >= 0.99, f"9-layer real-weights PCC {pccs_per_layer[-1]} below 0.99 gate"
