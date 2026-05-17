"""POC test: pi05_siglip_ops::LayerNorm Op-struct.

Validates the Op-struct port preserves LN's load-bearing gotchas:
  * Stage-A (mul_tiles accumulate) + Stage-B (single reduce_tile) in phases 1, 4
  * Mandatory binary_op_init_common reset between phase types

Compares against torch fp32 layer_norm (PCC) and against the monolithic
siglip_layernorm_kernel.cpp output (bit-identical).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Reuse the existing perf-test helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden_fc1 import make_real_activation, pcc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op_struct_layernorm_poc import (  # noqa: E402
    SigLIPLayerNormOpStruct,
    build_tensors_for_ln_test,
)

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VP = "paligemma_with_expert.paligemma.model.vision_tower."
M, D = 256, 1152
EPS = 1e-6


def load_layer0_ln1() -> tuple[torch.Tensor, torch.Tensor]:
    sd = load_file(PI05_WEIGHTS)
    gamma = sd[f"{VP}vision_model.encoder.layers.0.layer_norm1.weight"].to(torch.bfloat16)
    beta = sd[f"{VP}vision_model.encoder.layers.0.layer_norm1.bias"].to(torch.bfloat16)
    return gamma, beta


def _run_op(device, op_cls, gamma_torch, beta_torch, x_torch):
    tensors = build_tensors_for_ln_test(device, gamma_torch, beta_torch, x_torch, num_cores=8)
    op_cls.op(*tensors, num_cores=8, eps=EPS)
    import ttnn as _ttnn

    return _ttnn.to_torch(tensors[-1])


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_layernorm_op_struct_pcc(device):
    gamma_torch, beta_torch = load_layer0_ln1()
    x_torch = make_real_activation(seed=42)

    y_golden = F.layer_norm(
        x_torch.float(),
        normalized_shape=(D,),
        weight=gamma_torch.float(),
        bias=beta_torch.float(),
        eps=EPS,
    ).to(torch.bfloat16)

    y_device = _run_op(device, SigLIPLayerNormOpStruct, gamma_torch, beta_torch, x_torch)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (LN Op-struct vs torch fp32) = {p:.6f}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_layernorm_op_struct_matches_monolithic(device):
    from layernorm_op import SigLIPLayerNormOp  # noqa: E402

    gamma_torch, beta_torch = load_layer0_ln1()
    x_torch = make_real_activation(seed=42)

    y_op_struct = _run_op(device, SigLIPLayerNormOpStruct, gamma_torch, beta_torch, x_torch)
    y_mono = _run_op(device, SigLIPLayerNormOp, gamma_torch, beta_torch, x_torch)

    diff = (y_op_struct.float() - y_mono.float()).abs()
    max_diff = float(diff.max())
    print(f"\nmax abs diff (LN Op-struct vs monolithic) = {max_diff:.6e}")
    assert max_diff == 0.0, f"Op-struct path diverged from monolithic by {max_diff}"
