# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/rms_norm.py` vs an in-test fp32 torch reference. Gate: `G-RMS`.

The block: `out = x / sqrt(mean(x^2) + eps) * weight`, hidden 4096, `eps` from
`config.json:rms_norm_eps`. **Plain** RMSNorm — no Gemma `(1 + weight)` fold
(`bringup_log/00_MODEL_CARD.md` §3).

* **Input distribution:** standard normal. Stated because it must never be chosen to pass: the
  torch bf16 floor is identical under `rand[0,1)` and `randn` (recipe §2.1(b)), and randn is the
  harder of the two for a norm.
* **Reference dtype policy:** the reference weight and activations are **fp32**, all arithmetic
  fp32. A bf16-weight reference shares the device's own rounding and reports a flattered number —
  measured 0.9999867 versus 0.99995 on the same device output (recipe §2.1(a)).
* **Noise floor:** computed in-test — quantise exactly what the device *stores* (bf16 activations,
  bf16 norm weight, `DEC-022`) and do the rest in fp32.
* **Threshold:** PCC >= 0.9999 (`BRINGUP_RECIPE.md:2067`). The error ratio to the floor is
  **recorded, not asserted**: a correct module sits right on §2.2's 3x stage bound, so asserting it
  would gate on the wrong side of the noise (`BRINGUP_RECIPE.md:1290-1293`).
* **Negative control:** a zero-gain probe must produce `max|out| = 0.0`. A Gemma `(1 + weight)`
  fold would return the normalised input instead, so this is the discriminator for the one feature
  the nearest templates have and Llama does not.
* **The A/B:** `test_rms_norm_compute_kernel_config_ab` measures `fp32_dest_acc_en` True vs False
  on this box, so `DEC-014`'s claim is a number here rather than a quotation from the recipe.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    err_ratio,
    llama_config_dims,
    load_hf_state_dict,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm

PCC_THRESHOLD = 0.9999
SEQ_LENS = [32, 512, 4096]
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
WEIGHT_DTYPE = ttnn.bfloat16  # norm weights stay bf16 ROW_MAJOR


def _torch_rms_norm(x, weight, eps):
    """fp32 reference. `LlamaRMSNorm.forward` with every cast removed."""
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight.float()


def _layer0_norm_weight():
    """The real `model.layers.0.input_layernorm.weight`, fp32."""
    sd = load_hf_state_dict(prefixes=("model.layers.0.input_layernorm.",))
    return sd["model.layers.0.input_layernorm.weight"].float()


def _run_rms_norm(mesh_device, hf, x, weight, *, fp32_dest_acc_en=True):
    """Push `x` through the module and return the device output as fp32 torch."""
    norm = RMSNorm(mesh_device, hf, {"weight": weight}, fp32_dest_acc_en=fp32_dest_acc_en)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = norm(tt_x)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    tt_x.deallocate(True)
    tt_out.deallocate(True)
    return out


def _floor(x, weight, eps, ref):
    """PCC of the fp32 reference against a bf16-storage-quantised, fp32-arithmetic evaluation."""
    x_q = quantize_like_device(x, ACTIVATION_DTYPE)
    # Quantise the weight in the shape the device stores it in: (1, 1, hidden/32, 32).
    w_q = quantize_like_device(weight.reshape(1, 1, -1, ttnn.TILE_SIZE), WEIGHT_DTYPE).reshape(-1)
    _, floor = comp_pcc(ref, _torch_rms_norm(x_q, w_q, eps), 0.0)
    return float(floor)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_rms_norm_vs_ref_random_weight(mesh_device, seq_len, reset_seeds):
    """Random-but-identical weights on both sides, so the test needs no checkpoint."""
    hf = llama_config_dims()
    hidden, eps = hf["hidden_size"], hf["rms_norm_eps"]

    x = torch.randn(1, 1, seq_len, hidden)
    weight = torch.randn(hidden)

    ref = _torch_rms_norm(x, weight, eps)
    floor = _floor(x, weight, eps, ref)
    out = _run_rms_norm(mesh_device, hf, x, weight)

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), floor)
    logger.info(f"[G-RMS] random-weight seq={seq_len}: PCC={float(pcc):.7f} floor={floor:.7f} ratio={ratio:.2f}x")
    assert passing, f"below threshold {PCC_THRESHOLD}: {pcc}"


@torch.no_grad()
@requires_hf_reference
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_rms_norm_vs_ref_real_weight(mesh_device, seq_len, reset_seeds):
    """The same gate on the **real layer-0 norm weight**, which is what §2.1 measured against.

    A trained norm gain is not standard-normal — it is a narrow positive distribution — so this is
    a materially different numeric test from the random-weight case above, not a duplicate.
    """
    hf = llama_config_dims()
    hidden, eps = hf["hidden_size"], hf["rms_norm_eps"]

    x = torch.randn(1, 1, seq_len, hidden)
    weight = _layer0_norm_weight()
    assert weight.shape == (hidden,)

    ref = _torch_rms_norm(x, weight, eps)
    floor = _floor(x, weight, eps, ref)
    out = _run_rms_norm(mesh_device, hf, x, weight)

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), floor)
    logger.info(f"[G-RMS] real-weight   seq={seq_len}: PCC={float(pcc):.7f} floor={floor:.7f} ratio={ratio:.2f}x")
    assert passing, f"below threshold {PCC_THRESHOLD}: {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rms_norm_zero_gain_negative_control(mesh_device, reset_seeds):
    """**The negative control.** A zero gain must give exactly zero out.

    With a Gemma `(1 + weight)` fold the module would return the *normalised input* — magnitude
    ~1 per channel — so `max|out| = 0.0` is the discriminator between plain and folded RMSNorm.
    """
    hf = llama_config_dims()
    hidden = hf["hidden_size"]

    x = torch.randn(1, 1, 512, hidden)
    out = _run_rms_norm(mesh_device, hf, x, torch.zeros(hidden))

    max_abs = float(out.abs().max())
    logger.info(f"[G-RMS] control: zero-gain max|out| = {max_abs}")
    assert max_abs == 0.0, f"zero gain produced non-zero output ({max_abs}) — a Gemma (1+w) fold is present"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 512], ids=lambda s: f"s{s}")
def test_rms_norm_compute_kernel_config_ab(mesh_device, seq_len, reset_seeds):
    """`fp32_dest_acc_en` True vs False, measured on this box (`DEC-014`'s falsifier).

    Recipe §2.4 puts the op-level gap at ~25x and the module-level gap at ~7x. Asserting only the
    *direction* — True must be at least as close to the reference as False — keeps this a
    measurement rather than a second threshold fitted to a number already seen.
    """
    hf = llama_config_dims()
    hidden, eps = hf["hidden_size"], hf["rms_norm_eps"]

    x = torch.randn(1, 1, seq_len, hidden)
    weight = torch.randn(hidden)
    ref = _torch_rms_norm(x, weight, eps)
    floor = _floor(x, weight, eps, ref)

    _, pcc_true = comp_pcc(ref, _run_rms_norm(mesh_device, hf, x, weight, fp32_dest_acc_en=True), 0.0)
    _, pcc_false = comp_pcc(ref, _run_rms_norm(mesh_device, hf, x, weight, fp32_dest_acc_en=False), 0.0)
    r_true, r_false = err_ratio(float(pcc_true), floor), err_ratio(float(pcc_false), floor)

    logger.info(
        f"[G-RMS] A/B seq={seq_len}: fp32_dest_acc_en=True PCC={float(pcc_true):.7f} ({r_true:.2f}x) "
        f"vs False PCC={float(pcc_false):.7f} ({r_false:.2f}x); floor={floor:.7f}; "
        f"gain={(1 - float(pcc_false)) / (1 - float(pcc_true)):.2f}x"
    )
    assert float(pcc_true) >= float(pcc_false), "fp32_dest_acc_en=True is not better here — recheck §2.4"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rms_norm_refuses_to_build_weightless(mesh_device, expect_error):
    """No `state_dict` and no cache path must fail loud, not run on a `None` gain.

    Appendix B's "cache-only build silently wrong" row: an un-cached weight with no source must
    raise rather than default (`models/demos/minimax_m3/tt/mlp.py` does the same for its bias).
    """
    with expect_error(ValueError, "tensor_cache_path"):
        RMSNorm(mesh_device, llama_config_dims(), {})
