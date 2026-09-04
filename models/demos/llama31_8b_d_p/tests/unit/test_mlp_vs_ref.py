# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/mlp.py` vs an in-test fp32 torch reference. Gate: `G-MLP`.

The block: `down(silu(gate(x)) * up(x))`, hidden 4096, intermediate 14336, **no biases**
(`bringup_log/00_MODEL_CARD.md` §2). `(1,1)` mesh, so TP=1 and the module's TP all-reduce tail is
not executed here — `bringup_log/04_CCL_PLAN.md` §5 row 2 is P8's.

* **Input distribution:** standard normal for `x`; projection weights `randn * 0.02`, the scale both
  templates use (`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:169-176`,
  `models/demos/minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py:98-101`). Stated because it must
  never be chosen to pass: at this scale `gate`/`up` land at std ~1.3, i.e. SiLU is exercised across
  the whole non-linear part of its curve rather than in a locally-linear tail.
* **Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic, `torch.nn.functional.silu`.
* **Noise floor:** computed in-test, per dtype. Recipe §2.2 defines it as "round **inputs and
  weights** to the device dtype, do all remaining math in fp32", so the bf16 *intermediates* the
  device also stores are deliberately **not** quantised — the conservative reading, since
  quantising them would lower the floor and flatter every ratio.
* **Thresholds:** PCC >= 0.999 @bf8_b and >= 0.9995 @bf16, **and <= 3x the floor at each dtype**
  (`BRINGUP_RECIPE.md:1792`). Unlike `G-RMS` the ratio bound is *asserted* here, because Appendix A
  states one for this gate.
* **Negative control:** SiLU applied to `up` instead of `gate` must collapse (the recipe measured
  0.6462). Driven by swapping the `gate_proj` / `up_proj` entries of the state dict, so the control
  runs the **real module** rather than a hand-copied device path.
* **Two A/Bs, both measurements rather than thresholds:** `fp32_dest_acc_en` True vs False on the
  three matmuls (recipe §2.4 predicts 96x-1168x for `False` on a bare `ttnn.linear`), and the fused
  vs separate SiLU spelling (`DEC-039`).

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.mlp import MLP

SEQ_LENS = [32, 512, 4096]
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
WEIGHT_SCALE = 0.02

# Per-dtype absolute threshold, `BRINGUP_RECIPE.md:1792`.
PCC_THRESHOLD = {ttnn.bfloat8_b: 0.999, ttnn.bfloat16: 0.9995}
# Recipe §2.2's stage budget, asserted for this gate.
MAX_ERR_RATIO = 3.0

_DTYPE_IDS = {ttnn.bfloat8_b: "bf8_b", ttnn.bfloat16: "bf16"}


def _torch_mlp(x, w):
    """fp32 reference. `LlamaMLP.forward` with no biases: `down(silu(gate(x)) * up(x))`."""
    gate = x.float() @ w["gate_proj"].float().transpose(-1, -2)
    up = x.float() @ w["up_proj"].float().transpose(-1, -2)
    return (torch.nn.functional.silu(gate) * up) @ w["down_proj"].float().transpose(-1, -2)


def _random_weights(hidden, intermediate):
    """HF `[out, in]` layout, identical on both sides of the comparison."""
    return {
        "gate_proj": torch.randn(intermediate, hidden) * WEIGHT_SCALE,
        "up_proj": torch.randn(intermediate, hidden) * WEIGHT_SCALE,
        "down_proj": torch.randn(hidden, intermediate) * WEIGHT_SCALE,
    }


def _quantize_weights(w, weight_dtype):
    """Quantise each weight in the **shape the device stores it in**: `[1, 1, in, out]`, tilized.

    `bfloat8_b` shares one exponent per tile row, so quantising the `[out, in]` HF orientation would
    block along the wrong axis and produce a floor that no kernel could ever hit.
    """
    out = {}
    for name, tensor in w.items():
        stored = tensor.transpose(-1, -2).unsqueeze(0).unsqueeze(0)  # exactly what `_prep` builds
        out[name] = quantize_like_device(stored, weight_dtype)[0, 0].transpose(-1, -2)
    return out


def _state_dict(w, *, swap_gate_up=False):
    """`{"<proj>.weight": tensor}` as `substate` would hand it to the module.

    `swap_gate_up` is the negative control: it makes the module compute
    `down(silu(up(x)) * gate(x))` while the reference keeps SiLU on `gate`.
    """
    gate_key, up_key = ("up_proj", "gate_proj") if swap_gate_up else ("gate_proj", "up_proj")
    return {
        "gate_proj.weight": w[gate_key],
        "up_proj.weight": w[up_key],
        "down_proj.weight": w["down_proj"],
    }


def _run_mlp(mesh_device, hf, x, state, *, weight_dtype, fused_silu=True, fp32_dest_acc_en=True):
    """Push `x` through the module and return the device output as fp32 torch."""
    mlp = MLP(
        mesh_device,
        hf,
        state,
        weight_dtype=weight_dtype,
        activation_dtype=ACTIVATION_DTYPE,
        fused_silu=fused_silu,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = mlp(tt_x)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    tt_x.deallocate(True)
    tt_out.deallocate(True)
    return out


def _floor(x, w, ref, weight_dtype):
    """PCC of the fp32 reference against a storage-quantised, fp32-arithmetic evaluation."""
    x_q = quantize_like_device(x, ACTIVATION_DTYPE)
    _, floor = comp_pcc(ref, _torch_mlp(x_q, _quantize_weights(w, weight_dtype)), 0.0)
    return float(floor)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", list(PCC_THRESHOLD), ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_mlp_vs_ref(mesh_device, weight_dtype, seq_len, reset_seeds):
    """Dense SwiGLU vs the fp32 reference, at both weight dtypes. Both are run and both recorded."""
    hf = llama_config_dims()
    hidden, intermediate = hf["hidden_size"], hf["intermediate_size"]

    x = torch.randn(1, 1, seq_len, hidden)
    w = _random_weights(hidden, intermediate)

    ref = _torch_mlp(x, w)
    floor = _floor(x, w, ref, weight_dtype)
    out = _run_mlp(mesh_device, hf, x, _state_dict(w), weight_dtype=weight_dtype)

    threshold = PCC_THRESHOLD[weight_dtype]
    passing, pcc = comp_pcc(ref, out, threshold)
    ratio = err_ratio(float(pcc), floor)
    logger.info(
        f"[G-MLP] {_DTYPE_IDS[weight_dtype]} seq={seq_len}: PCC={float(pcc):.7f} "
        f"floor={floor:.7f} ratio={ratio:.2f}x (threshold {threshold}, max ratio {MAX_ERR_RATIO}x)"
    )
    assert passing, f"below threshold {threshold}: {pcc}"
    assert ratio <= MAX_ERR_RATIO, f"{ratio:.2f}x off the noise floor — investigate before recording a PASS"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", list(PCC_THRESHOLD), ids=lambda d: _DTYPE_IDS[d])
def test_mlp_silu_on_wrong_branch_negative_control(mesh_device, weight_dtype, reset_seeds):
    """**The negative control.** SiLU on `up` instead of `gate` must collapse (recipe: 0.6462).

    This is what proves the fused unary is on the argument the module's comment claims. Note how
    *high* a completely wrong activation placement still scores — the whole argument of recipe §2.1.
    """
    hf = llama_config_dims()
    hidden, intermediate = hf["hidden_size"], hf["intermediate_size"]

    x = torch.randn(1, 1, 512, hidden)
    w = _random_weights(hidden, intermediate)
    ref = _torch_mlp(x, w)

    out = _run_mlp(mesh_device, hf, x, _state_dict(w, swap_gate_up=True), weight_dtype=weight_dtype)
    _, pcc = comp_pcc(ref, out, 0.0)

    logger.info(f"[G-MLP] control ({_DTYPE_IDS[weight_dtype]}): SiLU on `up` -> PCC={float(pcc):.5f}")
    assert float(pcc) < PCC_THRESHOLD[weight_dtype], (
        f"the negative control did not collapse ({pcc}) — the gate is not sensitive to which "
        f"branch SiLU is applied to"
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", list(PCC_THRESHOLD), ids=lambda d: _DTYPE_IDS[d])
def test_mlp_compute_kernel_config_ab(mesh_device, weight_dtype, reset_seeds):
    """`fp32_dest_acc_en` True vs False on the three matmuls, measured on this box (recipe §2.4).

    §2.4 predicts `False` costs 96x on a bf8_b `ttnn.linear` and 1168x at bf16, and that `True` is
    bit-identical to passing nothing. Asserting only the *direction* keeps this a measurement rather
    than a second threshold fitted to a number already seen.
    """
    hf = llama_config_dims()
    hidden, intermediate = hf["hidden_size"], hf["intermediate_size"]

    x = torch.randn(1, 1, 512, hidden)
    w = _random_weights(hidden, intermediate)
    state = _state_dict(w)
    ref = _torch_mlp(x, w)
    floor = _floor(x, w, ref, weight_dtype)

    _, pcc_true = comp_pcc(ref, _run_mlp(mesh_device, hf, x, state, weight_dtype=weight_dtype), 0.0)
    _, pcc_false = comp_pcc(
        ref, _run_mlp(mesh_device, hf, x, state, weight_dtype=weight_dtype, fp32_dest_acc_en=False), 0.0
    )
    r_true, r_false = err_ratio(float(pcc_true), floor), err_ratio(float(pcc_false), floor)

    logger.info(
        f"[G-MLP] A/B {_DTYPE_IDS[weight_dtype]}: fp32_dest_acc_en=True PCC={float(pcc_true):.7f} "
        f"({r_true:.2f}x) vs False PCC={float(pcc_false):.7f} ({r_false:.2f}x); floor={floor:.7f}; "
        f"cost of False = {(1 - float(pcc_false)) / (1 - float(pcc_true)):.2f}x"
    )
    assert float(pcc_true) >= float(pcc_false), "fp32_dest_acc_en=True is not better here — recheck §2.4"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", list(PCC_THRESHOLD), ids=lambda d: _DTYPE_IDS[d])
def test_mlp_fused_vs_separate_silu(mesh_device, weight_dtype, reset_seeds):
    """The two spellings of `silu(gate) * up`, measured — `DEC-039`'s evidence.

    Recipe P5.4 says to use `ttnn.mul(..., input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` "if
    available — check, and log which". It **is** available on this build even though
    `ttnn.mul.__doc__` does not list the keyword; the binding is
    `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:1469`. Both spellings must land
    within a hair of each other, or "fused" is doing something other than SiLU.
    """
    hf = llama_config_dims()
    hidden, intermediate = hf["hidden_size"], hf["intermediate_size"]

    x = torch.randn(1, 1, 512, hidden)
    w = _random_weights(hidden, intermediate)
    state = _state_dict(w)
    ref = _torch_mlp(x, w)
    floor = _floor(x, w, ref, weight_dtype)

    _, pcc_fused = comp_pcc(ref, _run_mlp(mesh_device, hf, x, state, weight_dtype=weight_dtype), 0.0)
    _, pcc_split = comp_pcc(ref, _run_mlp(mesh_device, hf, x, state, weight_dtype=weight_dtype, fused_silu=False), 0.0)
    r_fused, r_split = err_ratio(float(pcc_fused), floor), err_ratio(float(pcc_split), floor)

    logger.info(
        f"[G-MLP] SiLU spelling {_DTYPE_IDS[weight_dtype]}: fused PCC={float(pcc_fused):.7f} "
        f"({r_fused:.2f}x) vs separate ttnn.silu PCC={float(pcc_split):.7f} ({r_split:.2f}x); "
        f"floor={floor:.7f}"
    )
    assert float(pcc_fused) >= PCC_THRESHOLD[weight_dtype], f"fused SiLU below threshold: {pcc_fused}"
    assert float(pcc_split) >= PCC_THRESHOLD[weight_dtype], f"separate SiLU below threshold: {pcc_split}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mlp_refuses_to_build_weightless(mesh_device, expect_error):
    """No `state_dict` and no cache path must fail loud, not build three `None` projections."""
    with expect_error(ValueError, "tensor_cache_path"):
        MLP(mesh_device, llama_config_dims(), {})


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mlp_refuses_scatter_output(mesh_device, expect_error, reset_seeds):
    """The scheme-B seam is wired and must **refuse**, not half-implement (`DEC-025`, `DEC-038`).

    `BRINGUP_RECIPE.md:992-994`: wire `scatter_output` from day one so the switch is a flag, but make
    a module that cannot honour it refuse loudly.
    """
    hf = llama_config_dims()
    w = _random_weights(hf["hidden_size"], hf["intermediate_size"])
    with expect_error(NotImplementedError, "scatter_output=True"):
        MLP(mesh_device, hf, _state_dict(w), scatter_output=True)
