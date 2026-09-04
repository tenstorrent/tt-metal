# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-MLP` — `tt/mlp.py` dense SwiGLU vs a torch reference, `(1,1)` mesh.

The reference is the one-line HF body of ``transformers.models.llama.modeling_llama.LlamaMLP``
(``down(silu(gate(x)) * up(x))``, no bias) evaluated in **fp32** on the SAME random weights the TT
module is built from, so nothing about the comparison depends on a checkpoint.

**How this gate is judged — the noise floor, not a borrowed number (``DEC-032``).**
``BRINGUP_RECIPE.md`` Appendix E sets thresholds by copying what ``models/tt_transformers`` measured
(``G-MLP >= 0.999 @bf8_b``, from its 0.9995823). That method is unsound as a *target*: the oracle's
reference loads HF weights at the checkpoint's ``torch_dtype: bfloat16``, so its reference shares the
device's own rounding and its PCC is flattered. Its number is therefore not comparable to a gate
whose reference is fp32.

What is comparable, and distribution-stable, is the **torch-side noise floor**: quantise the inputs
and weights to exactly the dtypes the device will hold them in, do all the arithmetic in fp32, and
PCC *that* against the fp32 reference. Quantisation goes through ttnn itself
(``ttnn.from_torch(..., dtype=...)`` -> ``ttnn.to_torch``), so ``bfloat8_b``'s shared-exponent
blocking is reproduced exactly rather than approximated. Every case below reports

* ``measured`` — device vs the fp32 reference,
* ``floor`` — dtype-quantised-fp32-math vs the fp32 reference,
* ``err ratio`` = ``(1 - measured) / (1 - floor)`` — how many times the unavoidable quantisation
  error the implementation actually costs. **1.0 means the module is exactly at the floor.**

The recipe's absolute thresholds are kept as floors that must be cleared, and the error ratio is the
diagnostic: clearing 0.999 while sitting 20x off the noise floor is a finding, not a clean pass.

**Input distribution** (mandated by ``DEC-026`` / ``R-018``): ``torch.randn``, which is also the
oracle's own (``models/tt_transformers/tests/test_mlp.py:96``). **Reference dtype policy:** fp32
weights, fp32 activations, fp32 arithmetic — strictly harder than the oracle's bf16-weight
reference.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_mlp_vs_ref.py -x -q
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, err_ratio, quantize_like_device
from models.demos.llama31_8b_d_p.tt.mlp import MLP, default_compute_kernel_config

# 03_OUTLINE.md §5 / Appendix E. Kept as hard floors; the error ratio below is the real diagnostic.
PCC_THRESHOLD = {ttnn.bfloat8_b: 0.999, ttnn.bfloat16: 0.9995}
ORACLE_PCC_BF8 = 0.9995823  # tt_transformers test_mlp.py, seq 512, bf8_b — NOT a target (see above)

# How far off the torch noise floor a module may sit before the gate calls it a finding. 3x is
# generous: an implementation that only adds device rounding on top of the same quantisation lands
# at ~1-2x. DEC-032.
MAX_ERR_RATIO = 3.0

# Weight magnitude. 0.02 is the order of a real Llama projection weight, and it matters for a
# block-float dtype: bf8_b shares one exponent per block, so the *relative* spread inside a block is
# what the quantisation sees.
WEIGHT_SCALE = 0.02


# `quantize_like_device` and `err_ratio` were DEFINED here in P5 (`DEC-037`) and promoted to
# `tests/test_factory.py` in P6 (`DEC-046`), which could not delete these copies at the time. P9
# finished the promotion (`DEC-124`): this file now imports them like the other nine test files do,
# so the primitive every gate's error ratio is built on has exactly one definition.


def _random_mlp_state(hf_config, *, seed: int) -> dict:
    """HF-layout ``[out, in]`` weights for the three projections, no bias (``mlp_bias: false``)."""
    gen = torch.Generator().manual_seed(seed)
    h, i = hf_config.hidden_size, hf_config.intermediate_size
    return {
        "gate_proj.weight": torch.randn(i, h, generator=gen) * WEIGHT_SCALE,
        "up_proj.weight": torch.randn(i, h, generator=gen) * WEIGHT_SCALE,
        "down_proj.weight": torch.randn(h, i, generator=gen) * WEIGHT_SCALE,
    }


def _torch_mlp(x: torch.Tensor, state: dict) -> torch.Tensor:
    """``LlamaMLP.forward``: ``down(silu(gate(x)) * up(x))``, fp32, no bias."""
    gate = torch.nn.functional.linear(x, state["gate_proj.weight"])
    up = torch.nn.functional.linear(x, state["up_proj.weight"])
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, state["down_proj.weight"])


def _mlp_noise_floor(x: torch.Tensor, state: dict, weight_dtype) -> torch.Tensor:
    """The best any implementation can do: device dtypes for the stored tensors, fp32 arithmetic.

    Activations are ``bfloat16`` (``03_OUTLINE.md`` §1 convention 11); the three weights are
    ``weight_dtype``. The weights are quantised in the ttnn ``[1, 1, in, out]`` layout the module
    actually stores them in, because bf8_b blocking is layout-dependent.
    """
    x_q = quantize_like_device(x, ttnn.bfloat16)
    q = {
        k: quantize_like_device(w.transpose(-1, -2).unsqueeze(0).unsqueeze(0), weight_dtype)[0, 0].transpose(-1, -2)
        for k, w in state.items()
    }
    return _torch_mlp(x_q, q)


def _run_tt_mlp(
    mesh_device, hf_config, mesh_config, state, x, *, weight_dtype, scatter_output=False, compute_kernel_config=None
):
    tt_mlp = MLP(
        mesh_device,
        hf_config,
        state,
        mesh_config=mesh_config,
        ccl_manager=None,  # TP=1: no collective is entered
        weight_dtype=weight_dtype,
        scatter_output=scatter_output,
        compute_kernel_config=compute_kernel_config,
    )
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_mlp(tt_x)
    return ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 512, 4096], ids=["s32", "s512", "s4096"])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_mlp_vs_ref(mesh_device, seq_len, weight_dtype, reset_seeds):
    """Dense SwiGLU on device vs fp32 torch, identical random weights. See the module docstring."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config, mesh_config = objs["hf_config"], objs["mesh_config"]
    threshold = PCC_THRESHOLD[weight_dtype]

    assert (
        hf_config.hidden_size == 4096 and hf_config.intermediate_size == 14336
    ), f"unexpected MLP dims {hf_config.hidden_size} / {hf_config.intermediate_size}"
    assert hf_config.hidden_act == "silu"

    state = _random_mlp_state(hf_config, seed=0)
    # The oracle's own distribution (models/tt_transformers/tests/test_mlp.py:96) — R-018/DEC-026.
    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.float32)

    ref = _torch_mlp(x, state)
    out = _run_tt_mlp(mesh_device, hf_config, mesh_config, state, x, weight_dtype=weight_dtype)
    floor_out = _mlp_noise_floor(x, state, weight_dtype)

    assert out.shape == ref.shape == floor_out.shape == (1, 1, seq_len, hf_config.hidden_size)
    passing, pcc = comp_pcc(ref, out, threshold)
    _, floor_pcc = comp_pcc(ref, floor_out, 0.0)
    ratio = err_ratio(pcc, floor_pcc)

    logger.info(comp_allclose(ref, out))
    logger.info(
        f"[G-MLP] seq_len={seq_len} weight_dtype={weight_dtype}: measured PCC = {pcc} | "
        f"torch noise floor = {floor_pcc} | err ratio = {ratio:.2f}x | threshold {threshold} | "
        f"oracle @bf8_b {ORACLE_PCC_BF8} (not a target, DEC-032)"
    )
    assert passing, f"[G-MLP] seq_len={seq_len} {weight_dtype} below {threshold}: {pcc}"
    assert ratio <= MAX_ERR_RATIO, (
        f"[G-MLP] seq_len={seq_len} {weight_dtype}: PCC {pcc} clears {threshold} but sits "
        f"{ratio:.1f}x off the torch noise floor {floor_pcc} — investigate before recording a PASS "
        f"(DEC-032)"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_fp32_dest_acc_is_load_bearing(mesh_device, weight_dtype, reset_seeds):
    """A/B the compute-kernel config: ``fp32_dest_acc_en=True`` must not be worse than ``False``.

    Measured evidence for ``DEC-031``. Three configurations on identical inputs and weights:
    the module default (HiFi4 + ``fp32_dest_acc_en=True``), the same with fp32 accumulation off,
    and the op's own default (no config passed at all — what a copy of
    ``models/demos/minimax_m3/tt/dense_mlp.py:89`` would ship). The numbers go into the ``G-MLP``
    detail block; the assert only guards the direction, because a regression here is silent.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config, mesh_config = objs["hf_config"], objs["mesh_config"]
    seq_len = 512

    state = _random_mlp_state(hf_config, seed=0)
    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.float32)
    ref = _torch_mlp(x, state)
    _, floor_pcc = comp_pcc(ref, _mlp_noise_floor(x, state, weight_dtype), 0.0)

    no_fp32 = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    variants = {
        "HiFi4+fp32_dest_acc (module default)": default_compute_kernel_config(mesh_device),
        "HiFi4, fp32_dest_acc=False": no_fp32,
        "op default (no config)": None,
    }
    pccs = {}
    for name, cfg in variants.items():
        out = _run_tt_mlp(
            mesh_device, hf_config, mesh_config, state, x, weight_dtype=weight_dtype, compute_kernel_config=cfg
        )
        _, pcc = comp_pcc(ref, out, 0.0)
        pccs[name] = float(pcc)
        logger.info(
            f"[G-MLP] compute-kernel A/B ({weight_dtype}, seq {seq_len}): {name}: PCC = {pcc} | "
            f"err ratio = {err_ratio(pcc, floor_pcc):.2f}x (floor {floor_pcc})"
        )

    best = pccs["HiFi4+fp32_dest_acc (module default)"]
    for name, pcc in pccs.items():
        assert best >= pcc - 1e-9, (
            f"[G-MLP] the module default ({best}) is worse than {name} ({pcc}) at {weight_dtype}; "
            f"DEC-031 chose fp32_dest_acc_en=True on measured evidence — re-measure and re-decide"
        )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_silu_is_on_the_gate_branch(mesh_device, reset_seeds):
    """Negative control: the fused SiLU must sit on ``gate``, not on ``up``.

    ``ttnn.mul(gate, up, input_tensor_a_activations=[SILU])`` applies the unary to
    ``input_tensor_a`` only, so swapping the two arguments is a silent, plausible-looking bug —
    both branches have the same shape and the same weight distribution, and the output stays the
    right size. This is the ``test_rope_vs_ref.py:140`` negative-control pattern: build the
    reference the WRONG way round and show the PCC collapses, otherwise the positive test above
    cannot distinguish a pass from a symmetric error on both sides.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config, mesh_config = objs["hf_config"], objs["mesh_config"]
    seq_len = 128

    state = _random_mlp_state(hf_config, seed=0)
    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.float32)

    gate = torch.nn.functional.linear(x, state["gate_proj.weight"])
    up = torch.nn.functional.linear(x, state["up_proj.weight"])
    swapped = torch.nn.functional.linear(
        gate * torch.nn.functional.silu(up), state["down_proj.weight"]
    )  # SiLU on the WRONG operand

    out = _run_tt_mlp(mesh_device, hf_config, mesh_config, state, x, weight_dtype=ttnn.bfloat16)

    _, pcc_right = comp_pcc(_torch_mlp(x, state), out, 0.0)
    _, pcc_wrong = comp_pcc(swapped, out, 0.0)
    logger.info(f"[G-MLP] negative control: PCC vs correct = {pcc_right}, PCC vs SiLU-on-up = {pcc_wrong}")
    assert float(pcc_wrong) < 0.99, (
        f"SiLU-on-up scored {pcc_wrong}; the fused-activation operand is not being tested by the " f"positive gate"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_scatter_output_is_a_noop_at_tp1(mesh_device, reset_seeds):
    """``scatter_output`` is wired but inert at TP=1 — scheme B stays a flag (``DEC-018``).

    At TP=1 neither branch of the ``if tp > 1`` tail runs, so both settings must return the same
    full-width tensor bit-for-bit. This is the cheap proof that the parameter is plumbed through
    (P8 measures the scheme-B collective itself).
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config, mesh_config = objs["hf_config"], objs["mesh_config"]
    assert mesh_config.tp == 1, f"this test asserts the TP=1 behaviour; mesh_config.tp is {mesh_config.tp}"
    seq_len = 128

    state = _random_mlp_state(hf_config, seed=0)
    x = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.float32)

    a = _run_tt_mlp(mesh_device, hf_config, mesh_config, state, x, weight_dtype=ttnn.bfloat16, scatter_output=False)
    b = _run_tt_mlp(mesh_device, hf_config, mesh_config, state, x, weight_dtype=ttnn.bfloat16, scatter_output=True)

    assert a.shape == b.shape == (1, 1, seq_len, hf_config.hidden_size)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    logger.info("[G-MLP] scatter_output True/False are bit-identical at TP=1 (no collective entered)")


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_mlp_rejects_a_non_silu_config(mesh_device, reset_seeds, expect_error):
    """A config whose ``hidden_act`` is not ``silu`` is refused at construction, not silently run.

    ``03_OUTLINE.md`` §1 convention 12: assert on features Llama lacks instead of branching. The
    fused ``ttnn.mul(..., input_tensor_a_activations=[SILU])`` hard-codes the activation, so a
    ``gelu`` config would otherwise be computed as SwiGLU with no error anywhere.
    """
    from dataclasses import replace

    objs = TestFactory.setup_test(mesh_device)
    hf_config, mesh_config = objs["hf_config"], objs["mesh_config"]

    with expect_error(AssertionError, "hidden_act"):
        MLP(
            mesh_device,
            replace(hf_config, hidden_act="gelu"),
            _random_mlp_state(hf_config, seed=0),
            mesh_config=mesh_config,
        )
    with expect_error(AssertionError, "mlp_bias"):
        MLP(
            mesh_device,
            replace(hf_config, mlp_bias=True),
            _random_mlp_state(hf_config, seed=0),
            mesh_config=mesh_config,
        )
