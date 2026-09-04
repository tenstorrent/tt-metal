# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The shape of a gate test. Copy this file, replace the module and the reference.

Five properties make a PCC number mean something. A test missing any of them can
report 0.999 and still be hiding a bug:

1. **Identical weights on both sides.** Random is fine and needs no checkpoint --
   what matters is that the reference and the device see the same numbers.
2. **An fp32 reference.** A reference built at the device's own dtype shares its
   rounding and reports a flattered PCC. See BRINGUP_RECIPE.md section 2.1.
3. **A computed noise floor**, and a gate on the *ratio* of errors, not on an
   absolute PCC copied from somewhere else.
4. **A negative control.** Break the thing the test is supposed to be sensitive
   to and assert the PCC collapses. Without it a pass cannot be distinguished
   from a symmetric bug on both sides. Measured examples from one bring-up: a
   RoPE wrong at every position still scored 0.87 at layer level; a wrong GQA
   head map scored 0.55; a *rotated* KV head-to-column mapping still scored
   0.9989 -- which is why a layout bug needs bit-equality, not PCC.
5. **The recorded context**: input distribution and reference dtype policy, so
   the number can be compared with anything later.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# from <your package>.tests.test_factory import err_ratio, quantize_like_device


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 512, 4096], ids=lambda s: f"s{s}")
def test_module_vs_ref(mesh_device, seq_len, reset_seeds):
    """<MODULE> vs a torch reference. Gate: G-<NAME>.

    Input distribution: standard normal (state it; never choose it to pass).
    Reference dtype policy: fp32 weights, fp32 activations, fp32 arithmetic.
    """
    dtype = ttnn.bfloat8_b
    torch.manual_seed(0)

    x = torch.randn(1, 1, seq_len, HIDDEN)
    weights = {"w": torch.randn(HIDDEN, HIDDEN) * 0.02}

    # --- reference, fp32 -------------------------------------------------------
    ref = torch_reference(x.float(), {k: v.float() for k, v in weights.items()})

    # --- noise floor: quantise what the device STORES, rest in fp32 ------------
    floor_in = {k: quantize_like_device(v, dtype) for k, v in weights.items()}
    floor_out = torch_reference(quantize_like_device(x, dtype), floor_in)
    _, floor = comp_pcc(ref, floor_out, 0.0)

    # --- device ----------------------------------------------------------------
    module = YourModule(mesh_device, weights, weight_dtype=dtype)
    tt_out = module(to_device(x, mesh_device))
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()

    passing, pcc = comp_pcc(ref, out, THRESHOLD)
    ratio = err_ratio(float(pcc), float(floor))
    logger.info(f"[G-<NAME>] seq={seq_len}: PCC={pcc} floor={floor} err_ratio={ratio:.2f}x")

    assert passing, f"below threshold {THRESHOLD}: {pcc}"
    assert ratio <= MAX_RATIO, f"{ratio:.2f}x off the noise floor -- investigate before recording a PASS"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_module_negative_control(mesh_device, reset_seeds):
    """Break the property the gate is sensitive to; assert the PCC collapses.

    If this test passes with a high PCC, the positive test above proves nothing.
    """
    ...  # e.g. transpose a weight, rotate a head mapping, skip the activation
    assert pcc < 0.99, f"negative control did not collapse ({pcc}) -- the gate is not sensitive"
