# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The activation vs a torch reference, at the model's exact variant and constants.
Pattern: ``minimax_m3/tests/unit/test_swiglu_vs_ref.py``.

Mistral's variant is PLAIN silu SwiGLU: ``silu(gate) * up``, where ``silu(z) = z * sigmoid(z)``.
There is no clamp, no ``alpha`` and no ``(up + 1)`` term — the donor (M3) uses clamped SwiGLU-OAI
(``swiglu_limit`` / ``swiglu_alpha``), so this test's whole job is to pin the difference: an
accidental import of the donor's activation would still produce plausible-looking activations and
would only surface as a whole-model accuracy loss.

There is no ``ttnn.swiglu``; the MLP composes it from ``ttnn.silu`` + ``ttnn.multiply``, which is
exactly the decomposition checked here. Run at the model's real intermediate width per TP shard.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC

from ..test_factory import from_device_replicated, parametrize_mesh, to_device

# The per-TP-shard intermediate width the MLP's activation actually sees at the spec's TP=8.
LOCAL_INTERMEDIATE = C.INTERMEDIATE_SIZE // SPEC.tp


def _tt_swiglu(gate, up):
    """The MLP's own composition, isolated: silu(gate) * up."""
    return ttnn.multiply(ttnn.silu(gate), up)


@parametrize_mesh()
@pytest.mark.parametrize("tokens", [128, 512], ids=["t128", "t512"])
def test_swiglu_vs_ref(mesh_device, device_params, tokens, reset_seeds):
    """``ttnn.silu(gate) * up`` vs the torch reference, at the real per-shard intermediate width."""
    gate = torch.randn(1, 1, tokens, LOCAL_INTERMEDIATE)
    up = torch.randn(1, 1, tokens, LOCAL_INTERMEDIATE)

    ref = reference.golden_silu(gate.float()) * up.float()

    out_tt = _tt_swiglu(to_device(gate, mesh_device), to_device(up, mesh_device))
    out = from_device_replicated(out_tt, (1, 1, tokens, LOCAL_INTERMEDIATE))

    passing, pcc = comp_pcc(ref, out, SPEC.pcc)
    logger.info(f"swiglu tokens={tokens} width={LOCAL_INTERMEDIATE}: pcc={pcc}")
    assert passing, f"SwiGLU PCC fail (tokens={tokens}): {pcc}"


@parametrize_mesh()
def test_swiglu_is_plain_not_clamped_oai(mesh_device, device_params, reset_seeds):
    """The plain and clamped-OAI variants must be distinguishable, and the device is the plain one.

    The donor's activation is ``(clamp(up) + 1) * (g * sigmoid(alpha * g))`` with ``g`` clamped at
    ``swiglu_limit``; over inputs large enough to clamp, that differs sharply from ``silu(g) * up``.
    Driving the comparison with wide inputs is deliberate: at small magnitudes the two agree closely
    enough that the test would prove nothing.
    """
    tokens, width, alpha, limit = 64, LOCAL_INTERMEDIATE, 1.702, 7.0
    gate = torch.randn(1, 1, tokens, width) * 6.0  # wide enough that the OAI clamp bites
    up = torch.randn(1, 1, tokens, width) * 6.0

    out = from_device_replicated(
        _tt_swiglu(to_device(gate, mesh_device), to_device(up, mesh_device)), (1, 1, tokens, width)
    )

    plain = reference.golden_silu(gate.float()) * up.float()
    g_c = gate.float().clamp(max=limit)
    u_c = up.float().clamp(min=-limit, max=limit)
    clamped_oai = (u_c + 1.0) * (g_c * torch.sigmoid(alpha * g_c))

    ok_plain, pcc_plain = comp_pcc(plain, out, SPEC.pcc)
    _, pcc_oai = comp_pcc(clamped_oai, out, SPEC.pcc)
    logger.info(f"swiglu plain pcc={pcc_plain} vs clamped-OAI pcc={pcc_oai}")
    assert ok_plain, f"device activation does not match plain silu SwiGLU: {pcc_plain}"
    assert pcc_oai < pcc_plain, "plain and clamped-OAI SwiGLU are indistinguishable here; test proves nothing"
