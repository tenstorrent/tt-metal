# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm output vs the torch reference. Pattern: ``minimax_m3/tests/unit/test_norm_vs_ref.py``.

Mistral uses a PLAIN RMSNorm (``out = x_normed * weight``) — there is no Gemma ``(1 + weight)`` fold
and no ``use_gemma_norm`` key in the config — so the fold is exercised only to prove it stays OFF: a
future config that switched it on must change this test, not silently change every layer's output.

Both norms in a decoder layer are this class (``input_layernorm`` and ``post_attention_layernorm``),
and so is the model's final norm — the M2 table's "final-norm instance" row is the same test applied
to the tail instance, which ``test_model_sp_vs_ref.py`` covers in place.

Random weights, identical on both sides. Target mesh (SP=4 x TP=8); the residual stream is replicated
across TP in this package, so the norm sees full hidden width on every device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.rms_norm import RMSNorm

from ..test_factory import from_device_replicated, parametrize_mesh, to_device


@parametrize_mesh()
@pytest.mark.parametrize(
    "tokens, width",
    [
        (128, C.HIDDEN_SIZE),  # the real decoder-norm geometry
        (512, C.HIDDEN_SIZE),  # a longer chunk, several tiles of tokens
        (32, C.HIDDEN_SIZE),  # a single tile of tokens
    ],
    ids=["t128", "t512", "t32"],
)
def test_rms_norm_vs_ref(mesh_device, device_params, tokens, width, reset_seeds):
    """The tt RMSNorm class vs the HF-derived torch reference, random weights."""
    eps = C.RMS_NORM_EPS
    x = torch.randn(1, 1, tokens, width)
    weight = torch.randn(width) * 0.1 + 1.0

    ref = reference.rms_norm_reference(x, weight, eps)

    norm = RMSNorm(
        mesh_device=mesh_device,
        hf_config=SimpleNamespace(rms_norm_eps=eps),
        state_dict={"weight": weight},
        tensor_cache_path=None,
    )
    out_tt = norm(to_device(x, mesh_device))
    out = from_device_replicated(out_tt, (1, 1, tokens, width))

    passing, pcc = comp_pcc(ref, out, SPEC.pcc)
    logger.info(f"rms_norm tokens={tokens} width={width}: pcc={pcc}")
    assert passing, f"RMSNorm PCC fail (tokens={tokens}, width={width}): {pcc}"


@parametrize_mesh()
def test_rms_norm_is_plain_not_gemma(mesh_device, device_params, reset_seeds):
    """The Gemma ``(1 + w)`` fold must stay OFF: the two forms differ, and Mistral is the plain one.

    Without this, a stray ``use_gemma_norm`` would shift every norm in the model by a factor that
    still PCCs ~1.0 against itself and only shows up as a whole-model accuracy loss.
    """
    eps, tokens, width = C.RMS_NORM_EPS, 64, C.HIDDEN_SIZE
    x = torch.randn(1, 1, tokens, width)
    weight = torch.randn(width) * 0.1 + 1.0

    norm = RMSNorm(
        mesh_device=mesh_device,
        hf_config=SimpleNamespace(rms_norm_eps=eps),
        state_dict={"weight": weight},
        tensor_cache_path=None,
    )
    assert norm.use_gemma_norm is False, "Mistral RMSNorm must not fold +1 into the weight"

    out = from_device_replicated(norm(to_device(x, mesh_device)), (1, 1, tokens, width))
    plain = reference.golden_rms_norm(x, weight, eps)
    gemma = reference.golden_rms_norm(x, weight + 1.0, eps)

    ok_plain, pcc_plain = comp_pcc(plain, out, SPEC.pcc)
    _, pcc_gemma = comp_pcc(gemma, out, SPEC.pcc)
    logger.info(f"rms_norm plain pcc={pcc_plain} vs gemma-form pcc={pcc_gemma}")
    assert ok_plain, f"device norm does not match the plain form: {pcc_plain}"
    assert pcc_gemma < pcc_plain, "the two norm forms are indistinguishable here; the test proves nothing"
