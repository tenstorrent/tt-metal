# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-RMS` — `tt/rms_norm.py` vs the torch reference, `(1,1)` mesh, TP=1, no CCL.

The block: ``out = x * rsqrt(mean(x^2) + eps) * weight``. **Plain** RMSNorm — no Gemma ``+1``
weight fold (`00_MODEL_CARD.md` §2, `DEC-004` of P0's card). `eps` is `rms_norm_eps` = 1e-05, read
from the bundled `config.json` via `llama_hf_config`.

Oracle for the *threshold*: the existing `models/tt_transformers` implementation scores
**0.9999867 / 0.9999886** on this exact box with real weights
(`BRINGUP_RECIPE.md` Appendix E), so the gate is **PCC >= 0.9999** — not the 0.999 the recipe's
inline text guessed. A fresh module landing materially below what an existing implementation
already achieves on the same op and dtype is a bug, not a precision limit.

Oracle for the *math*: `_rms_norm` from `tests/unit/test_reference_model.py:136`, which `G-REF`
proved bit-exact against HF's `LlamaRMSNorm` inside a full decoder layer (PCC 1.0). Reused rather
than rewritten, so this file cannot drift from the gate-validated reference.

Both sides are driven by **identical** random weights; the reference runs in fp32 on the fp32
input, and the device gets the bfloat16 cast of that same input — so the measured PCC includes the
activation-quantisation error, which is the honest number.

The norm gain is centred on 1 but not constant (`1 + 0.1 * randn`): a gain of exactly ones makes
the weight multiply a no-op and would hide a whole class of bug.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_rms_norm_vs_ref.py -x -q
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory
from models.demos.llama31_8b_d_p.tests.unit.test_reference_model import _rms_norm
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm

# BRINGUP_RECIPE.md Appendix E (measured), not the inline 0.999 guess.
PCC_THRESHOLD = 0.9999


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", TestFactory.NORM_SEQ_LENS, ids=[f"s{s}" for s in TestFactory.NORM_SEQ_LENS])
@torch.no_grad()
def test_rms_norm_vs_ref(mesh_device, seq_len, reset_seeds):
    """One RMSNorm on `[1, 1, S, 4096]`, device vs torch, identical weights. PCC >= 0.9999."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    hidden = hf_config.hidden_size

    assert (
        hidden == 4096 and hf_config.rms_norm_eps == 1e-05
    ), f"unexpected norm dims: hidden={hidden}, eps={hf_config.rms_norm_eps}"

    weight = 1.0 + 0.1 * torch.randn(hidden, dtype=torch.float32)
    x = torch.randn(1, 1, seq_len, hidden, dtype=torch.float32)

    # --- reference (fp32, gate-validated math)
    ref = _rms_norm(x, weight, hf_config.rms_norm_eps)

    # --- device
    tt_norm = RMSNorm(mesh_device, hf_config, {"weight": weight}, mesh_config=objs["mesh_config"])
    assert tt_norm.is_distributed is False, "is_distributed must default to False this iteration (DEC-024)"
    assert tuple(tt_norm.tt_weight.shape) == (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)

    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_norm(tt_x)
    out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    assert out.shape == ref.shape == (1, 1, seq_len, hidden)

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(comp_allclose(ref, out))
    logger.info(f"[G-RMS] seq_len={seq_len}: PCC = {pcc} (threshold {PCC_THRESHOLD}, tt_transformers oracle 0.9999867)")
    assert passing, f"[G-RMS] seq_len={seq_len} below {PCC_THRESHOLD}: {pcc}"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_rms_norm_has_no_gemma_unit_offset(mesh_device, reset_seeds):
    """A zero gain must produce a zero output.

    This is the cheap, decisive check that no ``(1 + weight)`` fold crept in: under the Gemma fold a
    zero gain would return the normalised input unchanged instead of zeros. Llama's config has no
    ``use_gemma_norm``/``add_unit_offset`` key at all (`00_MODEL_CARD.md` §3), so the fold must be
    absent, not merely disabled.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    hidden = hf_config.hidden_size

    zero_gain = torch.zeros(hidden, dtype=torch.float32)
    x = torch.randn(1, 1, 32, hidden, dtype=torch.float32)

    tt_norm = RMSNorm(mesh_device, hf_config, {"weight": zero_gain}, mesh_config=objs["mesh_config"])
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.to_torch(tt_norm(tt_x), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    max_abs = out.abs().max().item()
    logger.info(f"[G-RMS] zero-gain probe: max|out| = {max_abs} (must be 0.0; a Gemma +1 fold would give ~1)")
    assert max_abs == 0.0, f"weight=0 produced a non-zero output ({max_abs}); a Gemma-style +1 fold is present"
