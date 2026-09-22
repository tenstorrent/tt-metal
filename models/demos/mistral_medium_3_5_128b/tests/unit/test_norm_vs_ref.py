# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite row 1: device RMSNorm vs the torch reference.

Recipe row: ``minimax_m3/tests/unit/test_norm_vs_ref.py``. The row calls out "including the Gemma
``(1 + weight)`` fold if used" — this model does **not** use it, and the second test below is what
makes that a measurement rather than a claim: with ``weight = 0`` plain RMSNorm gives exactly zero
and the Gemma form gives the normalized input, so the two cannot both pass.

Full width (hidden 12288) on the target mesh. The norm is local — every chip holds the whole
hidden vector for its own tokens — so this test also pins the activation layout: if the hidden dim
were TP-sharded instead of replicated, the local reduction would be over a quarter of the
elements and the PCC would collapse.

``test_rms_norm_scale`` is deliberately *not* a PCC check. PCC is invariant to a scale factor, so
it cannot see the failure mode this op actually has at this width: a bf16 accumulation of the
12288 squares comes out ~23% low and scales the output up by ~1.14 while leaving PCC at 0.9998
(see ``tt/rms_norm.py``). The absolute-RMS assertion is the only thing that catches it.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, MistralRMSNorm
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_sp, to_mesh
from models.demos.mistral_medium_3_5_128b.tt.rms_norm import RMSNorm

SEQ = 2048  # multiple of TILE_SIZE * sp = 256


def _reference(cfg, weight, x):
    ref = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, REF_DTYPE)
    with torch.no_grad():
        ref.weight.copy_(weight)
        return ref(x)


@pytest.mark.parametrize("seq", [SEQ])
def test_rms_norm_vs_ref(galaxy, mesh_config, cfg, seq):
    """RMSNorm at real width against the torch reference, same weights on both sides."""
    torch.manual_seed(0)
    weight = (1.0 + 0.02 * torch.randn(cfg.hidden_size)).to(REF_DTYPE)
    x = torch.randn(1, 1, seq, cfg.hidden_size).to(REF_DTYPE)

    ref_out = _reference(cfg, weight, x)

    norm = RMSNorm(galaxy, cfg, {"weight": weight}, mesh_config=mesh_config)
    tt_x = to_mesh(galaxy, x, dims=[-2, None])
    tt_out = norm(tt_x)
    out = from_mesh_sp(galaxy, tt_out)

    assert_pcc("rms_norm", ref_out, out)


@pytest.mark.parametrize("seq", [SEQ])
def test_rms_norm_scale(galaxy, mesh_config, cfg, seq):
    """The output magnitude must match the reference, not just its direction.

    With ``weight == 1`` every output row has RMS ``1`` by construction, so the check is a direct
    read of the reduction's accuracy: a sum of squares that is low by a factor ``f`` shows up here
    as a row RMS of ``1/sqrt(f)``. Asserted per row, because the bf16-accumulation error is
    data-dependent and a mean over rows would dilute it.
    """
    torch.manual_seed(1)
    weight = torch.ones(cfg.hidden_size, dtype=REF_DTYPE)
    x = (torch.randn(1, 1, seq, cfg.hidden_size) * 0.1).to(REF_DTYPE)

    norm = RMSNorm(galaxy, cfg, {"weight": weight}, mesh_config=mesh_config)
    out = from_mesh_sp(galaxy, norm(to_mesh(galaxy, x, dims=[-2, None])))[0, 0].float()

    row_rms = out.pow(2).mean(-1).sqrt()
    worst = (row_rms - 1.0).abs().max().item()
    print(f"rms_norm row RMS: min {row_rms.min():.6f} max {row_rms.max():.6f} (want 1.0)")
    # 1% covers bf16 storage of the output; the accumulation bug is 7-20% per row.
    assert worst < 0.01, (
        f"RMSNorm output row RMS is off by {worst:.4f} with a unit weight. The sum of squares is "
        f"being accumulated in bf16 — the compute kernel config needs fp32_dest_acc_en=True."
    )


def test_rms_norm_is_not_gemma(galaxy, mesh_config, cfg):
    """A zero weight must give a zero output.

    Plain RMSNorm scales by ``w``; the Gemma variant scales by ``1 + w``. With ``w = 0`` the first
    returns zeros and the second returns the normalized input, so this distinguishes them without
    needing a second reference implementation. The borrowed sources carry a ``use_gemma_norm``
    switch; this package deleted it, and this test is why that deletion is safe.
    """
    seq = 256
    x = torch.randn(1, 1, seq, cfg.hidden_size).to(REF_DTYPE)
    weight = torch.zeros(cfg.hidden_size, dtype=REF_DTYPE)

    norm = RMSNorm(galaxy, cfg, {"weight": weight}, mesh_config=mesh_config)
    tt_out = norm(to_mesh(galaxy, x, dims=[-2, None]))
    out = from_mesh_sp(galaxy, tt_out)

    assert out.float().abs().max() == 0.0, "zero weight must give zero output; this looks like a Gemma (1+w) fold"
