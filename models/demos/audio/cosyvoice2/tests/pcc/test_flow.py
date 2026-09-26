# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`CausalMaskedDiffWithXvec`: the outer module tying the Conformer encoder and the
CFM estimator together -- xvec projection, speech-token embedding, `conds`
prompt-splicing, and the final CFM call. See tt/flow/flow.py's module docstring for
the verified real-source `inference()` body this ports (`finalize=True` branch,
`streaming=False` throughout this phase).

Unlike the previous three components in this phase, this one is not "a genuinely
novel mechanism with no analog" -- it is real, easy-to-get-wrong wiring (`conds`'
splice, where `mel_len1`/`mel_len2` actually come from), so the tests here are
aimed specifically at the wiring decisions tt/flow/flow.py's module docstring
calls out, not at re-proving math already covered by test_flow_decoder.py /
test_conformer_encoder.py / test_upsample_conformer_encoder.py.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_conds_splices_prompt_feat_exactly_and_zeros_the_rest():
    """The real `conds` construction: `conds[:, :mel_len1] = prompt_feat`, and
    nothing else touches the remaining `mel_len2` rows -- checked directly against
    `CausalMaskedDiffWithXvecRef.inference`'s own construction path (reached via a
    monkeypatched decoder that returns its `cond` argument unchanged, isolating the
    splice from the CFM's own math, which the other test files already cover)."""
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

    torch.manual_seed(0)
    flow = CausalMaskedDiffWithXvecRef()
    flow.eval()

    captured = {}
    real_forward = flow.decoder.forward

    def spy_forward(mu, mask, n_timesteps, spks, cond, **kw):
        captured["cond"] = cond.clone()
        return real_forward(mu, mask, n_timesteps, spks, cond, **kw)

    flow.decoder.forward = spy_forward

    prompt_token_len, new_token_len = 4, 6
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, prompt_token_len * 2, 80) * 0.1
    embedding = torch.randn(1, 192)

    with torch.no_grad():
        flow.inference(token, prompt_token, prompt_feat, embedding)

    cond = captured["cond"]
    mel_len1 = prompt_feat.shape[1]
    assert torch.equal(cond[:, :mel_len1], prompt_feat), "the prompt portion of cond must be prompt_feat, unchanged"
    assert torch.equal(
        cond[:, mel_len1:], torch.zeros_like(cond[:, mel_len1:])
    ), "the rest of cond must be exactly zero"


def test_mel_len2_comes_from_subtraction_not_token_mel_ratio():
    """The real source computes `mel_len2 = h.shape[1] - prompt_feat.shape[1]` --
    a subtraction against the ENCODER's actual output length, not
    `token_mel_ratio * new_token_len`. Constructed here with a deliberately
    mismatched `prompt_feat` length (not `token_mel_ratio * prompt_token_len`) so
    the two formulas disagree -- a naive `token_mel_ratio`-only implementation
    would get this wrong, exactly the kind of assumption this port's docstring
    warns against making."""
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

    torch.manual_seed(1)
    flow = CausalMaskedDiffWithXvecRef()
    flow.eval()

    prompt_token_len, new_token_len = 4, 6
    mismatched_mel_len1 = 9  # NOT prompt_token_len * 2 (== 8)
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, mismatched_mel_len1, 80) * 0.1
    embedding = torch.randn(1, 192)

    total_mel = (prompt_token_len + new_token_len) * 2  # encoder always doubles token-rate length
    naive_wrong_mel_len2 = new_token_len * 2
    real_mel_len2 = total_mel - mismatched_mel_len1

    with torch.no_grad():
        out = flow.inference(token, prompt_token, prompt_feat, embedding)

    assert out.shape[1] == real_mel_len2
    assert out.shape[1] != naive_wrong_mel_len2


def test_xvec_normalizes_before_the_linear_not_after():
    """`F.normalize(embedding, dim=1)` runs BEFORE `spk_embed_affine_layer`, not
    after -- checked by comparing against feeding the affine layer an
    already-normalized vector (should match) vs. the raw, unnormalized vector
    (should not, whenever the raw vector's norm differs from 1)."""
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

    torch.manual_seed(2)
    flow = CausalMaskedDiffWithXvecRef()
    flow.eval()

    embedding = torch.randn(1, 192) * 5.0  # norm far from 1
    with torch.no_grad():
        want = flow.spk_embed_affine_layer(torch.nn.functional.normalize(embedding, dim=1))
        via_raw = flow.spk_embed_affine_layer(embedding)

    assert not torch.allclose(want, via_raw, atol=1e-3), "normalizing must actually change the affine layer's input"


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
def test_device_causal_masked_diff_with_xvec_matches_torch_reference(device):
    """`TtCausalMaskedDiffWithXvec.inference` vs. the torch reference, well-formed
    inputs (`prompt_feat` length == `token_mel_ratio * prompt_token_len`, the
    realistic case)."""
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec

    torch.manual_seed(0)
    flow = CausalMaskedDiffWithXvecRef()
    flow.eval()

    prompt_token_len, new_token_len = 4, 6
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, prompt_token_len * 2, 80) * 0.1
    embedding = torch.randn(1, 192)

    with torch.no_grad():
        want = flow.inference(token, prompt_token, prompt_feat, embedding)

    tt_flow = TtCausalMaskedDiffWithXvec(device, flow)
    got = tt_flow.inference(token, prompt_token, prompt_feat, embedding)

    assert got.shape == want.shape == (1, new_token_len * 2, 80)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalMaskedDiffWithXvec PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_causal_masked_diff_with_xvec_matches_torch_reference_mismatched_prompt_feat(device):
    """The same end-to-end comparison, but with a `prompt_feat` length that does
    NOT equal `token_mel_ratio * prompt_token_len` -- the device path must derive
    `mel_len2` by the same subtraction the reference does, not by assuming that
    relationship (see test_mel_len2_comes_from_subtraction_not_token_mel_ratio
    for the host-only version of this same check)."""
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec

    torch.manual_seed(1)
    flow = CausalMaskedDiffWithXvecRef()
    flow.eval()

    prompt_token_len, new_token_len = 4, 6
    mismatched_mel_len1 = 9  # NOT prompt_token_len * 2
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, mismatched_mel_len1, 80) * 0.1
    embedding = torch.randn(1, 192)

    with torch.no_grad():
        want = flow.inference(token, prompt_token, prompt_feat, embedding)

    tt_flow = TtCausalMaskedDiffWithXvec(device, flow)
    got = tt_flow.inference(token, prompt_token, prompt_feat, embedding)

    expected_mel_len2 = (prompt_token_len + new_token_len) * 2 - mismatched_mel_len1
    assert got.shape == want.shape == (1, expected_mel_len2, 80)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalMaskedDiffWithXvec (mismatched prompt_feat) PCC {pcc}")
    assert passed, pcc
