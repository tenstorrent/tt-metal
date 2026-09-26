# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Real CosyVoice2-0.5B checkpoint weights (`flow.pt`, from
`FunAudioLLM/CosyVoice2-0.5B`) loaded into
`CausalMaskedDiffWithXvecRef`/`TtCausalMaskedDiffWithXvec`, checked with the
SAME TT-vs-torch-with-shared-weights isolation `test_flow.py`'s existing
device tests already use -- just with the random init those tests use
replaced by real trained weights. See `tt/checkpoint.py` for the download,
and each submodule's own `from_checkpoint` classmethod
(`UpsampleConformerEncoderRef.from_checkpoint`,
`CausalConditionalDecoderRef.from_checkpoint`,
`CausalMaskedDiffWithXvecRef.from_checkpoint`) for the exact key mapping.

Third module in this bring-up's checkpoint-loading order (HiFT vocoder, F0
predictor, done -- see test_hift_checkpoint.py/test_f0_predictor_checkpoint.py
-- then flow decoder, here, then LLM backbone).

**The real, non-obvious finding here was architectural, not numerical**:
`flow.pt`'s `decoder.estimator.*` keys (910 of them) did not match
`CausalConditionalDecoderRef`'s own attribute names directly, and at first
glance that looked like a missing-architecture gap (a `down_blocks`/
`up_blocks`/`mid_blocks` UNet-style container with nested attention blocks
this package's decoder didn't appear to have). Traced fully against the real
checkpoint's own tensor shapes before concluding anything: it is a pure
container/naming difference, not a missing feature -- real upstream wraps the
exact same resnet/transformer-block/conv pieces this package already builds
(`down_resnet`/`down_tbs`/`down_conv`, etc.) inside `nn.ModuleList`s, and
every real tensor shape at every real index matches this package's own
modules exactly once unwrapped. See `CausalConditionalDecoderRef.from_checkpoint`
for the exact remapping and the reasoning. `encoder`'s only structural
difference (`LinearNoSubsampling`'s `out.0`/`out.1` Sequential vs. this
package's flat `linear`/`norm`) is much smaller -- see
`UpsampleConformerEncoderRef.from_checkpoint`.

**Checked bf16 before assuming either way** (per this bring-up's discipline
-- `test_hift_checkpoint.py` needed fp32, `test_f0_predictor_checkpoint.py`
didn't; neither result transfers automatically). Measured directly: bf16 PCC
is 0.999+ despite this being the most numerically involved module so far (a
10-step Euler ODE solve through the CFM estimator, not a single forward
pass) -- real weights do not destabilize this module's bf16 precision the way
they did HiFT's `conv_post`/`exp()`.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


@needs_l1_small
def test_device_causal_masked_diff_with_xvec_matches_torch_reference_real_checkpoint(device):
    """`TtCausalMaskedDiffWithXvec.inference` vs. the torch reference, both
    built from the SAME real `flow.pt` weights (not random init) -- the
    real-weight counterpart of `test_flow.py`'s
    `test_device_causal_masked_diff_with_xvec_matches_torch_reference`. Random
    synthetic speech tokens/prompt mel/xvec, same as that test (a real
    upstream LLM/prompt pipeline is a separate, later concern)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec

    flow_sd = load_checkpoint_file("flow.pt")
    ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
    ref.eval()

    torch.manual_seed(5)
    prompt_token_len, new_token_len = 4, 6
    prompt_token = torch.randint(0, 6561, (1, prompt_token_len))
    token = torch.randint(0, 6561, (1, new_token_len))
    prompt_feat = torch.randn(1, prompt_token_len * 2, 80) * 0.1
    embedding = torch.randn(1, 192)

    with torch.no_grad():
        want = ref.inference(token, prompt_token, prompt_feat, embedding)

    tt_flow = TtCausalMaskedDiffWithXvec(device, ref, dtype=ttnn.bfloat16)
    got = tt_flow.inference(token, prompt_token, prompt_feat, embedding)

    assert got.shape == want.shape == (1, new_token_len * 2, 80)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  real-checkpoint device CausalMaskedDiffWithXvec PCC {pcc}")
    assert passed, pcc


def test_estimator_down_blocks_container_shapes_match_real_checkpoint():
    """Direct shape check against the real checkpoint's own tensors for the
    piece that looked, before investigation, like it might be a missing
    architecture (see module docstring): `down_blocks.0.0` (the resnet) has
    `block1` reading `in_channels=320`, matching this package's own
    `down_resnet = CausalResnetBlock1DRef(in_channels, channels, ...)`."""
    from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

    flow_sd = load_checkpoint_file("flow.pt")
    est_sub = sub_state_dict(flow_sd, "decoder.estimator.")
    assert est_sub["down_blocks.0.0.block1.block.0.weight"].shape == (256, 320, 3)
    assert est_sub["down_blocks.0.1.0.attn1.to_q.weight"].shape == (512, 256)
    assert est_sub["up_blocks.0.0.block1.block.0.weight"].shape == (256, 512, 3)
    # 12 mid stages, each with a resnet (index 0) and 4 transformer blocks (index 1.0-1.3)
    for i in range(12):
        assert f"mid_blocks.{i}.0.res_conv.weight" in est_sub
        for j in range(4):
            assert f"mid_blocks.{i}.1.{j}.attn1.to_q.weight" in est_sub
