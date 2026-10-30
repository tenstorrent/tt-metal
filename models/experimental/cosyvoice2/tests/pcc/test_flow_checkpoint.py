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
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file
    from models.experimental.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec

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
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

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


# Streaming encoder, final chunk under bucketing (real weights). Gates over the affected region (the
# last partial up-rate chunk): PCC >= ENCODER_REGION_PCC and max|diff| <= ENCODER_REGION_MAX_ABS.
# Measured 2026-09-27 at valid 70/110/137/263: bucketed vs TT exact 0.016-0.055, bucketed vs the torch
# chunk-causal reference 0.039-0.067 (region PCC >= 0.99984); the chunk-only control 0.23-0.39 (region
# PCC 0.988-0.996, while its whole-output PCC stayed 0.998-0.9998 -- whole-output PCC misses it).
ENCODER_REGION_PCC = 0.999
ENCODER_REGION_MAX_ABS = 0.125


@pytest.fixture(scope="module")
def real_flow_encoder():
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.experimental.cosyvoice2.tt.flow.encoder import UpsampleConformerEncoderRef

    flow_sd = load_checkpoint_file("flow.pt")
    ref = UpsampleConformerEncoderRef.from_checkpoint(sub_state_dict(flow_sd, "encoder."))
    ref.eval()
    return ref, flow_sd["input_embedding.weight"].float()


@needs_l1_small
@pytest.mark.parametrize("valid", [70, 137, 263])  # none a multiple of CHUNK_SIZE=25
def test_device_streaming_encoder_final_chunk_real_checkpoint(device, monkeypatch, real_flow_encoder, valid):
    """Real `flow.pt` encoder, real token embeddings, a final (non-chunk-aligned) streaming call
    under bucketing, with large random values in the padded rows. Over the last partial chunk:
      * TT bucketed must match the torch chunk-causal reference at the EXACT length, and TT
        exact-length (the key-padding term hides the padding);
      * negative control: the same bucketed run with a chunk-only mask (no key-padding term) must
        fail that gate against TT exact-length."""
    import models.experimental.cosyvoice2.tt.flow.encoder as encoder_module
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.encoder import (
        CHUNK_SIZE,
        CHUNK_SIZE_UP,
        TtUpsampleConformerEncoder,
        bucket_length,
        chunk_causal_bias_torch,
    )

    ref, input_embedding = real_flow_encoder
    assert valid % CHUNK_SIZE != 0
    torch.manual_seed(valid)
    xs = input_embedding[torch.randint(0, input_embedding.shape[0], (1, valid))]
    bucket = bucket_length(valid)
    tt_enc = TtUpsampleConformerEncoder(device, ref)
    monkeypatch.setattr(
        tt_enc,
        "_bucket_padding",
        lambda b, rows: ttnn.from_torch(
            torch.randn(b, rows, xs.shape[-1]) * 100.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
    )

    def run(t_len, content, **kw):
        x = ttnn.from_torch(content, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.to_torch(tt_enc(x, t_len, 1, streaming=True, **kw)).float()

    padded = torch.cat([xs, torch.zeros(1, bucket - valid, xs.shape[-1])], dim=1)
    exact = run(valid, xs)
    bucketed = run(bucket, padded, valid_length=valid)
    with monkeypatch.context() as m:
        m.setattr(
            encoder_module,
            "streaming_attn_bias_torch",
            lambda size, v, chunk_size, neg=-30000.0: chunk_causal_bias_torch(size, chunk_size, neg),
        )
        chunk_only = run(bucket, padded, valid_length=valid)
    with torch.no_grad():
        want = ref(xs, streaming=True)

    v2 = 2 * valid
    lo = (v2 // CHUNK_SIZE_UP) * CHUNK_SIZE_UP  # start of the last (partial) up-rate chunk

    def region(a, b):
        wa, wb = a[:, lo:v2], b[:, lo:v2]
        return float(comp_pcc(wa, wb, GATE_BF16)[1]), (wa - wb).abs().max().item()

    def ok(stats):
        return stats[0] >= ENCODER_REGION_PCC and stats[1] <= ENCODER_REGION_MAX_ABS

    vs_ref, vs_exact, control = region(want, bucketed), region(exact, bucketed), region(exact, chunk_only)
    print(
        f"\n  valid={valid} bucket={bucket} last chunk [{lo},{v2}) PCC / max|diff|: bucketed vs torch "
        f"{vs_ref[0]:.6f} / {vs_ref[1]:.4g}; bucketed vs TT exact {vs_exact[0]:.6f} / {vs_exact[1]:.4g}; "
        f"chunk-only control vs TT exact {control[0]:.6f} / {control[1]:.4g}"
    )
    assert ok(vs_ref), vs_ref
    assert ok(vs_exact), vs_exact
    assert not ok(control), f"negative control passed the gate: {control}"


# Non-streaming bucketing (tt/flow/flow.py `bucket_tokens`), real weights, bucketed vs exact-length over the valid
# frames. Measured 2026-09-27:
# - a real corpus prompt (175 tokens): max |diff| 0.150 / 0.177, PCC 0.99985 / 0.99981 at 69 / 64 padded positions,
#   and 0.0 at 5;
# - this test's synthetic inputs: max |diff| 0.254 / 0.229, PCC 0.999696 / 0.999665.
# The naive control (the padding run as content, no masks): max |diff| 2.4-3.9, PCC 0.955-0.983. The gate sits
# between, with at least 1.6x headroom on the bucketed side and 6x on the control's.
BUCKETED_MAX_ABS, BUCKETED_PCC = 0.4, 0.9995


@needs_l1_small
@pytest.mark.parametrize("n_prompt, n_gen, bucket", [(96, 60, 192), (160, 150, 384)])
def test_device_nonstreaming_bucketed_flow_matches_exact_real_checkpoint(device, n_prompt, n_gen, bucket):
    """`inference(..., bucket_tokens=B)` against the exact-length call, real `flow.pt`, bf16 (the pipeline's), over
    the valid frames. The negative control runs the same padded tokens with no masks at all (the padding treated as
    content) and must fail the same gate: the masks are what make bucketing exact enough."""
    import ttnn
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file
    from models.experimental.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec

    ref = CausalMaskedDiffWithXvecRef.from_checkpoint(load_checkpoint_file("flow.pt"))
    ref.eval()
    g = torch.Generator().manual_seed(n_prompt + n_gen)
    prompt_token = torch.randint(0, 6561, (1, n_prompt), generator=g)
    token = torch.randint(0, 6561, (1, n_gen), generator=g)
    # a speech-like log-mel prompt: around -6, floored at CosyVoice2's log(1e-5)
    prompt_feat = (torch.randn(1, 2 * n_prompt, 80, generator=g) * 2.0 - 6.0).clamp(min=-11.5129)
    embedding = torch.randn(1, 192, generator=g)

    flow = TtCausalMaskedDiffWithXvec(device, ref, dtype=ttnn.bfloat16)
    exact = flow.inference(token, prompt_token, prompt_feat, embedding)
    bucketed = flow.inference(token, prompt_token, prompt_feat, embedding, bucket_tokens=bucket)
    padded = torch.nn.functional.pad(token, (0, bucket - n_prompt - n_gen))
    naive = flow.inference(padded, prompt_token, prompt_feat, embedding)[:, : 2 * n_gen]

    assert bucketed.shape == exact.shape == (1, 2 * n_gen, 80)
    got = {}
    for name, out in (("bucketed", bucketed), ("naive control", naive)):
        max_abs = (out - exact).abs().max().item()
        pcc = float(comp_pcc(exact, out, BUCKETED_PCC)[1])
        got[name] = (max_abs <= BUCKETED_MAX_ABS and pcc >= BUCKETED_PCC, max_abs, pcc)
        print(
            f"\n  {n_prompt}+{n_gen} tokens in a {bucket} bucket, {name} vs exact: max|diff| {max_abs:.4f}, PCC {pcc:.6f}"
        )
    assert got["bucketed"][0], got["bucketed"]
    assert not got["naive control"][0], f"negative control passed the gate: {got['naive control']}"
