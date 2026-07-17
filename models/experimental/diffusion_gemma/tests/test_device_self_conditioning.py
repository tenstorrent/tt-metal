# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC test for the ttnn self-conditioning gated MLP (#47461/#47463).

Validates the device module ``post_norm(inputs_embeds + down(gelu_tanh(gate(
pre_norm(signal))) * up(pre_norm(signal))))`` against the pure-torch reference
oracle (`reference/self_conditioning.py`). Checkpoint-free: random weights are
generated in the reference module and loaded verbatim into the device module, so
this isolates the module's compute (RMSNorm conventions + GeGLU) from weight
loading (which `tests/test_weight_mapping.py` already validates against the real
checkpoint).

Run on QB2:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_self_conditioning.py
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.tt.self_conditioning import TtSelfConditioning
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
    ),
    # One device open/teardown for the whole module — avoid the QB2 erisc-29-25
    # teardown hang from repeated CreateDevice (see test_device_bidirectional_sdpa).
    pytest.mark.use_module_device,
]

# 26B-A4B dims: hidden 2816, self-conditioning intermediate = dense intermediate 2112 (NOT moe 704).
HIDDEN, INTER, EPS = 2816, 2112, 1e-6


def _build(seed):
    """Reference module (random weights) + a device module loaded from its weights."""
    torch.manual_seed(seed)
    ref = SelfConditioning(HIDDEN, INTER, eps=EPS, activation="gelu_pytorch_tanh").eval()
    state = {
        "pre_norm.weight": ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": ref.gate_proj.weight.data.clone(),
        "up_proj.weight": ref.up_proj.weight.data.clone(),
        "down_proj.weight": ref.down_proj.weight.data.clone(),
    }
    return ref, state


def _to_dev(t, device):
    return ttnn.from_torch(t.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # [1,B,L,H]


@pytest.mark.parametrize("seq_len", [256])
def test_self_conditioning_pcc(device, seq_len):
    ref, state = _build(0)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    signal = torch.randn(1, seq_len, HIDDEN)

    with torch.no_grad():
        golden = ref(emb, signal)  # [1, L, H]

    out = ttnn.to_torch(tt.forward(_to_dev(emb, device), _to_dev(signal, device)))[0]  # [1,B,L,H] -> [B,L,H]
    assert_with_pcc(golden, out, 0.99)


def _embed_to_dev(embed_w, device):
    """Tied embedding table [vocab, hidden] -> ttnn [1,1,vocab,hidden] TILE (matmul operand)."""
    return ttnn.from_torch(
        embed_w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("vocab", [256])
def test_condition_full_path_pcc(device, seq_len, vocab):
    """Full self-conditioning: soft-embed prev logits (softmax @ embed) THEN the
    gated MLP — the production decoder path, vs the reference condition()."""
    ref, state = _build(2)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    prev_logits = torch.randn(1, seq_len, vocab)
    embed_w = torch.randn(vocab, HIDDEN)

    with torch.no_grad():
        golden = ref.condition(emb, prev_logits, embed_w, enabled=True)  # [1, L, H]

    out = ttnn.to_torch(
        tt.condition(_to_dev(emb, device), _to_dev(prev_logits, device), _embed_to_dev(embed_w, device))
    )[0]
    assert_with_pcc(golden, out, 0.99)


@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("vocab", [256])
def test_condition_none_logits_is_post_norm(device, seq_len, vocab):
    """prev_logits=None (first step / encoder) -> post_norm(inputs_embeds), device == ref."""
    ref, state = _build(3)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    embed_w = torch.randn(vocab, HIDDEN)
    with torch.no_grad():
        golden = ref.condition(emb, None, embed_w, enabled=False)  # == post_norm(emb)

    out = ttnn.to_torch(tt.condition(_to_dev(emb, device), None, _embed_to_dev(embed_w, device)))[0]
    assert_with_pcc(golden, out, 0.99)


@pytest.mark.parametrize("seq_len", [256])
def test_zero_signal_is_post_norm_of_embeds(device, seq_len):
    """Zero signal -> post_norm(inputs_embeds), NOT inputs_embeds unchanged."""
    ref, state = _build(1)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    signal = torch.zeros(1, seq_len, HIDDEN)

    with torch.no_grad():
        golden = ref(emb, signal)  # == ref.post_norm(emb)

    out = ttnn.to_torch(tt.forward(_to_dev(emb, device), _to_dev(signal, device)))[0]
    assert_with_pcc(golden, out, 0.99)
    # sanity: the module rescales the embeddings (post_norm), so it is not identity
    assert not torch.allclose(out.float(), emb, atol=1e-3)
