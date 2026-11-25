# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the model terminal: embedding -> hc streams, hyper-head + final norm, lm_head.

Graded against transformers' ``Glm5NextTextModel`` (``embed_tokens``, ``hc_head``, ``norm``) and the untied
``lm_head``, on a small config. The last test runs a two-layer model (one ``kda_dense`` and one ``kda_moe`` block)
token by token on the device and compares the logits with one HF forward over the whole sequence.

The upstream ``tt/model.py::Terminal`` is covered for its plain-ttnn half (``embed_streams``, ``hidden``); its
lm_head is ``optimized_decoder.DSLinear``, which needs ``ttnn...matmul.DramReaderPlacement`` (absent from this build).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tests.test_kda_block import (
    EXPERT_DTYPE,
    HC,
    assert_bf4_experts,
    bf4_round_experts,
    bf16_round,
    init_like_checkpoint,
    pcc,
    small_config,
    to_dev,
    to_host,
)
from models.experimental.glm53_flash.tt.ds_v4.streams import _strip_prefix
from models.experimental.glm53_flash.tt.kda_block.decoder_layer import GLMKDADecoderLayer
from models.experimental.glm53_flash.tt.kda_block.terminal import GLMEmbedding, GLMHead

transformers_glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

VOCAB = 1024


def terminal_config():
    cfg = small_config()
    cfg.vocab_size = VOCAB
    cfg.pad_token_id = None
    return cfg


@pytest.fixture(scope="module")
def cfg():
    return terminal_config()


@pytest.fixture(scope="module")
def hf_model(cfg):
    """(HF text model, lm_head) with checkpoint-like weights."""
    torch.manual_seed(0)
    model = bf4_round_experts(init_like_checkpoint(transformers_glm.Glm5NextTextModel(cfg), seed=3))
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    lm_head.weight.data.copy_(bf16_round(torch.randn(cfg.vocab_size, cfg.hidden_size) / cfg.hidden_size**0.5))
    return model, lm_head.float().eval()


def head_weights(model, lm_head) -> dict:
    return {"norm.weight": model.norm.weight, "lm_head.weight": lm_head.weight}


@pytest.mark.parametrize("b, s", [(1, 1), (4, 1), (1, 32)], ids=["decode1", "decode4", "prefill32"])
def test_embedding_matches_hf(device, cfg, hf_model, b, s):
    """``embed_tokens`` + the ``hc`` identical streams, exactly (a bf16 table lookup)."""
    model, _ = hf_model
    ids = torch.randint(0, VOCAB, (b, s))
    emb = GLMEmbedding(cfg, model.embed_tokens.weight, device)
    got = to_host(emb(emb.token_ids(ids)))
    with torch.no_grad():
        ref = model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, HC, -1)
    assert tuple(got.shape) == (b, s, HC, cfg.hidden_size)
    assert torch.equal(got, bf16_round(ref))


@pytest.mark.parametrize("b, s", [(1, 1), (4, 1), (1, 32)], ids=["decode1", "decode4", "prefill32"])
def test_head_hidden_matches_hf(device, cfg, hf_model, b, s):
    """``norm(hc_head(streams))``: the 4-stream mean then the final RMSNorm (``last_hidden_state``)."""
    model, lm_head = hf_model
    streams = bf16_round(torch.randn(b, s, HC, cfg.hidden_size) * 2)
    head = GLMHead(cfg, head_weights(model, lm_head), device)
    got = to_host(head.hidden(to_dev(streams, device))).reshape(b, s, -1)
    with torch.no_grad():
        ref = model.norm(model.hc_head(streams))
    assert pcc(got, ref) > 0.9999, f"PCC {pcc(got, ref):.6f}"


@pytest.mark.parametrize("b, s", [(1, 1), (4, 1), (1, 32)], ids=["decode1", "decode4", "prefill32"])
def test_head_logits_match_hf(device, cfg, hf_model, b, s):
    model, lm_head = hf_model
    streams = bf16_round(torch.randn(b, s, HC, cfg.hidden_size) * 2)
    head = GLMHead(cfg, head_weights(model, lm_head), device)
    got = to_host(head(to_dev(streams, device))).reshape(b * s, VOCAB)
    with torch.no_grad():
        ref = lm_head(model.norm(model.hc_head(streams))).reshape(b * s, VOCAB)
    assert pcc(got, ref) > 0.999, f"PCC {pcc(got, ref):.6f}"
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.9, f"top-1 agreement {agree:.2f}"


# ---------------------------------------------------------------------------------------------
# Upstream tt/model.py::Terminal (packed [1, 1, R, 4 * 4096] fp32 streams, full hidden size)
# ---------------------------------------------------------------------------------------------

FULL_HIDDEN = 4096


def _upstream_terminal(device, embed: torch.Tensor, norm: torch.Tensor):
    """`Terminal` with its plain-ttnn members only (its __init__ also builds the DSLinear lm_head)."""
    from models.experimental.glm53_flash.tt.model import Terminal
    from models.experimental.glm53_flash.tt.precision import hifi4_fp32

    t = Terminal.__new__(Terminal)
    t.mesh = device
    t.eps = 1e-5
    t.embed_w = ttnn.from_torch(embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t.norm_w = ttnn.from_torch(
        norm.reshape(1, 1, 1, FULL_HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    t.ckc_norm = hifi4_fp32(device)
    return t


@pytest.mark.parametrize("rows", [1, 32])
def test_upstream_terminal_embed_streams(device, rows):
    torch.manual_seed(30)
    embed = bf16_round(torch.randn(VOCAB, FULL_HIDDEN))
    term = _upstream_terminal(device, embed, torch.ones(FULL_HIDDEN))
    ids = torch.randint(0, VOCAB, (1, 32))
    tokens = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got = to_host(term.embed_streams(tokens, rows)).reshape(rows, HC * FULL_HIDDEN)
    ref = embed[ids[0, :rows]].repeat(1, HC)
    assert torch.equal(got, ref)


def test_upstream_terminal_hidden_matches_hf(device):
    """`Terminal.hidden`: mean of the 4 packed streams -> RMSNorm, against HF hc_head + norm."""
    torch.manual_seed(31)
    rows = 32
    norm_w = bf16_round(1 + 0.1 * torch.randn(FULL_HIDDEN))
    term = _upstream_terminal(device, torch.zeros(32, FULL_HIDDEN), norm_w)
    streams = torch.randn(rows, HC, FULL_HIDDEN)
    packed = to_dev(streams.reshape(1, 1, rows, HC * FULL_HIDDEN), device, ttnn.float32)
    got = to_host(term.hidden(packed)).reshape(rows, FULL_HIDDEN)
    ref_norm = transformers_glm.Glm5NextTextRMSNorm(FULL_HIDDEN, eps=1e-5)
    ref_norm.weight.data.copy_(norm_w)
    with torch.no_grad():
        ref = ref_norm(transformers_glm.Glm5NextTextHyperHead()(streams.unsqueeze(0)))[0]
    assert pcc(got, ref) > 0.9999, f"PCC {pcc(got, ref):.6f}"


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="Terminal's lm_head is optimized_decoder.DSLinear, which needs ttnn...matmul.DramReaderPlacement "
    "(not in this build); GLMHead is the Wormhole lm_head",
)
def test_upstream_terminal_lm_head_builds():
    from models.experimental.glm53_flash.tt.optimized_decoder import DSLinear  # noqa: F401


# ---------------------------------------------------------------------------------------------
# A two-layer model, decoded token by token
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("users", [1, 2])
def test_two_layer_model_decode_matches_hf(device, cfg, hf_model, users):
    """embedding -> kda_dense block -> kda_moe block -> head, one token per step for each user, against one HF
    ``Glm5NextTextModel`` forward over each user's whole sequence: logits PCC and greedy-token agreement per step."""
    model, lm_head = hf_model
    sd = model.state_dict()
    emb = GLMEmbedding(cfg, sd["embed_tokens.weight"], device)
    blocks = [
        GLMKDADecoderLayer(cfg, i, _strip_prefix(sd, f"layers.{i}"), device, expert_dtype=EXPERT_DTYPE)
        for i in range(cfg.num_hidden_layers)
    ]
    for i, blk in enumerate(blocks):
        if cfg.mlp_layer_types[i] == "sparse":
            assert_bf4_experts(blk.mlp.moe)
    head = GLMHead(cfg, head_weights(model, lm_head), device)
    states = [blk.allocate_state(users) for blk in blocks]

    torch.manual_seed(32 + users)
    steps = 5
    ids = torch.randint(0, VOCAB, (users, steps))
    with torch.no_grad():
        # No cache: DynamicCache.get_seq_length raises when every layer is linear attention.
        ref = lm_head(model(input_ids=ids, use_cache=False).last_hidden_state)  # [users, steps, V]

    for t in range(steps):
        x = emb(emb.token_ids(ids[:, t : t + 1]))
        for blk, st in zip(blocks, states):
            x = blk.decode(x, st)
        got = to_host(head(x)).reshape(users, VOCAB)
        for u in range(users):
            r = ref[u, t]
            assert pcc(got[u], r) > 0.99, f"user {u} step {t}: logits PCC {pcc(got[u], r):.5f}"
            assert got[u].argmax() == r.argmax() or r.topk(2).values.diff().abs() < 0.05, f"user {u} step {t}: top-1"
