# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of reference/mtp.py (pure torch, no device).

HF's hy_v4 code does not implement the MTP layer, so there is no module to compare against; what can be pinned:

* on a small random layer: shapes, finiteness, and that the three ways the layer is run are one computation (a first
  pass in one piece or continued from a cache; a draft step against a first pass over the same rows; a chain of two
  draft steps against a first pass);
* the parameter map covers exactly the checkpoint's tensors of ``model.mtp_layers.0`` (needs $HY4_HF_MODEL);
* with the real weights, on real text (needs $HY4_HF_MODEL, about 60 GB of free memory and the L78 decode goldens):
  the layer's draft of the token two positions on equals what HF's backbone generated far above chance, when it is
  fed the final normed hidden as vLLM does, and never when fed the hidden before the final norm.

``HY4_MTP_SEQ`` may name a file of a recorded sequence (``hidden`` [S, hidden] the final normed hidden of every
position, ``input_ids`` [S + 1], ``prompt_len``); the last test then drafts 3 tokens after every generated position
with the layer's cache holding the whole sequence, and prints the acceptance.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference import mtp as M
from models.experimental.hy4_preview.reference.attention import AttentionConfig
from models.experimental.hy4_preview.reference.decoder import DecoderConfig
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.reference.indexer import IndexerConfig
from models.experimental.hy4_preview.reference.moe import MoEConfig

HIDDEN, VOCAB = 96, 211


def _small():
    cfg = M.MTPConfig(
        decoder=DecoderConfig(
            attention=AttentionConfig(
                hidden_size=HIDDEN,
                num_heads=4,
                q_lora_rank=48,
                kv_lora_rank=24,
                qk_nope_head_dim=12,
                qk_rope_head_dim=8,
                v_head_dim=16,
            ),
            moe=MoEConfig(hidden_size=HIDDEN, num_experts=16, top_k=4, intermediate_size=40),
            ihc=IHCConfig(dim=HIDDEN),
            index_topk=64,
        ),
        indexer=IndexerConfig(
            hidden_size=HIDDEN, q_lora_rank=48, num_heads=2, head_dim=16, qk_rope_head_dim=8, topk=64
        ),
    )
    torch.manual_seed(7)
    layer = M.Hy4MTP(cfg).eval()
    with torch.no_grad():
        for p in (layer.enorm, layer.hnorm, layer.final_layernorm, layer.input_layernorm):
            p.add_(0.1 * torch.randn_like(p))
        layer.self_attn.learnable_sink_param.add_(torch.randn(4))
        layer.mlp.router_bias.add_(0.05 * torch.randn(16))
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "linear_gate"):
            getattr(layer.self_attn, name).mul_(20)  # attention that is not uniform
    embedding = torch.randn(VOCAB, HIDDEN)
    lm_head = torch.randn(VOCAB, HIDDEN)
    return layer, embedding, lm_head


def _sequence(S, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(S, HIDDEN, generator=g), torch.randint(0, VOCAB, (S,), generator=g)


def test_parameter_map_is_the_27_tensors():
    assert len(M.MTP_PARAMETERS) == 27
    layer, _, _ = _small()
    for name, path in M.MTP_PARAMETERS.items():
        layer.get_parameter(path)  # raises if the path is wrong
    root = os.environ.get(hf_harness.HY4_HF_MODEL_ENV)
    if not root:
        pytest.skip(f"{hf_harness.HY4_HF_MODEL_ENV} is not set: the checkpoint's own list is not compared")
    weight_map = json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]
    in_checkpoint = sorted(k for k in weight_map if k.startswith("model.mtp_layers."))
    assert in_checkpoint == sorted(M.checkpoint_names(0))


def test_first_pass_shapes_and_continuation():
    layer, embedding, _ = _small()
    hidden, tokens = _sequence(20)
    with torch.no_grad():
        rec = {}
        out, cache = layer.fill(hidden, embedding[tokens], record=rec)
        assert out.shape == (20, HIDDEN) and torch.isfinite(out).all()
        assert cache.latent.shape == (20, 24 + 8) and rec["router_indices"].shape == (20, 4)
        # the same rows in two pieces: 12 rows, then 8 continuing the cache
        head, cache_a = layer.fill(hidden[:12], embedding[tokens[:12]])
        tail, cache_b = layer.fill(hidden[12:], embedding[tokens[12:]], cache=cache_a)
    assert torch.allclose(torch.cat([head, tail]), out, atol=1e-5)
    assert torch.allclose(cache_b.latent, cache.latent, atol=1e-6)
    # a row's output does not depend on later rows (causal)
    with torch.no_grad():
        again = layer.fill(hidden[:12], embedding[tokens[:12]])[0]
    assert torch.allclose(again, out[:12], atol=1e-5)


def test_embedding_is_zeroed_at_position_zero():
    layer, embedding, _ = _small()
    hidden, tokens = _sequence(4)
    other = (tokens + 1) % VOCAB
    with torch.no_grad():
        a = layer.fill(hidden, embedding[tokens])[0]
        b = layer.fill(hidden, embedding[torch.cat([other[:1], tokens[1:]])])[0]
        c = layer.fill(hidden, embedding[torch.cat([tokens[:1], other[1:2], tokens[2:]])])[0]
    assert torch.equal(a, b), "the token after position 0 must not matter"
    assert not torch.allclose(a[1:], c[1:], atol=1e-4), "the token after position 1 must"


def test_draft_steps_are_first_pass_rows():
    """A draft step is the layer's ordinary row with another hidden: step j of an anchor equals the last row of a
    first pass over the anchor's prefix followed by its draft rows."""
    layer, embedding, lm_head = _small()
    S, depth = 14, 3
    hidden, tokens = _sequence(S)
    anchors = torch.tensor([3, 9, 13])
    drafts, outputs = M.draft(layer, embedding, lm_head, hidden, tokens, depth=depth, anchors=anchors)
    assert drafts.shape == (3, depth) and outputs.shape == (depth, 3, HIDDEN) and torch.isfinite(outputs).all()
    with torch.no_grad():
        for a, p in enumerate(anchors.tolist()):
            h, t = hidden[: p + 1], tokens[: p + 1]
            for j in range(depth):
                out = layer.fill(h, embedding[t])[0]
                assert torch.allclose(out[-1], outputs[j, a], atol=2e-5), (p, j)
                assert int(M.greedy(out[-1:], lm_head)) == int(drafts[a, j])
                h = torch.cat([h, out[-1:]])  # the next draft row is conditioned on this output
                t = torch.cat([t, drafts[a, j : j + 1]])


def test_accepted_lengths():
    tokens = torch.arange(100, 120)
    anchors = torch.tensor([0, 5, 10])
    drafts = torch.tensor([[102, 103, 104], [107, 0, 109], [0, 113, 114]])
    assert M.accepted_lengths(drafts, tokens, anchors).tolist() == [3, 1, 0]


def test_context_longer_than_topk_is_refused():
    layer, embedding, _ = _small()
    hidden, tokens = _sequence(65)
    with pytest.raises(AssertionError, match="index_topk"):  # allow-pytest.raises: a host assertion, no device
        layer.fill(hidden, embedding[tokens])


# ---------------------------------------------------------------------------------------------
# the real weights
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    if not os.environ.get(hf_harness.HY4_HF_MODEL_ENV):
        pytest.skip(f"{hf_harness.HY4_HF_MODEL_ENV} is not set")
    available_gb = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) / 2**20
    if available_gb < 60:
        pytest.skip(f"only {available_gb:.0f} GB of host memory available (the layer is 39 GB in fp32)")
    layer = M.Hy4MTP.from_checkpoint()
    embedding, lm_head = hf_harness.checkpoint_tensors(["model.embed_tokens.weight", "lm_head.weight"], None, None)
    return layer, embedding, lm_head


@pytest.mark.parametrize("prompt_len", (31, 33))
def test_draft_one_on_the_goldens(real, prompt_len):
    """A decode golden holds the final hidden of its decode rows only, so the layer's cache lacks the prompt here:
    a lower bound of what it does with the whole sequence. Measured: 29 of 31 and 26 of 31 rows."""
    layer, embedding, lm_head = real
    golden = hf_harness.find_decode_golden(78, prompt_len)
    if golden is None:
        pytest.skip(f"no 78-layer decode golden of a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    N, steps, ids = golden["meta"]["prompt_len"], golden["meta"]["steps"], golden["input_ids"]
    positions = torch.arange(N - 1, N - 1 + steps)
    committed = ids[positions[:-1] + 2]  # what HF generated two positions after each row
    hits = {}
    for kind in ("hidden", "hc_head"):
        with torch.no_grad():
            out, cache = layer.fill(golden[kind].float(), embedding[ids[positions + 1]].float(), positions)
        assert out.shape == (steps, 6144) and torch.isfinite(out).all()
        assert cache.latent.shape == (steps, 576)
        hits[kind] = int((M.greedy(out, lm_head)[:-1] == committed).sum())
    print(f"P{prompt_len}: draft 1 == HF's token on {hits['hidden']} of {steps - 1} rows (final normed hidden), "
          f"{hits['hc_head']} (the hidden before the final norm)")  # fmt: skip
    assert hits["hidden"] >= 0.6 * (steps - 1), hits  # chance is 1 in 120832
    assert hits["hc_head"] <= 0.1 * (steps - 1), hits


def test_draft_three_on_a_recorded_sequence(real):
    path = os.environ.get("HY4_MTP_SEQ")
    if not path or not Path(path).is_file():
        pytest.skip("HY4_MTP_SEQ does not name a recorded sequence")
    layer, embedding, lm_head = real
    seq = torch.load(path, weights_only=False)
    ids, N = seq["input_ids"], seq["prompt_len"]
    S = seq["hidden"].shape[0]
    anchors = torch.arange(N - 1, S - 3)  # every position whose next 4 tokens are known
    drafts, outputs = M.draft(layer, embedding, lm_head, seq["hidden"].float(), ids[1 : S + 1], 3, anchors)
    assert torch.isfinite(outputs).all()
    accepted = M.accepted_lengths(drafts, ids, anchors)
    by_depth = [float((accepted > j).float().mean()) for j in range(3)]
    print(f"{anchors.numel()} anchors: accepted >= 1, 2, 3 on {by_depth}; mean accepted {accepted.float().mean():.3f}")
    assert by_depth[0] >= 0.3
