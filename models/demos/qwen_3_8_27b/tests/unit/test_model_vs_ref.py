# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2/M3 model test suite on the 8x4 mesh, random weights (identical on both sides).

* test_parallel_embedding_vs_ref — lookup vs F.embedding, hidden sharded on TP (vocab replicated).
  (pattern: minimax_m3/tests/unit/test_parallel_embedding_vs_ref.py; the vocab-on-SP mode is not used)
* test_lm_head_vs_ref — vocab column-sharded projection. (pattern: gemma4/tests/unit/test_lm_head.py)
* test_model_sp_vs_ref — the whole model at SP=8 x TP=4 vs the composed torch reference: sequence
  SP-sharded through every layer, both layer types, per-layer weight slicing and type dispatch, every
  layer's carried state, one-shot and 2-chunk. Full width, **reduced depth (8 = two hybrid cycles)**:
  a diagnostic of the stack; the graded full-depth run is the real-weight acceptance test (P1/P2).
  (pattern: minimax_m3/tests/unit/test_model_sp_vs_ref.py)
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec, ttnn_dtype
from models.demos.qwen_3_8_27b.reference import golden
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.embedding import TtEmbedding, TtLMHead
from models.demos.qwen_3_8_27b.tt.kv_cache import cache_capacity
from models.demos.qwen_3_8_27b.tt.model import StateDictWeights, TtQwen38Model, gather_hidden

SEED = 17
NUM_LAYERS = 8


def test_parallel_embedding_vs_ref(mesh_config):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(QWEN38.vocab_size, QWEN38.hidden_size, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, QWEN38.vocab_size, (2048,), generator=g)
    ids[:4] = torch.tensor([0, QWEN38.vocab_size - 1, 1, QWEN38.vocab_size - 2])  # table edges
    emb = TtEmbedding(mesh_config, w)
    got = from_sp(emb(emb.make_tokens(ids)), mesh_config)[0, 0]
    assert_pcc("embedding", got, F.embedding(ids, w).float())


def test_lm_head_vs_ref(mesh_config):
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(QWEN38.vocab_size, QWEN38.hidden_size, generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(1, 1, 256, QWEN38.hidden_size, generator=g).to(torch.bfloat16)
    head = TtLMHead(mesh_config, w, weight_dtype=ttnn_dtype(PrefillSpec.load().weight_dtype_default))
    logits = ttnn.to_torch(
        head(
            ttnn.from_torch(
                x,
                device=mesh_config.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_config.shard(2, None),
            )
        ),
        mesh_composer=mesh_config.compose(2, 3),
    ).float()
    assert_pcc("lm_head", logits, (x.float() @ w.float().T))


@pytest.fixture(scope="module")
def model_and_weights(mesh_config, ccl_manager):
    sd = golden.random_weight_state_dicts(QWEN38, SEED, NUM_LAYERS)
    model = TtQwen38Model(
        mesh_config,
        ccl_manager,
        QWEN38,
        PrefillSpec.load(),
        StateDictWeights(sd, f"random{SEED}"),
        num_layers=NUM_LAYERS,
        cache=None,
    )
    return model


def _ids(T):
    return torch.randint(0, QWEN38.vocab_size, (1, T), generator=torch.Generator().manual_seed(3))


def _compare_states(model, caches, want_states, T, period, tag):
    for i in range(NUM_LAYERS):
        a, b = model.read_layer_state(caches, i, n_tokens=T, period=period)
        ws = want_states[i]
        wa, wb = (ws["k"], ws["v"]) if "k" in ws else (ws["recurrent_state"], ws["conv_state"])
        names = ("k", "v") if "k" in ws else ("recurrent", "conv")
        assert_pcc(f"{tag}_layer{i}_{names[0]}", a, wa.float())
        assert_pcc(f"{tag}_layer{i}_{names[1]}", b, wb.float())


def test_model_sp_vs_ref_one_shot(mesh_config, model_and_weights):
    model = model_and_weights
    T = 5120
    ids = _ids(T)
    want, want_states = golden.stream_forward(QWEN38, ids, seed=SEED, num_layers=NUM_LAYERS, dtype=torch.bfloat16)
    caches = model.allocate_caches(cache_capacity(T, [T]))
    out = model.forward(model.embedding.make_tokens(ids[0]), PrefillCtx(caches, 0, 0, T, T))
    assert_pcc("model_8L_one_shot_hidden", gather_hidden(out, mesh_config), want.float())
    _compare_states(model, caches, want_states, T, T, "model_8L_one_shot")
    caches.reset_gdn()


def test_model_sp_vs_ref_chunked(mesh_config, model_and_weights):
    model = model_and_weights
    chunk, T = 2560, 5120
    ids = _ids(T)
    want, want_states = golden.stream_forward(QWEN38, ids, seed=SEED, num_layers=NUM_LAYERS, dtype=torch.bfloat16)
    caches = model.allocate_caches(cache_capacity(T, [chunk]))
    outs = []
    for c in range(T // chunk):
        s = c * chunk
        o = model.forward(
            model.embedding.make_tokens(ids[0, s : s + chunk]), PrefillCtx(caches, 0, s, s + chunk, chunk)
        )
        outs.append(gather_hidden(o, mesh_config))
    assert_pcc("model_8L_chunked_hidden_chunk1", outs[1], want[:, chunk:].float())
    _compare_states(model, caches, want_states, T, chunk, "model_8L_chunked")
    caches.reset_gdn()
