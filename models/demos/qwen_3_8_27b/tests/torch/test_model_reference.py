# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 (host only): the whole-model reference.

* test_full_depth_reference_vs_inline_golden — all 64 layers (the real hybrid schedule), reduced
  width, random weights: TextModel vs the independent inline golden. Reduced width => diagnostic.
  (pattern: minimax_m3/tests/unit/test_reference_model.py, widened to the full model)
* test_stream_forward_matches_model — the layer-streaming runner (used for goldens) == TextModel.
* test_golden_cache_roundtrip — no repo pattern; authored: a second run loads instead of recomputing,
  and a changed ReferenceCacheKey field is a miss, not a stale hit.
"""

import torch

from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.reference import golden
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.tests.torch.test_reference_model import g_model, pcc

WIDE_SHALLOW = dict(
    hidden_size=256,
    intermediate_size=512,
    vocab_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    mrope_section=(3, 3, 2),
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
)
FULL_DEPTH = QWEN38.reduced(**WIDE_SHALLOW)  # 64 layers, reduced width


def test_full_depth_reference_vs_inline_golden():
    assert FULL_DEPTH.num_hidden_layers == 64 and FULL_DEPTH.layer_types == QWEN38.layer_types
    model = ref.init_random_(ref.TextModel(FULL_DEPTH, with_lm_head=False), seed=5).float().eval()
    ids = torch.randint(0, FULL_DEPTH.vocab_size, (1, 96), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        h, st = model(ids)
        gh, gst = g_model(model, ids, FULL_DEPTH)
    assert pcc(h, gh) > 0.999
    for i, (a, b) in enumerate(zip(st, gst)):
        for k in a:
            assert pcc(a[k], b[k]) > 0.999, (i, k)


def test_stream_forward_matches_model():
    cfg = QWEN38.reduced(num_hidden_layers=8, **WIDE_SHALLOW)
    seed = 7
    ids = torch.randint(0, cfg.vocab_size, (1, 64), generator=torch.Generator().manual_seed(1))
    h, st = golden.stream_forward(cfg, ids, seed=seed)
    model = ref.TextModel(cfg, with_lm_head=False).float().eval()
    sd = golden.random_weight_state_dicts(cfg, seed, cfg.num_hidden_layers)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    with torch.no_grad():
        h2, st2 = model(ids)
    assert pcc(h, h2) > 0.99999
    for a, b in zip(golden.flatten_states(st), golden.flatten_states(st2)):
        assert pcc(a, b) > 0.99999


def test_golden_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN38_REF_CACHE", str(tmp_path))
    calls = []

    def compute(tag):
        def f():
            calls.append(tag)
            return [torch.full((2, 2), float(len(calls)))], [torch.zeros(1)]

        return f

    key = golden.cache_key("random", "synthetic", 64, 8)
    a, _ = golden.cached_forward(key, compute("first"))
    b, _ = golden.cached_forward(key, compute("second"))  # must load, not recompute
    assert calls == ["first"] and torch.equal(a[0], b[0])
    key2 = golden.cache_key("random", "synthetic", 64, 9)  # one field changed -> miss
    c, _ = golden.cached_forward(key2, compute("third"))
    assert calls == ["first", "third"] and not torch.equal(a[0], c[0])
    try:
        golden.cached_forward(golden.cache_key("random", "synthetic", 128, 8), compute("x"), require_cached=True)
        raise AssertionError("expected a loud miss")
    except FileNotFoundError:
        pass
