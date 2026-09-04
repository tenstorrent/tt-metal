# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden-cache round-trip (recipe M1, third row — "No donor — author it").

Asserts the two properties the cache exists to guarantee:

* a second run **loads** instead of recomputing;
* a changed ``ReferenceCacheKey`` field forces a **miss** rather than silently reusing a stale
  result.

Host-only. Uses a reduced config so it costs a second, not an hour.
"""

import torch

from models.demos.llama3_1_8b_d_p.reference import golden
from models.demos.llama3_1_8b_d_p.reference.config import LlamaConfig


def _key(cfg, **over):
    base = dict(
        weight_type="random",
        seed=0,
        input_source="random_ids",
        seq_len=32,
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_chunks=1,
        dtype="float32",
    )
    base.update(over)
    return golden.ReferenceCacheKey(**base)


def _cfg():
    return LlamaConfig.from_json().reduced(num_hidden_layers=2, intermediate_size=128, vocab_size=256)


def test_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv(golden.CACHE_ENV, str(tmp_path))
    cfg = _cfg()
    key = _key(cfg)

    assert golden.load_reference_cache(key) is None, "cache must start empty"
    first = golden.run_golden(cfg, key)
    assert golden.cache_path(key).exists(), "run_golden must persist the trace"

    # Second run must LOAD, not recompute: require_hit makes a recompute an error.
    second = golden.run_golden(cfg, key, require_hit=True)
    torch.testing.assert_close(first["logits"], second["logits"])
    for a, b in zip(first["k"], second["k"]):
        torch.testing.assert_close(a, b)


def test_changed_key_field_forces_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(golden.CACHE_ENV, str(tmp_path))
    cfg = _cfg()
    golden.run_golden(cfg, _key(cfg))

    for field, value in [("seed", 1), ("seq_len", 64), ("input_source", "abc"), ("num_chunks", 2)]:
        assert golden.load_reference_cache(_key(cfg, **{field: value})) is None, f"{field} must key the cache"


def test_require_hit_fails_loudly_on_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(golden.CACHE_ENV, str(tmp_path))
    cfg = _cfg()
    try:
        golden.run_golden(cfg, _key(cfg), require_hit=True)
    except RuntimeError as e:
        assert "cache miss" in str(e)
    else:
        raise AssertionError("require_hit must raise on a cache miss, not recompute")


def test_chunked_golden_matches_one_shot(tmp_path, monkeypatch):
    """The chunked reference path (num_chunks>1) must produce the same KV as one-shot.

    This is the reference-side twin of the P2 device gate; if it did not hold, a chunked device run
    would be measured against a golden that is itself wrong.
    """
    monkeypatch.setenv(golden.CACHE_ENV, str(tmp_path))
    cfg = _cfg()
    one = golden.run_golden(cfg, _key(cfg, seq_len=64, num_chunks=1))
    many = golden.run_golden(cfg, _key(cfg, seq_len=64, num_chunks=4))
    torch.testing.assert_close(one["input_ids"], many["input_ids"])
    for i, (k1, k4) in enumerate(zip(one["k"], many["k"])):
        torch.testing.assert_close(k1, k4, atol=2e-4, rtol=2e-4, msg=lambda m: f"L{i} K: {m}")
    for i, (v1, v4) in enumerate(zip(one["v"], many["v"])):
        torch.testing.assert_close(v1, v4, atol=2e-4, rtol=2e-4, msg=lambda m: f"L{i} V: {m}")
