# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the DeepSeek-V4.1 oracle package (host only, no device).

Semantics are checked against the vendored reference itself (captures reproduce reference outputs and
state), the chunk contract with a prompt that is not a multiple of the chunk (padded tail), and the disk
cache (miss/hit, key sensitivity, bit-identical reload). Small dims; real dims only at the args level,
plus the Engram hash alignment against the full released layout (needs the HF tokenizer).
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as o
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.engram import EngramLayout

SEQ = 41  # > window (16), odd (ratio-2 carry), not a multiple of the 16-token test chunk


@pytest.fixture(autouse=True)
def _few_threads():
    # the small reference is dominated by per-op overhead; many threads on a shared host slow it 100x
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prev)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "CACHE_DIR", tmp_path)
    return tmp_path


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _assert_same(a, b, path="result"):
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(_bits(a), _bits(b)), path
    else:
        assert a == b, path


def _small():
    spec = o.small_spec(SEQ)
    return spec, o.random_tokens(spec)


def test_real_spec_keeps_v41_roles():
    spec = o.real_spec((0, 1, 2, 3, 20, 21, 24), 2048, candidate_topk_blocks=96)
    a, full = spec.args, o.released_args(2048)
    assert (a.dim, a.n_heads, a.head_dim, a.n_routed_experts, a.window_size) == (5120, 64, 512, 384, 128)
    assert spec.layer_ids == (0, 1, 2, 3, 20, 21, 24) and spec.model_args == full
    assert a.n_layers == 7 and a.compress_ratios == (0, 0, 2, 2, 1, 1, 1)  # ratio 0: SWA only, base RoPE
    assert a.kv_source_layers == (2, 4)  # V4.1 2, 20
    assert a.index_source_layers == (2, 4, 6)  # V4.1 2, 20, 24
    assert a.candidate_source_layer == 4 and a.candidate_topk_blocks == 96
    assert a.engram_layer_ids == (1,) and a.engram_num_embeddings == (384006168,)
    assert a.n_mtp_layers == 0 and a.dspark_block_size == 0 and a.vision_n_layers == 0

    ds = o.real_spec((20, 36, 37, 38, 39), 1024, dspark=True).args
    assert ds.compress_ratios == (1, 1, 1, 1, 1, 0, 0, 0) and ds.n_mtp_layers == 3
    assert ds.dspark_target_layer_ids == (2, 3, 4) and ds.dspark_block_size == 5
    assert ds.kv_source_layers == (0,) and ds.index_source_layers == (0, 1) and ds.candidate_source_layer == 0


def test_real_spec_rejects_incomplete_schedules(expect_error):
    with expect_error(ValueError, r"layers \[2\]"):
        o.real_spec((3,), 128)  # ratio-2 consumer without its KV/index source
    with expect_error(ValueError, r"layers \[20\]"):
        o.real_spec((24,), 128)  # candidate-constrained index source reads index-K and candidates of 20
    with expect_error(ValueError, "strictly increasing"):
        o.real_spec((21, 20), 128)
    with expect_error(ValueError, r"layers \[37, 38, 39\]"):
        o.real_spec((20, 36), 128, dspark=True)


@pytest.mark.skipif(not (o.HF_SNAPSHOT / "tokenizer.json").is_file(), reason="needs the HF V4.1 tokenizer")
def test_engram_hash_of_layer_subset_matches_full_model():
    """V4.1 layer 14 alone hashes exactly like layer 14 of the full model (primes, offsets, multipliers)."""
    spec = o.real_spec((14,), 16)
    tok = o._tokenizer(spec)
    sub = v41.NgramHashState(spec.args, EngramLayout.from_args(spec.args), tok)
    o._align_engram_hash(sub, spec)
    full = v41.NgramHashState(spec.model_args, EngramLayout.from_args(spec.model_args), tok)
    ids = o.random_tokens(spec)
    expected = full(ids, 0)[:, :, spec.model_args.engram_layer_ids.index(14)]
    assert torch.equal(sub(ids, 0)[:, :, 0], expected)


def test_synthetic_engram_rows_are_row_local(cache, expect_error):
    rows = torch.tensor([3, 5, 9])
    q, s = o.synthetic_engram_rows(0, 1, rows, 64)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (3, 64) and s.shape == (3, 2)
    q5, s5 = o.synthetic_engram_rows(0, 1, torch.tensor([5]), 64)
    assert torch.equal(_bits(q[1:2]), _bits(q5)) and torch.equal(_bits(s[1:2]), _bits(s5))
    assert not torch.equal(_bits(o.synthetic_engram_rows(0, 14, rows, 64)[0]), _bits(q))
    assert not torch.equal(_bits(o.synthetic_engram_rows(1, 1, rows, 64)[0]), _bits(q))

    spec, tokens = _small()
    model = o.build_reference(spec)
    emb = model.layers[0].engram.embed
    with expect_error(RuntimeError, "Engram rows were not loaded"):
        emb(torch.tensor([[1, 2]]))
    o.load_engram_rows(model, spec, tokens)
    ids = emb.oracle_rows[[0, 2, 2, 1]].unsqueeze(0)
    rq, rs = o.synthetic_engram_rows(spec.seed, 0, ids[0], emb.dim)
    expected = (rq.float().unflatten(-1, (-1, 32)) * rs.float().unsqueeze(-1)).flatten(-2).bfloat16()
    assert torch.equal(emb(ids)[0], expected)  # the reference lookup, on the loaded rows


def test_oracle_captures_reproduce_reference(cache):
    spec, tokens = _small()
    model = o.build_reference(spec)
    r = o.oracle(spec, tokens, model)
    ids, S = spec.layer_ids, SEQ
    blocks, shared, state = r["blocks"], r["shared"], r["state"]

    raw = v41.shared_attn.topk_idxs[0]  # after the oracle run: published by the last index source (5)
    assert torch.equal(shared[5]["topk_idxs"], torch.where(raw >= 0, raw - S, -1).int())
    assert torch.equal(shared[3]["candidates"], v41.shared_attn.candidates[0])

    # hooks observe without perturbing: a hook-free prefill of a fresh build gives the same logits
    fresh = o.build_reference(spec)
    o.load_engram_rows(fresh, spec, tokens)
    assert torch.equal(o.prefill(fresh, tokens)[0][0], r["logits"])
    assert torch.equal(r["tokens"], tokens[0])

    # the stream chain: block outputs feed the next block (through Engram where present)
    assert torch.equal(blocks[0]["pre_in"], v41.make_identity_pre_mix(tokens.unsqueeze(-1), 4)[0])
    for a, b in zip(ids, ids[1:]):
        nxt = blocks[b].get("engram_in", blocks[b]["x_in"])
        assert torch.equal(blocks[a]["x_out"], nxt) and torch.equal(blocks[a]["pre_out"], blocks[b]["pre_in"])
    assert not torch.equal(blocks[1]["engram_in"], blocks[1]["x_in"])
    assert blocks[1]["engram_hash_ids"].shape == (S, 6)

    # teacher-forced blocks without upstream shared state (SWA-only 0, KV+index source 1) reproduce
    o._reset_state(model)
    with torch.inference_mode(), v41.set_dtype(torch.bfloat16):
        for i in (0, 1):
            x, pre = model.layers[i](blocks[i]["x_in"][None], 0, blocks[i]["pre_in"][None], None)
            assert torch.equal(x[0], blocks[i]["x_out"]) and torch.equal(pre[0], blocks[i]["pre_out"])
    assert torch.equal(model.layers[1].attn.compress_kv_cache[0, : S // 2], shared[1]["compress_kv"])

    # shared state as the sources published it
    assert set(shared) == {1, 3, 5}
    assert shared[1]["compress_kv"].shape == (S // 2, 64) and shared[1]["index_k"].shape == (S // 2, 64)
    assert shared[3]["compress_kv"].shape == (S, 64) and shared[3]["candidates"].shape == (S, S)
    for src, ratio in ((1, 2), (3, 1), (5, 1)):
        idx = shared[src]["topk_idxs"]
        visible = (torch.arange(1, S + 1) // ratio).unsqueeze(1)
        assert ((idx == -1) | ((idx >= 0) & (idx < visible))).all()

    # window KV per position rebuilds every final ring; odd prompt leaves a ratio-2 carry of one token
    for lid in ids:
        assert torch.equal(o.window_ring(blocks[lid]["window_kv"], S, 16), state["window"][lid])
    carry = state["carry"][1]
    assert carry["kv"][0].abs().sum() > 0 and torch.isinf(carry["score"][1]).all()
    assert set(state["carry"]) == {1}
    assert state["engram_tokens"].shape == (S,)
    assert state["main_hidden"].shape == (S, 2 * 256) and state["dspark_window"][0].abs().sum() > 0

    # repeat run on the same model is bit-identical
    _assert_same(o._run(model, spec, tokens), r)


def test_oracle_cache_hit_miss_and_reload(cache, monkeypatch):
    spec, tokens = _small()
    model = o.build_reference(spec)
    weights = sorted(cache.glob("weights-*.pt"))
    assert len(weights) == 10  # embed, 6 layers, norm, head, 1 DSpark layer

    # weights reload bit-identically (and keep Linear's weight.scale)
    again = o.build_reference(spec)
    assert sorted(cache.glob("weights-*.pt")) == weights
    _assert_same(dict(model.state_dict()), dict(again.state_dict()))
    assert again.layers[1].attn.wq_a.weight.scale is again.layers[1].attn.wq_a.scale
    # a unit's weights depend on its name and parameters, not on args that shape no parameter
    other = o.build_reference(o.small_spec(SEQ, window_size=8))
    _assert_same(dict(model.layers[2].state_dict()), dict(other.layers[2].state_dict()))

    miss = o.oracle(spec, tokens, model)
    path = o.cache_path(spec, tokens)
    assert path.is_file()

    def no_run(*args):
        raise AssertionError("cache hit must not run the reference")

    monkeypatch.setattr(o, "_run", no_run)
    _assert_same(o.oracle(spec, tokens), miss)  # hit: bit-identical reload

    other_keys = {
        o.cache_path(o.small_spec(SEQ, seed=1), tokens),
        o.cache_path(spec, o.random_tokens(spec, seed=2)),
        o.cache_path(spec, tokens[:, :-1]),
        o.cache_path(o.small_spec(SEQ, window_size=8), tokens),
        o.cache_path(replace(spec, checkpoint=Path("/snapshots/0123abcd")), tokens),
    }
    monkeypatch.setattr(o, "PACKAGE_VERSION", o.PACKAGE_VERSION + 1)
    other_keys.add(o.cache_path(spec, tokens))
    assert path not in other_keys and len(other_keys) == 6


def test_tail_logits_match_reference_and_noise_floor(cache, monkeypatch):
    spec, tokens = _small()
    model = o.build_reference(spec)
    result = o.oracle(spec, tokens, model)
    tail = o.tail_logits(spec, tokens, 5, model)
    assert tail.shape == (5, spec.args.vocab_size) and tail.dtype == torch.float32
    assert torch.equal(tail[-1], result["logits"])  # the last position is the oracle's logits
    noisy = o.tail_logits(spec, tokens, 5, model, noise=(0.045, 4e-3, 0))
    assert not torch.equal(noisy, tail) and torch.equal(
        noisy, o.tail_logits(spec, tokens, 5, model, noise=(0.045, 4e-3, 0))
    )
    assert not torch.equal(noisy, o.tail_logits(spec, tokens, 5, model, noise=(0.045, 4e-3, 1)))

    drift = o.noise_drift(spec, tokens, 5, (0.045, 4e-3, 0), model)
    assert set(drift) == set(spec.layer_ids)
    assert all(0.5 < d[k] < 1.0 for d in drift.values() for k in ("all", "tail")), drift
    assert torch.equal(noisy, o.tail_logits(spec, tokens, 5, model, noise=(0.045, 4e-3, 0)))  # same noisy run

    def no_prefill(*args):
        raise AssertionError("cache hit must not run the reference")

    monkeypatch.setattr(o, "prefill", no_prefill)
    assert torch.equal(o.tail_logits(spec, tokens, 5), tail)  # hit: bit-identical reload
    assert o.noise_drift(spec, tokens, 5, (0.045, 4e-3, 0)) == drift


def test_window_ring_hand_values():
    kv = torch.arange(10.0).unsqueeze(1)
    assert o.window_ring(kv, 7, 4).flatten().tolist() == [4, 5, 6, 3]  # positions 3..6 at slot p % 4
    assert o.window_ring(kv, 2, 4).flatten().tolist() == [0, 1, 0, 0]  # still filling


def test_chunk_expectations_padded_tail(cache, expect_error):
    spec, tokens = _small()
    r = o.oracle(spec, tokens)
    chunks = o.chunk_expectations(r, chunk_len=16, pad_multiple=8)
    assert [(c["start"], c["length"], c["padded_length"]) for c in chunks] == [(0, 16, 16), (16, 16, 16), (32, 9, 16)]

    for lid, rec in r["blocks"].items():
        for key in rec:
            assert torch.equal(torch.cat([c["blocks"][lid][key] for c in chunks]), rec[key]), (lid, key)
    # rows written per chunk: [start//r, (start+L)//r); the odd tail token writes no ratio-2 row (carry)
    assert [(c["rows"][1]["row_start"], c["rows"][1]["row_end"]) for c in chunks] == [(0, 8), (8, 16), (16, 20)]
    assert [(c["rows"][3]["row_start"], c["rows"][3]["row_end"]) for c in chunks] == [(0, 16), (16, 32), (32, 41)]
    for src in (1, 3):
        for key in ("compress_kv", "index_k"):
            assert torch.equal(torch.cat([c["rows"][src][key] for c in chunks]), r["shared"][src][key])
    for lid in (1, 3, 5):
        assert torch.equal(torch.cat([c["topk_idxs"][lid] for c in chunks]), r["shared"][lid]["topk_idxs"])
    assert [tuple(c["candidates"][3].shape) for c in chunks] == [(16, 16), (16, 32), (9, 41)]
    assert torch.equal(chunks[1]["candidates"][3], r["shared"][3]["candidates"][16:32, :32])

    # state after each chunk: rings from real rows only; the last chunk carries the final state
    kv0 = r["blocks"][0]["window_kv"]
    assert torch.equal(chunks[0]["window"][0], kv0[:16])  # positions 0..15 fill slots 0..15
    assert torch.equal(chunks[2]["window"][0], torch.cat([kv0[32:41], kv0[25:32]]))  # 25..40 at p % 16
    assert all(torch.equal(chunks[-1]["window"][lid], r["state"]["window"][lid]) for lid in spec.layer_ids)
    assert torch.equal(chunks[0]["engram_history"], r["state"]["engram_tokens"][13:16])
    assert torch.equal(chunks[-1]["engram_history"], r["state"]["engram_tokens"][38:41])
    assert ["logits" in c for c in chunks] == [False, False, True] and chunks[-1]["state"] is r["state"]

    with expect_error(ValueError, "multiple of pad_multiple"):
        o.chunk_expectations(r, chunk_len=20, pad_multiple=8)
    with expect_error(ValueError, "multiple of every compress ratio"):
        o.chunk_expectations(r, chunk_len=9, pad_multiple=3)
