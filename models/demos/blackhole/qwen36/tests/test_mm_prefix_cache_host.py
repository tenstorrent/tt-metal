# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for multimodal prefix caching (QWEN36_MM_PREFIX_CACHE=1, tt/mm_prefix.py):

- the GDN prefix-state key of an image prompt covers the images: same token ids + different image -> no snapshot past
  the image is shared; same image -> full reuse; text prompts keep their exact digests;
- MMVision encodes only the images the resumed suffix [s, L) reaches and reports the row base the splice subtracts;
- MTPSpecEngine._prefill_row plans with the image key, encodes after the plan, restages every grid for M-RoPE, and
  resets the row base; _prefill_fast saves under the image key and builds M-RoPE with no rows to splice;
- Qwen36MTPContractForCausalLM.prefill_forward hands the engine an MMVision only for well-formed image rows.

Fakes of test_spec_fastprefill_host.py / test_spec_prefix_fast_host.py / test_spec_mm_host.py.
"""
import hashlib
import types

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import (
    VOCAB,
    _dispatch_engine,
    _engine,
    _fake_model,
    _row,
    fake_ttnn,
)
from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache
from models.demos.blackhole.qwen36.tt import mm_prefix, qwen36_vllm
from models.demos.blackhole.qwen36.tt.mm_prefix import (
    KEY_WORDS,
    MMItem,
    MMKeyedTokens,
    MMPayloadMissing,
    MMVision,
    parse_items,
    placeholders_match,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import _key_material
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

__all__ = ["fake_ttnn"]  # fixture re-export

IMG = 7  # image placeholder id of the fakes


def _prompt(L, images, seed=0):
    """L tokens (ids >= 10) with IMG runs at the (offset, length) of each image."""
    toks = [10 + (seed * 31 + 3 * i) % 900 for i in range(L)]
    for off, n in images:
        toks[off : off + n] = [IMG] * n
    return toks


def _items(images, ids):
    return [MMItem(off, n, i) for (off, n), i in zip(images, ids)]


def _digest(buf, k):
    return hashlib.blake2b(buf[:k].tobytes(), digest_size=16).digest()


# --------------------------------------------------------------------------- #
# Key material
# --------------------------------------------------------------------------- #
def test_text_key_material_is_the_plain_token_digest():
    toks = list(range(10, 300))
    buf, kpos, n = _key_material(torch.tensor(toks))
    assert n == 290 and [kpos(p) for p in (0, 128, 290)] == [0, 128, 290]
    assert buf.dtype == np.int32 and np.array_equal(buf, np.array(toks, dtype=np.int32))


def test_mm_key_covers_an_image_from_its_first_placeholder_on():
    images = [(200, 64)]
    toks = _prompt(400, images)
    a = MMKeyedTokens(toks, _items(images, ["hash-A"]))
    b = MMKeyedTokens(toks, _items(images, ["hash-B"]))
    plain = np.array(toks, dtype=np.int32)
    for pos in (0, 128, 200):  # up to the image start: equal to each other and to the text digest
        assert _digest(a.buf, a.kpos(pos)) == _digest(b.buf, b.kpos(pos)) == _digest(plain, pos)
    for pos in (201, 256, 264, 384, 400):  # from one placeholder in: the image hash is in the key
        assert a.kpos(pos) == pos + KEY_WORDS
        assert _digest(a.buf, a.kpos(pos)) != _digest(b.buf, b.kpos(pos))
        assert _digest(a.buf, a.kpos(pos)) != _digest(plain, pos)
    same = MMKeyedTokens(toks, _items(images, ["hash-A"]))
    assert np.array_equal(a.buf, same.buf)


def test_image_order_and_position_are_in_the_key():
    images = [(100, 32), (300, 32)]
    toks = _prompt(500, images)
    ab = MMKeyedTokens(toks, _items(images, ["A", "B"]))
    ba = MMKeyedTokens(toks, _items(images, ["B", "A"]))
    assert _digest(ab.buf, ab.kpos(128)) != _digest(ba.buf, ba.kpos(128))


def test_gdn_cache_never_serves_another_images_state(fake_ttnn):
    """Real GdnPrefixStateCache (device copies faked): snapshots of prompt A at 128 (before the image) and 384 (after)."""
    images = [(200, 64)]
    toks = _prompt(600, images)
    cache = _cache()
    key_a = MMKeyedTokens(toks, _items(images, ["image-A"]))
    for p in (128, 384):
        assert cache.save(key_a, p)
    # same text, different image: only the snapshot before the image matches
    assert cache.lookup(MMKeyedTokens(toks, _items(images, ["image-B"])), 512)[0] == 128
    # same image: the snapshot after it matches
    assert cache.lookup(MMKeyedTokens(toks, _items(images, ["image-A"])), 512)[0] == 384
    # the same ids as a text prompt (no image key) never match past the image either
    assert cache.lookup(torch.tensor(toks), 512)[0] == 128
    # and a text-saved snapshot is still found by text lookups exactly as before
    cache.save(torch.tensor(toks), 512)
    assert cache.lookup(torch.tensor(toks), 600)[0] == 512


def test_pin_uses_the_image_key(fake_ttnn):
    images = [(64, 32)]
    toks = _prompt(400, images)
    cache = _cache()
    key = MMKeyedTokens(toks, _items(images, ["A"]))
    cache.save(key, 256)
    assert cache.pin(key, 256, owner=0)
    assert not cache.pin(MMKeyedTokens(toks, _items(images, ["B"])), 256, owner=1)


def test_parse_items_sorts_and_rejects_overlap():
    assert parse_items([(50, 4, "b"), (10, 4, "a")]) == [MMItem(10, 4, "a"), MMItem(50, 4, "b")]
    for bad in ([(10, 8, "a"), (12, 4, "b")], [(10, 0, "a")], [(10, 4, "")]):
        with pytest.raises(ValueError):
            parse_items(bad)


def test_placeholders_must_match_the_spans():
    images = [(20, 8), (60, 8)]
    toks = _prompt(100, images)
    items = _items(images, ["a", "b"])
    assert placeholders_match(toks, items, IMG)
    stray = list(toks)
    stray[5] = IMG  # a placeholder id outside every span (e.g. typed by the user)
    assert not placeholders_match(stray, items, IMG)
    assert not placeholders_match(toks, items[:1], IMG)  # a span missing


# --------------------------------------------------------------------------- #
# MMVision
# --------------------------------------------------------------------------- #
def _vision(images, ids, pixels=None, L=600):
    toks = _prompt(L, images)
    enc = []
    pv = pixels if pixels is not None else [torch.full((4 * n, 3), float(k)) for k, (_, n) in enumerate(images)]
    grids = [torch.tensor([[1, 2 * n, 2]]) for _, n in images]

    def encode(p, g):
        enc.append((p[:, 0].unique().tolist(), g.tolist()))
        return f"ROWS{len(enc)}"

    return MMVision(toks, _items(images, ids), pv, grids, encode), enc, toks


def test_rows_for_encodes_only_the_items_the_suffix_reaches():
    v, enc, _ = _vision([(100, 32), (300, 32)], ["A", "B"])
    assert v.rows_for(0) == ("ROWS1", 0) and enc[0] == ([0.0, 1.0], [[1, 64, 2], [1, 64, 2]])
    # s = 256: image A ([100, 132)) is cached, image B is not
    assert v.rows_for(256) == ("ROWS2", 32) and enc[1] == ([1.0], [[1, 64, 2]])
    # s inside image B: B is still encoded whole, the splice starts at its row s - 300
    assert v.rows_for(320) == ("ROWS2", 32) and len(enc) == 2  # memo: no second tower run for the same first item
    # past every image: nothing to encode, M-RoPE still needs the grids
    assert v.rows_for(384) == (None, 64) and len(enc) == 2
    assert v.grids.tolist() == [[1, 64, 2], [1, 64, 2]]


def test_rows_for_needs_the_pixels_of_items_it_reaches():
    images = [(100, 32), (300, 32)]
    v, _, _ = _vision(images, ["A", "B"], pixels=[None, torch.ones(128, 3)])
    assert v.rows_for(256)[1] == 32  # A is cached: its stripped pixels are not needed
    with pytest.raises(MMPayloadMissing):
        v.rows_for(0)


def test_row_offset_subtracts_the_base():
    m = types.SimpleNamespace(
        _req_video_grid_thw=None,
        args=types.SimpleNamespace(hf_config=types.SimpleNamespace(image_token_id=IMG, video_token_id=IMG + 1)),
    )
    m._vision_placeholder_token_id = lambda: Qwen36Model._vision_placeholder_token_id(m)
    ids = torch.tensor([_prompt(600, [(100, 32), (300, 32)])])
    assert Qwen36Model._vis_row_offset_for(m, ids, 320) == 52  # 32 + 20 placeholders before 320
    m._vis_row_base = 32  # rows start at image B
    assert Qwen36Model._vis_row_offset_for(m, ids, 320) == 20
    assert Qwen36Model._vis_row_offset_for(m, ids, 256) == 0
    with pytest.raises(ValueError):
        Qwen36Model._vis_row_offset_for(m, ids, 128)  # a segment before the first encoded image


# --------------------------------------------------------------------------- #
# Engine: _prefill_row with an MMVision
# --------------------------------------------------------------------------- #
def _mm_row_engine(cache, calls):
    e = _dispatch_engine(8, fast=True, recorded=True, calls=calls)
    m = e.model
    m._prefix_cache = cache
    m._req_image_grid_thw = m._req_video_grid_thw = None
    m.args = types.SimpleNamespace(
        max_batch_size=8, hf_config=types.SimpleNamespace(image_token_id=IMG, video_token_id=IMG + 1)
    )
    m.plan_spec_prefix = lambda *a, **k: calls.append(("plan", a[0], a[2])) or Qwen36Model.plan_spec_prefix(m, *a, **k)

    def _fast(*a, **k):
        calls.append(("fast", dict(k), getattr(m, "_vis_row_base", 0), m._req_image_grid_thw))
        a[6][0] = True
        return torch.arange(VOCAB, dtype=torch.float32)

    e._prefill_fast = _fast
    return e


def test_cold_image_prompt_plans_with_the_image_key_and_encodes_everything(fake_ttnn):
    calls, cache = [], _cache()
    v, enc, toks = _vision([(100, 32), (300, 32)], ["A", "B"], L=9000)
    e = _mm_row_engine(cache, calls)
    MTPSpecEngine._prefill_row(e, 1, toks, _row(200), None, start_pos=0, vision_tokens=v)
    (plan_call,) = [c for c in calls if c[0] == "plan"]
    assert plan_call[1] is v.key and plan_call[2] == 0
    (fast,) = [c for c in calls if c[0] == "fast"]
    kw, base, grids = fast[1], fast[2], fast[3]
    assert kw["vision_tokens"] == "ROWS1" and base == 0 and len(enc) == 1
    assert kw["mm_key"] is v.key and kw["mm_rope"] is True and kw["plan"][0] == 0  # saves the snapshots, from 0
    assert grids.tolist() == v.grids.tolist()  # every image's grid staged for M-RoPE
    assert e.model.__dict__.get("_vis_row_base", 0) == 0  # reset after the prefill


def test_warm_turn_resumes_past_the_images_and_skips_the_tower(fake_ttnn):
    calls, cache = [], _cache()
    images = [(100, 32), (300, 32)]
    v, enc, toks = _vision(images, ["A", "B"], L=1200)
    cache.save(v.key, 1024)  # what the first turn left
    e = _mm_row_engine(cache, calls)
    # next turn: same images, vLLM hit of 1088 tokens -> resume at 1024
    MTPSpecEngine._prefill_row(e, 1, toks, _row(), None, start_pos=1088, vision_tokens=v)
    (fast,) = [c for c in calls if c[0] == "fast"]
    kw, base = fast[1], fast[2]
    assert kw["plan"][0] == 1024 and "vision_tokens" not in kw and enc == []
    assert kw["mm_rope"] is True and base == 64


def test_resume_between_images_encodes_only_the_later_one(fake_ttnn):
    calls, cache = [], _cache()
    images = [(100, 32), (300, 32)]
    v, enc, toks = _vision(images, ["A", "B"], L=1200)
    cache.save(v.key, 256)
    e = _mm_row_engine(cache, calls)
    MTPSpecEngine._prefill_row(e, 1, toks, _row(), None, start_pos=320, vision_tokens=v)
    (fast,) = [c for c in calls if c[0] == "fast"]
    assert fast[1]["plan"][0] == 256 and fast[1]["vision_tokens"] == "ROWS1" and fast[2] == 32
    assert enc == [([1.0], [[1, 64, 2]])]


def test_other_image_same_text_resumes_only_before_the_image(fake_ttnn):
    calls, cache = [], _cache()
    images = [(200, 64)]
    va, _, toks = _vision(images, ["A"], L=1200)
    for p in (128, 1024):
        cache.save(va.key, p)
    vb, enc_b, _ = _vision(images, ["B"], L=1200)
    e = _mm_row_engine(cache, calls)
    MTPSpecEngine._prefill_row(e, 1, toks, _row(), None, start_pos=1088, vision_tokens=vb)
    (fast,) = [c for c in calls if c[0] == "fast"]
    assert fast[1]["plan"][0] == 128 and fast[1]["vision_tokens"] == "ROWS1" and fast[2] == 0
    assert len(enc_b) == 1  # image B encoded


def test_eager_fallback_with_a_plan_recomputes_from_zero_with_all_images(fake_ttnn):
    calls, cache = [], _cache()
    images = [(100, 32), (300, 32)]
    v, enc, toks = _vision(images, ["A", "B"], L=1200)
    cache.save(v.key, 1024)
    e = _mm_row_engine(cache, calls)
    e.fast_prefill = False  # no traced prefill
    seen = {}

    def _eager(prompt, pt_row, Tp, on_chunk, slot=0, plan=None, vision_tokens=None):
        seen.update(plan=plan, vt=vision_tokens, base=getattr(e.model, "_vis_row_base", 0))
        on_chunk("HIDDEN", 0, Tp)
        return torch.arange(VOCAB + 8, dtype=torch.float32)

    e.model.prefill_for_spec = _eager
    MTPSpecEngine._prefill_row(e, 1, toks, _row(), None, start_pos=1088, vision_tokens=v)
    assert seen == dict(plan=None, vt="ROWS1", base=0) and enc[0][0] == [0.0, 1.0]


def test_re_prefill_of_an_image_row_reuses_its_mmvision(fake_ttnn):
    calls, cache = [], _cache()
    v, enc, toks = _vision([(100, 32)], ["A"], L=1200)
    e = _mm_row_engine(cache, calls)
    MTPSpecEngine._prefill_row(e, 2, toks, _row(), None, start_pos=0, vision_tokens=v)
    e._prompts[2] = (toks, _row(), 0)
    MTPSpecEngine._reprefill(e, 2)
    fasts = [c for c in calls if c[0] == "fast"]
    assert len(fasts) == 2 and len(enc) == 1  # memoized rows, no second tower run


# --------------------------------------------------------------------------- #
# Engine: _prefill_fast with a plan for an image prompt
# --------------------------------------------------------------------------- #
def test_prefill_fast_saves_under_the_image_key_and_builds_mrope(fake_ttnn, monkeypatch):
    calls, cache = [], _cache()
    images = [(100, 32)]
    v, _, toks = _vision(images, ["A"], L=1600)
    model = _fake_model(calls, trace_lengths=(1024, 1600 - 1024))  # segments [0, 1024) and [1024, 1600)
    model._prefix_cache = cache
    rope = []
    model._build_request_rope = lambda p, vt, **k: rope.append((vt, k))
    snap = []
    e = _engine(8, model=model)
    e._snap_run_plan = lambda *a: snap.append(a) or (None, None)
    plan = (0, None, [1024])
    MTPSpecEngine._prefill_fast(
        e,
        3,
        toks,
        torch.tensor([toks], dtype=torch.int32),
        _row(),
        1600,
        lambda *a: None,
        [False],
        plan=plan,
        mtp_prompt=[toks],
        mm_key=v.key,
        mm_rope=True,
    )
    assert rope[0] == (None, {"mm": True})
    assert snap == []  # no in-replay snapshots for an image prompt
    assert cache.lookup(v.key, 1100)[0] == 1024
    assert cache.lookup(torch.tensor(toks), 1100)[0] == 0  # not under the plain token key


def test_prefill_fast_refuses_an_image_plan_without_its_key(fake_ttnn):
    e = _engine(8, model=_fake_model([], trace_lengths=(100,)))
    with pytest.raises(RuntimeError, match="mm_key"):
        MTPSpecEngine._prefill_fast(
            e,
            3,
            [1] * 100,
            torch.ones(1, 100, dtype=torch.int32),
            _row(),
            100,
            lambda *a: None,
            [False],
            plan=(0, None, []),
            mtp_prompt=[[1] * 100],
            vision_tokens="ROWS",
        )


# --------------------------------------------------------------------------- #
# prefill_forward
# --------------------------------------------------------------------------- #
class _FakeEngine:
    def __init__(self):
        self.calls = []

    def prefill(self, slot, ids, row, *args, **kw):
        self.calls.append((slot, ids.clone(), args, kw))
        return torch.arange(VOCAB, dtype=torch.float32)


def _gen(monkeypatch, enabled=True):
    g = Qwen36MTPContractForCausalLM.__new__(Qwen36MTPContractForCausalLM)
    eng = _FakeEngine()
    tower = []
    model = types.SimpleNamespace(
        args=types.SimpleNamespace(hf_config=types.SimpleNamespace(image_token_id=IMG, video_token_id=IMG + 1))
    )
    model.get_image_features = lambda pix, grid: tower.append(("image", tuple(pix.shape))) or "IMG_ROWS"
    model.get_video_features = lambda pix, grid: tower.append(("video", tuple(pix.shape))) or "VID_ROWS"
    g.model = [model]
    g._ensure_ready = lambda kv_cache=None: eng
    g._table = lambda pt: pt.to(torch.int32)
    monkeypatch.setattr(qwen36_vllm, "_spec_prefix_cache_enabled", lambda: True)
    monkeypatch.setenv("QWEN36_MM_PREFIX_CACHE", "1" if enabled else "0")
    return g, eng, tower


def _forward(g, spans, tokens_row, video=False):
    tokens = torch.tensor([tokens_row], dtype=torch.int32)
    kw = dict(
        pixel_values=[[torch.zeros(16, 8)]],
        image_grid_thw=[[torch.tensor([1, 4, 4])]],
        mm_item_spans=[spans],
        start_pos=torch.tensor([64]),
    )
    if video:
        kw.update(pixel_values_videos=[[torch.zeros(8, 8)]], video_grid_thw=[[torch.tensor([2, 2, 2])]])
    return g.prefill_forward(
        tokens, torch.ones(1, 4, dtype=torch.int32), None, torch.tensor([200]), empty_slots=[0], **kw
    )


def test_prefill_forward_hands_an_mmvision_to_the_engine(monkeypatch):
    g, eng, tower = _gen(monkeypatch)
    toks = _prompt(200, [(20, 4)])
    _forward(g, [(20, 4, "hash-A")], toks)
    ((_, _, _, kw),) = eng.calls
    assert kw["start_pos"] == 64 and isinstance(kw["vision_tokens"], MMVision) and tower == []  # tower runs lazily
    assert kw["vision_tokens"].items == [MMItem(20, 4, "hash-A")]


@pytest.mark.parametrize(
    "case",
    ["flag_off", "bad_spans", "video", "no_spans"],
)
def test_prefill_forward_falls_back_to_the_plain_mm_path(monkeypatch, case):
    g, eng, tower = _gen(monkeypatch, enabled=case != "flag_off")
    toks = _prompt(200, [(20, 4)])
    spans = None if case == "no_spans" else [(24, 4, "hash-A")] if case == "bad_spans" else [(20, 4, "hash-A")]
    _forward(g, spans, toks, video=case == "video")
    ((_, _, _, kw),) = eng.calls
    assert kw["start_pos"] is None and isinstance(kw["vision_tokens"], str)  # recompute from 0, rows precomputed


def test_capability_follows_the_env():
    caps = Qwen36MTPContractForCausalLM.model_capabilities
    assert "accepts_mm_prefix_cache" in caps
    assert caps["accepts_mm_prefix_cache"] == (
        mm_prefix.mm_prefix_cache_enabled() and qwen36_vllm._spec_prefix_cache_enabled()
    )
