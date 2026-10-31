# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests of the image-aware prefix cache keys (DSV41_APC_IMAGES): content hashes in the key, the boundary rule, text prompts unchanged.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_prefix_images.py
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.prefix_cache import (
    PrefixCache,
    PrefixSnapshot,
    PrefixStore,
    image_spans,
    key_tokens,
    straddles,
)
from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionSeq

IMG = 99999


def prompt(n=4000, spans=((600, 100),)):
    t = torch.arange(1, n + 1) % 50000
    for s, k in spans:
        t[s : s + k] = IMG
    return t


def vseq(t, spans, keys):
    types = torch.full((t.numel(),), -1, dtype=torch.int64)
    rows = []
    for s, k in spans:
        types[s : s + k] = 0
        rows.append((s, torch.zeros(k, 4, dtype=torch.bfloat16)))
    return VisionSeq(types, rows, keys)


def cache():
    c = PrefixCache(None, 512, 10**7)
    c.image_token_id = IMG
    return c


def put(c, kt, t, s):
    snap = PrefixSnapshot(s=s, toks=t[:s].clone(), pool={0: torch.zeros(100, dtype=torch.bfloat16)})
    snap.ktoks = None if kt is t else kt[:s].clone()
    return c.store.put(snap)


def test_text_prompts_are_untouched(monkeypatch):
    for v in ("0", "1"):
        monkeypatch.setenv("DSV41_APC_IMAGES", v)
        c = cache()
        t = torch.arange(1, 4001) % 50000
        assert c._keyed(t, 4000, None) is t  # the very same tensor: the key chain is the pre-image one
        assert put(c, t, t, 1024)
        assert c.lookup(0, t, 4000).s == 1024
        assert next(iter(c.store.snaps))[3] == PrefixStore(512, 1)._chain(t, 1024)[1024][3]


def test_image_prompts_bypass_the_cache_by_default(monkeypatch):
    monkeypatch.delenv("DSV41_APC_IMAGES", raising=False)
    c = cache()
    t = prompt()
    assert c._keyed(t, 4000, vseq(t, [(600, 100)], ["aa"])) is None
    assert c.lookup(0, t, 4000, vs=vseq(t, [(600, 100)], ["aa"])) is None


def test_same_image_hits_different_image_does_not(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt()
    vsA, vsB = vseq(t, [(600, 100)], ["aa" * 20]), vseq(t, [(600, 100)], ["bb" * 20])
    ktA = c._keyed(t, 1024, vsA, 1024)
    assert ktA is not None and not bool((ktA == IMG).any())
    assert put(c, ktA, t, 1024)
    assert c.lookup(0, t, 4000, vs=vsA).s == 1024  # the same image: hit
    assert c.lookup(0, t, 4000, vs=vsB) is None  # same text, other image: no hit
    t2 = t.clone()
    t2[3000] += 1  # text after the snapshot may differ
    assert c.lookup(0, t2, 4000, vs=vsA).s == 1024
    t3 = t.clone()
    t3[10] += 1  # text before the image differs: no hit
    assert c.lookup(0, t3, 4000, vs=vseq(t3, [(600, 100)], ["aa" * 20])) is None


def test_same_image_at_another_position_does_not_hit(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt(spans=((600, 100),))
    put(c, c._keyed(t, 1024, vseq(t, [(600, 100)], ["aa" * 20]), 1024), t, 1024)
    t2 = prompt(spans=((700, 100),))  # the image comes 100 tokens later (other text before it)
    assert c.lookup(0, t2, 4000, vs=vseq(t2, [(700, 100)], ["aa" * 20])) is None


def test_boundary_rule_no_snapshot_inside_an_image_span(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt(spans=((400, 300),))  # image 400..700 crosses 512
    vs = vseq(t, [(400, 300)], ["cc" * 20])
    assert (
        straddles(image_spans(vs), 512) and not straddles(image_spans(vs), 1024) and not straddles(image_spans(vs), 400)
    )
    assert c._keyed(t, 512, vs, 512) is None  # no snapshot at 512
    assert c._keyed(t, 1024, vs, 1024) is not None
    # a lookup of this prompt only finds the boundary after the span, and only when it was stored
    put(c, c._keyed(t, 1024, vs, 1024), t, 1024)
    assert c.lookup(0, t, 1100, vs=vs) is None  # 1100 - 128 = 972 < 1024: the hit position stays <= end - 128
    assert c.lookup(0, t, 1200, vs=vs).s == 1024


def test_unexplained_placeholders_are_never_cached(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt(spans=((600, 100), (900, 100)))
    assert c._keyed(t, 1500, vseq(t, [(600, 100)], ["aa" * 20])) is None  # the second image has no span / key


def test_min_s_and_images_after_the_snapshot(monkeypatch):
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt(spans=((600, 100), (3000, 100)))
    vs = vseq(t, [(600, 100), (3000, 100)], ["aa" * 20, "dd" * 20])
    put(c, c._keyed(t, 1024, vs, 1024), t, 1024)
    assert c.lookup(0, t, 4000, vs=vs).s == 1024  # a later image does not matter for a snapshot before it
    assert c.lookup(0, t, 4000, vs=vs, min_s=1024) is None  # a continuation already past it gains nothing
    kt, ok = key_tokens(t[:1024], image_spans(vs), IMG)
    assert ok and kt.shape[0] == 1024


# ---- step 2: window_needs_sync and the restore wave (image users) ----
def _load(name):
    """a function / method of tt/dsv41_model.py compiled stand-alone (the module itself needs ttnn)."""
    import ast
    import os
    import textwrap

    src = open(os.path.join(os.path.dirname(__file__), "..", "tt", "dsv41_model.py")).read()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns[name]


def test_window_needs_sync_image_users_only_with_the_switch():
    window_needs_sync = _load("window_needs_sync")

    wants = lambda s: s == 1024  # noqa: E731
    users = [(3, 512, 4096)]  # a window ending at 1024, not the last of the call
    vis = {3: object()}
    assert (
        window_needs_sync(users, 512, True, wants, None, None) is True
    )  # text user: a wanted snapshot forces the sync (unchanged)
    assert (
        window_needs_sync(users, 512, True, wants, vis, None) is False
    )  # image user, switch off: unchanged (never snapshotted)
    assert window_needs_sync(users, 512, True, wants, vis, None, True) is True  # switch on: it is snapshotted too


class _Pool:
    def __init__(self):
        self.grown = []

    def grow(self, b, n):
        self.grown.append((b, n))

    def sync_page_table(self):
        pass


class _Apc:
    def __init__(self, c):
        self.c, self.restored = c, []

    def images_enabled(self):
        return self.c.images_enabled()

    def lookup(self, *a, **k):
        return self.c.lookup(*a, **k)

    def restore_group(self, snap, bs):
        self.restored.append((snap.s, list(bs)))


class _Model:
    _apc_restore_wave = _load("_apc_restore_wave")

    def __init__(self):
        self.pool = _Pool()


def _model():
    return _Model()


def test_restore_wave_text_unchanged_and_image_continuation_jumps_ahead(monkeypatch):
    c = cache()
    t = prompt(spans=((600, 100),))
    vs = vseq(t, [(600, 100)], ["aa" * 20])
    txt = torch.arange(1, 4001) % 50000
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    put(c, c._keyed(t, 1024, vs, 1024), t, 1024)
    put(c, txt, txt, 1024)
    apc = _Apc(c)
    m = _model()
    # new text prompt + continuation of text (st > 0: never looked up) + continuation of the image prompt at st = 512
    wave = [(0, txt, 0, 2000), (1, txt, 512, 2000), (2, t, 512, 2000)]
    out = m._apc_restore_wave(wave, apc, {2: vs})
    assert out == [(0, txt, 1024, 2000), (1, txt, 512, 2000), (2, t, 1024, 2000)]
    assert sorted(apc.restored) == [(1024, [0]), (1024, [2])] or sorted(apc.restored) == [(1024, [0, 2])]
    # the image continuation already at 1024: nothing to gain
    assert m._apc_restore_wave([(2, t, 1024, 2000)], _Apc(c), {2: vs}) == [(2, t, 1024, 2000)]
    # switch off: the image user is left alone
    monkeypatch.setenv("DSV41_APC_IMAGES", "0")
    assert m._apc_restore_wave([(2, t, 0, 2000)], _Apc(c), {2: vs}) == [(2, t, 0, 2000)]


def test_cache_salt_changes_the_image_key(monkeypatch):
    from models.demos.blackhole.deepseek_v41_flash.tt.vision_state import VisionBook

    k = "aa" * 20
    assert VisionBook._salted(k, "a") != VisionBook._salted(k, "b") != k
    assert VisionBook._salted(k, "a") == VisionBook._salted(k, "a")
    monkeypatch.setenv("DSV41_APC_IMAGES", "1")
    c = cache()
    t = prompt()
    vsA, vsB = vseq(t, [(600, 100)], [VisionBook._salted(k, "a")]), vseq(t, [(600, 100)], [VisionBook._salted(k, "b")])
    put(c, c._keyed(t, 1024, vsA, 1024), t, 1024)
    assert c.lookup(0, t, 4000, vs=vsA).s == 1024 and c.lookup(0, t, 4000, vs=vsB) is None
