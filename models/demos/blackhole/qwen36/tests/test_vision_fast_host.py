# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device, no ttnn) for tt/vision/vision_fast.py: the vision flags default to the release behaviour, the
bucket rules and their warm-up grids, the per-grid cache, the rotary-table padding, the linear patch embedding, the splice
buffer book-keeping and the [vision_timing] line."""
import re

import pytest
import torch

from models.demos.blackhole.qwen36.tt.vision import vision_fast as vf

FLAGS = (
    "QWEN36_VISION_TIMING",
    "QWEN36_VISION_SPLICE_SEG",
    "QWEN36_VISION_GRID_CACHE",
    "QWEN36_VISION_RESIDENT_GRIDS",
    "QWEN36_VISION_BUCKET_EXACT",
    "QWEN36_VISION_BUCKET_1024",
    "QWEN36_VISION_PIPELINE",
    "QWEN36_VISION_PATCH_LINEAR",
    "QWEN36_VISION_LEAN",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    yield
    vf._current = None


# ---- flags ---------------------------------------------------------------------------------------------------------
def test_flags_default_off():
    assert not vf.timing_enabled() and not vf.splice_seg_enabled() and not vf.pipeline_enabled()
    assert not vf.bucket_exact_enabled() and not vf.bucket_1024_enabled() and not vf.bucket_flags_on()
    assert not vf.patch_linear_enabled() and vf.grid_cache_cap() == 0
    assert not any(vf.lean(x) for x in vf._LEAN_ALL)
    assert vf.flags_summary() == "none"
    assert vf.timing_current() is None


def test_lean_list(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_LEAN", "1")
    assert all(vf.lean(x) for x in vf._LEAN_ALL)
    monkeypatch.setenv("QWEN36_VISION_LEAN", "rotpad, hostref")
    assert vf.lean("rotpad") and vf.lean("hostref") and not vf.lean("maskskip") and not vf.lean("hostrows")
    monkeypatch.setenv("QWEN36_VISION_LEAN", "rotpad,typo")
    with pytest.raises(ValueError):
        vf.lean("rotpad")


# ---- buckets -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 255, 1023, 1024, 1025, 1936, 2047, 2048, 2049, 3888, 4095, 4096, 4097, 6143, 6144, 6400])
def test_default_bucket_is_the_release_rule(n):
    assert vf.vision_bucket(n) == (n // 2048 + 1) * 2048


def test_exact_rule_keeps_multiples():
    b = lambda n: vf.vision_bucket(n, exact=True, b1024=False)
    assert [b(n) for n in (1, 2047, 2048, 2049, 4096, 4097, 6144, 8192)] == [2048, 2048, 2048, 4096, 4096, 6144, 6144, 8192]
    # only exact multiples move, and they move down one bucket
    for n in range(1, 8300):
        old, new = vf.vision_bucket(n, False, False), b(n)
        assert new == (old - 2048 if n % 2048 == 0 else old) and new >= n


def test_bucket_1024():
    assert vf.vision_bucket(1023, exact=False, b1024=True) == 1024
    assert vf.vision_bucket(1024, exact=False, b1024=True) == 2048  # the release rule for the multiple
    assert vf.vision_bucket(1024, exact=True, b1024=True) == 1024
    assert vf.vision_bucket(1025, exact=True, b1024=True) == 2048
    for n in range(1, 8300):
        for ex in (False, True):
            got = vf.vision_bucket(n, ex, True)
            assert got >= n and (got == 1024 or got % 2048 == 0)  # the tower's row constraints


def test_bucket_env(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_BUCKET_EXACT", "1")
    assert vf.vision_bucket(4096) == 4096 and vf.vision_bucket(1000) == 2048
    monkeypatch.setenv("QWEN36_VISION_BUCKET_1024", "1")
    assert vf.vision_bucket(1024) == 1024 and vf.bucket_flags_on()


RELEASE_GRIDS = vf.parse_grids("1x32x32;1x48x64;1x64x64;1x80x80")


def test_warm_grids_unchanged_without_flags():
    assert vf.warm_grids(RELEASE_GRIDS, False, False) == RELEASE_GRIDS
    assert vf.warmed_buckets(RELEASE_GRIDS, False, False) == [2048, 4096, 6144, 8192]


@pytest.mark.parametrize("exact,b1024", [(True, False), (False, True), (True, True)])
def test_warm_grids_cover_every_release_bucket(exact, b1024):
    grids = vf.warm_grids(RELEASE_GRIDS, exact, b1024)
    assert grids[: len(RELEASE_GRIDS)] == RELEASE_GRIDS
    want = {2048, 4096, 6144, 8192} | ({1024} if b1024 else set())
    assert set(vf.warmed_buckets(grids, exact, b1024)) == want
    for t, h, w in grids:
        assert t == 1 and h % 2 == 0 and w % 2 == 0 and h > 0 and w > 0


def test_grid_for_bucket_px512_and_px262():
    # max_pixels 512,000 -> at most 2,000 patches; 262,144 -> at most 1,024
    assert vf.vision_bucket(2000, True, True) == 2048
    assert vf.vision_bucket(1024, True, True) == 1024 and vf.vision_bucket(1024, False, True) == 2048
    for b in (1024, 2048, 4096, 6144, 8192):
        t, h, w = vf.grid_for_bucket(b, True, True)
        assert vf.vision_bucket(h * w, True, True) == b


# ---- per-grid cache ------------------------------------------------------------------------------------------------
def test_lru():
    c = vf.LRU(2)
    assert c.get("a") is None and c.misses == 1
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1 and c.hits == 1
    c.put("c", 3)  # evicts b (a was used last)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3 and len(c) == 2
    off = vf.LRU(0)
    off.put("a", 1)
    assert off.get("a") is None and len(off) == 0


def test_grid_key_separates_bucket_and_width():
    g = torch.tensor([[1, 44, 44]], dtype=torch.int32)
    assert vf.grid_key(g, 2048, 72) == (1, 44, 44, 2048, 72)
    assert len({vf.grid_key(g, 2048, 72), vf.grid_key(g, 2048, 96), vf.grid_key(g, 4096, 72)}) == 3
    assert vf.grid_key(torch.tensor([1, 44, 46]), 2048, 72) != vf.grid_key(torch.tensor([1, 46, 44]), 2048, 72)


def test_pad_rot_tables_matches_release_padding():
    torch.manual_seed(0)
    cos, sin = torch.rand(1936, 72), torch.rand(1936, 72)
    # the release code (vision/model.py)
    ref_c = torch.nn.functional.pad(cos, (0, 0, 0, 2048 - 1936), value=1).unsqueeze(0).unsqueeze(0)
    ref_s = torch.nn.functional.pad(sin, (0, 0, 0, 2048 - 1936), value=0).unsqueeze(0).unsqueeze(0)
    c, s = vf.pad_rot_tables(cos, sin, 1936, 2048)
    assert torch.equal(c, ref_c) and torch.equal(s, ref_s)
    # host column pad == what ttnn.pad(.., 1.0) / (.., 0.0) to padded_head_dim does per block on device
    c96, s96 = vf.pad_rot_tables(cos, sin, 1936, 2048, width=96)
    assert c96.shape == s96.shape == (1, 1, 2048, 96)
    assert torch.equal(c96[..., :72], ref_c) and torch.equal(s96[..., :72], ref_s)
    assert bool((c96[..., 72:] == 1).all()) and bool((s96[..., 72:] == 0).all())
    # exact bucket: no row padding
    c0, _ = vf.pad_rot_tables(cos, sin, 1936, 1936)
    assert c0.shape == (1, 1, 1936, 72)
    with pytest.raises(AssertionError):
        vf.pad_rot_tables(cos, sin, 1936, 1024)


def test_patch_embed_linear_matches_conv3d():
    torch.manual_seed(1)
    C, T, P, D = 3, 2, 16, 64

    class PatchEmbed(torch.nn.Module):  # the HF Qwen3VLVisionPatchEmbed forward
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv3d(C, D, kernel_size=[T, P, P], stride=[T, P, P], bias=True)

        def forward(self, x):
            x = x.view(-1, C, T, P, P).to(dtype=self.proj.weight.dtype)
            return self.proj(x).view(-1, D)

    pe = PatchEmbed()
    px = torch.randn(200, C * T * P * P)
    with torch.no_grad():
        ref, got = pe(px), vf.patch_embed_linear(pe, px)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=0, atol=2e-5)
    # how far apart after the bf16 cast the device input goes through (reported in VISION_FAST.md, not a gate)
    diff = (got.to(torch.bfloat16).float() - ref.to(torch.bfloat16).float()).abs().max().item()
    assert diff <= 2 ** -6 * max(1.0, ref.abs().max().item())


# ---- splice buffers ------------------------------------------------------------------------------------------------
def test_splice_lengths():
    buckets = (128, 256, 512, 1024, 2048)
    assert vf.splice_lengths(buckets, 2048, 8192) == [128, 256, 512, 1024, 2048]
    assert vf.splice_lengths(buckets, 2048, 2048) == [128, 256, 512, 1024]  # no big chunk: 2048 is the release pair
    assert vf.splice_lengths(buckets, 4096, 8192) == [128, 256, 512, 1024, 2048, 4096]
    assert vf.splice_lengths(buckets, None, 2048) == [128, 256, 512, 1024]


def _set(keys=()):
    ss = vf.SpliceSet()
    ss.add(None, "V", "M", "Z")
    for k in keys:
        ss.add(k, f"V{k}", f"M{k}", f"Z{k}")
    return ss


def test_splice_set_release_behaviour():
    ss = _set()
    assert ss.pick(2048) is None and ss.pick(None) is None and ss.get(None) == ("V", "M", "Z")
    assert ss.to_clear(None) == [None]  # every clear uploads the zero mask, as the release
    ss.staged(None)
    assert ss.to_clear(None) == [None]
    ss.cleared([None])
    assert ss.to_clear(None, skip_clean=True) == []  # maskskip: known zero
    ss.staged(None)
    assert ss.to_clear(None, skip_clean=True) == [None]


def test_splice_set_per_length():
    ss = _set((512, 1024, 2048))
    assert ss.pick(2048) == 2048 and ss.pick(1536) is None and ss.pick(8192) is None and ss.pick(None) is None
    assert ss.get(1024) == ("V1024", "M1024", "Z1024")
    # the trace body and the staging pick the same pair for one length
    for length in (128, 512, 1024, 1536, 2048, 8192):
        assert ss.pick(length) == ss.pick(length)
    ss.staged(2048)
    # a text replay of another length also zeroes the mask the image left behind
    assert sorted(ss.to_clear(512), key=str) == [2048, 512]
    ss.cleared([2048, 512])
    assert ss.to_clear(512) == [512] and ss.to_clear(512, skip_clean=True) == []
    ss.staged(1024)
    ss.staged(None)
    assert set(ss.to_clear(1024, skip_clean=True)) == {1024, None}
    with pytest.raises(AssertionError):
        ss.add(512, 0, 0, 0)


# ---- timing --------------------------------------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_timing_line_stages_and_sync(monkeypatch):
    clk, syncs = _Clock(), []

    def sync():
        syncs.append(clk.t)
        clk.t += 0.010  # every device wait costs 10 ms

    rec = vf.VisionTiming(sync=sync, clock=clk)
    rec.note("images", 2)
    rec.note("tokens", 1500)
    for grid in ((1, 44, 44), (1, 42, 46)):
        rec.image(grid, grid[1] * grid[2], 2048)
        clk.t += 0.003
        rec.lap("host_rot")
        clk.t += 0.012
        rec.lap("host_patch")
        clk.t += 0.004
        rec.lap("host_pos")
        clk.t += 0.001
        rec.lap("host_rot")  # second part of the rotary stage accumulates
        clk.t += 0.002
        rec.lap("up_rot", sync=True, nbytes=1000)
        clk.t += 0.005
        rec.lap("up_patch", sync=True, nbytes=2000)
        clk.t += 0.090
        rec.lap("tower", sync=True, disp=True)
        clk.t += 0.004
        rec.lap("merger", sync=True, disp=True)
        clk.t += 0.006
        rec.lap("readback")
        clk.t += 0.001
        rec.lap("trim")
    rec.add("tower_call_ms", 1e3 * clk.t)
    rec.start()
    clk.t += 0.030
    rec.lap("splice_host", image=False)
    clk.t += 0.040
    rec.lap("splice_up", sync=True, image=False, nbytes=8192 * 5120 * 2)
    rec.add("splice_n", 1)
    clk.t += 0.100  # trace replays etc.
    line = rec.line()
    kv = dict(p.split("=", 1) for p in line.split()[1:])
    assert line.startswith("[vision_timing] ")
    assert kv["images"] == "2" and kv["tower_images"] == "2" and kv["tokens"] == "1500"
    assert kv["grids"] == "1x44x44,1x42x46" and kv["patches"] == "1936,1932" and kv["buckets"] == "2048,2048"
    assert kv["host_rot_ms"] == "4.00,4.00" and kv["host_patch_ms"] == "12.00,12.00"
    assert kv["up_rot_ms"] == "12.00,12.00" and kv["up_rot_bytes"] == "1000,1000"  # 2 ms host + the 10 ms wait
    assert kv["tower_disp_ms"] == "90.00,90.00" and kv["tower_ms"] == "100.00,100.00"
    assert kv["merger_disp_ms"] == "4.00,4.00" and kv["merger_ms"] == "14.00,14.00"
    assert kv["splice_host_ms"] == "30.00" and kv["splice_up_ms"] == "50.00" and kv["splice_up_bytes"] == str(8192 * 5120 * 2)
    total = float(kv["total_ms"])
    assert abs(total - 1e3 * clk.t) < 1e-6
    # token prefill = the rest of the call: here the 100 ms after the splice
    assert abs(float(kv["token_prefill_ms"]) - 100.0) < 0.02
    assert len(syncs) == 2 * 4 + 1
    assert kv["flags"] == "none"
    # nothing but counts, sizes and times
    assert re.fullmatch(r"[\[\]a-z_ =0-9x.,\-A-Z]+", line)


def test_timing_begin_end_current(monkeypatch):
    lines, n = [], [0]
    rec = vf.timing_begin(sync=lambda: n.__setitem__(0, n[0] + 1))
    assert vf.timing_current() is rec and n[0] == 1 and "pre_sync_ms" in rec.req
    rec.note("images", 1)
    out = vf.timing_end(rec, lines.append)
    assert vf.timing_current() is None and n[0] == 2 and lines == [out]
    assert "end_sync_ms=" in out and "token_prefill_ms=" in out and "standalone=0" in out
    # standalone (warm-up / tower-only): no token-prefill figure
    rec = vf.timing_begin(standalone=True)
    out = vf.timing_end(rec)
    assert "standalone=1" in out and "token_prefill_ms" not in out
    # an aborted request leaves no open record and no line
    rec = vf.timing_begin()
    vf.timing_abort(rec)
    assert vf.timing_current() is None
    assert vf.timing_end(None) is None


def test_flags_summary(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_TIMING", "1")
    monkeypatch.setenv("QWEN36_VISION_GRID_CACHE", "16")
    monkeypatch.setenv("QWEN36_VISION_LEAN", "rotpad, hostref")
    assert vf.flags_summary() == "TIMING=1,GRID_CACHE=16,LEAN=rotpad,hostref"
