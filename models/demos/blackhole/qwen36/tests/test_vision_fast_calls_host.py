# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device; ttnn calls replaced by recording stand-ins) for the QWEN36_VISION_* levers in the tower loop
(tt/vision/model.py) and the splice staging (tt/model.py):

- flags unset: the tower loop and the splice make the release's device calls, in the release's order;
- each result-preserving lever returns the same rows / stages the same buffer content as flags-off, and removes the work
  it is meant to remove (table uploads, position interpolation, the 8,192-row upload, zero-mask uploads);
- the per-length splice pairs: the trace body and the staging pick the same pair.

The stand-in tower is a deterministic function of its uploaded inputs, so "same rows" means the same data reached it.
``tower_log`` / ``splice_log`` are also used from outside the repo to diff this tree against the release tree."""
import os
import types

import pytest
import torch

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
HEAD, PADDED, OUT_H, DIM = 72, 96, 16, 32
C, T, P = 3, 2, 4  # small patches: rows of C * T * P * P pixels


class Shape(tuple):
    @property
    def rank(self):
        return len(self)


class H:
    """Stand-in tensor: ``data`` (torch), where it lives, a name for the call log."""

    n = 0

    def __init__(self, data, dev, tag=""):
        H.n += 1
        self.id, self.data, self.dev, self.tag, self.freed = H.n, data, dev, tag, False

    @property
    def shape(self):
        return Shape(self.data.shape)

    def __getitem__(self, idx):
        return H(self.data[idx], self.dev, f"slice({self.tag})")


class FakeTtnn:
    def __init__(self):
        self.log = []
        self.bfloat16, self.TILE_LAYOUT, self.DRAM_MEMORY_CONFIG = "bf16", "tile", "dram"

    def install(self, monkeypatch):
        import ttnn

        for name in (
            "from_torch",
            "to_device",
            "to_torch",
            "deallocate",
            "synchronize_device",
            "copy_host_to_device_tensor",
            "ReplicateTensorToMesh",
            "ConcatMeshToTensor",
            "ShardTensor2dMesh",
            "where",
            "event_synchronize",
            "record_event",
        ):
            monkeypatch.setattr(ttnn, name, getattr(self, name), raising=False)
        for name in ("bfloat16", "TILE_LAYOUT", "DRAM_MEMORY_CONFIG"):
            monkeypatch.setattr(ttnn, name, getattr(self, name), raising=False)
        return self

    # -- recorded calls ----------------------------------------------------------------------------------------------
    def from_torch(self, t, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        h = H(t.detach().clone(), device is not None, "from_torch")
        self.log.append(("from_torch", "dev" if device is not None else "host", tuple(t.shape), mesh_mapper))
        return h

    def to_device(self, h, mesh, memory_config=None):
        self.log.append(("to_device", tuple(h.data.shape)))
        return H(h.data, True, "to_device")

    def to_torch(self, h, mesh_composer=None):
        assert not h.freed, "read of a freed tensor"
        self.log.append(("to_torch", tuple(h.data.shape)))
        return h.data.clone()

    def deallocate(self, h):
        h.freed = True
        self.log.append(("deallocate", h.tag))

    def synchronize_device(self, mesh):
        self.log.append(("sync",))

    def copy_host_to_device_tensor(self, src, dst):
        assert tuple(src.data.shape) == tuple(dst.data.shape), (src.data.shape, dst.data.shape)
        dst.data = src.data.clone()
        self.log.append(("copy", dst.tag, tuple(src.data.shape)))

    def where(self, mask, vis, x):
        self.log.append(("where", mask.tag, vis.tag))
        return H(torch.where(mask.data.bool(), vis.data, x.data), True, "where")

    def event_synchronize(self, e):
        self.log.append(("event_sync",))

    def record_event(self, mesh):
        return "event"

    ReplicateTensorToMesh = staticmethod(lambda mesh: "rep")
    ConcatMeshToTensor = staticmethod(lambda mesh, dim: ("concat", dim))
    ShardTensor2dMesh = staticmethod(lambda mesh, dims=None, mesh_shape=None: "shard")

    def calls(self, *kinds):
        return [c for c in self.log if c[0] in kinds]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    yield
    try:
        from models.demos.blackhole.qwen36.tt.vision import vision_fast as vf

        vf._current = None
    except ImportError:  # the release tree (outside-the-repo diff)
        pass


# ---- tower ---------------------------------------------------------------------------------------------------------
class _PatchEmbed(torch.nn.Module):  # the HF patch embedding: Conv3d with kernel == stride
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.proj = torch.nn.Conv3d(C, DIM, kernel_size=[T, P, P], stride=[T, P, P], bias=True)

    def forward(self, x):
        return self.proj(x.view(-1, C, T, P, P).to(self.proj.weight.dtype)).view(-1, DIM)


class _Ref:
    def __init__(self):
        self.patch_embed = _PatchEmbed()
        self.pos_calls = 0

    def fast_pos_embed_interpolate(self, grid_thw):
        self.pos_calls += 1
        t, h, w = (int(x) for x in grid_thw[0])
        return (torch.arange(t * h * w * DIM, dtype=torch.float32).reshape(-1, DIM) % 13) * 0.25 * (1 + h / 100.0)


class _Tower:
    """Stand-in VisionTransformer: output = a fixed function of the uploaded patch rows and rotary tables, with the
    per-block 72 -> 96 pad of vision_attention applied when the tables arrive 72 wide."""

    def __init__(self, fake):
        self.fake = fake

    def prepare_input(self, patch_input, seq_len):
        x = torch.nn.functional.pad(patch_input.detach(), (0, 0, 0, seq_len - patch_input.shape[0]))
        self.fake.log.append(("prepare_input", int(seq_len)))
        return H(x.to(torch.bfloat16), True, "tt_input")

    def __call__(self, tt_input, unpadded_seq_len, rot_mats, trim_before_merger=True, _mark=None):
        cos, sin = rot_mats
        assert not (tt_input.freed or cos.freed or sin.freed)
        c, s = cos.data.float(), sin.data.float()
        if c.shape[-1] != PADDED:  # vision_attention's device pad
            c = torch.nn.functional.pad(c, (0, PADDED - c.shape[-1]), value=1.0)
            s = torch.nn.functional.pad(s, (0, PADDED - s.shape[-1]), value=0.0)
        self.fake.log.append(("tower", int(tt_input.data.shape[0]), int(cos.data.shape[-1]), bool(trim_before_merger)))
        if _mark is not None:
            _mark("tower")
        x = tt_input.data.float()
        rows = x.shape[0] // 4
        mix = (c[0, 0].sum(-1) + 2 * s[0, 0].sum(-1))[: rows * 4].reshape(rows, 4).sum(-1, keepdim=True)
        out = x[: rows * 4].reshape(rows, 4, -1).sum(1)[:, :OUT_H] + mix
        if _mark is not None:
            _mark("merger")
        return H(out.reshape(1, 1, rows, OUT_H).to(torch.bfloat16), True, "tt_out")


def _tower(monkeypatch):
    from models.demos.blackhole.qwen36.tt.vision import model as vmod

    fake = FakeTtnn().install(monkeypatch)
    monkeypatch.setattr(vmod, "convert_rope_style_hf_to_meta", lambda c, s: (c, s))
    d = object.__new__(vmod.DropInVisionTransformer)
    torch.nn.Module.__init__(d)
    d.reference_model = _Ref()
    d.debug = False
    d.tt_model = _Tower(fake)
    vc = types.SimpleNamespace(spatial_merge_size=2, out_hidden_size=OUT_H)
    d.model_args = types.SimpleNamespace(
        mesh_device=types.SimpleNamespace(get_num_devices=lambda: 8),
        head_dim=HEAD,
        padded_head_dim=PADDED,
        hf_config=types.SimpleNamespace(vision_config=vc),
    )
    return d, fake


def _images(grids, seed=3):
    g = torch.tensor(grids, dtype=torch.int32)
    torch.manual_seed(seed)
    return torch.randn(int(g.prod(dim=1).sum()), C * T * P * P), g


GRIDS2 = [[1, 44, 44], [1, 30, 32]]


def tower_log(monkeypatch, grids=GRIDS2):
    """(per-image rows, device-call log) of one host_trim tower call."""
    d, fake = _tower(monkeypatch)
    px, g = _images(grids)
    with torch.no_grad():
        rows = d.forward(px, g, host_trim=True, return_host_rows=True)
    return rows, fake.log, d


def test_tower_flags_off_is_the_release_call_sequence(monkeypatch):
    rows, log, d = tower_log(monkeypatch)
    per_image = lambda bucket: [
        ("from_torch", "dev", (1, 1, bucket, HEAD), "rep"),
        ("from_torch", "dev", (1, 1, bucket, HEAD), "rep"),
        ("prepare_input", bucket),
        ("tower", bucket, HEAD, False),
        ("deallocate", "tt_input"),
        ("deallocate", "from_torch"),
        ("deallocate", "from_torch"),
        ("deallocate", "from_torch"),
        ("deallocate", "from_torch"),
        ("to_torch", (1, 1, bucket // 4, OUT_H)),
        ("deallocate", "tt_out"),
    ]
    assert log == per_image(2048) + per_image(2048)  # 1,936 and 960 patches: both in the 2,048 bucket
    assert [tuple(r.shape) for r in rows] == [(484, OUT_H), (240, OUT_H)] and all(r.dtype == torch.bfloat16 for r in rows)
    assert d.reference_model.pos_calls == 2


def _same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def test_grid_cache_same_rows_and_skips_host_work(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_GRID_CACHE", "4")
    d, fake = _tower(monkeypatch)
    px, g = _images(GRIDS2)
    with torch.no_grad():
        first = d.forward(px, g, host_trim=True, return_host_rows=True)
        n_host = len([c for c in fake.calls("from_torch") if c[1] == "host"])
        again = d.forward(px, g, host_trim=True, return_host_rows=True)
    assert _same(first, base) and _same(again, base)
    assert d.reference_model.pos_calls == 2  # second call: both grids hit
    assert n_host == 4 and len([c for c in fake.calls("from_torch") if c[1] == "host"]) == 4  # no new tilize on a hit
    assert len(fake.calls("to_device")) == 8  # the tables are still uploaded per image (nothing kept on device)
    assert d._grid_cache.hits == 2 and d._grid_cache.misses == 2
    # a different image with a known grid: its own rows, constants from the cache
    px2, _ = _images(GRIDS2, seed=9)
    with torch.no_grad():
        other = d.forward(px2, g, host_trim=True, return_host_rows=True)
    assert not _same(other, base) and d.reference_model.pos_calls == 2


def test_grid_cache_keeps_buckets_apart(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_GRID_CACHE", "4")
    d, fake = _tower(monkeypatch)
    px, g = _images([[1, 30, 32]])
    with torch.no_grad():
        a = d.forward(px, g, host_trim=True, return_host_rows=True)
        monkeypatch.setenv("QWEN36_VISION_BUCKET_1024", "1")
        b = d.forward(px, g, host_trim=True, return_host_rows=True)
    assert [c[1] for c in fake.calls("tower")] == [2048, 1024]  # not the cached 2,048-row tables
    assert d._grid_cache.misses == 2 and a[0].shape == b[0].shape == (240, OUT_H)


def test_resident_tables_same_rows_no_upload_no_free(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    d, fake = _tower(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_RESIDENT_GRIDS", "1x44x44")  # read by warmup_vision; staged by hand here
    keys = d.stage_resident_rot([(1, 44, 44)])
    assert keys == [(1, 44, 44, 2048, HEAD)]
    staged = list(fake.log)
    assert [c[:2] for c in staged] == [("from_torch", "dev")] * 2
    px, g = _images(GRIDS2)
    with torch.no_grad():
        for _ in range(2):
            rows = d.forward(px, g, host_trim=True, return_host_rows=True)
            assert _same(rows, base)
    cos, sin = d._resident_rot[keys[0]]
    assert not cos.freed and not sin.freed
    # per call: only the second image (not listed) uploads its two tables
    assert len(fake.calls("to_device")) == 2 * 2
    assert d.stage_resident_rot([(1, 44, 44)]) == keys and len(fake.log) > len(staged)  # idempotent


def test_rotpad_same_rows_wide_tables(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_LEAN", "rotpad")
    rows, log, _ = tower_log(monkeypatch)
    assert _same(rows, base)
    assert [c[2] for c in log if c[0] == "tower"] == [PADDED, PADDED]  # nothing left for the blocks to pad


def test_pipeline_same_rows_reads_after_all_dispatches(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_PIPELINE", "1")
    rows, log, _ = tower_log(monkeypatch)
    assert _same(rows, base)
    kinds = [c[0] for c in log if c[0] in ("tower", "to_torch")]
    assert kinds == ["tower", "tower", "to_torch", "to_torch"]
    # one image: nothing to overlap, the read-back follows its tower
    rows1, log1, _ = tower_log(monkeypatch, grids=[[1, 44, 44]])
    assert [c[0] for c in log1 if c[0] in ("tower", "to_torch")] == ["tower", "to_torch"] and _same(rows1, base[:1])


def test_patch_linear_rows_close(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_PATCH_LINEAR", "1")
    rows, _, _ = tower_log(monkeypatch)
    for a, b in zip(rows, base):
        assert a.shape == b.shape and torch.allclose(a.float(), b.float(), rtol=2e-2, atol=2e-2)


def test_all_result_preserving_levers_together(monkeypatch):
    base, _, _ = tower_log(monkeypatch)
    monkeypatch.setenv("QWEN36_VISION_GRID_CACHE", "8")
    monkeypatch.setenv("QWEN36_VISION_PIPELINE", "1")
    monkeypatch.setenv("QWEN36_VISION_LEAN", "1")
    d, fake = _tower(monkeypatch)
    d.stage_resident_rot([(1, 44, 44)])
    px, g = _images(GRIDS2)
    with torch.no_grad():
        for _ in range(3):
            assert _same(d.forward(px, g, host_trim=True, return_host_rows=True), base)


def test_bucket_flags_change_the_tower_rows_only_where_expected(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_BUCKET_EXACT", "1")
    monkeypatch.setenv("QWEN36_VISION_BUCKET_1024", "1")
    _, log, _ = tower_log(monkeypatch, grids=[[1, 44, 44], [1, 30, 32], [1, 32, 32], [1, 64, 64]])
    assert [c[1] for c in log if c[0] == "tower"] == [2048, 1024, 1024, 4096]
    for f in ("QWEN36_VISION_BUCKET_EXACT", "QWEN36_VISION_BUCKET_1024"):
        monkeypatch.delenv(f)
    _, log, _ = tower_log(monkeypatch, grids=[[1, 44, 44], [1, 30, 32], [1, 32, 32], [1, 64, 64]])
    assert [c[1] for c in log if c[0] == "tower"] == [2048, 2048, 2048, 6144]


def test_timing_line_from_the_tower(monkeypatch):
    from loguru import logger

    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        base, base_log, _ = tower_log(monkeypatch)
        monkeypatch.setenv("QWEN36_VISION_TIMING", "1")
        rows, log, _ = tower_log(monkeypatch)
        monkeypatch.setenv("QWEN36_VISION_PIPELINE", "1")
        rows_p, log_p, _ = tower_log(monkeypatch)
    finally:
        logger.remove(sink)
    assert _same(rows, base) and _same(rows_p, base)
    vt = [ln for ln in lines if "[vision_timing]" in ln]
    assert len(vt) == 2 and all("standalone=1" in ln and "grids=1x44x44,1x30x32" in ln for ln in vt)
    for key in ("host_rot_ms=", "host_patch_ms=", "host_pos_ms=", "up_rot_ms=", "up_patch_ms=", "tower_ms=", "merger_ms="):
        assert all(key in ln for ln in vt), key
    assert "readback_ms=" in vt[0] and "buckets=2048,2048" in vt[0] and "up_patch_bytes=" in vt[0]
    # pre-sync + 4 stage waits per image + end-sync; the device calls between them are the release's
    assert len([c for c in log if c[0] == "sync"]) == 1 + 2 * 4 + 1
    assert [c for c in log if c[0] != "sync"] == base_log
    assert len([c for c in log_p if c[0] == "sync"]) == 2  # PIPELINE: no per-stage wait


# ---- splice --------------------------------------------------------------------------------------------------------
IMG = 7


def _model(monkeypatch, big=8192, chunk=2048):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    fake = FakeTtnn().install(monkeypatch)
    m = types.SimpleNamespace(
        _vis_buf=None,
        _vis_mask_buf=None,
        _vis_zero_mask_host=None,
        _vis_set=None,
        _vis_host_refs=None,
        num_devices=8,
        mesh_device="mesh",
        args=types.SimpleNamespace(dim=DIM, cluster_shape=(1, 8)),
        _PREFILL_MASK_BUCKETS=Qwen36Model._PREFILL_MASK_BUCKETS,
        _chunked_chunk_size=chunk,
    )
    m._vision_placeholder_token_id = lambda: IMG
    m.vision_rows_host = lambda vt: Qwen36Model.vision_rows_host(m, vt)
    Qwen36Model._alloc_vision_merge_buffers(m, "dev", max(chunk, big))
    pairs = {None: (m._vis_buf, m._vis_mask_buf, None)} if m._vis_set is None else m._vis_set.pairs  # None: release tree
    for key, (vis, mask, _zero) in pairs.items():
        vis.tag, mask.tag = f"vis{key}", f"mask{key}"
    fake.log.clear()
    return Qwen36Model, m, fake


def _segment(n_tokens=1500, img=(100, 484)):
    ids = torch.full((1, n_tokens), 11, dtype=torch.int32)
    ids[0, img[0] : img[0] + img[1]] = IMG
    rows = (torch.arange(img[1] * DIM, dtype=torch.float32).reshape(img[1], DIM) % 97).to(torch.bfloat16)
    return ids, rows


def splice_log(monkeypatch, length=2048):
    """Device calls of staging one image segment, then a text segment, as the six call sites do (length given)."""
    Q, m, fake = _model(monkeypatch)
    ids, rows = _segment()
    try:
        Q._set_vision_merge(m, ids, rows, 0, length=length)
        Q._set_vision_merge(m, ids, None, 0, length=length)
    except TypeError:  # the release tree: no length argument
        Q._set_vision_merge(m, ids, rows, 0)
        Q._set_vision_merge(m, ids, None, 0)
    return fake.log, m


def test_splice_flags_off_is_the_release_call_sequence(monkeypatch):
    log, m = splice_log(monkeypatch)
    assert sorted(m._vis_set.pairs, key=str) == [None]
    assert log == [
        ("from_torch", "host", (1, 1, 8192, DIM), "shard"),
        ("copy", "visNone", (1, 1, 8192, DIM)),
        ("from_torch", "host", (1, 1, 8192, 1), "rep"),
        ("copy", "maskNone", (1, 1, 8192, 1)),
        ("copy", "maskNone", (1, 1, 8192, 1)),  # the text segment's zero mask
    ]
    assert m._vis_host_refs is None


def test_splice_seg_uploads_the_bucket_and_same_rows(monkeypatch):
    Q, m0, _ = _model(monkeypatch)
    ids, rows = _segment()
    Q._set_vision_merge(m0, ids, rows, 0, length=2048)
    ref_vis, ref_mask = m0._vis_buf.data.clone(), m0._vis_mask_buf.data.clone()

    monkeypatch.setenv("QWEN36_VISION_SPLICE_SEG", "1")
    Q, m, fake = _model(monkeypatch)
    assert sorted(k for k in m._vis_set.pairs if k is not None) == [128, 256, 512, 1024, 2048]
    Q._set_vision_merge(m, ids, rows, 0, length=2048)
    assert fake.log == [
        ("from_torch", "host", (1, 1, 2048, DIM), "shard"),
        ("copy", "vis2048", (1, 1, 2048, DIM)),
        ("from_torch", "host", (1, 1, 2048, 1), "rep"),
        ("copy", "mask2048", (1, 1, 2048, 1)),
    ]
    vis, mask, _ = m._vis_set.get(2048)
    # the rows a 2,048-row replay reads are the release's
    assert torch.equal(vis.data, ref_vis[:, :, :2048]) and torch.equal(mask.data, ref_mask[:, :, :2048])
    assert not bool(m._vis_buf.data.any()) and not bool(m._vis_mask_buf.data.any())  # the 8,192-row pair is untouched
    # the trace body of that length reads the same pair, whole (no slice); another length keeps the release pair
    fake.log.clear()
    x = H(torch.ones(1, 1, 2048, DIM), True, "x")
    out = Q._apply_vision_merge(m, x, 2048)
    assert fake.calls("where") == [("where", "mask2048", "vis2048")]
    assert torch.equal(out.data[0, 0, 100:584].to(torch.bfloat16), rows) and bool((out.data[0, 0, :100] == 1).all())
    fake.log.clear()
    Q._apply_vision_merge(m, H(torch.ones(1, 1, 8192, DIM), True, "x"), 8192)
    Q._apply_vision_merge(m, H(torch.ones(1, 1, 1536, DIM), True, "x"), 1536)
    assert fake.calls("where") == [("where", "maskNone", "visNone"), ("where", "slice(maskNone)", "slice(visNone)")]


def test_splice_seg_text_replay_clears_what_an_image_left(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_SPLICE_SEG", "1")
    Q, m, fake = _model(monkeypatch)
    ids, rows = _segment()
    Q._set_vision_merge(m, ids, rows, 0, length=2048)
    fake.log.clear()
    Q._set_vision_merge(m, ids[:, :512], None, 0, length=512)  # a text request on another bucket
    assert sorted(fake.calls("copy")) == [("copy", "mask2048", (1, 1, 2048, 1)), ("copy", "mask512", (1, 1, 512, 1))]
    assert not bool(m._vis_set.get(2048)[1].data.any()) and m._vis_set.dirty == set()
    fake.log.clear()
    Q._set_vision_merge(m, ids[:, :512], None, 0, length=512)
    assert fake.calls("copy") == [("copy", "mask512", (1, 1, 512, 1))]  # release behaviour: a clear per text replay
    # an unknown length, or a caller without one, stages the release pair (which the trace of that length reads)
    fake.log.clear()
    Q._set_vision_merge(m, ids, rows, 0, length=1536)
    Q._set_vision_merge(m, ids, rows, 0)
    assert [c[1] for c in fake.calls("copy")] == ["visNone", "maskNone", "visNone", "maskNone"]
    # a segment without placeholders is a clear
    fake.log.clear()
    Q._set_vision_merge(m, torch.full((1, 2048), 11, dtype=torch.int32), rows, 0, length=2048)
    assert ("copy", "mask2048", (1, 1, 2048, 1)) in fake.log and not fake.calls("from_torch")


def test_maskskip_and_hostref(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_LEAN", "maskskip,hostref")
    Q, m, fake = _model(monkeypatch)
    ids, rows = _segment()
    Q._set_vision_merge(m, ids, None, 0, length=2048)
    assert fake.log == []  # allocated zero: nothing to upload
    Q._set_vision_merge(m, ids, rows, 0, length=2048)
    assert [c[0] for c in fake.log] == ["from_torch", "copy", "from_torch", "copy"]  # the release's order
    assert m._vis_host_refs is not None and len(m._vis_host_refs) == 2
    fake.log.clear()
    Q._set_vision_merge(m, ids, None, 0, length=2048)
    Q._set_vision_merge(m, ids, None, 0, length=2048)
    assert fake.calls("copy") == [("copy", "maskNone", (1, 1, 8192, 1))]  # once after the image, then known zero
    assert not bool(m._vis_mask_buf.data.any())


def test_splice_timing_counts(monkeypatch):
    from models.demos.blackhole.qwen36.tt.vision import vision_fast as vf

    monkeypatch.setenv("QWEN36_VISION_SPLICE_SEG", "1")
    Q, m, fake = _model(monkeypatch)
    ids, rows = _segment()
    rec = vf.timing_begin(sync=lambda: fake.log.append(("sync",)))
    Q._set_vision_merge(m, ids, rows, 0, length=2048)
    Q._set_vision_merge(m, ids, None, 0, length=2048)
    line = vf.timing_end(rec)
    kv = dict(p.split("=", 1) for p in line.split()[1:])
    assert kv["splice_n"] == "1" and kv["splice_rows"] == "484" and kv["splice_len"] == "2048"
    assert kv["splice_up_bytes"] == str(2048 * DIM * 2 + 2048 * 2 * 8) and kv["mask_clear_n"] == "1"
    assert [c for c in fake.log if c[0] != "sync"][:4] == [
        ("from_torch", "host", (1, 1, 2048, DIM), "shard"),
        ("copy", "vis2048", (1, 1, 2048, DIM)),
        ("from_torch", "host", (1, 1, 2048, 1), "rep"),
        ("copy", "mask2048", (1, 1, 2048, 1)),
    ]
    assert os.environ.get("QWEN36_VISION_TIMING") is None  # a record opened by the caller is enough


# ---- warm-up, bucket guard, hostrows (tt/qwen36_vllm.py) -------------------------------------------------------------
def _gen(monkeypatch, max_pixels=512000):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    FakeTtnn().install(monkeypatch)
    warmed, staged = [], []
    vc = types.SimpleNamespace(patch_size=16, in_channels=3, temporal_patch_size=2)
    model = types.SimpleNamespace(
        vision_model=types.SimpleNamespace(stage_resident_rot=lambda grids: staged.extend(grids) or list(grids)),
        vision_args=types.SimpleNamespace(hf_config=types.SimpleNamespace(vision_config=vc)),
        get_image_features=lambda pix, grid: warmed.append(tuple(int(x) for x in grid[0])) or H(pix[:1], True, "feat"),
    )
    gen = object.__new__(qv.Qwen36ForCausalLM)
    gen.model = [model]
    gen.mesh_device = "mesh"
    gen._vllm_mm_max_pixels = max_pixels
    return gen, warmed, staged


def test_warmup_flags_off_is_the_release_grid_list(monkeypatch):
    gen, warmed, staged = _gen(monkeypatch)
    gen.warmup_vision()
    assert warmed == [(1, 32, 32), (1, 48, 64), (1, 64, 64), (1, 80, 80)] and staged == []
    assert gen._vision_warmed_max_bucket == 8192 and not hasattr(gen, "_vision_warmed_buckets")
    gen._check_vision_bucket(torch.tensor([[1, 44, 44]]))
    with pytest.raises(ValueError, match="max_pixels"):
        gen._check_vision_bucket(torch.tensor([[1, 100, 100]]))


@pytest.mark.parametrize("exact,b1024,extra", [("1", "0", {6144}), ("0", "1", {1024}), ("1", "1", {2048, 6144})])
def test_warmup_with_bucket_flags_warms_every_bucket(monkeypatch, exact, b1024, extra):
    from models.demos.blackhole.qwen36.tt.vision import vision_fast as vf

    monkeypatch.setenv("QWEN36_VISION_BUCKET_EXACT", exact)
    monkeypatch.setenv("QWEN36_VISION_BUCKET_1024", b1024)
    gen, warmed, _ = _gen(monkeypatch)
    gen.warmup_vision()
    assert warmed[:4] == [(1, 32, 32), (1, 48, 64), (1, 64, 64), (1, 80, 80)]
    want = {2048, 4096, 6144, 8192} | ({1024} if b1024 == "1" else set())
    assert gen._vision_warmed_buckets == want and gen._vision_warmed_max_bucket == 8192
    assert extra <= {vf.vision_bucket(t * h * w) for t, h, w in warmed[4:]}
    for grid in ((1, 44, 44), (1, 30, 32), (1, 32, 32), (1, 64, 64), (1, 62, 64)):
        gen._check_vision_bucket(torch.tensor([grid]))  # every bucket a px512 / 1 MP image can land in is warmed
    # a bucket below the maximum that nothing warmed is refused before the engine runs
    gen._vision_warmed_buckets = want - {4096}
    with pytest.raises(ValueError, match="not warmed"):
        gen._check_vision_bucket(torch.tensor([[1, 48, 64]]))


def test_warmup_stages_resident_grids_before_the_first_tower_call(monkeypatch):
    monkeypatch.setenv("QWEN36_VISION_RESIDENT_GRIDS", "1x44x44;1x42x46")
    gen, warmed, staged = _gen(monkeypatch)
    order = []
    model = gen.model[0]
    stage, feat = model.vision_model.stage_resident_rot, model.get_image_features
    model.vision_model.stage_resident_rot = lambda g: order.append("stage") or stage(g)
    model.get_image_features = lambda p, g: order.append("tower") or feat(p, g)
    gen.warmup_vision()
    assert staged == [(1, 44, 44), (1, 42, 46)] and order[0] == "stage" and order.count("stage") == 1


def test_hostrows_takes_host_rows_directly(monkeypatch):
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv

    calls = []
    gen = object.__new__(qv.Qwen36MTPContractForCausalLM)
    rows = torch.ones(4, 8, dtype=torch.bfloat16)

    def compute(model, kwargs, i, **kw):
        calls.append(kw)
        return rows if kw.get("host_rows") else "DEVICE_ROWS"

    gen._compute_vision_tokens = compute
    gen._mm_vision_for = lambda *a, **k: None
    model = types.SimpleNamespace(vision_rows_host=lambda vt: rows if vt == "DEVICE_ROWS" else vt)
    seen = []
    eng = types.SimpleNamespace(prefill=lambda *a, **k: seen.append(k.get("vision_tokens")) or torch.zeros(5))
    args = (eng, model, {}, 0, 3, [0], torch.zeros(1, 3, dtype=torch.int32), torch.zeros(1, 4), None, [False], True)
    gen._prefill_row(*args)
    assert calls == [{}] and seen[-1] is rows  # release: device rows, read back by vision_rows_host
    monkeypatch.setenv("QWEN36_VISION_LEAN", "hostrows")
    gen._prefill_row(*args)
    assert calls[-1] == {"host_rows": True} and seen[-1] is rows
