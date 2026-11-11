# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the per-item HOST vision cache (Qwen36ForCausalLM._vision_cached, QWEN36_MM_VISION_CACHE=N).

The device-tensor cache of 8c2faf2ec95 returned the same device tensor on a hit while prefill_paged_slots deallocated
it after the prefill: hits spliced freed DRAM. Now the cache holds host rows per item and every call uploads a fresh
device tensor; a request that adds an item hits on the items it already sent."""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

H = 4


def _rows(tag, n):
    return torch.full((n, H), float(tag), dtype=torch.bfloat16)


def _model():
    m = types.SimpleNamespace(tower=[], uploads=[], _req_image_grid_thw=None, _req_video_grid_thw=None)

    def host(modality, pv, grids):
        # one item = rows tagged by the item's first pixel value; n_tokens = patches / 4 (2x2 merge)
        m.tower.append((modality, grids.tolist()))
        sizes = [int(g.prod()) for g in grids]
        out, off = [], 0
        for n in sizes:
            out.append(_rows(pv[off, 0].item(), n // 4))
            off += n
        m._req_image_grid_thw = grids
        return out

    def upload(rows):
        t = types.SimpleNamespace(rows=rows.clone(), uid=len(m.uploads))
        m.uploads.append(t)
        return t

    m.get_visual_features_host = host
    m.upload_vision_rows = upload
    return m


def _req(tags, n=16):
    """One user's items: item k has n patches all equal to tags[k]; grid (1, 4, n/4)."""
    pv = torch.cat([torch.full((n, 3), float(t)) for t in tags])
    grid = torch.tensor([[1, 4, n // 4]] * len(tags), dtype=torch.int32)
    return pv, grid


def _call(g, m, tags, ids=None, monkeypatch=None):
    return Qwen36ForCausalLM._vision_cached(
        g, m, "image", _req(tags), lambda pv, grid: pytest.fail("encode() must not run with the cache on"), ids=ids
    )


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.setenv("QWEN36_MM_VISION_CACHE", "8")
    return Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)


def test_hit_returns_a_fresh_upload_with_the_same_rows(gen):
    m = _model()
    a = _call(gen, m, [1, 2])
    b = _call(gen, m, [1, 2])
    assert len(m.tower) == 1  # second call: all hits, no tower
    assert a is not b and a.uid != b.uid  # never the same device tensor: the caller may free it
    assert torch.equal(a.rows, b.rows) and a.rows[:, 0].tolist() == [1.0] * 4 + [2.0] * 4


def test_interleaved_other_request_does_not_disturb_hits(gen):
    m = _model()
    first = _call(gen, m, [1, 2]).rows
    other = _call(gen, m, [3, 4]).rows
    again = _call(gen, m, [1, 2]).rows
    assert torch.equal(first, again) and not torch.equal(first, other)
    assert [t[1] for t in m.tower] == [[[1, 4, 4], [1, 4, 4]], [[1, 4, 4], [1, 4, 4]]]


def test_adding_an_image_only_encodes_the_new_one(gen):
    m = _model()
    _call(gen, m, [1, 2])
    out = _call(gen, m, [1, 2, 5]).rows
    assert len(m.tower) == 2 and m.tower[1][1] == [[1, 4, 4]]  # only image 5 ran
    assert out[:, 0].tolist() == [1.0] * 4 + [2.0] * 4 + [5.0] * 4  # in request order
    assert m._req_image_grid_thw.tolist() == [[1, 4, 4]] * 3  # full grid list staged for M-RoPE


def test_order_follows_the_request(gen):
    m = _model()
    _call(gen, m, [1, 2])
    assert _call(gen, m, [2, 1]).rows[:, 0].tolist() == [2.0] * 4 + [1.0] * 4


def test_content_ids_key_the_items(gen):
    m = _model()
    _call(gen, m, [1], ids=["hash-x"])
    # same pixels, different content hash (e.g. different processor kwargs): a miss, never a wrong hit
    _call(gen, m, [1], ids=["hash-y"])
    assert len(m.tower) == 2
    _call(gen, m, [1], ids=["hash-x"])
    assert len(m.tower) == 2


def test_lru_is_bounded_in_items(gen, monkeypatch):
    monkeypatch.setenv("QWEN36_MM_VISION_CACHE", "2")
    m = _model()
    _call(gen, m, [1, 2])
    _call(gen, m, [3])  # evicts item 1
    _call(gen, m, [2])
    assert len(m.tower) == 2
    _call(gen, m, [1])
    assert len(m.tower) == 3 and len(gen._vision_lru) == 2


def test_off_calls_encode_unchanged(monkeypatch):
    monkeypatch.setenv("QWEN36_MM_VISION_CACHE", "0")
    g = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    seen = []
    out = Qwen36ForCausalLM._vision_cached(g, _model(), "image", _req([1]), lambda pv, grid: seen.append(1) or "ROWS")
    assert out == "ROWS" and seen == [1]


def test_upload_waits_for_the_write_before_releasing_the_host_rows(monkeypatch):
    """rc1e d30u20 11-11: an in-flight host->device write read freed host memory (last image wrong). upload_rows must
    record an event after the write and block on it before returning (the host sources die on return)."""
    import ttnn

    from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer

    order = []
    monkeypatch.setattr(ttnn, "from_torch", lambda rows, **k: order.append(("host", tuple(rows.shape))) or "HOST")
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda mesh, dim: ("shard", dim))
    monkeypatch.setattr(ttnn, "to_device", lambda h, mesh, memory_config=None: order.append(("write", h)) or "DEV")
    monkeypatch.setattr(ttnn, "record_event", lambda mesh, *a, **k: order.append(("record", mesh)) or "EV")
    monkeypatch.setattr(ttnn, "event_synchronize", lambda ev: order.append(("wait", ev)))
    me = types.SimpleNamespace(model_args=types.SimpleNamespace(mesh_device="MESH"))
    out = DropInVisionTransformer.upload_rows(me, torch.zeros(1024, H, dtype=torch.bfloat16))
    assert out == "DEV"
    assert order == [("host", (1024, H)), ("write", "HOST"), ("record", "MESH"), ("wait", "EV")]
