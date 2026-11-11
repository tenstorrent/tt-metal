# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 media split as tt-d-gen runs it (encoder service -> embedding store -> spliced prefill), host only:
the stored value's header (grid for M-RoPE), the adapter's splice and the PrefillMediaRef layout, the runner's fetch +
checks, M-RoPE positions from the stored grids vs HF get_rope_index, and the service emulation's request path (limits,
resize by the request's image count, media keys, data-parallel workers, cache hits) with stub embeddings. The Mooncake
round trip runs only where Mooncake's client is installed."""

import io
import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.vision import format as media_format
from models.demos.blackhole.qwen36.vision import splice
from models.demos.blackhole.qwen36.vision.media import Qwen35Media
from models.demos.blackhole.qwen36.vision.service_emu import DictStore, EncodeWorker, Front, Limits, LocalWorker

IMAGE_PAD = 248056
CONFIG = {
    "image_token_id": IMAGE_PAD,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "vision_config": {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 64},
}
HIDDEN = 64


def png(h, w, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([x / w, y / h, (x + y) / (h + w)], axis=-1) * 200
    arr = np.clip(base + rng.normal(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def stub_media():
    m = Qwen35Media(CONFIG)
    m.load(None, "", "image", {"stub_embeddings": True})
    return m


def service(k_groups, store=None, batch=False):
    store = store if store is not None else DictStore()
    workers = [LocalWorker(EncodeWorker(stub_media(), store)) for _ in range(k_groups)]
    return Front(workers, IMAGE_PAD, Limits(), batch=batch), store


# ---- stored value -------------------------------------------------------------------------------------------------


def test_header_round_trip_and_value_size():
    h = media_format.Header(n_tokens=6 * 8, hidden=HIDDEN, grid_thw=(1, 12, 16), n_refs=3)
    buf = memoryview(bytearray(media_format.stored_nbytes(h.n_tokens, HIDDEN)))
    assert len(buf) == 64 + 48 * HIDDEN * 2
    media_format.embeddings(buf, HIDDEN)[:] = 7
    media_format.write_header(buf, h)
    got, rows = media_format.parse(buf)
    assert got == h and rows.shape == (48, HIDDEN) and (rows == 7).all()
    assert rows.ctypes.data == np.frombuffer(buf, np.uint8).ctypes.data + 64  # in place


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: b.__setitem__(slice(0, 4), b"XXXX"), "not a Qwen3.5"),
        (lambda b: b.__setitem__(slice(4, 6), (2).to_bytes(2, "little")), "not a Qwen3.5"),
        (lambda b: b.__setitem__(slice(6, 8), (2).to_bytes(2, "little")), "only bf16 images"),
        (lambda b: b.__setitem__(slice(24, 28), (14).to_bytes(4, "little")), "holds"),
    ],
)
def test_header_rejects(expect_error, mutate, message):
    h = media_format.Header(n_tokens=48, hidden=HIDDEN, grid_thw=(1, 12, 16))
    buf = bytearray(media_format.stored_nbytes(48, HIDDEN))
    media_format.write_header(memoryview(buf), h)
    mutate(buf)
    with expect_error(ValueError, message):
        media_format.read_header(buf)


def test_header_rejects_size_mismatch(expect_error):
    h = media_format.Header(n_tokens=48, hidden=HIDDEN, grid_thw=(1, 12, 16))
    buf = bytearray(media_format.stored_nbytes(48, HIDDEN))
    media_format.write_header(memoryview(buf), h)
    with expect_error(ValueError, "its header says"):
        media_format.read_header(buf + bytes(HIDDEN * 2))
    with expect_error(ValueError, "shorter than"):
        media_format.read_header(buf[:10])
    with expect_error(ValueError, "not a positive grid"):
        media_format.Header(n_tokens=48, hidden=HIDDEN, grid_thw=(1, 13, 16))


def test_stub_encode_stores_the_grid():
    media = stub_media()
    from models.demos.blackhole.qwen36.vision.service_emu import decode_image

    item = media.prepare("image", decode_image(png(300, 500)), n_refs=2)
    buf = memoryview(bytearray(media.value_size(item)))
    media.encode([item], [buf])
    h, rows = media_format.parse(buf)
    assert h.grid_thw == item.grid_thw and h.n_tokens == item.n_tokens and h.n_refs == 2
    assert rows.shape == (item.n_tokens, HIDDEN) and rows.any()


# ---- adapter splice / refs --------------------------------------------------------------------------------------


def test_expand_spans_and_refs(expect_error):
    template = [1, 2, 248053, IMAGE_PAD, 248054, 3, 248053, IMAGE_PAD, 248054, 4]
    ids, spans = splice.expand(template, IMAGE_PAD, [[(IMAGE_PAD, 4)], [(IMAGE_PAD, 6)]])
    assert spans == [(3, 7), (10, 16)]
    assert ids[3:7] == [IMAGE_PAD] * 4 and ids[10:16] == [IMAGE_PAD] * 6 and ids[-1] == 4 and len(ids) == 18
    with expect_error(ValueError, "2 image markers in the prompt for 1 images"):
        splice.expand(template, IMAGE_PAD, [[(IMAGE_PAD, 4)]])
    keys = [bytes([1]) * 32, bytes([2]) * 32]
    assert [r.media_key for r in splice.refs_for_chunk(0, 0, 8, spans, keys)] == keys[:1]
    assert [r.media_key for r in splice.refs_for_chunk(1, 6, 11, spans, keys)] == keys
    assert splice.refs_for_chunk(2, 16, 18, spans, keys) == []
    assert [r.media_key for r in splice.refs_for_chunk(1, 7, 10, spans, keys)] == []
    r = splice.MediaRef(3, 10, 16, keys[1])
    raw = r.pack()
    assert len(raw) == splice.REF_BYTES == 44 and splice.MediaRef.unpack(raw) == r


def _hf_rope_index(ids, grid):
    """HF Qwen3.5 get_rope_index on one prompt (mm_token_type_ids = image positions)."""
    from types import SimpleNamespace

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    class _Stub:
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
        get_rope_index = Qwen3_5Model.get_rope_index

    t = torch.tensor([ids])
    pos, delta = _Stub().get_rope_index(t, (t == IMAGE_PAD).int(), image_grid_thw=torch.tensor(grid))
    return pos[:, 0].numpy(), int(delta[0, 0])


@pytest.mark.parametrize("grids", [[(1, 32, 32)], [(1, 44, 44), (1, 44, 44)], [(1, 32, 48), (1, 48, 32), (1, 64, 64)]])
def test_mrope_from_stored_grids_matches_hf(grids):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    template = [5, 6, 7]
    for _ in grids:
        template += [248053, IMAGE_PAD, 248054, 8, 9]
    ids, _ = splice.expand(template, IMAGE_PAD, [[(IMAGE_PAD, t * h * w // 4)] for t, h, w in grids])
    want, delta = _hf_rope_index(ids, grids)
    got = splice.mrope_positions(ids, grids, IMAGE_PAD)
    assert np.array_equal(got, want)
    assert splice.rope_delta(ids, grids, IMAGE_PAD) == delta
    # a transposed grid (same token count) gives different positions: why the grid must reach the runner
    if grids[0][1] != grids[0][2]:
        swapped = [(t, w, h) for t, h, w in grids]
        assert not np.array_equal(splice.mrope_positions(ids, swapped, IMAGE_PAD), got)


# ---- the service's request path ----------------------------------------------------------------------------------


@pytest.mark.parametrize("k_groups", [1, 2, 4])
def test_service_resizes_by_request_count_and_runner_reads_grids(k_groups):
    front, store = service(k_groups)
    imgs = [png(1024, 1024, seed=i) for i in range(2)]
    reply = front.encode(imgs, "prefill-host:1")
    items = reply["items"]
    assert [it["grid"] for it in items] == [(1, 44, 44)] * 2  # 2 refs: the 724^2 cap -> 704^2
    assert all(it["runs"] == [(IMAGE_PAD, 484)] for it in items)
    assert all(store.values[it["media_key"]][1] == "prefill-host:1" for it in items)
    one = front.encode(imgs[:1], "prefill-host:1")["items"][0]
    assert one["grid"] == (1, 64, 64) and one["media_key"] != items[0]["media_key"]  # 1 ref: 1 MP, another key
    # runner: template -> splice -> refs -> fetch -> headers
    template = [1, 248053, IMAGE_PAD, 248054, 248053, IMAGE_PAD, 248054, 2]
    ids, spans = splice.expand(template, IMAGE_PAD, [it["runs"] for it in items])
    refs = [splice.MediaRef(0, s, e, bytes.fromhex(it["media_key"])) for (s, e), it in zip(spans, items)]
    got = splice.fetch_media(store, refs)
    assert splice.image_grid_thw([h for h, _ in got]).tolist() == [[1, 44, 44]] * 2
    assert all(r.shape == (484, HIDDEN) for _, r in got)


def test_service_spreads_images_over_workers_one_per_call():
    front, _ = service(4)
    reply = front.encode([png(512, 512, seed=i) for i in range(10)], "p:1")
    w = reply["timing"]["workers"]
    assert sorted(len(x["encode_calls_ms"]) for x in w) == [2, 2, 3, 3]  # 10 images over 4 groups, 1 per call
    batched, _ = service(4, batch=True)
    w = batched.encode([png(512, 512, seed=i) for i in range(10)], "p:1")["timing"]["workers"]
    assert [len(x["encode_calls_ms"]) for x in w] == [1, 1, 1, 1]


def test_service_cache_hit_and_nonce():
    front, store = service(2)
    imgs = [png(512, 512, seed=i) for i in range(3)]
    a = front.encode(imgs, "p:1")
    b = front.encode(imgs, "p:1")
    assert [i["media_key"] for i in a["items"]] == [i["media_key"] for i in b["items"]]
    assert sum(x["cache_hits"] for x in b["timing"]["workers"]) == 3
    c = front.encode(imgs, "p:1", nonce=b"x")
    assert sum(x["cache_hits"] for x in c["timing"]["workers"]) == 0 and len(store.values) == 6


def test_service_limits_and_runner_checks(expect_error):
    front, store = service(1)
    assert "error" in front.encode([], "p:1")
    assert "exceed the limit" in front.encode([png(64, 64)] * 11, "p:1")["error"]
    assert "limit" in Front(front.workers, IMAGE_PAD, Limits(max_bytes=100)).encode([png(64, 64)], "p:1")["error"]
    it = front.encode([png(512, 512)], "p:1")["items"][0]
    with expect_error(ValueError, "for a span of"):
        splice.fetch_media(store, [splice.MediaRef(0, 10, 11, bytes.fromhex(it["media_key"]))])


# ---- Mooncake (where the client is installed) ----------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mooncake_put_from_and_in_place_read(tmp_path):
    pytest.importorskip("mooncake.store")
    from models.common.multimedia_store.config import MediaEmbeddingStoreConfig
    from models.common.multimedia_store.mooncake_store import MooncakeEmbeddingReader
    from models.demos.blackhole.qwen36.vision.service_emu import MooncakeWriter

    import mooncake as mooncake_pkg

    binary = shutil.which("mooncake_master") or os.path.join(os.path.dirname(mooncake_pkg.__file__), "mooncake_master")
    rpc, http, metrics = _free_port(), _free_port(), _free_port()
    log = open(tmp_path / "master.log", "w")
    master = subprocess.Popen(
        [binary, f"--rpc_port={rpc}", "--enable_http_metadata_server=true", f"--http_metadata_server_port={http}",
         "--allow_evict_soft_pinned_objects=false", f"--metrics_port={metrics}"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(2)
        meta = f"http://127.0.0.1:{http}/metadata"
        reader = MooncakeEmbeddingReader.open(MediaEmbeddingStoreConfig(
            master=f"127.0.0.1:{rpc}", segment=f"127.0.0.1:{_free_port()}", segment_bytes=64 << 20, metadata=meta,
            protocol="tcp"))
        writer = MooncakeWriter(f"127.0.0.1:{rpc}", meta, f"127.0.0.1:{_free_port()}", 16 << 20)
        try:
            front = Front([LocalWorker(EncodeWorker(stub_media(), writer, writer.pool))], IMAGE_PAD)
            items = front.encode([png(512, 512, seed=i) for i in range(3)], reader.segment)["items"]
            template = [1] + [248053, IMAGE_PAD, 248054] * 3
            ids, spans = splice.expand(template, IMAGE_PAD, [it["runs"] for it in items])
            refs = [splice.MediaRef(0, s, e, bytes.fromhex(it["media_key"])) for (s, e), it in zip(spans, items)]
            got = splice.fetch_media(reader, refs)
            assert [h.grid_thw for h, _ in got] == [(1, 32, 32)] * 3
            assert reader.stats.snapshot()["local_hits"] == 3  # placed on the prefill host's segment, read in place
            assert writer.is_exist(items[0]["media_key"])
        finally:
            writer.close()
            reader.close()
    finally:
        master.terminate()
        master.wait(timeout=30)
        log.close()
