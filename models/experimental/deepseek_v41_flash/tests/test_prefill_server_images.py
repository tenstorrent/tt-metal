# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Images through the disaggregated prefill server (``demo/prefill_server.py``), host only (no device, no model):

* ``test_resolve`` -- the server's ImageBook rewrites a decode server's image ids into its own rows, writing an
  image's rows the first time they come, and asks for the rows of an image it does not hold.
* ``test_evict`` -- rows run out: the least recently used image goes, and the prefix-cache slots holding it.
* ``test_need_round_trip`` -- the client sends a prompt again with the rows the server asked for, over a socket.
* ``test_wire_images`` -- the decode server names the images a prompt's ids use.
* ``test_no_cut_when_the_server_takes_images`` -- the decode scheduler leaves such a prompt whole.

    pytest models/experimental/deepseek_v41_flash/tests/test_prefill_server_images.py
"""

import socket
import threading
from collections import OrderedDict
from types import SimpleNamespace

import torch

from models.experimental.deepseek_v41_flash.demo import prefill_server as ps
from models.experimental.deepseek_v41_flash.demo import server
from models.experimental.deepseek_v41_flash.tt.prefill import d_p_wire as wire

V, H = 1000, 8  # a vocabulary and a hidden size
FIRST = V + 3  # the first image row id


class _FakeDP:
    def __init__(self, image_rows: int):
        self.vocab, self.image_rows = V, image_rows
        self.cfg = SimpleNamespace(dim=H)
        self.written = {}

    def write_image_rows(self, rows, ids):
        for row, i in zip(rows, ids):
            self.written[int(i)] = row.clone()


def _rows(n: int, seed: int) -> torch.Tensor:
    return torch.randn(n, H, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _blob(rows: torch.Tensor) -> bytearray:
    return bytearray(rows.contiguous().view(torch.int16).numpy().tobytes())


def _book(image_rows: int, slots: int = 2):
    cache = ps.PrefixCache(slots)
    return ps.ImageBook(_FakeDP(image_rows), cache), cache


def test_resolve(expect_error):
    book, _ = _book(3 + 64)
    # The decode server's slots 40..43 hold image "a": the span is start, 4 image rows, end.
    ids = torch.tensor([5, 6, V + 0] + [FIRST + s for s in (40, 41, 42, 43)] + [V + 2, 7], dtype=torch.int32)
    images = [{"key": "a", "slots": [40, 41, 42, 43], "rows": False}]
    with expect_error(ps.NeedImages, "send the rows") as need:
        book.resolve(ids, images, [])
    assert need.value.keys == ["a"]
    rows = _rows(4, 0)
    images[0]["rows"] = True
    out = book.resolve(ids, images, [_blob(rows)])
    local = book.entries["a"]
    assert out.tolist() == [5, 6, V + 0] + [FIRST + int(s) for s in local] + [V + 2, 7]
    assert all(torch.equal(book.d_p.written[FIRST + int(s)], rows[j]) for j, s in enumerate(local))
    # Held now: no rows needed, the same ids; another decode layout of the same image maps to the same rows.
    again = book.resolve(
        torch.tensor([FIRST + 7, FIRST + 9], dtype=torch.int32), [{"key": "a", "slots": [9, 7, 1, 2]}], []
    )
    assert again.tolist() == [FIRST + int(local[1]), FIRST + int(local[0])]
    # An image id no image names, and a span whose row count disagrees.
    with expect_error(ValueError, "no image of the request names"):
        book.resolve(
            torch.tensor([FIRST + 50], dtype=torch.int32), images[:0] + [{"key": "a", "slots": [1, 2, 3, 4]}], []
        )
    with expect_error(ValueError, "slots"):
        book.resolve(torch.tensor([FIRST + 1], dtype=torch.int32), [{"key": "a", "slots": [1, 2]}], [])


def test_evict(expect_error):
    book, cache = _book(3 + 8)
    a = book.resolve(
        torch.tensor([FIRST + 0], dtype=torch.int32),
        [{"key": "a", "slots": [0, 1, 2, 3, 4], "rows": True}],
        [_blob(_rows(5, 1))],
    )
    cache.keep(0, torch.tensor([1, 2, int(a[0])], dtype=torch.int32))  # a prompt holding image "a"
    cache.keep(1, torch.tensor([1, 2, 3], dtype=torch.int32))  # a text prompt
    # "b" needs 5 rows, 3 are free: "a" goes, and slot 0 with it.
    book.resolve(
        torch.tensor([FIRST + 0], dtype=torch.int32),
        [{"key": "b", "slots": [0, 1, 2, 3, 4], "rows": True}],
        [_blob(_rows(5, 2))],
    )
    assert list(book.entries) == ["b"] and cache.ids[0] is None and cache.ids[1] is not None
    # A request's own images are never evicted for it.
    with expect_error(ValueError, "more than"):
        book.resolve(
            torch.tensor([FIRST], dtype=torch.int32),
            [{"key": "b", "slots": [0, 1, 2, 3, 4]}, {"key": "c", "slots": [5, 6, 7, 8], "rows": True}],
            [_blob(_rows(4, 3))],
        )


def test_need_round_trip():
    a, b = socket.socketpair()
    rows = _rows(4, 4)
    seen = []

    def serve():
        header, blobs = wire.recv_msg(b)
        seen.append((header, blobs))
        wire.send_msg(b, {"need": ["k"]})
        header, blobs = wire.recv_msg(b)
        seen.append((header, blobs))
        wire.send_msg(b, {"progress": 128})
        wire.send_msg(
            b, {"seq_len": 128, "base": 0, "cached": 0, "seconds": 0.1, "queued": 0.0, "bytes": 1, "pages": 1}
        )

    thread = threading.Thread(target=serve)
    thread.start()
    client = wire.PrefillClient.__new__(wire.PrefillClient)
    client.sock, client.address, client.sent_images = a, "test", set()
    progress = []
    image = wire.WireImage("k", [3, 4, 5, 6], lambda: rows, send_rows=False)
    reply = client.prefill_push(list(range(128)), {"seq_len": 128}, on_progress=progress.append, images=[image])
    thread.join()
    assert reply["seq_len"] == 128 and progress == [128] and client.sent_images == {"k"}
    (first, first_blobs), (second, second_blobs) = seen
    assert first["images"] == [{"key": "k", "slots": [3, 4, 5, 6], "rows": False}] and len(first_blobs) == 1
    assert second["images"][0]["rows"] is True and len(second_blobs) == 2
    assert torch.equal(torch.frombuffer(second_blobs[1], dtype=torch.bfloat16).view(4, H), rows)
    a.close()
    b.close()


def test_wire_images(expect_error):
    book = server._ImageBook.__new__(server._ImageBook)
    book.vocab_size, book._lock = V, threading.Lock()
    rows_a, rows_b = _rows(3, 5), _rows(2, 6)
    book.entries = OrderedDict(
        a=server._Image(torch.zeros(5), None, slots=torch.tensor([10, 11, 12]), vision_rows=rows_a),
        b=server._Image(torch.zeros(4), None, slots=torch.tensor([20, 21]), vision_rows=rows_b),
        c=server._Image(torch.zeros(4), None),  # not encoded
    )
    ids = torch.tensor([1, 2, V, FIRST + 20, FIRST + 21, V + 2, 3])
    (image,) = book.wire_images(ids, sent=set())
    assert (image.key, image.slots, image.send_rows) == ("b", [20, 21], True) and torch.equal(image.rows(), rows_b)
    both = book.wire_images(torch.tensor([FIRST + 12, FIRST + 21]), sent={"a"})
    assert [(i.key, i.send_rows) for i in both] == [("a", False), ("b", True)]
    assert book.wire_images(torch.tensor([1, 2, V + 1]), sent=set()) == []
    with expect_error(RuntimeError, "no held image"):
        book.wire_images(torch.tensor([FIRST + 99]), sent=set())


def test_no_cut_when_the_server_takes_images():
    ids = [i % V for i in range(1000)] + [V + i for i in range(184)] + [1] * 20
    sched = server._PrefillFirst.__new__(server._PrefillFirst)
    sched.engine = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=V)),
        persistent_moe=True,
        users=[SimpleNamespace(pos=0, tokens=[])],
    )
    prefiller = server._RemotePrefiller.__new__(server._RemotePrefiller)
    prefiller.align, prefiller.seconds, prefiller.image_rows = 128, (lambda n: 0.1), 16381
    sched.server = SimpleNamespace(prefiller=prefiller, prefix_cache=None)
    sched.replay_rate = 200.0
    turn = SimpleNamespace(ids=ids, device_prefill=1152, prefill_from=0, slot=0, user_key="u")
    sched._prefill_text_only(turn)
    assert turn.device_prefill == 1152
    prefiller.image_rows = 0  # an older server: cut before the image
    sched._prefill_text_only(turn)
    assert turn.device_prefill == 896
