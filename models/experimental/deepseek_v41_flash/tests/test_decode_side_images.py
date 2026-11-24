# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Images on the persistent-MoE decode server (``demo/server.py --persistent-moe``), host only.

Neither the disaggregated prefill server nor the persistent stage's traced prefill takes image-span tokens (ids past
the vocabulary), so the server prefills a prompt up to its first image token and decode replays the rest, image runs
in verify blocks of image tokens only:

* ``test_text_prefill`` -- the prefill is cut before the first image token, down to the prefill's grid.
* ``test_kind_run`` -- a replay batch ends where the tokens change kind (image-span or text).
* ``test_prefill_text_only`` -- the scheduler cuts a remote or persistent prefill at the first image token, and
  leaves the stage pipeline's alone.
* ``test_spec_replay_batch_ends_at_a_change_of_kind`` -- with image blocks, DSpark's replay batches stop there.
* ``test_vision_flags`` -- ``--persistent-moe`` keeps ``--vision`` with DSpark on and takes fewer image rows by
  default; the stage pipeline's DSpark still turns it off.

    pytest models/experimental/deepseek_v41_flash/tests/test_decode_side_images.py
"""

from types import SimpleNamespace

import pytest

from models.experimental.deepseek_v41_flash.demo import server

V = 1000  # a vocabulary: ids >= V are image-span tokens


def _prompt(text: int, image: int, after: int) -> list[int]:
    return [i % V for i in range(text)] + [V + i for i in range(image)] + [i % V for i in range(after)]


def test_text_prefill():
    ids = _prompt(300, 184, 20)
    # The first image token is at 300: 256 on the 128 grid.
    assert server._text_prefill(ids, 0, 384, V, 128) == 256
    # A prefill that ends before the image is kept whole.
    assert server._text_prefill(ids, 0, 256, V, 128) == 256
    # Less than one grid step of text before the image: nothing for the prefill model.
    assert server._text_prefill(_prompt(100, 184, 20), 0, 256, V, 128) == 0
    # From a lead (prefill_from): the count is from the prefill's own start.
    assert server._text_prefill(ids, 40, 256, V, 128) == 256
    assert server._text_prefill(ids, 50, 384, V, 128) == 128
    # No image token at all.
    assert server._text_prefill(list(range(1024)), 0, 896, V, 128) == 896


def test_kind_run():
    ids = _prompt(6, 9, 3)
    assert server._kind_run(ids, 0, 18, V) == 6
    assert server._kind_run(ids, 6, 12, V) == 9
    assert server._kind_run(ids, 8, 4, V) == 4  # capped by n
    assert server._kind_run(ids, 15, 3, V) == 3
    assert server._kind_run(ids, 3, 0, V) == 0


def _scheduler(cls, prefiller, persistent=True, pos=0, image_blocks=True, history=None):
    sched = cls.__new__(cls)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=V), _image_blocks=image_blocks)
    tokens = list(history) if history is not None else [i % V for i in range(pos)]
    user = SimpleNamespace(pos=len(tokens), tokens=tokens)
    sched.engine = SimpleNamespace(model=model, persistent_moe=persistent, users=[user])
    sched.server = SimpleNamespace(prefiller=prefiller, prefix_cache=None)
    sched.replay_rate = 200.0
    return sched


def _remote(seconds=0.1):
    prefiller = server._RemotePrefiller.__new__(server._RemotePrefiller)
    prefiller.align = 128
    prefiller.seconds = lambda n: seconds
    return prefiller


def _turn(ids, device_prefill, prefill_from=0):
    return SimpleNamespace(ids=ids, device_prefill=device_prefill, prefill_from=prefill_from, slot=0, user_key="u")


def test_prefill_text_only():
    ids = _prompt(1000, 184, 20)
    # The prefill server takes the text up to the image, on its grid; decode replays the rest.
    turn = _turn(ids, 1024)
    _scheduler(server._PrefillFirst, _remote())._prefill_text_only(turn)
    assert turn.device_prefill == 896
    # Nothing to cut without an image token, or when the cut is past the prefill.
    turn = _turn([i % V for i in range(1200)], 1024)
    _scheduler(server._PrefillFirst, _remote())._prefill_text_only(turn)
    assert turn.device_prefill == 1024
    turn = _turn(_prompt(1100, 184, 20), 1024)
    _scheduler(server._PrefillFirst, _remote())._prefill_text_only(turn)
    assert turn.device_prefill == 1024
    # Less than a grid step of text, or a cut too short to beat replaying it: decode replays the whole prompt.
    turn = _turn(_prompt(100, 184, 20), 256)
    _scheduler(server._PrefillFirst, _remote())._prefill_text_only(turn)
    assert turn.device_prefill == 0
    turn = _turn(_prompt(200, 184, 20), 256)
    _scheduler(server._PrefillFirst, _remote(seconds=2.0))._prefill_text_only(turn)
    assert turn.device_prefill == 0
    # The stage pipeline's own prefill model takes image tokens: left alone.
    local = SimpleNamespace(seconds=lambda n: 0.1)
    turn = _turn(ids, 1024)
    _scheduler(server._PrefillFirst, local, persistent=False)._prefill_text_only(turn)
    assert turn.device_prefill == 1024
    # The persistent stage's prefill model is cut too, but continues a session on its own grid: replay instead.
    turn = _turn(ids, 1024)
    _scheduler(server._PrefillFirst, local)._prefill_text_only(turn)
    assert turn.device_prefill == 896
    turn = _turn(ids, 1024, prefill_from=40)
    _scheduler(server._PrefillFirst, local, pos=88)._prefill_text_only(turn)
    assert (turn.device_prefill, turn.prefill_from) == (0, 0)


def test_prefill_text_only_after_an_image_turn():
    # The prefill server is sent the conversation from position 0: an image in an earlier turn rules it out, even for
    # a text-only turn.
    text = [i % V for i in range(1200)]
    turn = _turn(text, 1024)
    _scheduler(server._PrefillFirst, _remote(), history=_prompt(256, 184, 72))._prefill_text_only(turn)
    assert (turn.device_prefill, turn.prefill_from) == (0, 0)
    # A text-only history is continued as before.
    turn = _turn(text, 1024)
    _scheduler(server._PrefillFirst, _remote(), pos=512)._prefill_text_only(turn)
    assert turn.device_prefill == 1024


def test_spec_replay_batch_ends_at_a_change_of_kind():
    ids = _prompt(6, 9, 3)
    sched = _scheduler(server._SpecScheduler, None)
    user = SimpleNamespace(pos=0)
    turn = SimpleNamespace(ids=ids, fed=0, prefix_cuts=[])
    assert sched._prefix_batch(turn, user, 18) == 6
    turn.fed = 6
    assert sched._prefix_batch(turn, user, 12) == 9
    # Without image blocks a batch runs on across the kinds (one-row steps take image tokens as they are).
    sched = _scheduler(server._SpecScheduler, None, image_blocks=False)
    turn.fed = 0
    assert sched._prefix_batch(turn, user, 18) == 18


@pytest.mark.parametrize(
    "argv, vision, image_rows",
    [
        (["--persistent-moe", "--dspark"], True, server._PERSISTENT_IMAGE_ROWS),
        (["--persistent-moe", "--no-dspark"], True, server._PERSISTENT_IMAGE_ROWS),
        (["--persistent-moe", "--no-vision"], False, server._PERSISTENT_IMAGE_ROWS),
        (["--persistent-moe", "--image-rows", "4096"], True, 4096),
        (["--no-persistent-moe", "--dspark"], False, server._IMAGE_ROWS),
        (["--no-persistent-moe", "--no-dspark"], True, server._IMAGE_ROWS),
    ],
)
def test_vision_flags(argv, vision, image_rows, monkeypatch):
    # parse_args writes a few environment variables for the model; keep them out of the other tests.
    for name in ("DEEPSEEK_V41_RUST_HOST", "DEEPSEEK_V4_SPEC_VERBOSE", "DEEPSEEK_V4_GCB_HEADROOM"):
        monkeypatch.delenv(name, raising=False)
    args = server.parse_args(argv)
    assert args.vision is vision
    assert args.image_rows == image_rows
