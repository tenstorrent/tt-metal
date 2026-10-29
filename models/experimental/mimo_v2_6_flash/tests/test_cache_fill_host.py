# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host: tt.common.cache_fill when several processes build one TTNN cache at once (serving/serve_rows.sh, one engine
per Galaxy row): each cold tag is converted by one process, and every other process loads only complete files; no
device."""

import multiprocessing
import os
import time

from models.experimental.mimo_v2_6_flash.tt.common import cache_fill, cache_name

TAGS = [f"layer{i}" for i in range(6)]
CHUNKS, CHUNK = 8, 4096


def build(cache_dir, log):
    """One engine's model construction: per tag, write the tensor file in place (as ttnn.as_tensor does, chunk by
    chunk) when the cache is cold, otherwise read it, which must find the whole file."""
    for tag in TAGS:
        path = cache_name(cache_dir, tag, "tensor")
        with cache_fill(cache_dir, tag) as need:
            if need:
                with open(path, "w") as f:
                    for _ in range(CHUNKS):
                        f.write("x" * CHUNK)
                        f.flush()
                        time.sleep(0.005)
                with open(log, "a") as f:
                    f.write(f"{os.getpid()} {tag}\n")
            else:
                assert os.path.getsize(path) == CHUNKS * CHUNK, f"{tag}: loaded a partial file"


def run_engines(cache_dir, log, engines):
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=build, args=(cache_dir, log)) for _ in range(engines)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    return [p.exitcode for p in procs]


def fills(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        return [line.split()[1] for line in f]


def test_concurrent_cold_fill_converts_each_tag_once(tmp_path):
    log = str(tmp_path / "fills.log")
    assert run_engines(str(tmp_path), log, engines=4) == [0, 0, 0, 0]
    assert sorted(fills(log)) == sorted(TAGS)  # each tag filled by exactly one engine
    for tag in TAGS:
        assert os.path.exists(cache_name(str(tmp_path), tag, "complete"))
    # A warm restart converts nothing.
    assert run_engines(str(tmp_path), log, engines=4) == [0, 0, 0, 0]
    assert sorted(fills(log)) == sorted(TAGS)


def fill_top(cache_dir):
    with cache_fill(cache_dir, "top") as need:
        assert need


def test_failed_fill_is_not_marked_and_unlocks(tmp_path, expect_error):
    cache_dir = str(tmp_path)
    with expect_error(RuntimeError, "conversion failed"):
        with cache_fill(cache_dir, "top") as need:
            assert need
            raise RuntimeError("conversion failed")
    assert not os.path.exists(cache_name(cache_dir, "top", "complete"))
    # The lock was released: another process takes it at once and fills the tag.
    p = multiprocessing.get_context("fork").Process(target=fill_top, args=(cache_dir,))
    p.start()
    p.join(10)
    assert p.exitcode == 0
    assert os.path.exists(cache_name(cache_dir, "top", "complete"))


def test_no_cache_dir_always_converts():
    with cache_fill(None, "top") as need:
        assert need
