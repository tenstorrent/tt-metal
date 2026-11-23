# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU: ENGRAM_SHM_DIR keeps resident_bfp8's packed rows across runs (no device, no checkpoint).

A small fake table (raw e4m3 / e8m0 bytes in two files, as the checkpoint stores them) is packed three ways -- into
anonymous memory (ENGRAM_SHM_DIR=off), into a new file in the directory, and mapped back from that file by a later
"run" -- and every one must hold the same bytes. A directory without room for the file falls back to anonymous memory
and leaves nothing behind."""

import os
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

import models.demos.deepseek_v41_flash_d_p.tt.engram as engram

LAYER, ROWS, DIM, BLOCK = 1, 3000, 256, 1000


@pytest.fixture
def checkpoint(tmp_path):
    g = torch.Generator().manual_seed(0)
    weight = torch.randint(0, 256, (ROWS, DIM), dtype=torch.uint8, generator=g)
    weight[(weight & 0x7F) == 0x7F] = 0  # (no e4m3 NaN)
    scale = torch.randint(110, 140, (ROWS, DIM // 32), dtype=torch.uint8, generator=g)
    wname, sname = engram._names(LAYER)
    files = {wname: ("table.bin", weight, 64), sname: ("scale.bin", scale, 128)}
    for file, t, off in files.values():
        (tmp_path / file).write_bytes(bytes(off) + t.numpy().tobytes())

    def info(name):
        _, t, off = files[name]
        return "U8", tuple(t.shape), off, t.numel()

    return SimpleNamespace(path=str(tmp_path), weight_map={n: f for n, (f, _, _) in files.items()}, info=info)


def _packed(ck):
    engram._RESIDENT.clear()
    return engram.resident_bfp8(ck, LAYER, threads=4, block=BLOCK).clone()


def test_rows_are_kept_and_mapped_back(checkpoint, tmp_path, monkeypatch):
    store = tmp_path / "shm"
    monkeypatch.setenv("DS41_ENGRAM_SHM_DIR", "off")
    anonymous = _packed(checkpoint)
    monkeypatch.setenv("DS41_ENGRAM_SHM_DIR", str(store))
    built = _packed(checkpoint)
    (kept,) = os.listdir(store)
    assert kept.startswith(f"L{LAYER:02d}.bfp8.") and kept.endswith(".rows")
    assert os.path.getsize(store / kept) == anonymous.numel()
    mapped = _packed(checkpoint)
    assert torch.equal(built, anonymous) and torch.equal(mapped, anonymous)


def test_no_room_builds_in_anonymous_memory(checkpoint, tmp_path, monkeypatch):
    store = tmp_path / "shm"
    monkeypatch.setenv("DS41_ENGRAM_SHM_DIR", "off")
    anonymous = _packed(checkpoint)
    monkeypatch.setenv("DS41_ENGRAM_SHM_DIR", str(store))
    full = namedtuple("statvfs", "f_bavail f_frsize")(0, 4096)
    monkeypatch.setattr(engram.os, "statvfs", lambda path: full)
    assert torch.equal(_packed(checkpoint), anonymous)
    assert os.listdir(store) == []
