# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Engram table staging (host only, no device, no checkpoint).

A miniature checkpoint stands in for the real one: one safetensors file per Engram layer, ``weight`` and
``scale`` back to back, staged into a temporary ``DEEPSEEK_V41_ENGRAM_STAGE_DIR``. Checks that

* :func:`stage_engram_tables` copies them, is idempotent, and leaves no partial file behind;
* :func:`staged_copy` takes only a copy of the same length, and :func:`engram_ram_needed` reports 0 for a staged
  layer (the copy is resident tmpfs, so a run maps it instead of needing RAM of its own);
* :func:`load_engram_table` maps the staged copy to the same values as the checkpoint's own tensors;
* ``DEEPSEEK_V41_ENGRAM_STAGE=0`` turns staging off, and a full disk leaves the checkpoint as it was.

    pytest models/experimental/deepseek_v41_flash/tests/test_engram_staging.py
"""

import importlib
import json
import os
import struct
from pathlib import Path

import pytest
import torch

from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

LAYERS = (1, 14)
ROWS = 64
HEAD_DIM = 64
SCALE_COLS = HEAD_DIM // 32
_DTYPES = {"float8_e4m3fn": "F8_E4M3", "float8_e8m0fnu": "F8_E8M0"}


def _write_shard(path: Path, tensors: dict) -> None:
    """A one-header safetensors file (no metadata), the layout the checkpoint's shards use."""
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {
            "dtype": _DTYPES[str(tensor.dtype).removeprefix("torch.")],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _checkpoint(root: Path) -> Path:
    """A two-shard checkpoint, one Engram layer per shard, with the index the loader resolves."""
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for index, layer in enumerate(LAYERS):
        name = f"model-{index:05d}-of-{len(LAYERS):05d}.safetensors"
        weight = torch.randint(0, 120, (ROWS, HEAD_DIM), dtype=torch.uint8).view(torch.float8_e4m3fn)
        scale = torch.full((ROWS, SCALE_COLS), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        _write_shard(
            root / name,
            {f"layers.{layer}.engram.embed.weight": weight, f"layers.{layer}.engram.embed.scale": scale},
        )
        for part in ("weight", "scale"):
            weight_map[f"layers.{layer}.engram.embed.{part}"] = name
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


@pytest.fixture()
def staged(tmp_path, monkeypatch):
    """``engram_lookup`` re-imported against a temporary checkpoint and stage dir, so its module-level
    :data:`STAGE_DIR` is the temporary one and no real ``/dev/shm`` copy is touched."""
    checkpoint = _checkpoint(tmp_path / "ckpt")
    monkeypatch.setenv("DEEPSEEK_V41_ENGRAM_STAGE_DIR", str(tmp_path / "stage"))
    monkeypatch.delenv("DEEPSEEK_V41_ENGRAM_STAGE", raising=False)
    from models.experimental.deepseek_v41_flash.tt import engram_lookup

    try:
        yield importlib.reload(engram_lookup), DeepseekV4WeightLoader(checkpoint)
    finally:
        monkeypatch.undo()
        importlib.reload(engram_lookup)


def test_stage_copies_once_and_ram_needed_drops_to_zero(staged):
    lookup, loader = staged
    nbytes = lookup.engram_ram_needed(loader, LAYERS)
    shard_bytes = sum(lookup.engram_table_nbytes(loader, i) for i in LAYERS)
    assert nbytes == shard_bytes > 0

    shards, copied = lookup.stage_engram_tables(loader, LAYERS)
    assert copied == sum(path.stat().st_size for path in shards)
    assert all(lookup.staged_copy(path) is not None for path in shards)
    # A staged layer is resident tmpfs, so a run maps it instead of needing RAM of its own.
    assert lookup.engram_ram_needed(loader, LAYERS) == 0
    # Nothing to do the second time, and no partial left behind.
    assert lookup.stage_engram_tables(loader, LAYERS)[1] == 0
    assert not list(lookup.STAGE_DIR.glob("*.partial"))


def test_staged_copy_rejects_a_truncated_copy(staged):
    lookup, loader = staged
    shards, _ = lookup.stage_engram_tables(loader, LAYERS)
    copy = lookup.STAGE_DIR / shards[0].name
    os.truncate(copy, copy.stat().st_size - 1)
    assert lookup.staged_copy(shards[0]) is None
    # A short copy is staged again rather than mapped.
    assert lookup.stage_engram_tables(loader, LAYERS)[1] > 0
    assert lookup.staged_copy(shards[0]) is not None


def test_a_staged_table_maps_to_the_checkpoint_values(staged):
    lookup, loader = staged
    original = {layer: lookup.load_engram_table(loader, layer) for layer in LAYERS}
    lookup.stage_engram_tables(loader, LAYERS)
    for layer in LAYERS:
        mapped = lookup.load_engram_table(loader, layer)
        assert tuple(mapped[0].shape) == (ROWS, HEAD_DIM)
        assert tuple(mapped[1].shape) == (ROWS, SCALE_COLS)
        for from_copy, from_checkpoint in zip(mapped, original[layer]):
            assert torch.equal(from_copy.view(torch.uint8), from_checkpoint.view(torch.uint8))


def test_staging_off(staged, monkeypatch):
    lookup, loader = staged
    monkeypatch.setenv("DEEPSEEK_V41_ENGRAM_STAGE", "0")
    assert lookup.stage_engram_tables(loader, LAYERS) == ([], 0)
    assert lookup.engram_ram_needed(loader, LAYERS) > 0


def test_no_room_leaves_the_checkpoint_alone(staged, monkeypatch):
    lookup, loader = staged
    # A stage dir with no room cannot hold a copy, so staging gives up and the model reads the checkpoint.
    monkeypatch.setattr(lookup.shutil, "disk_usage", lambda path: os.statvfs(path) and type("U", (), {"free": 1})())
    shards, copied = lookup.stage_engram_tables(loader, LAYERS)
    assert copied == 0
    assert all(lookup.staged_copy(path) is None for path in shards)
    assert lookup.engram_ram_needed(loader, LAYERS) > 0
