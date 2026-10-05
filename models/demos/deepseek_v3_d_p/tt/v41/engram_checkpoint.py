# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 Engram parameters from the checkpoint (shards 47 / 48): the host-table placement.

``weights.py`` leaves the Engram tensors out. This module reads them: the small ``wkv`` / ``q_weight`` /
``k_weight`` of each Engram layer, and the layer's embedding table (384M rows of float8_e4m3fn [256] with
float8_e8m0fnu [8] scales, ~101 GB per layer in the shard) packed for ``V41EngramTable`` (``engram.pack_rows``:
320 B per row, 123 GB per layer).

``V41EngramTable`` packs a whole table in one torch pass, which needs the raw table, its scales and the packed
output resident at once (~224 GB per layer). Here the table is packed in row chunks straight from the safetensors
shard (``safe_open().get_slice``) into a file under ``TT_V41_WEIGHT_CACHE/engram/`` (one pass per layer, done
once), and that file is read back into host RAM on every later run (``np.fromfile``, ~123 GB per layer, at NVMe
speed). The host table is then exactly the branch's ``V41EngramTable`` with a prepacked ``rows`` tensor
(``V41EngramTable.from_packed``); per chunk the transformer gathers the rows the prompt hashes to and uploads them
(``TtV41Engram.prepare_host``). ``TT_V41_ENGRAM_MMAP=1`` maps the file instead of reading it (lazy page-ins on a
host short of RAM; slower first gathers).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tt.v41.engram import PACKED_WIDTH, SCALE_BLOCK, V41EngramTable, pack_rows
from models.demos.deepseek_v3_d_p.tt.v41.weights import CHECKPOINT_REVISION, V41Checkpoint, dequant_fp8_block

ROWS_PER_CHUNK = 1 << 21  # 2M rows: 512 MB of fp8 + 16 MB of scales in, 640 MB packed out, per step
PACKED_DIR = Path(os.environ.get("TT_V41_WEIGHT_CACHE", Path.home() / ".cache" / "tt-v41-weights")) / "engram"


def engram_names(layer: int) -> dict[str, str]:
    """Checkpoint tensor names of one Engram layer's parameters."""
    p = f"layers.{layer}.engram."
    return {
        "wkv": p + "wkv.weight",
        "wkv_scale": p + "wkv.scale",
        "q_weight": p + "q_weight",
        "k_weight": p + "k_weight",
        "embed": p + "embed.weight",
        "embed_scale": p + "embed.scale",
    }


def load_engram_weights(ckpt: V41Checkpoint, layer: int) -> dict:
    """``TtV41Engram`` ``weights``: ``wkv`` dequantized to bf16 ``[(hc_mult + 1) * hidden, 24 * head_dim]``,
    ``q_weight`` and ``k_weight`` ``[hc_mult, hidden]`` as stored (bf16)."""
    names = engram_names(layer)
    raw = ckpt.read([names["wkv"], names["wkv_scale"], names["q_weight"], names["k_weight"]])
    return {
        "wkv": dequant_fp8_block(raw[names["wkv"]], raw[names["wkv_scale"]], names["wkv"]),
        "q_weight": raw[names["q_weight"]],
        "k_weight": raw[names["k_weight"]],
    }


def packed_table_path(layer: int, config=DeepSeekV41FlashConfig) -> Path:
    """The packed table file of ``layer``: rows x ``PACKED_WIDTH`` int16 (the uint16 containers), no header; the
    row count is in the name, the checkpoint revision too so a different checkpoint cannot be read by mistake."""
    rows = dict(zip(config.ENGRAM_LAYER_IDS, config.ENGRAM_NUM_EMBEDDINGS))[layer]
    return PACKED_DIR / f"layer_{layer}.{CHECKPOINT_REVISION[:12]}.{rows}x{PACKED_WIDTH}.i16"


def pack_table_to_file(ckpt: V41Checkpoint, layer: int, path: Path, rows_per_chunk: int = ROWS_PER_CHUNK) -> Path:
    """Pack ``layer``'s table from its shard into ``path`` in row chunks; written to ``path.tmp`` and renamed, so
    a partial file is never read as complete."""
    names = engram_names(layer)
    shard = ckpt.root / ckpt.weight_map[names["embed"]]
    assert ckpt.weight_map[names["embed_scale"]] == ckpt.weight_map[names["embed"]], "table and scales in one shard"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    started = time.time()
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight, scale = handle.get_slice(names["embed"]), handle.get_slice(names["embed_scale"])
        rows, head_dim = weight.get_shape()
        assert tuple(scale.get_shape()) == (rows, head_dim // SCALE_BLOCK), (weight.get_shape(), scale.get_shape())
        with open(tmp, "wb") as f:
            f.truncate(rows * PACKED_WIDTH * 2)
        out = np.memmap(tmp, dtype=np.int16, mode="r+", shape=(rows, PACKED_WIDTH))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            packed = pack_rows(weight[start:end], scale[start:end])  # uint16 [n, PACKED_WIDTH]
            out[start:end] = packed.view(torch.int16).numpy()
            if (start // rows_per_chunk) % 16 == 0:
                logger.info(f"engram layer {layer}: packed {end}/{rows} rows ({time.time() - started:.0f} s)")
        out.flush()
        del out
    tmp.rename(path)
    logger.info(f"engram layer {layer}: packed table {path} ({rows} rows) in {time.time() - started:.0f} s")
    return path


def load_packed_table(path: Path, rows: int, resident: bool | None = None) -> torch.Tensor:
    """uint16 ``[rows, PACKED_WIDTH]`` from a packed table file: read into RAM (``resident``, the default), or
    mapped (``TT_V41_ENGRAM_MMAP=1``)."""
    if resident is None:
        resident = os.environ.get("TT_V41_ENGRAM_MMAP", "0") != "1"
    expected = rows * PACKED_WIDTH * 2
    assert path.stat().st_size == expected, f"{path}: {path.stat().st_size} bytes, expected {expected}"
    started = time.time()
    if resident:
        arr = np.fromfile(path, dtype=np.int16)
    else:
        arr = np.memmap(path, dtype=np.int16, mode="r")
    t = torch.from_numpy(arr).view(rows, PACKED_WIDTH).view(torch.uint16)
    how = "read" if resident else "mapped"
    logger.info(f"engram table {path.name}: {how} {expected / 1e9:.1f} GB in {time.time() - started:.0f} s")
    return t


def engram_table(ckpt: V41Checkpoint, layer: int, config=DeepSeekV41FlashConfig) -> V41EngramTable:
    """The full host table of ``layer``: packed once into the cache file, read into RAM after."""
    path = packed_table_path(layer, config)
    if not path.is_file():
        pack_table_to_file(ckpt, layer, path)
    rows = dict(zip(config.ENGRAM_LAYER_IDS, config.ENGRAM_NUM_EMBEDDINGS))[layer]
    return V41EngramTable.from_packed(load_packed_table(path, rows), config.ENGRAM_HEAD_DIM)


def build_engram_modules(mesh_device, config, ckpt: V41Checkpoint, layers, tokenizer):
    """``(engram, engram_hash)`` for ``TtV41Transformer``: a ``TtV41Engram`` with the real table and weights for
    every Engram layer in ``layers`` (none -> ``({}, None)``), and the n-gram hasher on the real tokenizer."""
    from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash

    wanted = [layer for layer in layers if layer in config.ENGRAM_LAYER_IDS]
    if not wanted:
        return {}, None
    engram_hash = V41EngramHash.from_tokenizer(config, tokenizer)
    engram = {}
    for layer in wanted:
        weights, table = load_engram_weights(ckpt, layer), engram_table(ckpt, layer, config)
        engram[layer] = TtV41Engram(mesh_device, config, layer, weights, table)
    return engram, engram_hash
