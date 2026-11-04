# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The Rust host runtime's Engram lookup (``tt/host_rs.py``, ``host_rs/``) against ``EngramHostLookup``. Host only.

* ``test_rust_engram_matches_python`` -- a prompt holding an image span (``__call__``, as prefill calls it), then
  decode steps on another session slot (``hash_session`` + ``gather``, as the decode packet path does), then a
  history export / import: hashes equal and rows bit-identical, NaN-carrying table bytes included.

The ~200 GB tables are sparse anonymous mappings: only the rows the hashes hit are written (with random bytes).
Build the extension first::

    models/experimental/deepseek_v41_flash/host_rs/build.sh
    pytest models/experimental/deepseek_v41_flash/tests/test_host_rs.py
"""

import mmap
import time

import pytest
import torch
from loguru import logger

from models.experimental.deepseek_v41_flash.tt import host_rs
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import TABLE_BLOCK, EngramHostLookup
from models.experimental.deepseek_v41_flash.tt.vision import DELIMITER_ROWS, FIRST_IMAGE_ROW, IMAGE_END, IMAGE_START
from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir

SESSIONS = 3
MAX_SEQ = 4096
DECODE_STEPS = 64


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


pytestmark = [
    pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}"),
    pytest.mark.skipif(not host_rs._EXTENSION.is_file(), reason="dsv41_host is not built (host_rs/build.sh)"),
]


def _sparse_table(rows: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight, scale)`` shaped like a real table, backed by untouched (zero, unallocated) anonymous pages."""
    tensors = []
    for width, dtype in ((head_dim, torch.float8_e4m3fn), (head_dim // TABLE_BLOCK, torch.float8_e8m0fnu)):
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_NORESERVE", 0x4000)  # Linux's value
        memory = mmap.mmap(-1, rows * width, flags=flags)
        tensors.append(torch.frombuffer(memory, dtype=torch.uint8).view(dtype).reshape(rows, width))
    return tuple(tensors)


def _fill(tables: dict, layout, hashes: torch.Tensor, generator: torch.Generator) -> None:
    """Random bytes in every row ``hashes`` (``[..., n_layout_layers, n_cols]``) hits: weights anywhere in fp8
    (NaN included), scales in a range that keeps the products finite."""
    for layer_id, (weight, scale) in tables.items():
        rows = hashes[..., layout.layer_ids.index(layer_id), :].reshape(-1).unique()
        w = torch.randint(0, 256, (rows.numel(), weight.shape[1]), dtype=torch.uint8, generator=generator)
        s = torch.randint(100, 150, (rows.numel(), scale.shape[1]), dtype=torch.uint8, generator=generator)
        weight.view(torch.uint8)[rows] = w
        scale.view(torch.uint8)[rows] = s


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def test_rust_engram_matches_python():
    from transformers import AutoTokenizer

    snapshot = _snapshot()
    config = load_config(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    tables = {
        layer_id: _sparse_table(rows, config.engram_head_dim)
        for layer_id, rows in zip(config.engram_layer_ids, config.engram_num_embeddings)
    }
    python = EngramHostLookup(config, snapshot, tables, 1, MAX_SEQ, tokenizer, sessions=SESSIONS)
    rust = host_rs.RustEngramLookup(config, snapshot, tables, 1, MAX_SEQ, tokenizer, sessions=SESSIONS)
    layout, vocab = python.layout, config.vocab_size
    generator = torch.Generator().manual_seed(0)

    # Prefill: a prompt with an image span, on session 0.
    text = tokenizer(
        "The quick brown fox jumps over the lazy dog. Describe the picture in a few words, please.",
        add_special_tokens=False,
    )["input_ids"]
    span = [vocab + DELIMITER_ROWS[IMAGE_START]] + [vocab + FIRST_IMAGE_ROW + i for i in range(6)]
    span.append(vocab + DELIMITER_ROWS[IMAGE_END])
    ids = torch.tensor([text[:7] + span + text[7:]])
    _fill(tables, layout, python.hash_session(SESSIONS - 1, ids, 0), generator)
    python.use_session(0)
    rust.use_session(0)
    expected, got = python(ids, 0), rust(ids, 0)
    assert expected.keys() == got.keys()
    for layer_id in expected:
        assert torch.equal(_bits(expected[layer_id]), _bits(got[layer_id])), f"layer {layer_id} prompt rows"

    # Decode: one token a step on session 1, continuing a copy of the prompt's history.
    history = python.export_history(0, ids.shape[1])
    assert torch.equal(history, rust.export_history(0, ids.shape[1]))
    python.import_history(1, history)
    rust.import_history(1, history)
    py_seconds = rs_seconds = 0.0
    for step in range(DECODE_STEPS):
        pos = ids.shape[1] + step
        token = torch.tensor([[int(torch.randint(0, len(tokenizer), (1,), generator=generator))]])
        t0 = time.perf_counter()
        py_hashes = python.hash_session(1, token, pos)
        py_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        rs_hashes = rust.hash_session(1, token, pos)
        rs_seconds += time.perf_counter() - t0
        assert torch.equal(py_hashes, rs_hashes), f"step {step} hashes"
        _fill(tables, layout, py_hashes, generator)
        for layer_id in tables:
            col = layout.layer_ids.index(layer_id)
            t0 = time.perf_counter()
            py_rows = python.gather(layer_id, py_hashes[:, :, col])
            py_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            rs_rows = rust.gather(layer_id, rs_hashes[:, :, col])
            rs_seconds += time.perf_counter() - t0
            assert torch.equal(_bits(py_rows), _bits(rs_rows)), f"step {step} layer {layer_id} rows"
    assert torch.equal(python.export_history(1, pos + 1), rust.export_history(1, pos + 1))
    logger.info(
        f"per decode step (hash + {len(tables)} gathers, through Python): EngramHostLookup "
        f"{py_seconds / DECODE_STEPS * 1e6:.0f} us, RustEngramLookup {rs_seconds / DECODE_STEPS * 1e6:.0f} us"
    )
