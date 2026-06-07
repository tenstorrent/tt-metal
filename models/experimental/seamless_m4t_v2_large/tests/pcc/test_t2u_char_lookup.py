# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T2U char/duration prep: precomputed lookup tables vs legacy string-walk path."""

from __future__ import annotations

import random
import time

import pytest
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tt.t2u_char_lookup import build_t2u_char_lookup_tables
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    _char_count_per_subword,
    _get_char_ids,
    _indices_to_subwords,
)


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _legacy_char_prep(gc, token_ids: list[int], *, pad_token_id: int) -> tuple[list[int], list[int]]:
    subwords = _indices_to_subwords(gc, token_ids)
    cc = _char_count_per_subword(token_ids, subwords, pad_token_id=pad_token_id)
    char_ids = _get_char_ids(gc, token_ids, subwords, cc, pad_token_id=pad_token_id)
    return cc, char_ids


def _table_char_prep(tables, token_ids: list[int]) -> tuple[list[int], list[int]]:
    char_t, cc_t = tables.prepare_speech(token_ids)
    return cc_t.tolist(), char_t.tolist()


def test_t2u_char_lookup_matches_legacy_on_random_sequences():
    weights_dir = _weights_dir_or_skip()
    hf_model, _, _ = load_hf_model_and_processor(weights_dir)
    gc = hf_model.generation_config
    pad = int(hf_model.config.pad_token_id)
    vs = int(hf_model.config.vocab_size)
    tables = build_t2u_char_lookup_tables(gc, vocab_size=vs, pad_token_id=pad)

    rng = random.Random(42)
    for trial in range(64):
        n = rng.randint(1, 120)
        ids = [rng.randrange(vs) for _ in range(n)]
        if rng.random() < 0.3:
            ids.append(pad)
        cc_old, chars_old = _legacy_char_prep(gc, ids, pad_token_id=pad)
        cc_new, chars_new = _table_char_prep(tables, ids)
        assert cc_new == cc_old, f"trial {trial}: char counts differ"
        assert chars_new == chars_old[: len(chars_new)], f"trial {trial}: char id prefix differs"
        if len(chars_old) > len(chars_new):
            assert all(x == pad for x in chars_old[len(chars_new) :]), f"trial {trial}: legacy tail not pad"


def test_t2u_char_lookup_table_memory_and_speed():
    weights_dir = _weights_dir_or_skip()
    hf_model, _, _ = load_hf_model_and_processor(weights_dir)
    gc = hf_model.generation_config
    pad = int(hf_model.config.pad_token_id)
    vs = int(hf_model.config.vocab_size)
    tables = build_t2u_char_lookup_tables(gc, vocab_size=vs, pad_token_id=pad)

    total_mb = tables.memory_bytes() / (1024 * 1024)
    logger.info(f"T2U char tables: vocab={vs} flat_chars={tables.char_ids_flat.numel()} " f"total_mb={total_mb:.2f}")

    rng = random.Random(0)
    seqs = [[rng.randrange(vs) for _ in range(rng.randint(20, 80))] for _ in range(200)]

    t0 = time.perf_counter()
    for ids in seqs:
        _legacy_char_prep(gc, ids, pad_token_id=pad)
    legacy_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for ids in seqs:
        _table_char_prep(tables, ids)
    table_ms = (time.perf_counter() - t0) * 1000

    speedup = legacy_ms / max(table_ms, 1e-6)
    logger.info(f"T2U char prep 200× seqs: legacy={legacy_ms:.2f}ms tables={table_ms:.2f}ms ratio={speedup:.2f}x")
    # Longer decode-like sequences (typical T2ST text length).
    long_seqs = [[rng.randrange(vs) for _ in range(rng.randint(40, 100))] for _ in range(500)]
    t0 = time.perf_counter()
    for ids in long_seqs:
        _legacy_char_prep(gc, ids, pad_token_id=pad)
    legacy_long_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for ids in long_seqs:
        _table_char_prep(tables, ids)
    table_long_ms = (time.perf_counter() - t0) * 1000
    long_ratio = legacy_long_ms / max(table_long_ms, 1e-6)
    logger.info(
        f"T2U char prep 500× long seqs: legacy={legacy_long_ms:.2f}ms "
        f"tables={table_long_ms:.2f}ms ratio={long_ratio:.2f}x"
    )
