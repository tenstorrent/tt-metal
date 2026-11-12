# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the prefill runner's weight-cache helpers and chunk-size env (no device, no checkpoint)."""

from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc


def test_format_missing_summarises_layer_ranges():
    assert (
        wc.format_missing({wc.EMBEDDING: [-1], wc.MOE_GATE: [3, 4, 5]}) == "embedding, moe_gate (layers 3..5, 3 total)"
    )


def test_chunk_size_env_selects_chunk_and_default_is_one_place(monkeypatch):
    from models.demos.deepseek_v3_d_p.utils import chunk_config

    monkeypatch.delenv("PREFILL_CHUNK_SIZE", raising=False)
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "2560")
    assert chunk_config.prefill_chunk_tokens() == 2560
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "")
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS
