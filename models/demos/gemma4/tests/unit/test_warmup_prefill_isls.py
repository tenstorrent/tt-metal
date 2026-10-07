# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the GEMMA4_WARMUP_PREFILL_ISLS ladder selection."""

from models.demos.gemma4.tt.warmup_isls import ENV_VAR, parse_warmup_isls, warmup_prefill_isls


def test_parse_is_tolerant_sorted_and_unique():
    assert parse_warmup_isls("8192, 4096,abc,,4096,-1,0,32768") == [4096, 8192, 32768]
    assert parse_warmup_isls(None) == [] and parse_warmup_isls("") == []


def test_unset_env_warms_nothing_extra():
    assert warmup_prefill_isls(262144, already_warmed=[32, 128, 512], env={}) == []


def test_ladder_is_capped_by_context_and_skips_already_warmed_lengths():
    env = {ENV_VAR: "128,512,4096,8192,16384,32768,65536"}
    assert warmup_prefill_isls(16384, already_warmed=[32, 128, 512, 4096], env=env) == [8192, 16384]
    assert warmup_prefill_isls(None, already_warmed=[], env=env) == [128, 512, 4096, 8192, 16384, 32768, 65536]
