# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The GLM / DeepSeek prefill KV chunk table must name its configs like blaze's decode table.

The KV manager loads both tables and pairs them by config NAME to assign ids. blaze names its decode
configs with at least two digits (kv_chunk_migration_helpers: ``width = max(2, len(str(n - 1)))``,
``f"{i:0{width}d}"``), so a table naming them "0"/"1" is rejected with

    [ERROR] KV config '0': name-to-id mismatch across the loaded tables
    [FATAL] KV chunk tables loaded but do not describe this instance

and every kv-manager exits 1. Observed on GLM-5.3 P/D (pdg-glm-a7, 2026-09-29): decode published
"00"/"01", this side published "0"/"1", 24/24 managers CrashLoopBackOff.
"""

from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import config_name, dflash_config_name


def _blaze_config_name(config_id: int, num_configs: int) -> str:
    """Verbatim from tt-blaze blaze/kv_chunk_migration_helpers.py:897-898."""
    width = max(2, len(str(num_configs - 1)))
    return f"{config_id:0{width}d}"


def test_config_names_match_blaze_zero_padding():
    # The GLM case: 2 block-cyclic configs (KVPE, index) -> blaze publishes "00"/"01".
    assert [config_name(i, 2) for i in range(2)] == ["00", "01"]
    # Parity with blaze's own formula across widths.
    for num_configs in (1, 2, 3, 9, 10, 99, 100, 120):
        for config_id in range(min(num_configs, 4)):
            assert config_name(config_id, num_configs) == _blaze_config_name(config_id, num_configs)
    assert config_name(7, 120) == "007"


def test_config_names_sort_in_id_order():
    # build_and_serialize_kv_chunk_table asserts names == sorted(names) because the protobuf
    # renumbers by name; padding is what keeps that true past nine configs.
    for num_configs in (2, 10, 120):
        names = [config_name(i, num_configs) for i in range(num_configs)]
        assert names == sorted(names)


def test_block_cyclic_names_sort_ahead_of_dflash():
    # Digits sort before letters, so the drafter configs keep their ids after the block-cyclic ones.
    names = [config_name(i, 2) for i in range(2)] + [dflash_config_name(k, h) for k in ("k", "v") for h in range(2)]
    assert names == sorted(names)
