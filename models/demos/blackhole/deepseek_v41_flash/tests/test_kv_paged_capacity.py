# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU: prints the capacity tables of docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md (``-s``) and checks the arithmetic.
Env DSV41_FREE_GIB: measured free DRAM per chip after the full model is resident (default 6.9)."""

import os

from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import (
    Formats,
    GiB,
    V41Geometry,
    capacity_table,
    comp_bytes_per_token,
    fixed_bytes_per_user,
    fmt_tokens,
    idx_bytes_per_token,
    max_context,
)

USERS = (1, 2, 4, 8, 16, 32)


def test_bytes_per_token():
    g = V41Geometry.from_config()
    bf16 = Formats.presets()["bf16"]
    assert (
        comp_bytes_per_token(g, bf16) == 3 * 512 + 1024
    )  # three ratio-2 sources + the ratio-1 source, bf16 rows of 512
    assert idx_bytes_per_token(g, bf16) == 3 * 128 + 256
    assert (
        comp_bytes_per_token(g, bf16, shared=False) == 18 * 512 + 20 * 1024
    )  # today: every reading layer keeps its own copy
    assert fixed_bytes_per_user(g, bf16)["ring"] == 40 * 128 * 512 * 2


def test_capacity_monotone_and_prints():
    g = V41Geometry.from_config()
    free = float(os.environ.get("DSV41_FREE_GIB", "6.9"))
    for name, fmt in Formats.presets().items():
        for shard in (1, 8):
            row = capacity_table(free, g, fmt, shard)
            assert all(a >= b for a, b in zip(row, row[1:]))
            print(
                f"CAP free={free} GiB fmt={name:7s} shard_cols={shard}: U/row {USERS} -> {[fmt_tokens(x) for x in row]}"
            )
    assert max_context(8, free * GiB, g, Formats.presets()["lean"], 8) >= 64 * 1024
