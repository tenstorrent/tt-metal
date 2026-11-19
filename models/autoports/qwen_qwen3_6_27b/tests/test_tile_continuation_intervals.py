# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only coverage of logical interval preservation and cache tile eligibility."""

import ast
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "tt/bounded_prefill.py"
node = next(
    n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "chunk_intervals"
)
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
intervals = namespace["chunk_intervals"]


def test_every_offset_preserves_logical_span_and_safe_tile_writes():
    for start in [*range(1024), 262111, 262112, 262143]:
        for length in (1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1025):
            end = min(start + length, 262144)
            pieces = list(intervals(start, end))
            cursor = start
            serial_tokens = 0
            for absolute, count in pieces:
                assert absolute == cursor and 0 < count <= 512
                assert absolute // 512 == (absolute + count - 1) // 512
                if absolute % 32:
                    assert absolute == start and count <= 512 - absolute % 512
                    serial_tokens += count
                else:
                    # Whole-tile writes cannot overwrite the existing prefix;
                    # the final padded write stays within the context allocation.
                    assert absolute + ((count + 31) // 32) * 32 <= 262144
                cursor += count
            assert cursor == end and serial_tokens <= 511


def test_diagnostic_offsets_reduce_only_unaligned_head_work():
    for start, expected in ((1, 511), (31, 481), (32, 0), (481, 31), (962, 62), (7696, 496)):
        assert sum(n for s, n in intervals(start, start + 512) if s % 32) == expected
