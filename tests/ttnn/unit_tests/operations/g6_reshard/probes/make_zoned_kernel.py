# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: write probes/kernels/reader_zoned.cpp, the shipped issuer kernel plus
timing markers (P-T-ARGS / P-T-ACC instants, P-ADDR per piece address, P-RUNS per
piece issue). Re-run after every kernel edit; the zones cost ~18 cycles each.

    python tests/ttnn/unit_tests/operations/g6_reshard/probes/make_zoned_kernel.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
SRC = ROOT / "ttnn/ttnn/operations/g6_reshard/kernels/g6_reshard_reader.cpp"
DST = Path(__file__).parent / "kernels" / "reader_zoned.cpp"

EDITS = [
    (
        "        const auto in_accessor = TensorAccessor(in_args, in_addr, aligned_page_bytes);\n",
        '        { DeviceZoneScopedN("P-T-ARGS"); }\n'
        "        const auto in_accessor = TensorAccessor(in_args, in_addr, aligned_page_bytes);\n"
        '        { DeviceZoneScopedN("P-T-ACC"); }\n',
    ),
    (
        "                    uint64_t src = in_accessor.get_noc_addr(page_row0 + piece_col0, 0, noc);\n"
        "                    uint32_t dst = out_row_addr + (piece_col0 - block_col_begin) * aligned_page_bytes;\n",
        "                    uint64_t src;\n"
        "                    {\n"
        '                        DeviceZoneScopedN("P-ADDR");\n'
        "                        src = in_accessor.get_noc_addr(page_row0 + piece_col0, 0, noc);\n"
        "                    }\n"
        "                    uint32_t dst = out_row_addr + (piece_col0 - block_col_begin) * aligned_page_bytes;\n"
        '                    DeviceZoneScopedN("P-RUNS");\n',
    ),
]


def main():
    text = SRC.read_text()
    for old, new in EDITS:
        assert text.count(old) == 1, f"anchor not found once in {SRC}: {old!r}"
        text = text.replace(old, new)
    DST.parent.mkdir(exist_ok=True)
    DST.write_text(text)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
