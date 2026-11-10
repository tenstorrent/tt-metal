# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""matmul_pack_tables.h (the block pack forms the matmul test kernels run) is generated from llk_pack_block of the metal API; it fails when the two drift."""

import importlib.util
import pathlib

GEN = pathlib.Path(__file__).resolve().parents[2] / "python_tests/gen_matmul_pack_tables.py"


def test_tables_match_api():
    spec = importlib.util.spec_from_file_location("gen_matmul_pack_tables", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUT.read_text() == mod.generate(), "matmul_pack_tables.h differs from llk_pack_tile_api.h: run tests/python_tests/gen_matmul_pack_tables.py --write"
