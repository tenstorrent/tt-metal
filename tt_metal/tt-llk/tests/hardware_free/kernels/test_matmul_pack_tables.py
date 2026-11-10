# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""matmul_pack_tables.h (the block pack forms the matmul test kernels run) is generated from llk_pack_block of the metal API; it fails when the two drift."""

import importlib.util
import pathlib
import re

GEN = pathlib.Path(__file__).resolve().parents[2] / "python_tests/gen_matmul_pack_tables.py"
API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"


def test_tables_match_api():
    spec = importlib.util.spec_from_file_location("gen_matmul_pack_tables", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OUT.read_text() == mod.generate(), "matmul_pack_tables.h differs from llk_pack_tile_api.h: run tests/python_tests/gen_matmul_pack_tables.py --write"
    assert mod.API.read_text() == mod.api_text(), "the generated region of llk_pack_tile_api.h differs from llk_pack_block: run tests/python_tests/gen_matmul_pack_tables.py --write"


def test_block_selection_matches_api():
    """block() in matmul_pack_production.h hand-mirrors how llk_pack_block picks the form table, the input family and the W offset run; these are the API lines it mirrors."""
    src = re.sub(r"\s+", " ", API.read_text())
    for line in (
        "const bool first_block = pack_w_last != output_id;",
        "first_block ? (pack_reconfigured ? llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_reconf_form, llk_pack_reconf_form_dc) : llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_init_form, llk_pack_init_form_dc)) : llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_next_form, llk_pack_next_form_dc), !first_block);",
        "return llk_pack_src32_family<is_fp32_dest_acc_en> ? u : dc;",
        "inline constexpr bool llk_pack_src32_family = is_fp32_dest_acc_en && (llk_pack_src32_mask(0) != 0 || llk_pack_src32_mask(32) != 0);",
        "const std::uint32_t max_tiles = llk_pack_dst_is_wide(output_id) ? LLK_PACK_W_MAX_TILES_W : LLK_PACK_W_MAX_TILES;",
        "const bool cont = pack_w_valid && pack_w_out == output_id && pack_w_page == page && pack_w_next == addr && pack_w_tiles + ntiles <= max_tiles;",
        "#define LLK_PACK_W_MAX_TILES 127",
        "constexpr std::uint32_t LLK_PACK_PAGE_WORDS_WIDE = 256;",
    ):
        assert line in src, f"llk_pack_block changed ({line!r}): port it to block() in tests/helpers/include/matmul_pack_production.h"
