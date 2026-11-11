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
        "if (__builtin_constant_p(output_id) && __builtin_constant_p(ntiles)) { const std::uint32_t ctx = llk_pack_context(); forms = ctx == 0 ? llk_pack_block_forms<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 0) : (ctx == 1 ? llk_pack_block_forms<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 1) : llk_pack_block_forms<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 2)); }",
        "if (forms == 1) { #ifdef LLK_PACK_W_OFFSET llk_pack_state_set_half(output_id + 1); #endif llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles); return; }",
        "const bool one_form = forms != 0 && (forms & (forms - 1)) == 0;",
        "if (LLK_PACK_OTHER_PROGRAM_ACTIVE() || (DST_SYNC_MODE == DstSync::SyncHalf && LLK_PACK_L1_ACC_ACTIVE())) { if (!one_form) { _llk_pack_close_fast_untilize_<DST_SYNC_MODE>(); } llk_pack_block_per_tile<is_fp32_dest_acc_en>(start_tile_index, output_id, ntiles); return; }",
        "const std::uint32_t state = llk_pack_state_half(); const bool first_block = state != output_id + 1u; const bool reconfigured = (state >> 8) != 0;",
        "const char form = forms == 2 ? 'B' : (first_block ? (reconfigured ? llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 1) : llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 0)) : llk_pack_block_form<is_fp32_dest_acc_en, matmul_kt>(output_id, ntiles, 2));",
        "if (!one_form && first_block) { _llk_pack_close_fast_untilize_<DST_SYNC_MODE>(); } llk_pack_state_set_half(output_id + 1); #else",
        "const bool take_w = form == 'W' && (forms == 0 || (forms & 4) != 0); if (llk_pack_block_is_contiguous(output_id)) {",
        "llk_pack_block_form_of<is_fp32_dest_acc_en, matmul_kt>(row, n, tables[state], state, ctx0 + ctx * step);",
        "llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_init_form, llk_pack_init_form_dc), llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_reconf_form, llk_pack_reconf_form_dc), llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_next_form, llk_pack_next_form_dc)};",
        "return llk_pack_src32_family<is_fp32_dest_acc_en> && llk_pack_unpack_to_dest ? u : dc;",
        "inline constexpr bool llk_pack_src32_family = is_fp32_dest_acc_en && (llk_pack_src32_mask(0) != 0 || llk_pack_src32_mask(32) != 0);",
        "const std::uint32_t max_tiles = llk_pack_dst_is_wide(output_id) ? LLK_PACK_W_MAX_TILES_W : LLK_PACK_W_MAX_TILES;",
        "const bool cont = pack_w_valid && pack_w_out == output_id && pack_w_page == page && pack_w_next == addr && pack_w_tiles + ntiles <= max_tiles; if (!cont) { program_packer_destination(addr);",
        "if (llk_pack_state_half() != output_id + 1u) { llk_pack_state_set_half(output_id + 1); return false; }",
        "#define LLK_PACK_W_MAX_TILES 127",
        "constexpr std::uint32_t LLK_PACK_PAGE_WORDS_WIDE = 256;",
    ):
        assert line in src, f"llk_pack_block changed ({line!r}): port it to block() in tests/helpers/include/matmul_pack_production.h"
