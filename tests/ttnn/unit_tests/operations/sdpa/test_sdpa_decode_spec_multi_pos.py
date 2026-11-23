# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``spec_multi_pos_tiles`` paged SDPA decode: post-commit smoke.

Full sweep: tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode_spec_multi_pos.py.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.spec_multi_pos_test_utils import (
    GROUP_CHUNK,
    HEAD_DIM,
    SCALE,
    TILE,
    build_inputs,
    check,
    check_bit_exact,
    check_groups,
    check_groups_bit_exact,
    config_for,
    minimal_spec_args,
    program_config,
    run_reference,
    torch_reference,
)

# One module-scoped device: the cases reuse the same two programs per (T, seq_len).
pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("T", [4, 7], ids=["T4", "T7"])
@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),  # p % 32 == 0, bounds inside one chunk
        (2048, 1021),  # bounds straddle a 128-chunk boundary -> TWO masked chunks
    ],
    ids=["s2k_p1024", "s2k_p1021_straddle"],
)
def test_spec_multi_pos_matches_batched(device, T, seq_len, p):
    torch.manual_seed(0)
    check(device, T, p, seq_len)


def test_spec_multi_pos_is_bit_exact(device):
    torch.manual_seed(5)
    check_bit_exact(device, T=4, p=1024, seq_len=2048)


def test_spec_multi_pos_first_block(device):
    """cur_pos inside the very first tile — the mask cuts inside column-tile 0."""
    torch.manual_seed(2)
    check(device, T=7, p=5, seq_len=512, seed=11)


def test_spec_multi_pos_dynamic_chunk(device):
    """In-kernel k-chunk: spec caps it at 4 tiles, the reference doesn't, hence the looser 0.999."""
    torch.manual_seed(3)
    pc = program_config(device, max_cores=64, k_chunk_size=0)
    check(device, T=4, p=1024, seq_len=2048, seed=13, program_config=pc, pcc_ref_vs_spec=0.999)


# ── Validation ─────────────────────────────────────────────────────────────


def test_rejects_page_table_batch_mismatch(device, expect_error):
    """Page table needs one row per Q batch group."""
    T = 4
    inp, pc = minimal_spec_args(device, T=T)
    page_table = inp["page_row"].unsqueeze(0).repeat(T, 1).contiguous()
    with expect_error(RuntimeError, "one row per Q batch group"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_q_row_count_mismatch(device, expect_error):
    """spec_multi_pos_tiles must equal the number of 32-row Q tiles."""
    T = 4
    inp, pc = minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "padded rows"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T + 1,
        )


def test_rejects_cur_pos_length_mismatch(device, expect_error):
    """cur_pos must carry exactly T bounds — one per candidate row-tile."""
    T = 4
    inp, pc = minimal_spec_args(device, T=T)
    short_cur_pos = ttnn.Tensor(torch.tensor([100, 101], dtype=torch.int32), ttnn.int32).to(device)
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_sliding_window(device, expect_error):
    """The sliding-window mask shares the causal mask machinery; not combined for now."""
    T = 4
    inp, pc = minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "sliding_window_size"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            sliding_window_size=256,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_legacy_path_unaffected(device):
    """spec_multi_pos_tiles omitted -> the pre-change op, bit-for-bit. Backward-compat guard."""
    torch.manual_seed(4)
    T = 4
    inp = build_inputs(device, T, p=1000, seq_len=2048, seed=5)
    pc = config_for(device, T)
    ref = run_reference(device, inp, T, pc)
    torch_ref = torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"legacy path regressed: {msg}"


# ── Batch groups (B >= 1) ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),  # both groups inside one chunk
        (2048, 1020),  # group 0 ends at 1023 (8 chunks), group 1 at 1027 (9) -> scan ranges differ
    ],
    ids=["s2k_p1024", "s2k_p1020_split_scan"],
)
def test_spec_multi_pos_groups_matches_batched(device, seq_len, p):
    """B=2 groups of Tg=4 at 55 cores/head, the full-grid config."""
    check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=55, seed=31)


def test_spec_multi_pos_groups_is_bit_exact(device):
    check_groups_bit_exact(device, seq_len=2048, p=1024)


def test_spec_multi_pos_groups_dynamic_chunk(device):
    """In-kernel k-chunk with the two groups on different chunk sizes (2 vs 4 tiles)."""
    B, Tg, p = 2, 4, 60
    assert (p + Tg - 1) // 32 + 1 == 2 and (p + 2 * Tg - 1) // 32 + 1 == 3  # 2 tiles vs 4 (pow2)
    check_groups(device, B=B, Tg=Tg, p=p, seq_len=512, max_cores=16, seed=47, k_chunk_size=0, pcc=0.999)


def test_spec_groups_b3_tg4_full_grid(device):
    """B=3, Tg=4, dynamic chunk: the T=12 config the model uses."""
    check_groups(device, B=3, Tg=4, p=1024, seq_len=2048, max_cores=36, seed=67, k_chunk_size=0, pcc=0.999)


def test_rejects_group_cur_pos_length(device, expect_error):
    """cur_pos must carry B*Tg bounds — Tg per batch group, not Tg in total."""
    B, Tg = 2, 4
    inp = build_inputs(device, B * Tg, p=100, seq_len=512, seed=61)
    short_cur_pos = ttnn.Tensor(torch.arange(Tg, dtype=torch.int32), ttnn.int32).to(device)
    q_spec = inp["q_batched"].reshape(1, B, Tg * TILE, HEAD_DIM)
    page_table = inp["page_row"].unsqueeze(0).repeat(B, 1).contiguous()
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(q_spec, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=program_config(device, max_cores=16, k_chunk_size=GROUP_CHUNK),
            spec_multi_pos_tiles=Tg,
        )
