# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device checks of the paged QSA KV mode: the torch references, and the PAGED opt-in discipline of the kernels
(the composed forms and the programs run in test_fused_qsa_paged_device.py)."""

import inspect
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.kv_page_scatter import kv_page_scatter

KERNELS = Path(qsa_block.__file__).parent / "kernels"
SCATTER_KERNEL = Path(kv_page_scatter.__code__.co_filename).parent / "kernels" / "scatter_rows.cpp"


@pytest.mark.parametrize("page_size", [32, 64, 128, 256])
def test_selection_mapping_is_a_bijection_on_lane_tokens_and_keeps_sentinels(page_size):
    generator = torch.Generator().manual_seed(1)
    blocks = 16
    table, pages = ref.random_page_table(4, blocks, generator)
    tokens = torch.randint(0, blocks * page_size, (4, 2080), generator=generator)
    tokens[:, ::37] = ref.MASKED
    out = ref.selection_row_paged_reference(tokens, table, page_size)
    assert torch.equal(out == ref.MASKED, tokens == ref.MASKED)
    live = tokens != ref.MASKED
    assert int(out[live].max()) < pages * page_size and int(out[live].min()) >= page_size  # never page 0
    for r in range(4):
        lane = live[r]
        assert torch.equal(out[r][lane] & (page_size - 1), tokens[r][lane] & (page_size - 1))
        # the lane's pages are its own: distinct tokens of distinct pages land on distinct rows
        unique_in = torch.unique(tokens[r][lane])
        unique_out = torch.unique(out[r][lane])
        assert unique_in.numel() == unique_out.numel()


def test_identity_page_table_is_the_flat_cache_with_a_lane_offset():
    blocks, page_size = 8, 32
    table = torch.zeros(32, blocks, dtype=torch.int32)
    for u in range(3):
        table[u] = torch.arange(blocks, dtype=torch.int32) + u * blocks  # lane u owns pages u*blocks ..
    tokens = torch.randint(0, blocks * page_size, (3, 64))
    out = ref.selection_row_paged_reference(tokens, table, page_size)
    for u in range(3):
        assert torch.equal(out[u], tokens[u] + u * blocks * page_size)  # the unpaged lane offset u * C


@pytest.mark.parametrize("page_size", [32, 64])
def test_block_writes_land_in_the_lane_pages_only(page_size):
    generator = torch.Generator().manual_seed(2)
    blocks, lanes = 8, 5
    table, pages = ref.random_page_table(lanes, blocks, generator)
    pool = torch.zeros(pages * page_size, 512)
    starts = (torch.randint(0, blocks * page_size, (lanes,), generator=generator)) & ~31
    data = torch.randn(lanes, 32, 512, generator=generator)
    bases = ref.kv_block_base_reference(table, starts, page_size)
    out = ref.pool_after_block_writes(pool, data, bases)
    assert int((out.abs().sum(dim=1) > 0).sum()) == lanes * 32
    assert not out[:page_size].any()  # the scratch page stays clean
    for u in range(lanes):
        lane_pages = {int(p) for p in table[u]}
        rows = torch.arange(int(bases[u]), int(bases[u]) + 32)
        assert {int(r) // page_size for r in rows} <= lane_pages
        assert torch.equal(out[rows], data[u])


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 1000, 4097])
@pytest.mark.parametrize("page_size", [32, 64])
def test_scatter_reference_copies_whole_groups(rows, page_size):
    generator = torch.Generator().manual_seed(3)
    blocks = -(-4128 // page_size // 8) * 8
    table, pages = ref.random_page_table(2, blocks, generator)
    src = torch.randn(4128, 512, generator=generator)
    pool = torch.randn(pages * page_size, 512, generator=generator)
    out = ref.pool_after_scatter(pool, src, table[1], rows, page_size)
    count = -(-rows // 32) * 32
    changed = (out != pool).any(dim=1)
    assert int(changed.sum()) == count
    mapped = ref.physical_rows(table[1], torch.arange(count), page_size)
    assert torch.equal(out[mapped], src[:count])


def test_paged_flag_is_compile_time_and_the_unpaged_program_is_untouched():
    selection = (KERNELS / "selection_row.cpp").read_text()
    writer = (KERNELS / "main_tail_writer_staging.cpp").read_text()
    for source in (selection, writer):
        assert source.count("#ifdef PAGED") >= 1
    assert qsa_block.PAGED_DEFINE == ("PAGED", "1")
    for function in (qsa_block.selection_row, qsa_block.selection_row_composed):
        parameters = inspect.signature(function).parameters
        assert parameters["page_table"].default is None and parameters["page_size"].default == 0
    parameters = inspect.signature(qsa_block.main_tail).parameters
    assert parameters["page_table"].default is None and parameters["page_size"].default == 0
    assert "defines=[PAGED_DEFINE] if paged else ()" in inspect.getsource(qsa_block.main_tail)
    assert "defines = [PAGED_DEFINE]" in inspect.getsource(qsa_block.selection_row)
    # unpaged: the same accessor arguments and the same CB size (8 KB) as before
    assert "paged_args, defines, cb_bytes = [], [], 8192" in inspect.getsource(qsa_block.selection_row)


def test_page_arguments_are_validated(expect_error):
    assert qsa_block.page_shift(32) == 5 and qsa_block.page_shift(256) == 8
    for bad in (0, 16, 48, 100):
        with expect_error(ValueError, "power of two"):
            qsa_block.page_shift(bad)
    assert "pt_l1[v >> shift] * bs + (v & (bs - 1))" in (KERNELS / "selection_row.cpp").read_text()
    assert "table[t0 >> shift] * bs + (t0 & (bs - 1))" in SCATTER_KERNEL.read_text()
