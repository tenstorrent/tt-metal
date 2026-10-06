# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the LM-head column chunk plan.

The DRAM-sharded matmul reader streams each weight block with the largest
whole-tile page that divides the block and is at most the 16 KB NOC burst
(``get_max_page_size_and_num_pages`` in
``matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp``).  A block
is one bank's chunk width times the K block, two tiles for K=2560 over 40
storage cores, so a per-bank width that is not a multiple of four tiles pages
below 16 KB: the former 4736-column chunk (148 tiles, 19 per bank) paged at
4 KB and ran 832 us against 375 us for each 8192 chunk.  These tests pin a
plan on which every chunk pages at 16 KB with the least bank padding, and
show that column chunking leaves every logit and greedy candidate
bit-identical: each output column is its own sequential K accumulation,
independent of which columns share its chunk.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)

HIDDEN_SIZE = 2560
LOCAL_VOCAB_SIZE = 62_080
TP_SIZE = 4
TILE_SIZE = 32
DRAM_BANKS = 8
STORAGE_CORES = 40
BF16_TILE_BYTES = TILE_SIZE * TILE_SIZE * 2
NOC_BURST_BYTES = 16_384  # Blackhole NOC_MAX_BURST_SIZE
PLAN = embedding_module.LM_HEAD_CHUNK_COLUMNS
FORMER_PLAN = (8192,) * 7 + (4736,)
CACHE_NAME_CHARACTERS = set("abcdefghijklmnopqrstuvwxyz-0123456789")


def _chunk_ranges(plan: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    offsets = [0]
    for width in plan:
        offsets.append(offsets[-1] + width)
    return tuple(zip(offsets[:-1], offsets[1:]))


def _bank_stream(width: int):
    """(tiles per bank, padding tiles per K row, in1 block tiles, reader page bytes, program config)."""

    mesh_device = SimpleNamespace(dram_grid_size=lambda: ttnn.CoreCoord(DRAM_BANKS, 1))
    weight = dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, width)
    _, program = dram_sharded_matmul_configs(mesh_device, HIDDEN_SIZE, width, num_cores=STORAGE_CORES)
    tiles_per_bank = weight.shard_spec.shape[1] // TILE_SIZE
    padding = tiles_per_bank * DRAM_BANKS - width // TILE_SIZE
    block_tiles = tiles_per_bank * program.in0_block_w
    block_bytes = block_tiles * BF16_TILE_BYTES
    page = NOC_BURST_BYTES // BF16_TILE_BYTES * BF16_TILE_BYTES
    while block_bytes % page and page >= BF16_TILE_BYTES:
        page -= BF16_TILE_BYTES
    return tiles_per_bank, padding, block_tiles, page, program


def _sequential_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-column fp32 accumulation in K order, rounded once to BF16 like the device output."""

    accumulator = torch.zeros((hidden.shape[0], weight.shape[1]), dtype=torch.float32)
    for k in range(weight.shape[0]):
        accumulator += hidden[:, k : k + 1].float() * weight[k].float()
    return accumulator.to(torch.bfloat16)


def test_chunk_plan_tiles_the_local_vocabulary_exactly() -> None:
    assert PLAN == (8192,) * 5 + (7040,) * 3
    assert sum(PLAN) == LOCAL_VOCAB_SIZE == embedding_module.LOCAL_VOCAB_SIZE
    assert all(width % TILE_SIZE == 0 and width <= 8192 for width in PLAN)
    ranges = _chunk_ranges(PLAN)
    assert ranges[0][0] == 0 and ranges[-1][1] == LOCAL_VOCAB_SIZE
    assert all(end == next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:]))

    # Every local column lands in exactly one chunk, at offset column - start.
    owner = torch.full((LOCAL_VOCAB_SIZE,), -1, dtype=torch.int64)
    local = torch.full((LOCAL_VOCAB_SIZE,), -1, dtype=torch.int64)
    for index, (start, end) in enumerate(ranges):
        assert torch.all(owner[start:end] == -1)
        owner[start:end] = index
        local[start:end] = torch.arange(end - start)
    assert torch.all(owner >= 0)
    starts = torch.tensor([start for start, _ in ranges])
    assert torch.equal(local, torch.arange(LOCAL_VOCAB_SIZE) - starts[owner])


def test_every_chunk_streams_whole_noc_bursts_from_all_eight_banks() -> None:
    total_padding = 0
    for width in PLAN:
        tiles_per_bank, padding, block_tiles, page, program = _bank_stream(width)
        assert (program.in0_block_w, program.per_core_M) == (2, 1)
        assert program.per_core_N == math.ceil(width / (TILE_SIZE * STORAGE_CORES))
        assert tiles_per_bank % 4 == 0 and tiles_per_bank <= 32
        assert padding < tiles_per_bank  # no padding-only bank: all eight workers stream
        assert page == NOC_BURST_BYTES
        assert block_tiles * BF16_TILE_BYTES == page * (tiles_per_bank // 4)
        total_padding += padding
    assert {(width, _bank_stream(width)[0]) for width in PLAN} == {(8192, 32), (7040, 28)}
    # Bank widths that are multiples of four tiles sum to a multiple of 32
    # tiles per K row; 62080 columns are 1940 tiles, so any such plan pads at
    # least (-1940) % 32 = 12 tiles, and this one pads exactly that.
    assert total_padding == (-(LOCAL_VOCAB_SIZE // TILE_SIZE)) % 32 == 12


def test_former_tail_chunk_paged_below_the_noc_burst() -> None:
    assert sum(FORMER_PLAN) == LOCAL_VOCAB_SIZE
    tiles_per_bank, padding, block_tiles, page, _ = _bank_stream(FORMER_PLAN[-1])
    assert (tiles_per_bank, padding, block_tiles) == (19, 4, 38)
    assert page == 4096
    assert all(_bank_stream(width)[3] == NOC_BURST_BYTES for width in FORMER_PLAN[:-1])


def test_column_chunking_leaves_every_logit_and_greedy_candidate_bit_identical() -> None:
    generator = torch.Generator().manual_seed(62_080)
    rows = 3
    hidden = torch.randn((rows, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
    weight = torch.randn((HIDDEN_SIZE, LOCAL_VOCAB_SIZE), generator=generator, dtype=torch.bfloat16)

    full = _sequential_logits(hidden, weight)
    for plan in (PLAN, FORMER_PLAN):
        chunked = torch.cat([_sequential_logits(hidden, weight[:, start:end]) for start, end in _chunk_ranges(plan)], 1)
        assert chunked.shape == full.shape and torch.equal(chunked, full)

    # The local greedy candidates and their global ids follow from the logits alone.
    chunked = torch.cat([_sequential_logits(hidden, weight[:, start:end]) for start, end in _chunk_ranges(PLAN)], 1)
    local_index = torch.argmax(chunked, dim=-1)
    assert torch.equal(local_index, torch.argmax(full, dim=-1))
    assert torch.equal(chunked.max(dim=-1).values, full.max(dim=-1).values)
    for shard in range(TP_SIZE):
        start = shard * LOCAL_VOCAB_SIZE
        assert torch.equal(start + local_index, start + torch.argmax(full, dim=-1))


def test_lm_head_modules_bind_the_chunk_plan_and_cache_names() -> None:
    loader = inspect.getsource(embedding_module.Qwen38TTNNModelIOWeights.from_checkpoint)
    assert "for chunk_index, chunk_size in enumerate(LM_HEAD_CHUNK_COLUMNS):" in loader
    assert "slice(start + offset, start + offset + width)" in loader
    assert (
        'name=f"lm-head-chunk-dram-sharded-{chunk_index:02d}-cols-{chunk_offset}-{chunk_offset + chunk_size}"' in loader
    )
    assert "memory_config=dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, chunk_size)" in loader
    assert "lm_head_chunk_sizes=LM_HEAD_CHUNK_COLUMNS," in loader
    assert 'raise AssertionError("LM-head chunks do not cover one exact vocabulary shard")' in loader

    names = [
        f"lm-head-chunk-dram-sharded-{index:02d}-cols-{start}-{end}"
        for index, (start, end) in enumerate(_chunk_ranges(PLAN))
    ]
    assert names[0] == "lm-head-chunk-dram-sharded-00-cols-0-8192"
    assert names[5] == "lm-head-chunk-dram-sharded-05-cols-40960-48000"
    assert names[-1] == "lm-head-chunk-dram-sharded-07-cols-55040-62080"
    assert all(set(name) <= CACHE_NAME_CHARACTERS for name in names)
    # Former-plan artifacts were named by index alone and cannot load under the new names.
    assert not {f"lm-head-chunk-dram-sharded-{index:02d}" for index in range(8)} & set(names)

    head_init = inspect.getsource(embedding_module.Qwen38TTNNLMHead.__init__)
    assert "mesh_device, HIDDEN_SIZE, width, num_cores=40, num_workers_per_dram_bank=workers" in head_init
    assert "for width in weights.lm_head_chunk_sizes" in head_init
    call = inspect.getsource(embedding_module.Qwen38TTNNLMHead.__call__)
    assert call.count("ttnn.to_memory_config(full_hidden, self.hidden_act_memory_config)") == 1
    assert call.count("ttnn.linear(") == 1
    assert "for weight, program_config in zip(self.weights.lm_head_chunks, self.chunk_program_configs):" in call
    assert "ttnn.concat(outputs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in call
