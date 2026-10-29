# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device: QWEN38_MM_WIDE_BLOCK off keeps today's in0_block_w, on takes the table value."""

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm

# (K, N, activation cores, readers, today's block, wide block)
CASES = [
    (2560, 2112, 8, 2, 5, 10),
    (768, 2560, 8, 2, 3, 6),
    (2560, 2304, 8, 1, 5, 20),
    (2560, 512, 5, 1, 8, 16),
    (2560, 352, 5, 1, 8, 16),
    (2560, 8000, 40, 2, 2, 10),
    (2560, 7040, 40, 2, 2, 16),
]


class _Mesh:
    def dram_grid_size(self):
        class G:
            x, y = 8, 1

        return G()


@pytest.mark.parametrize("k,n,cores,readers,today,wide", CASES)
@pytest.mark.parametrize("on", [False, True])
def test_in0_block_w(monkeypatch, k, n, cores, readers, today, wide, on):
    monkeypatch.setattr(dm, "_MM_WIDE_BLOCK_ON", on)
    _, cfg = dm.dram_sharded_matmul_configs(_Mesh(), k, n, num_cores=cores, num_workers_per_dram_bank=readers)
    assert cfg.in0_block_w == (wide if on else today)


def test_table_blocks_divide_k_tiles():
    for (k, _), block in dm.MM_WIDE_BLOCKS.items():
        assert (k // 32) % block == 0
