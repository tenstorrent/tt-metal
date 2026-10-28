# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("height", [4032, 4033])
def test_slice_row_major_batch_preserves_live_l1(device, start, height):
    # 4033 rows split over two cores produce 2017 and 2016 rows. Their
    # independently chosen batches are 64 and 63, while the shared CB holds 128.
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    values = (torch.arange(height * 256).reshape(1, 1, height, 256) % 127).bfloat16()
    source = ttnn.from_torch(values, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sentinel = 0x5A5A5A5A
    live = []

    def run():
        result = ttnn.slice(
            source,
            [0, 0, 0, start],
            [1, 1, height, start + 32],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=cores,
        )
        actual = ttnn.to_torch(result)
        ttnn.deallocate(result)
        return actual

    run()
    try:
        for words in (8192, 256, 16):
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
            )
            while True:
                try:
                    live.append(
                        ttnn.from_torch(
                            torch.full((2, words), sentinel, dtype=torch.int32),
                            dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device,
                            memory_config=memory,
                        )
                    )
                except RuntimeError:
                    break
        live.sort(key=lambda tensor: tensor.buffer_address())
        while live:
            try:
                actual = run()
                break
            except RuntimeError as exc:
                if "clash with L1 buffers" not in str(exc):
                    raise
                ttnn.deallocate(live.pop(0))
        assert live, "No live sentinels fit beside the slice program"
        changed = sum(int((ttnn.to_torch(tensor) != sentinel).sum()) for tensor in live)
        assert changed == 0, f"Slice overwrote {changed} words in unrelated live L1 tensors"
        assert torch.equal(actual, values[..., start : start + 32])
    finally:
        for tensor in live:
            ttnn.deallocate(tensor)
