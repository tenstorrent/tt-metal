# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-token logits readback on one chip.

On one device the qwen36 model has no on-device sampler (it needs a 1x4 or 1x8 mesh), so every decode
step reads the [1, 1, 32, vocab] bf16 TILE logits to the host and takes the argmax there. This times that
against a device argmax that returns one uint32 per row, and against untilizing the one real row on device
before the read.
    MESH_DEVICE=P150 pytest models/demos/blackhole/bonsai2/tp1/test_logits_readback.py -s
"""
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn

VOCAB = 248320
REPEATS = 30


def _ms(fn):
    fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_logits_readback(mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, VOCAB, dtype=torch.bfloat16)
    want = 123457
    x[..., want] = 100.0  # a unique maximum, so host and device argmax cannot differ on a tie
    logits = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def read_only():
        return ttnn.to_torch(logits)

    def host_argmax():
        return int(ttnn.to_torch(logits)[0, 0, 0].float().argmax())

    def device_argmax():
        idx = ttnn.argmax(logits, dim=-1)
        out = int(ttnn.to_torch(idx).flatten()[0])
        ttnn.deallocate(idx)
        return out

    def unpad_then_host_argmax():
        row = ttnn.untilize_with_unpadding(logits, output_tensor_end=[0, 0, 0, VOCAB - 1])
        out = int(ttnn.to_torch(row).flatten().float().argmax())
        ttnn.deallocate(row)
        return out

    assert host_argmax() == want
    assert device_argmax() == want
    assert unpad_then_host_argmax() == want
    ms = {
        "to_torch": _ms(read_only),
        "host_argmax": _ms(host_argmax),
        "device_argmax": _ms(device_argmax),
        "unpad_then_host_argmax": _ms(unpad_then_host_argmax),
    }
    for name, v in ms.items():
        logger.info(f"{name:>14}: {v:.3f} ms per token")
