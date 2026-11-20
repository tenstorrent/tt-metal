# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What a block-format weight tensor holds, by how it was created (no kernel of ours runs here: ttnn creates the
tensor, ttnn reads it back).

    host      ttnn.from_torch(float32, dtype, TILE) on the host, then moved to the device: one rounding
    sharded   ttnn.from_torch(float32, dtype, TILE, device=, memory_config=<sharded>): what the streamed pieces'
              DRAM tensors are made with
    cache     ttnn.as_tensor(..., cache_file_name=) with the same sharded memory config: what their prepared
              streams are made with

Each is compared with the host's single rounding (lab.stored) and with "BF16 first, then the block format"
(lab.stored_sharded): ttnn/core/tensor/py_to_tt_tensor.cpp compute_host_dtype asks for a BFLOAT16 host buffer when
the target is a block format and the memory config is sharded. One HY4_MK9 line per case. Board only."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.fidelity import lab as LAB

pytestmark = pytest.mark.use_module_device


def stream_config(rows):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (rows, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


@pytest.mark.parametrize("how", ["host", "sharded", "cache"])
@pytest.mark.parametrize("fmt", ["bfp8", "bfp4"])
def test_stream_reference(device, tmp_path, fmt, how):
    if LAB.MOCK:
        pytest.skip("reads the tensor back from the device")
    g = torch.Generator().manual_seed(4)
    w = torch.randn(256, 256, generator=g) * 0.05
    dtype = LAB.DTYPES[fmt]
    config = stream_config(256)
    if how == "host":
        t = ttnn.from_torch(w[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT).to(device, config)
    elif how == "sharded":
        t = ttnn.from_torch(w[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
    else:
        kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)
        cache = str(tmp_path / f"stream_{fmt}")
        ttnn.deallocate(ttnn.as_tensor(w[None, None], cache_file_name=cache, **kwargs))
        t = ttnn.as_tensor(None, cache_file_name=cache, **kwargs)  # loaded from the file, as the prepared streams
    back = ttnn.to_torch(t).float().reshape(256, 256)
    ttnn.deallocate(t)
    one = LAB.stored(w, fmt)
    two = LAB.stored_sharded(w, fmt)
    step = (back - one).abs()
    logger.info(
        f"HY4_MK9 stream reference {fmt} [{how}]: equal to the host's single rounding in "
        f"{float((back == one).float().mean()):.4f} of the entries, to BF16-then-{fmt} in "
        f"{float((back == two).float().mean()):.4f}; relative rms difference to the single rounding "
        f"{(step.double().pow(2).sum().sqrt() / one.double().pow(2).sum().sqrt()).item():.3e}, to the float32 "
        f"weights {((back - w).double().pow(2).sum().sqrt() / w.double().pow(2).sum().sqrt()).item():.3e} "
        f"(the single rounding: {((one - w).double().pow(2).sum().sqrt() / w.double().pow(2).sum().sqrt()).item():.3e})"
    )
    if how == "host":
        assert torch.equal(back, one)
    else:
        assert torch.equal(back, two), "a sharded block-format tensor is expected to be rounded to BF16 first"
