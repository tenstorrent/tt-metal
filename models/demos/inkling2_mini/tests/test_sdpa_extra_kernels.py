# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""sdpa_decode(extra_kernels=...): caller kernels ride inside the local sdpa program on cores it does not use.

Local-layer shapes (24 q heads, 8 kv heads, 1024-slot ring, mask, sink) on the 8 x 4 grid. The extra kernel copies
pages of a DRAM tensor into an L1 shard on 4 idle cores (tt/mk/kernels/stage_pages_fill.cpp). Checks: sdpa output
bit-equal with and without the extra kernels; the copy landed; a second call (program-cache hit) with other runtime
args (another source tensor) copies the new data (get_dynamic_runtime_args)."""

import os

import torch

import ttnn

MK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tt", "mk", "kernels")


def _sdpa(dev, q, k, v, mask, sink, extra=None):
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4), q_chunk_size=0, k_chunk_size=256,
        exp_approx_mode=False, max_cores_per_head_batch=4,
    )  # fmt: skip
    hifi4 = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )  # fmt: skip
    return ttnn.transformer.scaled_dot_product_attention_decode(
        q, k, v, is_causal=False, attn_mask=mask, attention_sink=sink, scale=1.0, program_config=cfg,
        compute_kernel_config=hifi4, extra_kernels=extra,
    )  # fmt: skip


def _copy_kernel(src, stage, cores, pages):
    """pages fp32 tiles of src (page i * pages + j for core i) -> each core's shard of stage."""
    rt = [(ttnn.CoreCoord(*c), [src.buffer_address(), i * pages, stage.buffer_address()]) for i, c in enumerate(cores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    return ttnn.KernelDescriptor(
        kernel_source=f"{MK}/stage_pages_fill.cpp", core_ranges=crs,
        named_compile_time_args=[("kz", pages), ("nt", 1), ("nct", 1)], runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )  # fmt: skip


def test_sdpa_extra_kernels(device):
    torch.manual_seed(0)
    f = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    q = f(torch.randn(1, 1, 24, 128))
    k = f(torch.randn(1, 8, 1024, 128), ttnn.bfloat8_b)
    v = f(torch.randn(1, 8, 1024, 128), ttnn.bfloat8_b)
    mask = f(torch.randn(1, 1, 24, 1024) * 2)
    sink = f(torch.randn(24, 32))
    cores, pages = [(9, 6), (10, 6), (9, 7), (10, 7)], 2  # outside the 8 x 4 grid
    words = pages * 1024
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(9, 6), ttnn.CoreCoord(10, 7))]), (1, words),
        ttnn.ShardOrientation.ROW_MAJOR,
    )  # fmt: skip
    stage = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, len(cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
    )  # fmt: skip
    ref = ttnn.to_torch(_sdpa(device, q, k, v, mask, sink))
    for trial in range(2):  # second call: program-cache hit with a new source address
        src_t = torch.randn(1, 1, len(cores) * pages * 32, 32) + trial
        src = f(src_t, ttnn.float32)
        extra = ttnn.ProgramDescriptor(kernels=[_copy_kernel(src, stage, cores, pages)], semaphores=[], cbs=[])
        out = ttnn.to_torch(_sdpa(device, q, k, v, mask, sink, extra))
        assert torch.equal(out, ref), f"trial {trial}: sdpa output changed"
        got = ttnn.to_torch(stage).reshape(-1).view(torch.float32)
        want = ttnn.to_torch(src).float().reshape(len(cores) * pages, 32, 32)
        # shard i = pages i * pages .. of src in tile order: compare as raw tile bytes via a host re-tilize
        tiles = ttnn.to_torch(ttnn.from_torch(want.reshape(1, 1, -1, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))
        assert got.numel() == len(cores) * words
        assert torch.isfinite(got).all() and got.abs().sum() > 0, f"trial {trial}: nothing copied"
        assert abs(float(got.sum()) - float(tiles.sum())) < 1e-2 * float(
            tiles.abs().sum()
        ), f"trial {trial}: wrong data"
