# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Board probe: does the Blackhole unpacker read a ternary tile stored as 256 B of Bfp2_b data with a forced
shared exponent (no exponent section)? out = act @ W on one core, act bf16."""

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.pack import pack_ternary_tiles, unpack_ternary_tiles

K = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"


def _run(device, act, words, shared_exp, force=1):
    words = np.asarray(words, dtype=np.uint32).reshape(-1, 64)
    n = words.shape[0]
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    act_t = ttnn.from_torch(act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w_t = ttnn.from_torch(
        torch.from_numpy(words.view(np.int32)).reshape(n, 64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([n * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    def cb(index, fmt, page, pages=2):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    reader_args = ttnn.TensorAccessorArgs(act_t).get_compile_time_args()
    reader_args.extend(ttnn.TensorAccessorArgs(w_t).get_compile_time_args())
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [act_t.buffer_address(), w_t.buffer_address(), n]
    wr_rt[0][0] = [out_t.buffer_address(), n]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=K + "probe_reader.cpp",
                core_ranges=core,
                compile_time_args=reader_args,
                runtime_args=rd_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=K + "probe_writer.cpp",
                core_ranges=core,
                compile_time_args=ttnn.TensorAccessorArgs(out_t).get_compile_time_args(),
                runtime_args=wr_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=K + "probe_compute.cpp",
                core_ranges=core,
                compile_time_args=[shared_exp, force, n],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            ),
        ],
        semaphores=[],
        cbs=[cb(0, ttnn.bfloat16, 2048), cb(1, ttnn.bfloat16, 256), cb(16, ttnn.bfloat16, 2048)],
    )
    out = ttnn.generic_op([act_t, w_t, out_t], program)
    return ttnn.to_torch(out).float()


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(3, 32, 32)).astype(np.int8)
    assert np.array_equal(unpack_ternary_tiles(pack_ternary_tiles(t)), t)


@pytest.mark.parametrize("shared_exp", [127])
def test_ternary_tile_identity(device, shared_exp):
    rng = np.random.default_rng(1)
    t = rng.integers(-1, 2, size=(32, 32)).astype(np.int8)
    words = pack_ternary_tiles(t)
    out = _run(device, torch.eye(32), words, shared_exp)
    vals = sorted(set(out.flatten().tolist()))
    print("distinct output values", vals[:10])
    nz = out[torch.from_numpy(t) != 0]
    scale = (nz.abs().max()).item()
    print("scale", scale, "match", torch.equal(out / scale, torch.from_numpy(t).float()))
    bad = (out / scale != torch.from_numpy(t).float()).nonzero()
    print("mismatches", bad.shape[0], bad[:8].tolist())
    assert torch.equal(out / scale, torch.from_numpy(t).float())


def test_ternary_tile_random_act(device):
    rng = np.random.default_rng(2)
    t = rng.integers(-1, 2, size=(32, 32)).astype(np.int8)
    act = torch.randn(32, 32).bfloat16()
    out = _run(device, act, pack_ternary_tiles(t), 127)
    ref = act.float() @ torch.from_numpy(t).float()
    scale = (out.abs().sum() / ref.abs().sum()).item()
    rel = ((out / scale - ref).norm() / ref.norm()).item()
    print("scale", scale, "rel err", rel)
    assert rel < 0.02
