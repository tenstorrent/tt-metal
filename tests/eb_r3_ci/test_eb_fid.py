# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): binary_ng FPU multiplies of bf16 and block-float operands, the cases the HiFi2 fidelity
rule reaches. Each test saves its output to $EB_BITS_DIR (when set) so that two hosts can be compared bit for bit, and the
device profiler plugin can time them. bf16 takes the FPU path only with fast_and_approximate_mode=True."""
import os
import zlib
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)


DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def _save(name, out):
    d = os.environ.get("EB_BITS_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        torch.save(ttnn.to_torch(out), os.path.join(d, name + ".pt"))


def _mem(kind, shape):
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if kind == "l1":
        return ttnn.L1_MEMORY_CONFIG
    strat, grid = {
        "hs8": (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=4)),
        "bs64": (ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=8, x=8)),
        "ws32": (ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=4, x=8)),
    }[kind]
    return ttnn.create_sharded_memory_config(shape, core_grid=grid, strategy=strat, orientation=ttnn.ShardOrientation.ROW_MAJOR)


CASES = [
    # (id, shape a, shape b, dtype a, dtype b, memory)
    ("bf16_hs8", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bf16", "bf16", "hs8"),
    ("bf16_bs64", (1, 1, 4096, 1280), (1, 1, 4096, 1280), "bf16", "bf16", "bs64"),
    ("bf16_ws32", (1, 1, 32, 4096), (1, 1, 32, 4096), "bf16", "bf16", "ws32"),
    ("bfp8_hs8", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bfp8", "hs8"),
    ("bfp8_bs64", (1, 1, 4096, 1280), (1, 1, 4096, 1280), "bfp8", "bfp8", "bs64"),
    ("bfp4_hs8", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp4", "bfp4", "hs8"),
    ("bf16_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bf16", "bf16", "dram"),
    ("bfp8_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bfp8", "dram"),
    ("bfp8_l1", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bfp8", "l1"),
    ("bf16_bfp8_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bf16", "bfp8", "dram"),
    ("bfp8_rowb_dram", (1, 1, 1024, 1024), (1, 1, 1, 1024), "bfp8", "bfp8", "dram"),
    ("bfp8_colb_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1), "bfp8", "bfp8", "dram"),
    ("bfp8_scalarb_dram", (1, 1, 1024, 1024), (1, 1, 1, 1), "bfp8", "bfp8", "dram"),
    ("bf16_rowb_dram", (1, 8, 512, 512), (1, 1, 1, 512), "bf16", "bf16", "dram"),
    ("bfp4_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp4", "bfp4", "dram"),
    ("bfp8_bfp4_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bfp4", "dram"),
    ("bfp8_ws32", (1, 1, 32, 4096), (1, 1, 32, 4096), "bfp8", "bfp8", "ws32"),
    ("bfp8_rowb_l1", (1, 8, 512, 512), (1, 1, 1, 512), "bfp8", "bfp8", "l1"),
    # mixed: SrcB (b) block-float is exact at HiFi2, SrcB bf16 is not
    ("bf16_bfp8_hs8", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bf16", "bfp8", "hs8"),
    ("bfp8_bf16_hs8", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bf16", "hs8"),
    ("bf16_bfp4_bs64", (1, 1, 4096, 1280), (1, 1, 4096, 1280), "bf16", "bfp4", "bs64"),
    ("bfp8_bf16_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bfp8", "bf16", "dram"),
    ("bf16_bfp8_rowb_dram", (1, 8, 512, 512), (1, 1, 1, 512), "bf16", "bfp8", "dram"),
    ("bf16_bfp8_colb_dram", (1, 1, 1024, 1024), (1, 1, 1024, 1), "bf16", "bfp8", "dram"),
]


@pytest.mark.parametrize("cid, sa, sb, da, db, mem", CASES, ids=[c[0] for c in CASES])
def test_fid_mul(device, cid, sa, sb, da, db, mem):
    torch.manual_seed(zlib.crc32(cid.encode()) % 100000)
    a = torch.randn(sa, dtype=torch.bfloat16) * 3
    b = torch.randn(sb, dtype=torch.bfloat16) * 3
    ma = _mem(mem, sa)
    mb = _mem(mem, sb) if sb == sa else ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ma)
    tb = ttnn.from_torch(b, dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mb)
    out = ttnn.multiply(ta, tb, fast_and_approximate_mode=True, memory_config=ma)
    _save(cid, out)
    ref = ttnn.to_torch(ta).float() * ttnn.to_torch(tb).float()
    got = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > (0.98 if "bfp4" in cid else 0.99)


@pytest.mark.parametrize("da", ["bf16", "bfp8"])
def test_fid_mul_scalar(device, da):
    torch.manual_seed(7)
    a = torch.randn((1, 1, 1024, 1024), dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.multiply(ta, 1.7265625, fast_and_approximate_mode=True)
    _save(f"scalar_{da}", out)


@pytest.mark.parametrize("da", ["bf16", "bfp8"])
def test_fid_mul_fp32_out(device, da):
    torch.manual_seed(11)
    a = torch.randn((1, 1, 512, 512), dtype=torch.bfloat16)
    b = torch.randn((1, 1, 512, 512), dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.multiply(ta, tb, dtype=ttnn.float32, fast_and_approximate_mode=True)
    _save(f"fp32out_{da}", out)


@pytest.mark.parametrize("act", ["relu", "silu"])
def test_fid_mul_post(device, act):
    torch.manual_seed(13)
    a = torch.randn((1, 1, 1024, 1024), dtype=torch.bfloat16)
    b = torch.randn((1, 1, 1024, 1024), dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    acts = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU if act == "relu" else ttnn.UnaryOpType.SILU)]
    out = ttnn.multiply(ta, tb, activations=acts)
    _save(f"post_{act}", out)


# binary_ng FPU multiplies into a Float32 output (fp32 DEST, the typecast post activation), interleaved, one tile per DEST section
@pytest.mark.parametrize("hw", [512, 1024, 2048])
@pytest.mark.parametrize("da", ["bf16", "bfp8"])
@pytest.mark.parametrize("mem", ["dram", "l1"])
def test_fid_mul_fp32_out_sizes(device, hw, da, mem):
    if mem == "l1" and hw > 1024:
        pytest.skip("L1 holds up to 1024x1024 here")
    torch.manual_seed(hw + len(da))
    a = torch.randn((1, 1, hw, hw), dtype=torch.bfloat16)
    b = torch.randn((1, 1, hw, hw), dtype=torch.bfloat16)
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    out = ttnn.multiply(ta, tb, dtype=ttnn.float32, fast_and_approximate_mode=True, memory_config=mc)
    _save(f"fp32out_{da}_{hw}_{mem}", out)
