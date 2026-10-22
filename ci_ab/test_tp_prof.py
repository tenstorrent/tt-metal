# Scratch, not for merge. Op-level boundedness cases for the block transpose and block pack callers (Blackhole).
# Each case runs one ttnn op; output sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

BF16, BFP8 = ttnn.bfloat16, ttnn.bfloat8_b
DRAM = ttnn.DRAM_MEMORY_CONFIG


def record(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


def hsharded(device, shape, cores):
    grid = device.compute_with_storage_grid_size()
    cs = ttnn.num_cores_to_corerangeset(cores, grid, True)
    rows = int(torch.Size(shape).numel() // shape[-1])
    spec = ttnn.ShardSpec(cs, [rows // cores, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def out_hsharded(device, shape, cores):
    s = list(shape)
    s[-1], s[-2] = s[-2], s[-1]
    return hsharded(device, s, cores)


# name: (shape, dtype, layout, kind)  kind: dram or hs<cores>
TR = {
    "k_1_8_1024_128": ((1, 8, 1024, 128), BF16, ttnn.TILE_LAYOUT, "dram"),
    "k_1_8_1024_128_b8": ((1, 8, 1024, 128), BFP8, ttnn.TILE_LAYOUT, "dram"),
    "k_1_32_2048_128": ((1, 32, 2048, 128), BF16, ttnn.TILE_LAYOUT, "dram"),
    "w_1_1_4096_4096": ((1, 1, 4096, 4096), BF16, ttnn.TILE_LAYOUT, "dram"),
    "dec_1_32_32_128": ((1, 32, 32, 128), BF16, ttnn.TILE_LAYOUT, "dram"),
    "small_1_1_64_64": ((1, 1, 64, 64), BF16, ttnn.TILE_LAYOUT, "dram"),
    "rm_1_8_1024_128": ((1, 8, 1024, 128), BF16, ttnn.ROW_MAJOR_LAYOUT, "dram"),
    "rm_1_1_1024_256": ((1, 1, 1024, 256), BF16, ttnn.ROW_MAJOR_LAYOUT, "dram"),
    "hs_32_2_128_64": ((32, 2, 128, 64), BF16, ttnn.TILE_LAYOUT, "hs32"),
    "hs_64_1_256_128": ((64, 1, 256, 128), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_128_b8": ((64, 1, 256, 128), BFP8, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_32": ((64, 1, 256, 32), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_128_64": ((64, 1, 128, 64), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_96": ((64, 1, 256, 96), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_192": ((64, 1, 256, 192), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_512_64": ((64, 1, 512, 64), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_256": ((64, 1, 256, 256), BF16, ttnn.TILE_LAYOUT, "hs64"),
    "hs_64_1_256_320": ((64, 1, 256, 320), BF16, ttnn.TILE_LAYOUT, "hs64"),
}


@pytest.mark.parametrize("case", list(TR))
def test_transpose(device, case, request):
    shape, dt, lay, kind = TR[case]
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    if kind == "dram":
        t = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=DRAM)
        out = ttnn.transpose(t, -2, -1)
    else:
        n = int(kind[2:])
        t = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=hsharded(device, shape, n))
        out = ttnn.transpose(t, -2, -1, memory_config=out_hsharded(device, shape, n))
    record(request, out)


PM = {
    "p0132_1_8_1024_128": ((1, 8, 1024, 128), (0, 1, 3, 2), BF16, ttnn.TILE_LAYOUT),
    "p0231_1_1024_8_128": ((1, 1024, 8, 128), (0, 2, 3, 1), BF16, ttnn.TILE_LAYOUT),
    "p0213_1_1024_8_128": ((1, 1024, 8, 128), (0, 2, 1, 3), BF16, ttnn.TILE_LAYOUT),
    "p0312_1_8_1024_128": ((1, 8, 1024, 128), (0, 3, 1, 2), BF16, ttnn.TILE_LAYOUT),
    "p0132_rm_1_8_1024_128": ((1, 8, 1024, 128), (0, 1, 3, 2), BF16, ttnn.ROW_MAJOR_LAYOUT),
    "p3210_2_4_256_256": ((2, 4, 256, 256), (3, 2, 1, 0), BF16, ttnn.TILE_LAYOUT),
}


@pytest.mark.parametrize("case", list(PM))
def test_permute(device, case, request):
    shape, perm, dt, lay = PM[case]
    torch.manual_seed(0)
    t = ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=dt, layout=lay, device=device, memory_config=DRAM)
    out = ttnn.permute(t, perm)
    record(request, out)


# binary_ng with every tensor sharded: the only case with more than one tile per DEST section
def sharded_cfg(device, shape, kind):
    rows, cols = shape[-2], shape[-1]
    grid = device.compute_with_storage_grid_size()
    if kind == "h64":
        g = ttnn.num_cores_to_corerangeset(64, grid, True)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                 ttnn.ShardSpec(g, [rows // 64, cols], ttnn.ShardOrientation.ROW_MAJOR))
    if kind == "w32":
        g = ttnn.num_cores_to_corerangeset(32, grid, True)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                 ttnn.ShardSpec(g, [rows, cols // 32], ttnn.ShardOrientation.ROW_MAJOR))
    g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(g, [rows // 8, cols // 8], ttnn.ShardOrientation.ROW_MAJOR))


BN = {
    "add_h64_4096_512": ("add", (1, 1, 4096, 512), "h64"),
    "mul_h64_4096_512": ("mul", (1, 1, 4096, 512), "h64"),
    "add_w32_32_8192": ("add", (1, 1, 32, 8192), "w32"),
    "mul_w32_32_4096": ("mul", (1, 1, 32, 4096), "w32"),
    "add_b8_2048_2048": ("add", (1, 1, 2048, 2048), "b8"),
    "mul_b8_2048_2048": ("mul", (1, 1, 2048, 2048), "b8"),
    "muls_b8_2048_2048": ("muls", (1, 1, 2048, 2048), "b8"),
    "muls_w32_32_8192": ("muls", (1, 1, 32, 8192), "w32"),
    "adds_b8_2048_2048": ("adds", (1, 1, 2048, 2048), "b8"),
    "adds_h64_4096_512": ("adds", (1, 1, 4096, 512), "h64"),
    "sub_b8_2048_2048": ("sub", (1, 1, 2048, 2048), "b8"),
    "add_dram_1_1_4096_4096": ("add", (1, 1, 4096, 4096), "dram"),
}


@pytest.mark.parametrize("case", list(BN))
def test_binary_ng(device, case, request):
    op, shape, kind = BN[case]
    torch.manual_seed(0)
    mem = DRAM if kind == "dram" else sharded_cfg(device, shape, kind)
    mk = lambda: ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    a = mk()
    if op == "muls":
        out = ttnn.mul(a, 1.5, memory_config=mem)
    elif op == "adds":
        out = ttnn.add(a, 1.5, memory_config=mem)
    else:
        out = {"add": ttnn.add, "mul": ttnn.mul, "sub": ttnn.subtract}[op](a, mk(), memory_config=mem)
    record(request, out)


LN = {
    "ln_1_1_2048_2048": ((1, 1, 2048, 2048), "layer"),
    "ln_1_1_1024_1024": ((1, 1, 1024, 1024), "layer"),
    "ln_1_1_4096_1024": ((1, 1, 4096, 1024), "layer"),
    "rms_1_1_4096_1024": ((1, 1, 4096, 1024), "rms"),
}


@pytest.mark.parametrize("case", list(LN))
def test_norm_welford(device, case, request):
    shape, kind = LN[case]
    torch.manual_seed(0)
    w = shape[-1]
    x = ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    mkw = lambda: ttnn.from_torch(torch.randn((1, 1, w // 32, 32)).bfloat16(), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cfg = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    if kind == "layer":
        out = ttnn.layer_norm(x, weight=mkw(), bias=mkw(), epsilon=1e-5, program_config=cfg)
    else:
        out = ttnn.rms_norm(x, weight=mkw(), epsilon=1e-5, program_config=cfg)
    record(request, out)


SD = {
    "sdpa_16_4096_128": ((1, 16, 4096, 128), 128, 128),
    "sdpa_32_2048_128": ((1, 32, 2048, 128), 128, 128),
    "sdpa_8_1024_128": ((1, 8, 1024, 128), 64, 128),
}


@pytest.mark.parametrize("case", list(SD))
def test_sdpa(device, case, request):
    shape, qc, kc = SD[case]
    torch.manual_seed(0)
    mk = lambda: ttnn.from_torch(torch.randn(shape).bfloat16(), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    q, k, v = mk(), mk(), mk()
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=qc,
                                k_chunk_size=kc, exp_approx_mode=False)
    cc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                          fp32_dest_acc_en=False, packer_l1_acc=False)
    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc, compute_kernel_config=cc)
    record(request, out)
