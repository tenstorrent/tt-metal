# r3 unary datacopy: bit identity of the ops the round 3 candidates touch. Runs every case once on deterministic inputs that
# carry every bf16 bit pattern (16-bit cases) or the 32-bit special values (+-0, +-inf, NaN payloads, denormals, the extremes)
# mixed with random values, and saves the raw output bits; bitid_cmp.py compares two trees' outputs bit for bit.
# usage: python bitid.py <out_dir> [case ...]
import os
import sys

import numpy as np
import torch
import ttnn

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock and no mock cluster descriptor")

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def specials_f32():
    v = [0.0, -0.0, float("inf"), float("-inf"), 1.0, -1.0, 3.4028234663852886e38, -3.4028234663852886e38, 1.1754943508222875e-38,
         -1.1754943508222875e-38]
    bits = [0x7FC00000, 0x7FC00001, 0xFFC00000, 0x7F800001, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x3F800001,
            0x00400000]
    a = np.array(v, dtype=np.float32)
    b = np.array(bits, dtype=np.uint32).view(np.float32)
    return np.concatenate([a, b])


def fill_f32(shape, seed):
    g = np.random.default_rng(seed)
    n = int(np.prod(shape))
    x = (g.standard_normal(n) * np.exp(g.uniform(-40, 40, n))).astype(np.float32)
    sp = specials_f32()
    idx = g.choice(n, size=min(n, 4096), replace=False)
    x[idx] = sp[np.arange(len(idx)) % len(sp)]
    return torch.from_numpy(x.reshape(shape))


def fill_i32(shape, seed):
    g = np.random.default_rng(seed)
    n = int(np.prod(shape))
    x = g.integers(-(2**31), 2**31 - 1, n, dtype=np.int64).astype(np.int32)
    sp = np.array([0, 1, -1, 2**31 - 1, -(2**31), 2**30, -(2**30), 0x7FFF, -0x8000, 0x10000], dtype=np.int64).astype(np.int32)
    idx = g.choice(n, size=min(n, 4096), replace=False)
    x[idx] = sp[np.arange(len(idx)) % len(sp)]
    return torch.from_numpy(x.reshape(shape))


def fill_bf16_all(shape, shift=0):
    n = int(np.prod(shape))
    bits = ((np.arange(n, dtype=np.int64) + shift) % 65536).astype(np.uint16)
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).reshape(shape)


def hs_mem(dev, shape, ncores):
    grid = dev.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    h = int(np.prod(shape[:-1]))
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (h // ncores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


def tt(dev, t, dt, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dt, layout=layout, device=dev, memory_config=mem)


def save(name, y):
    t = ttnn.to_torch(y)
    if t.dtype in (torch.float32, torch.int32):
        raw = t.contiguous().view(torch.int32)
    elif t.dtype == torch.uint32:
        raw = t.contiguous().view(torch.int32)
    elif t.dtype == torch.bfloat16:
        raw = t.contiguous().view(torch.int16)
    else:
        raw = t.contiguous()
    torch.save(raw.cpu(), os.path.join(OUT, f"{name}.pt"))


S = (1, 1, 4096, 256)
CASES = {}


def case(f):
    CASES[f.__name__] = f
    return f


@case
def add_int32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.add(tt(dev, fill_i32(S, 1), ttnn.int32, m), tt(dev, fill_i32(S, 2), ttnn.int32, m), memory_config=m)


@case
def add_fp32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.add(tt(dev, fill_f32(S, 3), ttnn.float32, m), tt(dev, fill_f32(S, 4), ttnn.float32, m), memory_config=m)


@case
def sub_fp32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.subtract(tt(dev, fill_f32(S, 5), ttnn.float32, m), tt(dev, fill_f32(S, 6), ttnn.float32, m), memory_config=m)


@case
def mul_int32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.multiply(tt(dev, fill_i32(S, 7), ttnn.int32, m), tt(dev, fill_i32(S, 8), ttnn.int32, m), memory_config=m)


@case
def add_int32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.add(tt(dev, fill_i32(s, 9), ttnn.int32), tt(dev, fill_i32(s, 10), ttnn.int32))


@case
def add_fp32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.add(tt(dev, fill_f32(s, 11), ttnn.float32), tt(dev, fill_f32(s, 12), ttnn.float32))


@case
def mul_bf16_hs64_all(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.multiply(tt(dev, fill_bf16_all(S), ttnn.bfloat16, m), tt(dev, fill_bf16_all(S, 12345), ttnn.bfloat16, m), memory_config=m)


@case
def mul_silu_bf16_hs64_all(dev):
    # the MLP multiply's form: SiLU on the lhs in the binary_ng preprocess pass
    m = hs_mem(dev, S, 64)
    return ttnn.mul(tt(dev, fill_bf16_all(S), ttnn.bfloat16, m), tt(dev, fill_bf16_all(S, 4321), ttnn.bfloat16, m),
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=m)


@case
def mul_silu_bf16_dram_all_b8(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.mul(tt(dev, fill_bf16_all(s, 99), ttnn.bfloat16), tt(dev, fill_bf16_all(s, 5555), ttnn.bfloat16),
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b)


@case
def mul_rowb_bf16_l1_all(dev):
    # binary_ng's SFPU row broadcast kernel (the TransFuser SE multiply's shape)
    return ttnn.multiply(tt(dev, fill_bf16_all((1, 1, 7040, 96), 7), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
                         tt(dev, fill_bf16_all((1, 1, 1, 96), 31000), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
                         memory_config=ttnn.L1_MEMORY_CONFIG)


@case
def mul_colb_bf16_all(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.multiply(tt(dev, fill_bf16_all(s, 11), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 1024, 1), 222), ttnn.bfloat16))


@case
def mul_scalarb_bf16_all(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.multiply(tt(dev, fill_bf16_all(s, 13), ttnn.bfloat16), tt(dev, torch.full((1, 1, 1, 1), 1.5).to(torch.bfloat16), ttnn.bfloat16))


@case
def mul_pyscalar_bf16_all(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.multiply(tt(dev, fill_bf16_all(s, 17), ttnn.bfloat16), 0.75)


@case
def mul_batchb_bf16_all(dev):
    return ttnn.multiply(tt(dev, fill_bf16_all((8, 1, 512, 768), 19), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 512, 768), 3333), ttnn.bfloat16))


@case
def where_ttt_bf16_all(dev):
    s = (1, 1, 1024, 1024)
    g = torch.Generator().manual_seed(23)
    c = (torch.rand(s, generator=g) > 0.5).to(torch.bfloat16)
    return ttnn.where(tt(dev, c, ttnn.bfloat16), tt(dev, fill_bf16_all(s, 29), ttnn.bfloat16), tt(dev, fill_bf16_all(s, 7777), ttnn.bfloat16))


@case
def mul_rowb_fp32(dev):
    return ttnn.multiply(tt(dev, fill_f32((1, 1, 1024, 1024), 41), ttnn.float32), tt(dev, fill_f32((1, 1, 1, 1024), 42), ttnn.float32))


@case
def mul_colb_fp32(dev):
    return ttnn.multiply(tt(dev, fill_f32((1, 1, 1024, 1024), 43), ttnn.float32), tt(dev, fill_f32((1, 1, 1024, 1), 44), ttnn.float32))


@case
def max_rowcol_bf16_all(dev):
    return ttnn.maximum(tt(dev, fill_bf16_all((1, 1, 1024, 1), 45), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 1, 1024), 46000), ttnn.bfloat16))


@case
def mul_rowcol_bf16_all(dev):
    return ttnn.multiply(tt(dev, fill_bf16_all((1, 1, 1024, 1), 47), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 1, 1024), 50000), ttnn.bfloat16))


@case
def max_bf16_hs64_all(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.maximum(tt(dev, fill_bf16_all(S), ttnn.bfloat16, m), tt(dev, fill_bf16_all(S, 777), ttnn.bfloat16, m), memory_config=m)


@case
def abs_int32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.abs(tt(dev, fill_i32(S, 13), ttnn.int32, m), memory_config=m)


@case
def neg_fp32_hs64(dev):
    m = hs_mem(dev, S, 64)
    return ttnn.neg(tt(dev, fill_f32(S, 14), ttnn.float32, m), memory_config=m)


@case
def neg_fp32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.neg(tt(dev, fill_f32(s, 15), ttnn.float32))


def _transpose_hs(dev, kind, seed):
    n, c, h, w = 64, 2, 128, 128
    grid = dev.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(64, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mk = lambda sh: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, sh, ttnn.ShardOrientation.ROW_MAJOR))
    x = fill_i32((n, c, h, w), seed) if kind == "int32" else fill_f32((n, c, h, w), seed)
    dt = ttnn.int32 if kind == "int32" else ttnn.float32
    return ttnn.transpose(tt(dev, x, dt, mk((n * c * h // 64, w))), -2, -1, memory_config=mk((n * c * w // 64, h)))


@case
def transpose_int32_hs64(dev):
    return _transpose_hs(dev, "int32", 16)


@case
def transpose_fp32_hs64(dev):
    return _transpose_hs(dev, "fp32", 17)


@case
def transpose_int32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.transpose(tt(dev, fill_i32(s, 18), ttnn.int32), -2, -1)


@case
def typecast_fp32_int32_hs32(dev):
    s = (1, 1, 2048, 1024)
    x = fill_f32(s, 19).clamp(-2e9, 2e9).nan_to_num(0.0)
    m = hs_mem(dev, s, 32)
    return ttnn.typecast(tt(dev, x, ttnn.float32, m), ttnn.int32)


@case
def typecast_fp32_bf16_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.typecast(tt(dev, fill_f32(s, 20), ttnn.float32), ttnn.bfloat16)


@case
def untilize_fp32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.untilize(tt(dev, fill_f32(s, 21), ttnn.float32))


@case
def untilize_int32_dram(dev):
    s = (1, 1, 1024, 1024)
    return ttnn.untilize(tt(dev, fill_i32(s, 22), ttnn.int32))


@case
def tilize_int32_dram(dev):
    s = (1, 1, 512, 512)
    return ttnn.tilize(tt(dev, fill_i32(s, 23), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))


@case
def tilize_fp32_dram(dev):
    s = (1, 1, 512, 1024)
    return ttnn.tilize(tt(dev, fill_f32(s, 24), ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT))


def _mm(dev, m_, k, n, sbh, sbw, untilize=False):
    g = torch.Generator().manual_seed(25)
    a = tt(dev, torch.randn((1, 1, m_, k), generator=g, dtype=torch.float32).to(torch.bfloat16), ttnn.bfloat16)
    b = tt(dev, torch.randn((1, 1, k, n), generator=g, dtype=torch.float32).to(torch.bfloat16), ttnn.bfloat16)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1), in0_block_w=2, out_subblock_h=sbh, out_subblock_w=sbw,
        per_core_M=m_ // 32, per_core_N=n // 32, fuse_batch=True, mcast_in0=True, untilize_out=untilize,
    )
    cc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    return ttnn.matmul(a, b, program_config=pc, compute_kernel_config=cc, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)


@case
def mm_fp32acc_sb2(dev):
    return _mm(dev, 128, 4096, 128, 1, 2)


@case
def mm_fp32acc_sb4(dev):
    return _mm(dev, 128, 4096, 128, 2, 2)


@case
def conv2d_fp32acc(dev):
    g = torch.Generator().manual_seed(26)
    x = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, 32 * 32, 64), generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16), dev)
    w = ttnn.from_torch(torch.randn((128, 64, 3, 3), generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16)
    cc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    return ttnn.conv2d(input_tensor=x, weight_tensor=w, in_channels=64, out_channels=128, device=dev, kernel_size=(3, 3), stride=(1, 1),
                       padding=(1, 1), batch_size=1, input_height=32, input_width=32, compute_config=cc)


def _conv3d(dev, c_in, c_out, k, t_in, pad_t, blk, seed):
    cib, cob, tob, hob, wob = blk
    x = fill_f32((1, t_in, 1, 1, c_in), seed).nan_to_num(0.0, 0.0, 0.0).clamp(-1e3, 1e3)
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(c_out, c_in, k, 1, 1, generator=g) * 0.02
    b = fill_f32((c_out,), seed + 1).nan_to_num(0.0, 0.0, 0.0).clamp(-1e3, 1e3)
    tx = ttnn.from_torch(x, device=dev, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tw = ttnn.experimental.prepare_conv3d_weights(weight_tensor=ttnn.from_torch(w, dtype=ttnn.float32, pad_value=0), groups=1, C_in_block=cib, alignment=32, device=dev)
    tb = ttnn.from_torch(b.reshape(1, -1), device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, pad_value=0)
    cfg = ttnn.Conv3dConfig(weights_dtype=ttnn.float32, output_layout=ttnn.ROW_MAJOR_LAYOUT, T_out_block=tob, W_out_block=wob, H_out_block=hob,
                            C_out_block=cob, C_in_block=cib, compute_with_storage_grid_size=dev.compute_with_storage_grid_size())
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    return ttnn.experimental.conv3d(input_tensor=tx, weight_tensor=tw, bias_tensor=tb, device=dev, config=cfg, output_channels=c_out, kernel_size=(k, 1, 1),
                                    stride=(1, 1, 1), padding=(pad_t, 0, 0), dilation=(1, 1, 1), padding_mode="zeros", dtype=ttnn.float32, compute_kernel_config=ckc)


@case
def conv3d_fp32_amp384(dev):
    return _conv3d(dev, 384, 384, 7, 766, 0, (128, 128, 16, 1, 1), 27)


@case
def conv3d_fp32_amp768(dev):
    return _conv3d(dev, 768, 768, 7, 386, 0, (256, 32, 64, 1, 1), 28)


@case
def conv3d_fp32_ups0(dev):
    return _conv3d(dev, 1536, 768, 11, 3054, 0, (128, 128, 32, 1, 1), 30)


def _gn(dev, x, c, nob):
    grid = ttnn.CoreGrid(y=8, x=8)
    g = torch.Generator().manual_seed(29)
    wt = torch.rand((c,), generator=g).to(torch.bfloat16)
    bs = torch.rand((c,), generator=g).to(torch.bfloat16)
    tx = tt(dev, x, ttnn.bfloat16)
    [tw, tb], tm = ttnn.dram_group_norm_params_from_torch([wt, bs], c, 32, dev, core_grid=grid, return_mask=True)
    return ttnn.group_norm(tx, num_groups=32, input_mask=tm, negative_mask=None, weight=tw, bias=tb, epsilon=1e-6,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=grid, num_out_blocks=nob, inplace=False, use_welford=True)


@case
def groupnorm_bf16_rand_vae512(dev):
    # finite inputs: with every bf16 pattern each group holds NaN and the output is NaN throughout
    g = torch.Generator().manual_seed(31)
    return _gn(dev, (torch.randn((1, 1, 256 * 256, 512), generator=g) * 3 + 0.5).to(torch.bfloat16), 512, 4)


@case
def groupnorm_bf16_rand_256(dev):
    g = torch.Generator().manual_seed(32)
    return _gn(dev, (torch.randn((1, 1, 256 * 256, 256), generator=g) * 3 + 0.5).to(torch.bfloat16), 256, 8)


@case
def groupnorm_bf16_all(dev):
    n, c, h, w = 1, 256, 256, 256
    grid = ttnn.CoreGrid(y=8, x=8)
    x = fill_bf16_all((n, 1, h * w, c))
    g = torch.Generator().manual_seed(29)
    wt = torch.rand((c,), generator=g).to(torch.bfloat16)
    bs = torch.rand((c,), generator=g).to(torch.bfloat16)
    tx = tt(dev, x, ttnn.bfloat16)
    [tw, tb], tm = ttnn.dram_group_norm_params_from_torch([wt, bs], c, 32, dev, core_grid=grid, return_mask=True)
    return ttnn.group_norm(tx, num_groups=32, input_mask=tm, negative_mask=None, weight=tw, bias=tb, epsilon=1e-6,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=grid, num_out_blocks=8, inplace=False, use_welford=True)


# 32-bit unpack-to-dest copies on every upper half (65536 patterns, each 4 times) with seeded lower halves: the opt-in per
# tile call (binary_ng's activation preprocess and the per tile operand copies of a Float32 / bf16 op; binary_ng pairs no
# other dtype with a 32-bit one), zs2's per tile bracket on UInt32 (typecast) and blk3's operand blocks on UInt32.
U = (1, 1, 1024, 256)


def upper32(shape, seed):
    n = int(np.prod(shape))
    lo = np.random.default_rng(seed).integers(0, 65536, n, dtype=np.int64)
    return ((np.arange(n, dtype=np.int64) % 65536) << 16) | lo


def f32_upper(shape, seed):
    return torch.from_numpy(upper32(shape, seed).astype(np.uint32).view(np.float32).reshape(shape))


def i32_upper(shape, seed):
    return torch.from_numpy(upper32(shape, seed).astype(np.uint32).view(np.int32).reshape(shape))


def u32_upper(shape, seed):
    return torch.from_numpy(upper32(shape, seed).reshape(shape))


def u32_rand(shape, seed):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 2**32, int(np.prod(shape)), dtype=np.int64).reshape(shape))


_ACT = lambda op: [ttnn.UnaryWithParam(op)]  # noqa: E731


@case
def f32_upper_act_abs_add(dev):
    return ttnn.add(tt(dev, f32_upper(U, 41), ttnn.float32), tt(dev, fill_f32(U, 42), ttnn.float32),
                    input_tensor_a_activations=_ACT(ttnn.UnaryOpType.ABS))


@case
def f32_upper_rhs_act_neg_mul(dev):
    return ttnn.multiply(tt(dev, fill_f32(U, 43), ttnn.float32), tt(dev, f32_upper(U, 44), ttnn.float32),
                         input_tensor_b_activations=_ACT(ttnn.UnaryOpType.NEG))


@case
def f32_upper_mixed_bf16_add(dev):
    return ttnn.add(tt(dev, f32_upper(U, 45), ttnn.float32), tt(dev, fill_bf16_all(U), ttnn.bfloat16))


@case
def bf16_mixed_f32_upper_mul(dev):
    return ttnn.multiply(tt(dev, fill_bf16_all(U, 7), ttnn.bfloat16), tt(dev, f32_upper(U, 46), ttnn.float32))


@case
def i32_upper_act_neg_add(dev):
    return ttnn.add(tt(dev, i32_upper(U, 47), ttnn.int32), tt(dev, fill_i32(U, 48), ttnn.int32),
                    input_tensor_a_activations=_ACT(ttnn.UnaryOpType.NEG))


@case
def u32_upper_add_dram(dev):
    return ttnn.add(tt(dev, u32_upper(U, 51), ttnn.uint32), tt(dev, u32_rand(U, 52), ttnn.uint32))


@case
def u32_upper_add_hs32(dev):
    m = hs_mem(dev, U, 32)
    return ttnn.add(tt(dev, u32_upper(U, 53), ttnn.uint32, m), tt(dev, u32_rand(U, 54), ttnn.uint32, m), memory_config=m)


@case
def u32_upper_typecast_f32(dev):
    return ttnn.typecast(tt(dev, u32_upper(U, 57), ttnn.uint32), ttnn.float32)



@case
def sampling_add_i32_ws8(dev):
    # the decode sampling index add: Int32 offsets in DRAM plus Int32 indices width sharded on 8 cores, one tile per core
    ws = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 32), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    a = fill_i32((1, 1, 32, 256), 61)
    b = fill_i32((1, 1, 32, 256), 62)
    return ttnn.add(tt(dev, a, ttnn.int32), tt(dev, b, ttnn.int32, ws), dtype=ttnn.int32, memory_config=ws)


@case
def add_fp32_ws8_dram_lhs(dev):
    ws = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 32), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    return ttnn.add(tt(dev, fill_f32((1, 1, 32, 256), 63), ttnn.float32), tt(dev, fill_f32((1, 1, 32, 256), 64), ttnn.float32, ws),
                    memory_config=ws)


@case
def mul_rowb_bf16_dram_32x4096_all(dev):
    # the SFPU row broadcast kernel at a decode size (one or two tiles per core), every bf16 pattern on both operands
    return ttnn.multiply(tt(dev, fill_bf16_all((1, 1, 32, 4096), 11), ttnn.bfloat16),
                         tt(dev, fill_bf16_all((1, 1, 1, 4096), 40000), ttnn.bfloat16))


@case
def add_rowb_bf16_dram_all(dev):
    return ttnn.add(tt(dev, fill_bf16_all((1, 1, 2048, 64), 3), ttnn.bfloat16),
                    tt(dev, fill_bf16_all((1, 1, 1, 64), 12345), ttnn.bfloat16))


@case
def mul_silu_rowb_bf16_all(dev):
    # the row broadcast kernel with an activation on the lhs (its preprocess pass)
    return ttnn.multiply(tt(dev, fill_bf16_all((1, 1, 2048, 64), 5), ttnn.bfloat16),
                         tt(dev, fill_bf16_all((1, 1, 1, 64), 777), ttnn.bfloat16),
                         input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])


# S4 (both operands' tiles under one unpack-to-dest handshake): every upper half on both operands, seeded lower halves
def _rot(t):
    return torch.roll(t.flatten(), 4099).reshape(t.shape)


@case
def f32_upper_both_add_dram(dev):
    a = f32_upper(U, 71)
    return ttnn.add(tt(dev, a, ttnn.float32), tt(dev, _rot(f32_upper(U, 72)), ttnn.float32))


@case
def f32_upper_both_sub_hs32(dev):
    m = hs_mem(dev, U, 32)
    return ttnn.subtract(tt(dev, f32_upper(U, 73), ttnn.float32, m), tt(dev, _rot(f32_upper(U, 74)), ttnn.float32, m), memory_config=m)


@case
def i32_upper_both_add_dram(dev):
    return ttnn.add(tt(dev, i32_upper(U, 75), ttnn.int32), tt(dev, _rot(i32_upper(U, 76)), ttnn.int32))


@case
def i32_upper_both_mul_hs32(dev):
    m = hs_mem(dev, U, 32)
    return ttnn.multiply(tt(dev, i32_upper(U, 77), ttnn.int32, m), tt(dev, _rot(i32_upper(U, 78)), ttnn.int32, m), memory_config=m)


@case
def u32_upper_both_add_l1(dev):
    return ttnn.add(tt(dev, u32_upper(U, 79), ttnn.uint32, ttnn.L1_MEMORY_CONFIG), tt(dev, _rot(u32_upper(U, 80)), ttnn.uint32, ttnn.L1_MEMORY_CONFIG),
                    memory_config=ttnn.L1_MEMORY_CONFIG)


@case
def sampling_add_u32out_ws8(dev):
    ws = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 32), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    return ttnn.add(tt(dev, fill_i32((1, 1, 32, 256), 81), ttnn.int32), tt(dev, fill_i32((1, 1, 32, 256), 82), ttnn.int32, ws),
                    dtype=ttnn.uint32, memory_config=ws)


# S4 on the broadcast, scalar, where and ternary kernels: every upper half on the full operands with seeded lower halves,
# the broadcast operand walking every upper half too where its shape allows; production forms (Qwen3.6 GDN decode in L1)
UB = (1, 1, 65536, 32)
UC = (1, 1, 65536, 1)
L1M = ttnn.L1_MEMORY_CONFIG


def _cond(shape, seed, dt):
    return (torch.from_numpy(np.random.default_rng(seed).integers(0, 2, int(np.prod(shape))).reshape(shape))).to(dt)


@case
def f32_upper_colb_mul_dram(dev):
    return ttnn.multiply(tt(dev, f32_upper(UB, 91), ttnn.float32), tt(dev, f32_upper(UC, 92), ttnn.float32))


@case
def i32_upper_colb_add_l1(dev):
    return ttnn.add(tt(dev, i32_upper(UB, 93), ttnn.int32, L1M), tt(dev, i32_upper(UC, 94), ttnn.int32, L1M), memory_config=L1M)


@case
def u32_upper_cola_sub_dram(dev):
    return ttnn.subtract(tt(dev, u32_upper(UC, 95), ttnn.uint32), tt(dev, u32_upper(UB, 96), ttnn.uint32))


@case
def f32_upper_scalarb_sub_dram(dev):
    return ttnn.subtract(tt(dev, f32_upper(U, 97), ttnn.float32), tt(dev, torch.full((1, 1, 1, 1), -1.5), ttnn.float32))


@case
def gdn_decay_exp_mul_l1(dev):
    g = -torch.from_numpy(np.abs(np.random.default_rng(98).standard_normal((1, 12, 1, 1))).astype(np.float32))
    return ttnn.multiply(tt(dev, f32_upper((1, 12, 128, 128), 99), ttnn.float32, L1M), tt(dev, g, ttnn.float32, L1M),
                         input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=L1M)


@case
def gdn_beta_mul_l1(dev):
    return ttnn.multiply(tt(dev, f32_upper((1, 12, 128, 128), 100), ttnn.float32, L1M),
                         tt(dev, fill_f32((1, 12, 1, 1), 101), ttnn.float32, L1M), memory_config=L1M)


@case
def f32_upper_pyscalar_mul_l1(dev):
    return ttnn.multiply(tt(dev, f32_upper(U, 102), ttnn.float32, L1M), 128 ** -0.5, memory_config=L1M)


@case
def f32_upper_pyscalar_mul_hs32(dev):
    m = hs_mem(dev, U, 32)
    return ttnn.multiply(tt(dev, f32_upper(U, 103), ttnn.float32, m), 0.7, memory_config=m)


@case
def i32_upper_pyscalar_add_dram(dev):
    return ttnn.add(tt(dev, i32_upper(U, 104), ttnn.int32), 3)


@case
def where_colb_tts_f32_upper(dev):
    return ttnn.where(tt(dev, _cond((1, 1, 1024, 1), 105, torch.float32), ttnn.float32), tt(dev, f32_upper(U, 106), ttnn.float32), 0.5)


@case
def where_colb_tst_f32_upper(dev):
    return ttnn.where(tt(dev, _cond((1, 1, 1024, 1), 107, torch.float32), ttnn.float32), 0.5, tt(dev, f32_upper(U, 108), ttnn.float32))


@case
def where_colb_tts_i32_upper(dev):
    return ttnn.where(tt(dev, _cond((1, 1, 1024, 1), 109, torch.int32), ttnn.int32), tt(dev, i32_upper(U, 110), ttnn.int32), 7)


@case
def where_scalarb_tst_f32_upper(dev):
    return ttnn.where(tt(dev, torch.ones((1, 1, 1, 1)), ttnn.float32), -2.0, tt(dev, f32_upper(U, 111), ttnn.float32))


@case
def where_tts_f32_upper(dev):
    return ttnn.where(tt(dev, _cond(U, 149, torch.float32), ttnn.float32), tt(dev, f32_upper(U, 150), ttnn.float32), 0.0)


@case
def where_tst_i32_upper_l1(dev):
    return ttnn.where(tt(dev, _cond(U, 151, torch.int32), ttnn.int32, L1M), 5, tt(dev, i32_upper(U, 152), ttnn.int32, L1M),
                      memory_config=L1M)


@case
def where_ttt_f32_upper(dev):
    return ttnn.where(tt(dev, _cond(U, 112, torch.float32), ttnn.float32), tt(dev, f32_upper(U, 113), ttnn.float32),
                      tt(dev, _rot(f32_upper(U, 114)), ttnn.float32))


@case
def where_ttt_i32_upper_l1(dev):
    return ttnn.where(tt(dev, _cond(U, 115, torch.int32), ttnn.int32, L1M), tt(dev, i32_upper(U, 116), ttnn.int32, L1M),
                      tt(dev, _rot(i32_upper(U, 117)), ttnn.int32, L1M), memory_config=L1M)


@case
def where_colb_ttt_f32_upper(dev):
    return ttnn.where(tt(dev, _cond((1, 1, 1024, 1), 118, torch.float32), ttnn.float32), tt(dev, f32_upper(U, 119), ttnn.float32),
                      tt(dev, _rot(f32_upper(U, 120)), ttnn.float32))


@case
def where_rowb_ttt_f32_upper(dev):
    return ttnn.where(tt(dev, _cond((1, 1, 32, 65536), 121, torch.float32), ttnn.float32),
                      tt(dev, f32_upper((1, 1, 1, 65536), 122), ttnn.float32), tt(dev, f32_upper((1, 1, 32, 65536), 123), ttnn.float32))


@case
def lerp_tts_f32_upper(dev):
    return ttnn.lerp(tt(dev, f32_upper(U, 124), ttnn.float32), tt(dev, _rot(f32_upper(U, 125)), ttnn.float32), 0.3)


@case
def lerp_tts_colb_f32_upper(dev):
    return ttnn.lerp(tt(dev, f32_upper(UB, 126), ttnn.float32), tt(dev, f32_upper(UC, 127), ttnn.float32), 0.3)


@case
def mac_tst_f32_upper(dev):
    return ttnn.mac(tt(dev, f32_upper(U, 128), ttnn.float32), 0.7, tt(dev, _rot(f32_upper(U, 129)), ttnn.float32))


@case
def mac_tst_colb_f32_upper(dev):
    return ttnn.mac(tt(dev, f32_upper(UB, 130), ttnn.float32), 0.7, tt(dev, f32_upper(UC, 131), ttnn.float32))


@case
def mac_tts_f32_upper(dev):
    return ttnn.mac(tt(dev, f32_upper(U, 132), ttnn.float32), tt(dev, _rot(f32_upper(U, 133)), ttnn.float32), 0.7)


@case
def addcmul_f32_upper(dev):
    return ttnn.addcmul(tt(dev, f32_upper(U, 134), ttnn.float32), tt(dev, _rot(f32_upper(U, 135)), ttnn.float32),
                        tt(dev, fill_f32(U, 136), ttnn.float32), value=0.5)


@case
def addcdiv_f32_upper(dev):
    return ttnn.addcdiv(tt(dev, f32_upper(U, 137), ttnn.float32), tt(dev, fill_f32(U, 138), ttnn.float32),
                        tt(dev, _rot(f32_upper(U, 139)), ttnn.float32), value=0.5)


@case
def addcmul_rowb_f32_upper(dev):
    r = (1, 1, 1, 65536)
    return ttnn.addcmul(tt(dev, f32_upper(r, 140), ttnn.float32), tt(dev, f32_upper((1, 1, 32, 65536), 141), ttnn.float32),
                        tt(dev, _rot(f32_upper(r, 142)), ttnn.float32), value=0.5)


@case
def addcmul_colb_f32_upper(dev):
    return ttnn.addcmul(tt(dev, f32_upper(UC, 143), ttnn.float32), tt(dev, f32_upper(UB, 144), ttnn.float32),
                        tt(dev, _rot(f32_upper(UB, 145)), ttnn.float32), value=0.5)


@case
def addcmul_colb_i32_upper(dev):
    return ttnn.addcmul(tt(dev, i32_upper(UC, 153), ttnn.int32), tt(dev, i32_upper(UB, 154), ttnn.int32),
                        tt(dev, _rot(i32_upper(UB, 155)), ttnn.int32), value=3)


@case
def addcmul_rowb_i32_upper(dev):
    r = (1, 1, 1, 65536)
    return ttnn.addcmul(tt(dev, i32_upper(r, 146), ttnn.int32), tt(dev, i32_upper((1, 1, 32, 65536), 147), ttnn.int32),
                        tt(dev, _rot(i32_upper(r, 148)), ttnn.int32), value=3)



# The bf16 row-broadcast addcmul (the repeated broadcast and copy inits skipped): every bf16 pattern on the full operand and
# on the broadcast rows, LTX-2's modulation form (shift and scale broadcast) and Wan's gated residual (gate broadcast)
@case
def addcmul_rowb_ac_bf16_all(dev):
    return ttnn.addcmul(tt(dev, fill_bf16_all((1, 1, 1, 65536), 11), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 32, 65536)), ttnn.bfloat16),
                        tt(dev, fill_bf16_all((1, 1, 1, 65536), 29), ttnn.bfloat16))


@case
def addcmul_rowc_ac_bf16_all(dev):
    return ttnn.addcmul(tt(dev, fill_bf16_all((1, 1, 32, 65536), 5), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 32, 65536)), ttnn.bfloat16),
                        tt(dev, fill_bf16_all((1, 1, 1, 65536), 13), ttnn.bfloat16), value=0.5)


@case
def where_colb_tts_ac_bf16_all(dev):
    return ttnn.where(tt(dev, fill_bf16_all((1, 1, 2048, 32), 3), ttnn.bfloat16), tt(dev, fill_bf16_all((1, 1, 2048, 1)), ttnn.bfloat16), 1.0)


@case
def where_colb_tst_ac_bf16_all(dev):
    c = fill_bf16_all((1, 1, 65536, 32), 9)
    return ttnn.where(tt(dev, c, ttnn.bfloat16), -2.0, tt(dev, fill_bf16_all((1, 1, 65536, 1), 7), ttnn.bfloat16))


# the opt-in per tile call in the bcast-dims kernel (mixed formats take it: gpt-oss's bf8_b experts by bf16 routing weights)
# and in the Python-scalar kernel's sharded two-tile sections
@case
def mul_colb_bf8_bf16_all(dev):
    return ttnn.multiply(tt(dev, fill_bf16_all((1, 1, 2048, 128), 19).to(torch.float32), ttnn.bfloat8_b),
                         tt(dev, fill_bf16_all((1, 1, 2048, 1), 23), ttnn.bfloat16))


@case
def mul_pyscalar_bf16_hs32_all(dev):
    s = (1, 1, 1024, 256)
    m = hs_mem(dev, s, 32)
    return ttnn.multiply(tt(dev, fill_bf16_all(s, 29), ttnn.bfloat16, m), 0.75, memory_config=m)


def main():
    names = sys.argv[2:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        for name in names:
            try:
                y = CASES[name](dev)
                ttnn.synchronize_device(dev)
                save(name, y)
                print(f"BITID {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"BITID {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
