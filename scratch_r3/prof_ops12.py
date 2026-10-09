# r3 unary_datacopy: device time of the production ops that pair an llk_unpack_A caller (copy_tile, transpose_tile)
# with a pack untilize, run under `python -m tracy -r -p --no-web-server -o <dir> prof_ops.py <cases>`; a signpost
# precedes the measured calls of every case, so the ops report splits them. Warm-up 2 calls, measured N_MEAS calls.
import gc
import sys
import torch
import ttnn
try:
    from tracy import signpost
except ImportError:  # CI scratch runs without the profiler
    def signpost(**_):
        pass

import os

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock and no mock cluster descriptor")

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def tile(t, dt=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG, dev=None):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)


def rm(t, dt=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG, dev=None):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)


def case_topk(shape, k):
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev),)

    def run(dev, x):
        return ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True)

    return setup, run


def case_transpose_rm(shape):
    def setup(dev):
        torch.manual_seed(0)
        return (rm(torch.randn(shape, dtype=torch.bfloat16), dev=dev),)

    def run(dev, x):
        return ttnn.transpose(x, -2, -1)

    return setup, run


def case_transpose_rm_sharded(shape, ncores):
    def setup(dev):
        torch.manual_seed(0)
        h = shape[-2] * shape[-3] * shape[0]
        grid = dev.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        spec = ttnn.ShardSpec(crs, (h // ncores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        return (rm(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev), mem)

    def run(dev, x, mem):
        return ttnn.transpose(x, -2, -1, memory_config=mem)

    return setup, run


def case_permute(shape, dims, layout):
    def setup(dev):
        torch.manual_seed(0)
        t = torch.randn(shape, dtype=torch.bfloat16)
        return ((tile if layout == "tile" else rm)(t, dev=dev),)

    def run(dev, x):
        return ttnn.permute(x, dims)

    return setup, run


def case_matmul_untilize(m, k, n, fp32_acc, l1_acc):
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn((1, 1, m, k), dtype=torch.bfloat16), dev=dev)
        b = tile(torch.randn((1, 1, k, n), dtype=torch.bfloat16), dev=dev)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=2,
            out_subblock_h=m // 32,
            out_subblock_w=n // 32,
            per_core_M=m // 32,
            per_core_N=n // 32,
            fuse_batch=True,
            mcast_in0=True,
            untilize_out=True,
        )
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=l1_acc
        )
        return (a, b, pc, cc)

    def run(dev, a, b, pc, cc):
        return ttnn.matmul(a, b, program_config=pc, compute_kernel_config=cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup, run


def case_moe_gate(batch_size, topk, enable_sigmoid, output_softmax):
    def setup(dev):
        input_shape = (batch_size, 8, 32)
        reshaped = (batch_size, 16, 16)
        torch.manual_seed(0)
        ti = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
        if enable_sigmoid or not output_softmax:
            ti = torch.sigmoid(ti)
        tb = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
        grid = dev.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        sh = ttnn.ShardSpec(crs, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
        tl = ttnn.Tile((32, 32))
        mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem, tile=tl)
        x = mk(torch.reshape(ti, reshaped), ttnn.bfloat16)
        b = mk(torch.transpose(torch.reshape(tb, reshaped), -2, -1), ttnn.bfloat16)
        idx = torch.arange(256, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).reshape(reshaped)
        ii = mk(torch.transpose(idx, -2, -1).to(torch.uint16), ttnn.uint16)
        o = mk(torch.zeros((batch_size, 1, 16), dtype=torch.bfloat16), ttnn.bfloat16)
        oi = mk(torch.zeros((batch_size, 1, 16), dtype=torch.uint16), ttnn.uint16)
        return (x, b, ii, o, oi)

    def run(dev, x, b, ii, o, oi):
        return ttnn.experimental.deepseek.moe.generalized_moe_gate(
            x, bias_tensor=b, input_indices_tensor=ii, output_tensor=o, output_indices_tensor=oi, eps=1e-20,
            scaling_factor=2.5, enable_sigmoid=enable_sigmoid, topk=topk, output_softmax=output_softmax,
        )

    return setup, run


def case_matmul_fp32acc(m, k, n, sbh, sbw, untilize=False, fp32=True):
    # 1-core matmul with fp32 DEST accumulation and no packer L1 accumulation: the partials reload is copy_block of
    # sbh x sbw tiles with fp32 partials (unpack to DEST)
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn((1, 1, m, k), dtype=torch.bfloat16), dev=dev)
        b = tile(torch.randn((1, 1, k, n), dtype=torch.bfloat16), dev=dev)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=2,
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            per_core_M=m // 32,
            per_core_N=n // 32,
            fuse_batch=True,
            mcast_in0=True,
            untilize_out=untilize,
        )
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        return (a, b, pc, cc)

    def run(dev, a, b, pc, cc):
        return ttnn.matmul(a, b, program_config=pc, compute_kernel_config=cc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup, run


def case_binary_bcast(shape_a, shape_b, op, dt=ttnn.bfloat16, swap=False):
    # binary_ng with a row broadcast operand (the LLK row broadcast kernel for bf16 / bfp8 on Blackhole)
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn(shape_a, dtype=torch.bfloat16), dt=dt, dev=dev)
        b = tile(torch.randn(shape_b, dtype=torch.bfloat16), dt=dt, dev=dev)
        return (b, a) if swap else (a, b)

    def run(dev, x, y):
        return op(x, y)

    return setup, run


def case_conv2d_fp32acc(cin, cout, hw, fp32=True):
    def setup(dev):
        torch.manual_seed(0)
        x = ttnn.from_torch(torch.randn((1, 1, hw * hw, cin), dtype=torch.bfloat16), dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn((cout, cin, 3, 3), dtype=torch.bfloat16), dtype=ttnn.bfloat16)
        x = ttnn.to_device(x, dev)
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        return (x, w, cc)

    def run(dev, x, w, cc):
        return ttnn.conv2d(
            input_tensor=x, weight_tensor=w, in_channels=cin, out_channels=cout, device=dev, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), batch_size=1, input_height=hw, input_width=hw, compute_config=cc,
        )

    return setup, run


def case_moe_gate512(batch_size, topk, enable_sigmoid, output_softmax):
    # the 512-expert two-block combine (process_block_to_run: copy_tile, pack_untilize_dest per block), shaped like
    # models/common/tests/modules/moe/test_generalized_moe_gate.py::test_generalized_moe_gate_512_global
    def setup(dev):
        nb, ne = 2, 512
        torch.manual_seed(0)
        ti = (2 * torch.rand((batch_size, ne), dtype=torch.bfloat16)) - 1
        if not enable_sigmoid:
            ti = torch.sigmoid(ti)
        tb = (2 * torch.rand((batch_size, ne), dtype=torch.bfloat16)) - 1
        logits = ti.reshape(batch_size, nb, 16, 16)
        bias = torch.transpose(tb.reshape(batch_size, nb, 16, 16), -2, -1).contiguous()
        grid = dev.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        mem = lambda shard: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR)
        )
        tl = ttnn.Tile((32, 32))
        mk = lambda t, dt, sh: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem(sh), tile=tl)
        multi, one = (nb * 32, 32), (32, 32)
        x = mk(logits, ttnn.bfloat16, multi)
        b = mk(bias, ttnn.bfloat16, multi)
        ar = torch.arange(256, dtype=torch.int32).reshape(1, 1, 16, 16)
        offs = (torch.arange(nb, dtype=torch.int32) * 256).reshape(1, nb, 1, 1)
        idx = torch.transpose(ar + offs, -2, -1).contiguous().to(torch.uint16).expand(batch_size, -1, -1, -1).contiguous()
        ii = mk(idx, ttnn.uint16, multi)
        o = mk(torch.zeros((batch_size, 1, 16), dtype=torch.bfloat16), ttnn.bfloat16, one)
        oi = mk(torch.zeros((batch_size, 1, 16), dtype=torch.uint16), ttnn.uint16, one)
        return (x, b, ii, o, oi)

    _, run = case_moe_gate(batch_size, topk, enable_sigmoid, output_softmax)
    return setup, run


def case_moe_gate_shift(batch_size, topk, enable_sigmoid, output_softmax, shift_tiles):
    # case_moe_gate behind a resident sharded bf16 tensor of shift_tiles tiles per core on the same cores, so the op's
    # CBs and tensors sit shift_tiles * 2 KB away from where case_moe_gate puts them (an L1 address sweep)
    setup0, run0 = case_moe_gate(batch_size, topk, enable_sigmoid, output_softmax)

    def setup(dev):
        pad = None
        if shift_tiles:
            grid = dev.compute_with_storage_grid_size()
            crs = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
            sh = ttnn.ShardSpec(crs, (32, 32 * shift_tiles), ttnn.ShardOrientation.ROW_MAJOR)
            mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sh)
            pad = ttnn.from_torch(
                torch.zeros((batch_size * 32, 32 * shift_tiles), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem,
            )
        return (pad,) + setup0(dev)

    def run(dev, pad, *args):
        return run0(dev, *args)

    return setup, run


def case_tilize(shape, kind):
    # ttnn.tilize of a row-major tensor; bf16 goes to fast tilize, fp32 (lossless) and int32 take tilize_block
    def setup(dev):
        torch.manual_seed(0)
        if kind == "int32":
            t, dt = torch.randint(-1000, 1000, shape, dtype=torch.int32), ttnn.int32
        else:
            t, dt = torch.randn(shape, dtype=torch.float32), ttnn.float32
        return (ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG),)

    def run(dev, x):
        return ttnn.tilize(x)

    return setup, run


def case_tilize_pad(shape, out_shape, kind):
    setup, _ = case_tilize(shape, kind)

    def run(dev, x):
        return ttnn.tilize_with_val_padding(x, out_shape, 0)

    return setup, run


def case_moe_fused_swiglu(emb, hidden, allocated, active, x_rm):
    # deepseek prefill routed expert, as tests/ttnn/nightly/.../test_moe_fused_swiglu.py runs it (grid 11x8, bfp4 weights);
    # its down and gate/up partials reload through copy_block_matmul_partials
    def setup(dev):
        torch.manual_seed(42)
        wg = torch.randn(hidden, emb, dtype=torch.float32) * 0.02
        wu = torch.randn(hidden, emb, dtype=torch.float32) * 0.02
        wd = torch.randn(emb, hidden, dtype=torch.float32) * 0.02
        td = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xin = torch.zeros(allocated, emb, dtype=torch.float32)
        xin[:active] = torch.randn(active, emb, dtype=torch.float32)
        x = td(xin.reshape(1, 1, allocated, emb), ttnn.bfloat16 if x_rm else ttnn.bfloat8_b, ttnn.ROW_MAJOR_LAYOUT if x_rm else ttnn.TILE_LAYOUT)
        w = [td(t.T, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for t in (wg, wu, wd)]
        idx = td(torch.tensor([0], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        counts = td(torch.tensor([active], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return (x, w[0], w[1], w[2], counts, idx)

    def run(dev, x, wg, wu, wd, counts, idx):
        return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            x, [wg], [wu], [wd], counts, idx, input_m_tiles=allocated // 32, core_grid=ttnn.CoreCoord(11, 8),
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    return setup, run


def _hs_mem(dev, shape, ncores):
    # height sharded over the first ncores cores of the grid, row major
    grid = dev.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    h = 1
    for d in shape[:-1]:
        h *= d
    spec = ttnn.ShardSpec(crs, (h // ncores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _rand(shape, kind):
    if kind == "int32":
        return torch.randint(-1000, 1000, shape, dtype=torch.int32), ttnn.int32
    if kind == "fp32":
        return torch.randn(shape, dtype=torch.float32), ttnn.float32
    return torch.randn(shape, dtype=torch.bfloat16), ttnn.bfloat16


def case_unary(op, shape, kind, ncores=0):
    # eltwise unary through eltwise_sfpu.cpp (copy_tile per tile); ncores > 0: height sharded in L1
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand(shape, kind)
        mem = _hs_mem(dev, shape, ncores) if ncores else ttnn.DRAM_MEMORY_CONFIG
        return (tile(t, dt=dt, mem=mem, dev=dev), mem)

    def run(dev, x, mem):
        return op(x, memory_config=mem)

    return setup, run


def case_transpose_t(shape, kind, mem):
    # ttnn.transpose of the last two dims, tile layout: transpose_wh.cpp, one transpose_tile per tile (32-bit: transpose_dest)
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand(shape, kind)
        return (tile(t, dt=dt, mem=mem, dev=dev),)

    def run(dev, x):
        return ttnn.transpose(x, -2, -1, memory_config=mem)

    return setup, run


def case_binary(op, shape_a, shape_b, kind, ncores=0, b_sharded=True):
    # binary_ng; ncores > 0: a height sharded in L1 (b too unless b_sharded is False, then b interleaved in DRAM)
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape_a, kind)
        tb, _ = _rand(shape_b, kind)
        mem_a = _hs_mem(dev, shape_a, ncores) if ncores else ttnn.DRAM_MEMORY_CONFIG
        mem_b = _hs_mem(dev, shape_b, ncores) if (ncores and b_sharded) else ttnn.DRAM_MEMORY_CONFIG
        return (tile(ta, dt=dt, mem=mem_a, dev=dev), tile(tb, dt=dt, mem=mem_b, dev=dev), mem_a)

    def run(dev, a, b, mem):
        return op(a, b, memory_config=mem)

    return setup, run


CASES = {
    "topk_qwen_vocab": case_topk((1, 1, 32, 151936), 50),
    "topk_llama_vocab": case_topk((1, 1, 32, 128256), 50),
    "topk_k512_w65536": case_topk((1, 1, 32, 65536), 512),
    "transpose_rm_1024x1024": case_transpose_rm((1, 1, 1024, 1024)),
    "transpose_rm_8x256x128": case_transpose_rm((1, 8, 256, 128)),
    "transpose_rm_hs_2048x64": case_transpose_rm_sharded((1, 1, 2048, 64), 8),
    "permute_rm_nchw_nhwc_64x128x128": case_permute((1, 64, 128, 128), (0, 2, 3, 1), "rm"),
    "permute_rm_nchw_nhwc_256x64x64": case_permute((1, 256, 64, 64), (0, 2, 3, 1), "rm"),
    "permute_tile_2301_32x32x64x64": case_permute((32, 32, 64, 64), (2, 3, 0, 1), "tile"),
    "matmul_untilize_fp32acc_64x8192x64": case_matmul_untilize(64, 8192, 64, True, False),
    "matmul_untilize_fp32acc_l1acc_64x8192x64": case_matmul_untilize(64, 8192, 64, True, True),
    "matmul_untilize_bf16_64x2048x128": case_matmul_untilize(64, 2048, 128, False, False),
    "moe_gate_b32_top8_sigmoid": case_moe_gate(32, 8, True, False),
    "moe_gate_b64_top8_softmax": case_moe_gate(64, 8, False, True),
    "moe_gate512_b32_top8_sigmoid": case_moe_gate512(32, 8, True, False),
    "moe_gate512_b32_top8_softmax": case_moe_gate512(32, 8, False, True),
}
CASES.update({
    "tilize_fp32_4096x32": case_tilize((1, 1, 4096, 32), "fp32"),
    "tilize_fp32_512x1024": case_tilize((1, 1, 512, 1024), "fp32"),
    "tilize_int32_4096x32": case_tilize((1, 1, 4096, 32), "int32"),
    "tilize_int32_512x512": case_tilize((1, 1, 512, 512), "int32"),
    "tilize_pad_fp32_1000x30": case_tilize_pad((1, 1, 1000, 30), [1, 1, 1024, 32], "fp32"),
    "tilize_pad_int32_1000x200": case_tilize_pad((1, 1, 1000, 200), [1, 1, 1024, 224], "int32"),
})
CASES.update({
    "mm_fp32acc_untilize_64x8192x64_sb4": case_matmul_fp32acc(64, 8192, 64, 2, 2, True),
    "mm_fp32acc_128x4096x128_sb4": case_matmul_fp32acc(128, 4096, 128, 2, 2),
    "mm_fp32acc_128x4096x128_sb8": case_matmul_fp32acc(128, 4096, 128, 2, 4),
    "mm_fp32acc_128x4096x128_sb2": case_matmul_fp32acc(128, 4096, 128, 1, 2),
    "conv2d_fp32acc_64x64_32x32": case_conv2d_fp32acc(64, 64, 32),
    "add_rowb_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 1, 4096), ttnn.add),
    "add_rowb_1x1x2048x4096": case_binary_bcast((1, 1, 2048, 4096), (1, 1, 1, 4096), ttnn.add),
    "add_rowb_8x1x512x768": case_binary_bcast((8, 1, 512, 768), (1, 1, 1, 768), ttnn.add),
    "mul_rowb_1x1x128x1024": case_binary_bcast((1, 1, 128, 1024), (1, 1, 1, 1024), ttnn.multiply),
    "sub_rowa_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1, 1024), ttnn.subtract, swap=True),
    "add_rowb_bfp8_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1, 1024), ttnn.add, dt=ttnn.bfloat8_b),
    "max_rowb_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1, 1024), ttnn.maximum),
    "min_rowa_1x1x256x2048": case_binary_bcast((1, 1, 256, 2048), (1, 1, 1, 2048), ttnn.minimum, swap=True),
})
CASES.update({
    # round 3: bf16 partials (the reload without unpack to dest) and a 3x3 conv whose K splits into window_h blocks
    "mm_bf16_128x4096x128_sb4": case_matmul_fp32acc(128, 4096, 128, 2, 2, fp32=False),
    "mm_bf16_128x4096x128_sb8": case_matmul_fp32acc(128, 4096, 128, 2, 4, fp32=False),
    "conv2d_fp32acc_64x128_32x32": case_conv2d_fp32acc(64, 128, 32),
    "conv2d_bf16_64x128_32x32": case_conv2d_fp32acc(64, 128, 32, fp32=False),
    "swiglu_dsv3_t3001_xrm": case_moe_fused_swiglu(7168, 2048, 5120, 3001, True),
    "swiglu_dsv3_t768_xtile": case_moe_fused_swiglu(7168, 2048, 5120, 768, False),
})
CASES.update({
    # issue verdicts (2026-10-06): the per tile copy (#58728), unpack to dest per tile (#58729), the 32-bit binary SFPU
    # copies (#58765) and the row broadcast's per tile inits (#58734), DRAM interleaved and height sharded in L1
    "relu_bf16_1024x1024": case_unary(ttnn.relu, (1, 1, 1024, 1024), "bf16"),
    "relu_bf16_hs64_4096x256": case_unary(ttnn.relu, (1, 1, 4096, 256), "bf16", 64),
    "abs_int32_1024x1024": case_unary(ttnn.abs, (1, 1, 1024, 1024), "int32"),
    "abs_int32_hs64_4096x256": case_unary(ttnn.abs, (1, 1, 4096, 256), "int32", 64),
    "neg_fp32_1024x1024": case_unary(ttnn.neg, (1, 1, 1024, 1024), "fp32"),
    "neg_fp32_hs64_4096x256": case_unary(ttnn.neg, (1, 1, 4096, 256), "fp32", 64),
    "add_int32_1024x1024": case_binary(ttnn.add, (1, 1, 1024, 1024), (1, 1, 1024, 1024), "int32"),
    "add_int32_hs64_4096x256": case_binary(ttnn.add, (1, 1, 4096, 256), (1, 1, 4096, 256), "int32", 64),
    "add_fp32_1024x1024": case_binary(ttnn.add, (1, 1, 1024, 1024), (1, 1, 1024, 1024), "fp32"),
    "add_fp32_hs64_4096x256": case_binary(ttnn.add, (1, 1, 4096, 256), (1, 1, 4096, 256), "fp32", 64),
    "add_rowb_hs64_4096x256": case_binary(ttnn.add, (1, 1, 4096, 256), (1, 1, 1, 256), "bf16", 64, b_sharded=False),
    "mul_bf16_hs64_4096x256": case_binary(ttnn.multiply, (1, 1, 4096, 256), (1, 1, 4096, 256), "bf16", 64),
    "mul_int32_hs64_4096x256": case_binary(ttnn.multiply, (1, 1, 4096, 256), (1, 1, 4096, 256), "int32", 64),
    "sub_fp32_hs64_4096x256": case_binary(ttnn.subtract, (1, 1, 4096, 256), (1, 1, 4096, 256), "fp32", 64),
    "transpose_int32_1024x1024": case_transpose_t((1, 1, 1024, 1024), "int32", ttnn.DRAM_MEMORY_CONFIG),
    "transpose_int32_l1_1024x1024": case_transpose_t((1, 1, 1024, 1024), "int32", ttnn.L1_MEMORY_CONFIG),
})
def case_binary_bcast_mem(shape_a, shape_b, op, mem_a, mem_b, dt=ttnn.bfloat16):
    # binary_ng with a broadcast operand, each operand in its own memory config
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn(shape_a, dtype=torch.bfloat16), dt=dt, mem=mem_a, dev=dev)
        b = tile(torch.randn(shape_b, dtype=torch.bfloat16), dt=dt, mem=mem_b, dev=dev)
        return (a, b)

    def run(dev, x, y):
        return op(x, y, memory_config=mem_a)

    return setup, run


def case_group_norm(n, c, h, w, num_out_blocks, groups=32):
    # ttnn.group_norm with use_welford=True on a DRAM interleaved tile tensor, as the SDXL VAE on Blackhole runs it
    # (vae/tt/tt_resnetblock2d.py:121-135): the two-pass statistics loop of welford_groupnorm.cpp, transpose per tile
    def setup(dev):
        torch.manual_seed(0)
        grid = ttnn.CoreGrid(y=8, x=8)
        x = torch.rand((n, 1, h * w, c), dtype=torch.bfloat16)
        wt = torch.rand((c,), dtype=torch.bfloat16)
        bs = torch.rand((c,), dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        [tw, tb], tm = ttnn.dram_group_norm_params_from_torch([wt, bs], c, groups, dev, core_grid=grid, return_mask=True)
        return (tx, tw, tb, tm, grid)

    def run(dev, x, w_, b_, m_, grid):
        return ttnn.group_norm(
            x, num_groups=groups, input_mask=m_, negative_mask=None, weight=w_, bias=b_, epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=grid, num_out_blocks=num_out_blocks, inplace=False,
            use_welford=True,
        )

    return setup, run


def case_conv3d_fp32(c_in, c_out, k, t_in, pad_t, blk):
    # ttnn.experimental.conv3d on Float32 data with an fp32 DEST: the fp32-exact reduction (reduce_block_fp32_sfpu) and
    # bias (add_bias_inplace_sfpu, unary_bcast<ROW> per output tile) tails; LTX-2 vocoder layers (models/tt_dit/utils/
    # conv3d.py _FP32_BLOCKINGS, layers/audio_ops.py)
    def setup(dev):
        torch.manual_seed(0)
        cib, cob, tob, hob, wob = blk
        x = torch.randn(1, t_in, 1, 1, c_in)
        w = torch.randn(c_out, c_in, k, 1, 1) * 0.02
        b = torch.randn(c_out)
        tx = ttnn.from_torch(x, device=dev, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        tw = ttnn.experimental.prepare_conv3d_weights(
            weight_tensor=ttnn.from_torch(w, dtype=ttnn.float32, pad_value=0), groups=1, C_in_block=cib, alignment=32, device=dev
        )
        tb = ttnn.from_torch(b.reshape(1, -1), device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, pad_value=0)
        cfg = ttnn.Conv3dConfig(
            weights_dtype=ttnn.float32, output_layout=ttnn.ROW_MAJOR_LAYOUT, T_out_block=tob, W_out_block=wob,
            H_out_block=hob, C_out_block=cob, C_in_block=cib, compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        )
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        return (tx, tw, tb, cfg, ckc)

    def run(dev, x, w_, b_, cfg, ckc):
        return ttnn.experimental.conv3d(
            input_tensor=x, weight_tensor=w_, bias_tensor=b_, device=dev, config=cfg, output_channels=c_out,
            kernel_size=(k, 1, 1), stride=(1, 1, 1), padding=(pad_t, 0, 0), dilation=(1, 1, 1), padding_mode="zeros",
            dtype=ttnn.float32, compute_kernel_config=ckc,
        )

    return setup, run


def _ws_mem(dev, shape, ncores):
    # width sharded over the first ncores cores of the grid, row major
    grid = dev.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    h = 1
    for d in shape[:-1]:
        h *= d
    spec = ttnn.ShardSpec(crs, (h, shape[-1] // ncores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


def case_mlp_mul_decode(width, ncores):
    # tt_transformers decode MLP: ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[SILU]) on bf16 width sharded L1
    # (models/tt_transformers/tt/mlp.py:318-324, ff1_3 outputs in L1_WIDTH_SHARDED): binary_ng SFPU, native L1 sharding
    def setup(dev):
        torch.manual_seed(0)
        shape = (1, 1, 32, width)
        mem = _ws_mem(dev, shape, ncores)
        a = tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev)
        b = tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev)
        return (a, b, mem)

    def run(dev, a, b, mem):
        return ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)

    return setup, run


def case_mlp_mul(shape, ncores, out_dtype):
    # the tt_transformers MLP multiply as the model runs it (models/tt_transformers/tt/mlp.py:318-324): SiLU on w1_out,
    # output in activation_dtype or bfloat8_b; ncores > 0: width sharded in L1 (decode), else DRAM interleaved (prefill)
    def setup(dev):
        torch.manual_seed(0)
        mem = _ws_mem(dev, shape, ncores) if ncores else ttnn.DRAM_MEMORY_CONFIG
        a = tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev)
        b = tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev)
        return (a, b, mem)

    def run(dev, a, b, mem):
        return ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=out_dtype, memory_config=mem)

    return setup, run


def case_binary_scalar(shape, op, scalar, ncores=0):
    # binary_ng with a Python scalar operand (the tensor-scalar SFPU kernel for bf16 on Blackhole)
    def setup(dev):
        torch.manual_seed(0)
        mem = _hs_mem(dev, shape, ncores) if ncores else ttnn.DRAM_MEMORY_CONFIG
        return (tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev), mem)

    def run(dev, a, mem):
        return op(a, scalar, memory_config=mem)

    return setup, run


def case_where(shape, scalar_false=None):
    # ttnn.where on bf16 DRAM tensors: TTT, or TTS with a scalar false value (ternary SFPU kernels)
    def setup(dev):
        torch.manual_seed(0)
        c = tile((torch.rand(shape) > 0.5).to(torch.bfloat16), dev=dev)
        a = tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev)
        b = tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev)
        return (c, a, b)

    def run(dev, c, a, b):
        return ttnn.where(c, a, b if scalar_false is None else scalar_false)

    return setup, run


def case_qkv_split_sbert():
    # sentence-BERT on P150: split_query_key_value_and_split_heads of a [8,384,2304] bf8_b BLOCK sharded 6x8 QKV
    # (models/demos/sentence_bert/ttnn/ttnn_sentencebert_self_attention.py:52): the legacy transpose_wh_sharded.cpp, K^T per tile
    def setup(dev):
        torch.manual_seed(0)
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))})
        bs = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, (384, 384), ttnn.ShardOrientation.ROW_MAJOR)
        )
        x = ttnn.from_torch(torch.randn(8, 1, 384, 2304), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=bs)
        return (x,)

    def run(dev, x):
        return ttnn.experimental.split_query_key_value_and_split_heads(
            x, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), num_heads=12,
        )

    return setup, run


def case_transpose_hs(n, c, h, w, kind, ncores):
    # ttnn.transpose(-2, -1) of a height sharded tile tensor whose shards hold whole H x W blocks:
    # TransposeWHShardedProgramFactory, transpose_wh_sharded_metal2.cpp (whole shard waited once, one tile per section)
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand((n, c, h, w), kind)
        grid = dev.compute_with_storage_grid_size()
        crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        mk = lambda sh: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs, sh, ttnn.ShardOrientation.ROW_MAJOR)
        )
        x = tile(t, dt=dt, mem=mk((n * c * h // ncores, w)), dev=dev)
        return (x, mk((n * c * w // ncores, h)))

    def run(dev, x, out_mem):
        return ttnn.transpose(x, -2, -1, memory_config=out_mem)

    return setup, run


def case_i2s_dtype(shape, ncores, out_dt):
    # ttnn.interleaved_to_sharded with an output dtype: the reader fills the whole shard, then eltwise_copy_metal2.cpp copies
    # it tile by tile (copy_tile per tile, the compute thread sets the rate)
    def setup(dev):
        torch.manual_seed(0)
        x = tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev)
        return (x, _hs_mem(dev, shape, ncores))

    def run(dev, x, mem):
        return ttnn.interleaved_to_sharded(x, mem, out_dt)

    return setup, run


def case_typecast(shape, kind, out_dt, ncores=0):
    # ttnn.typecast; ncores > 0: height sharded in L1 (the sharded factory when the tile sizes match)
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand(shape, kind)
        mem = _hs_mem(dev, shape, ncores) if ncores else ttnn.DRAM_MEMORY_CONFIG
        return (tile(t, dt=dt, mem=mem, dev=dev),)

    def run(dev, x):
        return ttnn.typecast(x, out_dt)

    return setup, run


def case_clone_dtype(shape, out_dt, mem):
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev),)

    def run(dev, x):
        return ttnn.clone(x, dtype=out_dt, memory_config=mem)

    return setup, run


def case_untilize(shape, kind):
    # ttnn.untilize of a 32-bit tile tensor in DRAM: pack_untilize_block (no fast untilize for Float32 / Int32)
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand(shape, kind)
        return (tile(t, dt=dt, dev=dev),)

    def run(dev, x):
        return ttnn.untilize(x)

    return setup, run


def case_copy_into(shape, src_dt, dst_dt):
    # ttnn.copy(src, dst) with a dtype change, both DRAM interleaved: the SameMemoryConfig factory, kernel/eltwise_copy.cpp
    # (qwen36 GDN state reset, models/demos/blackhole/qwen36/tt/gdn/tp.py:404)
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn(shape, dtype=torch.bfloat16), dt=src_dt, dev=dev)
        b = tile(torch.zeros(shape, dtype=torch.float32), dt=dst_dt, dev=dev)
        return (a, b)

    def run(dev, a, b):
        return ttnn.copy(a, b)

    return setup, run


_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG
CASES.update({
    # issue verdicts, round 2 (2026-10-06): the SFPU row broadcast kernel (#58734) on a TransFuser SE multiply
    # (models/experimental/transfuser/tt/bottleneck.py:230, L1) and a sharded bf16 maximum (#58728)
    "mul_rowb_transfuser_l1": case_binary_bcast_mem((1, 1, 7040, 96), (1, 1, 1, 96), ttnn.multiply, _L1, _L1),
    "mul_rowb_1x1x2048x4096": case_binary_bcast((1, 1, 2048, 4096), (1, 1, 1, 4096), ttnn.multiply),
    "max_bf16_hs64_4096x256": case_binary(ttnn.maximum, (1, 1, 4096, 256), (1, 1, 4096, 256), "bf16", 64),
    "mlp_mul_silu_decode_14336_ws56": case_mlp_mul_decode(14336, 56),
    "qkv_split_sbert": case_qkv_split_sbert(),
    "transpose_bf16_hs64": case_transpose_hs(64, 2, 128, 128, "bf16", 64),
    "transpose_int32_hs64": case_transpose_hs(64, 2, 128, 128, "int32", 64),
    "transpose_fp32_hs64": case_transpose_hs(64, 2, 128, 128, "fp32", 64),
    "i2s_bf16_bfp8_hs32": case_i2s_dtype((1, 1, 2048, 1024), 32, ttnn.bfloat8_b),
    "typecast_fp32_int32_hs32": case_typecast((1, 1, 2048, 1024), "fp32", ttnn.int32, 32),
    "typecast_fp32_bf16_1024x1024": case_typecast((1, 1, 1024, 1024), "fp32", ttnn.bfloat16),
    "clone_bf16_fp32_l1": case_clone_dtype((1, 1, 2048, 1024), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
    "copy_bf16_fp32_qwen_32x12x128x128": case_copy_into((32, 12, 128, 128), ttnn.bfloat16, ttnn.float32),
    "tilize_int32_32x128256": case_tilize((1, 1, 32, 128256), "int32"),
    "untilize_fp32_1024x1024": case_untilize((1, 1, 1024, 1024), "fp32"),
    "untilize_int32_1024x1024": case_untilize((1, 1, 1024, 1024), "int32"),
    # SDXL VAE on Blackhole: decoder up_blocks.1 and up_blocks.2 GroupNorms, and a shape that keeps its data in L1
    "gn_vae_65536x512": case_group_norm(1, 512, 256, 256, 4),
    "gn_vae_262144x256": case_group_norm(1, 256, 512, 512, 12),
    "gn_l1_65536x256": case_group_norm(1, 256, 256, 256, 8),
    # LTX-2 vocoder fp32 layers: stage 1 AMP k7 (per Galaxy device and single device), stage 0 AMP k7, ups[0]
    "c3d_fp32_amp384k7_t766": case_conv3d_fp32(384, 384, 7, 766, 0, (128, 128, 16, 1, 1)),
    "c3d_fp32_amp384k7_t6080": case_conv3d_fp32(384, 384, 7, 6080, 3, (128, 128, 16, 1, 1)),
    "c3d_fp32_amp768k7_t386": case_conv3d_fp32(768, 768, 7, 386, 0, (256, 32, 64, 1, 1)),
    "c3d_fp32_ups0_t3054": case_conv3d_fp32(1536, 768, 11, 3054, 0, (128, 128, 32, 1, 1)),
    # the MLP multiply with the model's bfloat8_b output, decode (width sharded) and prefill (DRAM)
    "mlp_mul_silu_decode_14336_ws56_b8": case_mlp_mul((1, 1, 32, 14336), 56, ttnn.bfloat8_b),
    "mlp_mul_silu_prefill_1024x14336_b8": case_mlp_mul((1, 1, 1024, 14336), 0, ttnn.bfloat8_b),
    # the other binary_ng SFPU kernels and the ternary SFPU kernels on bf16 (each operand form once)
    "mul_colb_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1024, 1), ttnn.multiply),
    "mul_scalarb_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1, 1), ttnn.multiply),
    "mul_batchb_8x1x512x768": case_binary_bcast((8, 1, 512, 768), (1, 1, 512, 768), ttnn.multiply),
    "add_rowcol_1x1x1024x1024": case_binary_bcast((1, 1, 1024, 1), (1, 1, 1, 1024), ttnn.add),
    "mul_pyscalar_1x1x1024x1024": case_binary_scalar((1, 1, 1024, 1024), ttnn.multiply, 0.5),
    "mul_pyscalar_hs64_4096x256": case_binary_scalar((1, 1, 4096, 256), ttnn.multiply, 0.5, 64),
    "where_ttt_bf16_1024x1024": case_where((1, 1, 1024, 1024)),
    "where_tts_bf16_1024x1024": case_where((1, 1, 1024, 1024), 0.0),
    # the two binary_ng SFPU kernels hold c7a did not reach: the bcast-dims kernel (32-bit broadcasts on Blackhole take it)
    # and the row-col broadcast kernel (maximum is SFPU only)
    "mul_rowb_fp32_1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1, 1024), ttnn.multiply, dt=ttnn.float32),
    "mul_colb_fp32_1024x1024": case_binary_bcast((1, 1, 1024, 1024), (1, 1, 1024, 1), ttnn.multiply, dt=ttnn.float32),
    "max_rowcol_bf16_1024x1024": case_binary_bcast((1, 1, 1024, 1), (1, 1, 1, 1024), ttnn.maximum),
    "mul_rowcol_bf16_1024x1024": case_binary_bcast((1, 1, 1024, 1), (1, 1, 1, 1024), ttnn.multiply),
})



# round 3 rework (2026-10-07 evening): the L1-interleaved 32-bit binary ops and the 32-bit ternary kernels (#58765, probe
# pw55), the Wormhole builds of the bcast-dims and TTS/TST kernels (#58734), the production callers of zs2 (#58729)
def case_binary_mem(op, shape, kind, mem):
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape, kind)
        tb, _ = _rand(shape, kind)
        return (tile(ta, dt=dt, mem=mem, dev=dev), tile(tb, dt=dt, mem=mem, dev=dev))

    def run(dev, a, b):
        return op(a, b, memory_config=mem)

    return setup, run


def case_lerp_tts(shape, kind, mem, weight=0.3):
    # ttnn.lerp with a Python weight: the ternary TTS kernel (ternary_sfpu_no_bcast_tts_tst.cpp), as the WAN CFG combine
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape, kind)
        tb, _ = _rand(shape, kind)
        return (tile(ta, dt=dt, mem=mem, dev=dev), tile(tb, dt=dt, mem=mem, dev=dev))

    def run(dev, a, b):
        return ttnn.lerp(a, b, weight, memory_config=mem)

    return setup, run


def case_where_ttt_k(shape, kind, mem):
    # ttnn.where with three tensors of one dtype: the ternary TTT kernel
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape, kind)
        tb, _ = _rand(shape, kind)
        c = (torch.rand(shape) > 0.5).to(ta.dtype)
        return (tile(c, dt=dt, mem=mem, dev=dev), tile(ta, dt=dt, mem=mem, dev=dev), tile(tb, dt=dt, mem=mem, dev=dev))

    def run(dev, c, a, b):
        return ttnn.where(c, a, b, memory_config=mem)

    return setup, run


def case_mixed_bcast(shape_a, shape_b, op, dt_a, dt_b):
    # binary_ng with mixed dtypes and a broadcast operand: no LLK broadcast on either arch (the bcast-dims SFPU kernel)
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape_a), dt=dt_a, dev=dev), tile(torch.randn(shape_b), dt=dt_b, dev=dev))

    def run(dev, a, b):
        return op(a, b)

    return setup, run


CASES.update({
    "add_int32_l1_1024x1024": case_binary_mem(ttnn.add, (1, 1, 1024, 1024), "int32", _L1),
    "add_fp32_l1_1024x1024": case_binary_mem(ttnn.add, (1, 1, 1024, 1024), "fp32", _L1),
    "mul_int32_l1_1024x1024": case_binary_mem(ttnn.multiply, (1, 1, 1024, 1024), "int32", _L1),
    "lerp_tts_fp32_1024x1024": case_lerp_tts((1, 1, 1024, 1024), "fp32", ttnn.DRAM_MEMORY_CONFIG),
    "lerp_tts_fp32_l1_1024x1024": case_lerp_tts((1, 1, 1024, 1024), "fp32", _L1),
    "lerp_tts_bf16_1024x1024": case_lerp_tts((1, 1, 1024, 1024), "bf16", ttnn.DRAM_MEMORY_CONFIG),
    "where_ttt_fp32_1024x1024": case_where_ttt_k((1, 1, 1024, 1024), "fp32", ttnn.DRAM_MEMORY_CONFIG),
    "where_ttt_fp32_l1_1024x1024": case_where_ttt_k((1, 1, 1024, 1024), "fp32", _L1),
    "where_ttt_int32_1024x1024": case_where_ttt_k((1, 1, 1024, 1024), "int32", ttnn.DRAM_MEMORY_CONFIG),
    "mul_colb_fp32_bf16_1024x1024": case_mixed_bcast((1, 1, 1024, 1024), (1, 1, 1024, 1), ttnn.multiply, ttnn.float32, ttnn.bfloat16),
    "mul_scalarb_fp32_bf16_1024x1024": case_mixed_bcast((1, 1, 1024, 1024), (1, 1, 1, 1), ttnn.multiply, ttnn.float32, ttnn.bfloat16),
})


def case_ds_b1_sampling(seed=2005, final_idx=100, ncores=101):
    # DeepSeek V3 b1 LMHeadSampling's top-k sampling (TopKSampling::Op, sampling.hpp: llk_unpack_A_top32_rm on the UInt32
    # indices CB, to DEST), as the SamplingOp micro-op on one device (test_sampling.py::test_sampling_argmax_single_device_101_cores)
    def setup(dev):
        from models.demos.deepseek_v3_b1.micro_ops.sampling.op import SamplingOp
        g = dev.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)][:ncores]
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores})
        fin = cores[final_idx]
        in_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                  ttnn.ShardSpec(crs, (1, 160), ttnn.ShardOrientation.ROW_MAJOR))
        out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(fin, fin)}), (1, 1), ttnn.ShardOrientation.ROW_MAJOR))
        torch.manual_seed(seed)
        n = 160 * ncores
        s = ttnn.from_torch(torch.randn((1, n), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=in_mc, tile=ttnn.Tile([1, 32]))
        i = ttnn.from_torch(torch.arange(n, dtype=torch.int32).reshape(1, n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=dev, memory_config=in_mc)
        o = ttnn.from_torch(torch.zeros((1, 1), dtype=torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=dev, memory_config=out_mc)
        return (SamplingOp, s, i, o, fin)

    def run(dev, op, s, i, o, fin):
        return op.op(scores_tensor=s, indices_tensor=i, output_index_tensor=o, k=1, p=1.0, final_core_coord=fin,
                     final_mesh_coord=None)

    return setup, run


def case_ptcb(shape, narrow):
    # DeepSeek prefill per_token_cast_back: fp8_e4m3 [M, H] row major with float32 [M, H / 128] scales, bf16 out; with
    # narrow_scales_to_bf16 the Float32 scale CB is copied to DEST (per_token_cast_back_program_factory.cpp:356-359)
    def setup(dev):
        torch.manual_seed(0)
        x = (torch.randn(shape) * 3.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        sc = (torch.rand(tuple(shape[:-1]) + (shape[-1] // 128,)) * 4.0 - 2.0).to(torch.float32)
        e = ttnn.from_torch(x, dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        st = ttnn.from_torch(sc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return (e, st)

    def run(dev, e, st):
        return ttnn.experimental.deepseek_prefill.per_token_cast_back(e, st, output_dtype=ttnn.bfloat16,
                                                                     narrow_scales_to_bf16=narrow)

    return setup, run


def case_topk_dt(shape, k, dt):
    # ttnn.topk on the last dim; Float32 values take the multi-core topk_local path with every tile transposed and
    # reloaded through unpack to DEST (topk_multi_core_program_factory.cpp:537-543); bf16 with 64 < k routes to
    # topk_large_indices (llk_unpack_A_topk_xl_copy, SrcA)
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.float32), dt=dt, dev=dev),)

    def run(dev, x):
        return ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True)

    return setup, run


def case_sparse_sdpa(q_dt, kv_fp8):
    # sparse MLA prefill (Blackhole), test_sparse_sdpa.py's output-dtype shape: H 32, S 64, T 256, TOPK 64, k chunk 32
    def setup(dev):
        from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import make_inputs, to_dev
        q, kv, idx = make_inputs(32, 64, 256, 64, 576, lambda s: 64)
        tq = to_dev(q.to(torch.bfloat16) if q_dt == ttnn.bfloat16 else q.to(torch.float32), dev, q_dt)
        tkv = to_dev(kv.to(torch.float32) if kv_fp8 else kv.to(torch.bfloat16), dev, ttnn.fp8_e4m3 if kv_fp8 else ttnn.bfloat16)
        ti = to_dev(idx.to(torch.int32), dev, ttnn.uint32)
        fmt = ttnn.transformer.SparseKVFormat.FP8_E4M3 if kv_fp8 else ttnn.transformer.SparseKVFormat.BF16
        return (tq, tkv, ti, fmt)

    def run(dev, tq, tkv, ti, fmt):
        return ttnn.transformer.sparse_sdpa(tq, tkv, ti, 512, kv_format=fmt, scale=576 ** -0.5, k_chunk_size=32)

    return setup, run


def case_sampling_add(out_dt=ttnn.uint32):
    # the decode sampling index add (models/common/sampling/tt_sampling.py:1117-1122): Int32 offsets [1, 1, 32, 256] in
    # DRAM plus Int32 top-k indices width sharded on 8 cores (1, 0)-(1, 7), one tile per core, output in the same config
    def setup(dev):
        torch.manual_seed(0)
        ws = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, 32), core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 7))}),
            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        off = (torch.arange(256, dtype=torch.int32) * 16032).reshape(1, 1, 1, 256).expand(1, 1, 32, 256).contiguous()
        a = tile(off, dt=ttnn.int32, dev=dev)
        b = tile(torch.randint(0, 16000, (1, 1, 32, 256), dtype=torch.int32), dt=ttnn.int32, mem=ws, dev=dev)
        return (a, b, ws)

    def run(dev, a, b, ws):
        return ttnn.add(a, b, dtype=out_dt, memory_config=ws)

    return setup, run


CASES.update({
    "ds_b1_sampling_k1_101": case_ds_b1_sampling(),
    "ptcb_narrow_640x7168": case_ptcb((640, 7168), True),
    "ptcb_fp32scale_640x7168": case_ptcb((640, 7168), False),
    "topk_fp32_32x8192_k32": case_topk_dt((1, 1, 32, 8192), 32, ttnn.float32),
    "topk_fp32_32x16384_k32": case_topk_dt((1, 1, 32, 16384), 32, ttnn.float32),
    "topk_large_bf16_32x8192_k128": case_topk_dt((1, 1, 32, 8192), 128, ttnn.bfloat16),
    "sparse_sdpa_qfp8": case_sparse_sdpa(ttnn.fp8_e4m3, False),
    "sparse_sdpa_kvfp8": case_sparse_sdpa(ttnn.bfloat16, True),
    "sampling_add_i32_ws8": case_sampling_add(),
})


CASES.update({
    # decode-sized row-broadcast multiplies (bf16 multiply takes binary_ng's SFPU path on Blackhole): one or two tiles per core
    "mul_rowb_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 1, 4096), ttnn.multiply),
    "mul_rowb_l1_1x1x32x4096": case_binary_bcast_mem((1, 1, 32, 4096), (1, 1, 1, 4096), ttnn.multiply, _L1, _L1),
})


CASES.update({
    # decode-sized forms of the other binary_ng SFPU kernels and the ternary kernels (one or two tiles per core)
    "mul_colb_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 32, 1), ttnn.multiply),
    "mul_scalarb_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 1, 1), ttnn.multiply),
    "mul_pyscalar_1x1x32x4096": case_binary_scalar((1, 1, 32, 4096), ttnn.multiply, 0.5),
    "max_rowcol_1x1x32x4096": case_binary_bcast((1, 1, 32, 1), (1, 1, 1, 4096), ttnn.maximum),
    "where_ttt_bf16_1x1x32x4096": case_where((1, 1, 32, 4096)),
    "lerp_tts_bf16_1x1x32x4096": case_lerp_tts((1, 1, 32, 4096), "bf16", ttnn.DRAM_MEMORY_CONFIG),
})


def case_sampling_add_alt():
    # the decode sampling index add in both production forms, alternated call by call in one process: the Int32 output
    # (sampling_1d.py:418) and the UInt32 output (tt_sampling.py:1117); the reports split them by output dtype
    setup0, _ = case_sampling_add()

    def run(dev, a, b, ws):
        ttnn.add(a, b, dtype=ttnn.int32, memory_config=ws)
        return ttnn.add(a, b, dtype=ttnn.uint32, memory_config=ws)

    return setup0, run


CASES.update({
    "sampling_add_i32out_ws8": case_sampling_add(ttnn.int32),
    "sampling_add_alt": case_sampling_add_alt(),
})


def case_add_fast(shape):
    # bf16 add with fast_and_approximate_mode: binary_ng's FPU kernel (it includes eltwise_utils_common.hpp)
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev), tile(torch.randn(shape, dtype=torch.bfloat16), dev=dev))

    def run(dev, a, b):
        return ttnn.add(a, b, fast_and_approximate_mode=True)

    return setup, run


def case_where_bcast(shape_c, shape_t, shape_f):
    # ttnn.where with a broadcast operand: the ternary broadcast kernels, or binary_ng's where SFPU kernel with a scalar
    def setup(dev):
        torch.manual_seed(0)
        c = tile((torch.rand(shape_c) > 0.5).to(torch.bfloat16), dev=dev)
        t = tile(torch.randn(shape_t, dtype=torch.bfloat16), dev=dev)
        f = tile(torch.randn(shape_f, dtype=torch.bfloat16), dev=dev) if shape_f else None
        return (c, t, f)

    def run(dev, c, t, f):
        return ttnn.where(c, t, f if f is not None else 0.0)

    return setup, run


CASES.update({
    "add_bf16_fast_1024x1024": case_add_fast((1, 1, 1024, 1024)),
    "where_rowb_ttt_bf16": case_where_bcast((1, 1, 1024, 1024), (1, 1, 1, 1024), (1, 1, 1024, 1024)),
    "where_colb_tts_bf16": case_where_bcast((1, 1, 1024, 1), (1, 1, 1024, 1024), None),
})


CASES.update({
    # decode sizes for #58734's skip in the bcast-dims kernel (32-bit broadcasts on Blackhole) and the row-col kernel, and a
    # no-broadcast control whose kernel neither the skip nor the opt-in extension changes
    "mul_rowb_fp32_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 1, 4096), ttnn.multiply, dt=ttnn.float32),
    "mul_colb_fp32_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 32, 1), ttnn.multiply, dt=ttnn.float32),
    "mul_colb_fp32_bf16_1x1x32x4096": case_mixed_bcast((1, 1, 32, 4096), (1, 1, 32, 1), ttnn.multiply, ttnn.float32, ttnn.bfloat16),
    "mul_rowcol_1x1x32x4096": case_binary_bcast((1, 1, 32, 1), (1, 1, 1, 4096), ttnn.multiply),
    "mul_nob_1x1x32x4096": case_binary_bcast((1, 1, 32, 4096), (1, 1, 32, 4096), ttnn.multiply),
})


def case_sampling_add_dram(out_dt, n=128):
    # Blackhole's form of the decode sampling index add (tt_sampling.py:1117-1122, sampling memory config DRAM by default,
    # tt_sampling.py:280-283; QB2 1x4: [1, 1, 32, 32 * 4]) and of the masked index add (tt_sampling.py:768), Int32 in DRAM
    def setup(dev):
        torch.manual_seed(0)
        a = tile((torch.arange(n, dtype=torch.int32) * 16032).reshape(1, 1, 1, n).expand(1, 1, 32, n).contiguous(), dt=ttnn.int32, dev=dev)
        b = tile(torch.randint(0, 16000, (1, 1, 32, n), dtype=torch.int32), dt=ttnn.int32, dev=dev)
        return (a, b)

    def run(dev, a, b):
        return ttnn.add(a, b, dtype=out_dt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup, run


def case_sampling_add_alt_dram(n=128):
    setup0, _ = case_sampling_add_dram(ttnn.int32, n)

    def run(dev, a, b):
        ttnn.add(a, b, dtype=ttnn.int32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(a, b, dtype=ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup0, run


def case_gdn_fp32(op, shape_a, shape_b):
    # Qwen3.6 GDN decode recurrence on Float32 in L1 (models/experimental/gated_attention_gated_deltanet/tt/
    # ttnn_delta_rule_ops.py:297 add(h, outer), :465 subtract(v_t, v_read); the default decode path on Blackhole)
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn(shape_a, dtype=torch.float32), dt=ttnn.float32, mem=_L1, dev=dev)
        b = tile(torch.randn(shape_b, dtype=torch.float32), dt=ttnn.float32, mem=_L1, dev=dev)
        return (a, b)

    def run(dev, a, b):
        return op(a, b, memory_config=_L1)

    return setup, run


CASES.update({
    "sampling_add_dram_u32out": case_sampling_add_dram(ttnn.uint32),
    "sampling_add_dram_i32out": case_sampling_add_dram(ttnn.int32),
    "sampling_add_alt_dram": case_sampling_add_alt_dram(),
    "gdn_sub_fp32_l1_1x12x128": case_gdn_fp32(ttnn.subtract, (1, 12, 128), (1, 12, 128)),
    "gdn_add_fp32_l1_1x12x128x128": case_gdn_fp32(ttnn.add, (1, 12, 128, 128), (1, 12, 128, 128)),
})


def case_gptoss_colb(E):
    # gpt-oss decode experts: next_states [1, E, 2880] bfloat8_b in L1 times routing weights [1, E, 1] bf16 in L1, in place
    # (models/demos/gpt_oss/tt/experts/decode.py:155-162; mixed dtypes take no LLK broadcast, so the bcast-dims SFPU kernel)
    def setup(dev):
        torch.manual_seed(0)
        a = tile(torch.randn((1, 1, E, 2880)), dt=ttnn.bfloat8_b, mem=_L1, dev=dev)
        w = torch.zeros((1, 1, E, 1))
        w[..., torch.randperm(E)[:4], :] = torch.softmax(torch.randn(4), 0).view(4, 1)
        b = tile(w.to(torch.bfloat16), mem=_L1, dev=dev)
        return (a, b)

    def run(dev, a, b):
        return ttnn.mul(a, b, output_tensor=a)

    return setup, run


def case_bf16_bin(op, shape_a, shape_b, mem_a=ttnn.DRAM_MEMORY_CONFIG, mem_b=ttnn.DRAM_MEMORY_CONFIG, out_mem=None):
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape_a, dtype=torch.bfloat16), mem=mem_a, dev=dev),
                tile(torch.randn(shape_b, dtype=torch.bfloat16), mem=mem_b, dev=dev))

    def run(dev, a, b):
        return op(a, b, memory_config=out_mem) if out_mem is not None else op(a, b)

    return setup, run


def case_bf16_pyscalar(shape, scalar, mem=ttnn.DRAM_MEMORY_CONFIG):
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.bfloat16), mem=mem, dev=dev),)

    def run(dev, a):
        return ttnn.multiply(a, scalar)

    return setup, run


CASES.update({
    "gptoss_colb_e128": case_gptoss_colb(128),
    "gptoss_colb_e32": case_gptoss_colb(32),
    # tt_sampling decode (gpt-oss on QB2): column broadcast [1,1,32,128] x [1,1,32,1] (tt_sampling.py:783-785) and the
    # Python-scalar multiply of one tile (tt_sampling.py:745)
    "tts_colb_1x1x32x128": case_bf16_bin(ttnn.multiply, (1, 1, 32, 128), (1, 1, 32, 1)),
    "tts_pyscalar_1x1x32x1": case_bf16_pyscalar((1, 1, 32, 1), 2.0 ** -6),
    # gemma4 E2B decode layer scalar (models/demos/gemma4/tt/layer.py:333), batch 1
    "gemma4_pyscalar_1x1x1x1536": case_bf16_pyscalar((1, 1, 1, 1536), 0.5),
    # Qwen3.6 attention decode q_norm weight (models/demos/blackhole/qwen36/tt/attention/tp.py:660): [1,1,6,256] L1 x [1,1,256]
    "qwen_qnorm_rowb": case_bf16_bin(ttnn.multiply, (1, 1, 6, 256), (1, 1, 256), _L1, ttnn.DRAM_MEMORY_CONFIG, _L1),
    # Qwen3.6 GDN decode conv tap at batch 32 (qwen36/tt/gdn/tp.py:1194): [1,32,2560] x [1,1,2560], L1 out
    "qwen_gdn_conv_rowb_b32": case_bf16_bin(ttnn.multiply, (1, 32, 2560), (1, 1, 2560), out_mem=_L1),
})


def case_k_bin(op, shape_a, shape_b, kind, mem=_L1, b_act=None):
    # binary_ng on two tensors of one dtype in mem, the output in mem, optionally an activation on the second operand
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape_a, kind)
        tb, _ = _rand(shape_b, kind)
        return (tile(ta, dt=dt, mem=mem, dev=dev), tile(tb, dt=dt, mem=mem, dev=dev))

    def run(dev, a, b):
        if b_act:
            return op(a, b, input_tensor_b_activations=b_act, memory_config=mem)
        return op(a, b, memory_config=mem)

    return setup, run


def case_k_pyscalar(op, shape, kind, scalar, mem=_L1):
    # binary_ng with a Python scalar: the scalar SFPU kernel
    def setup(dev):
        torch.manual_seed(0)
        t, dt = _rand(shape, kind)
        return (tile(t, dt=dt, mem=mem, dev=dev),)

    def run(dev, a):
        return op(a, scalar, memory_config=mem)

    return setup, run


def case_k_where(shape_c, shape_t, shape_f, kind, mem=ttnn.DRAM_MEMORY_CONFIG, scalar=None, scalar_true=False):
    # ttnn.where on one dtype: TTT (ternary kernels), or TTS / TST with a scalar (binary_ng's where kernels)
    def setup(dev):
        torch.manual_seed(0)
        ta, dt = _rand(shape_t, kind)
        c = (torch.rand(shape_c) > 0.5).to(ta.dtype)
        f = tile(_rand(shape_f, kind)[0], dt=dt, mem=mem, dev=dev) if shape_f else None
        return (tile(c, dt=dt, mem=mem, dev=dev), tile(ta, dt=dt, mem=mem, dev=dev), f)

    def run(dev, c, t, f):
        if f is not None:
            return ttnn.where(c, t, f, memory_config=mem)
        if scalar_true:
            return ttnn.where(c, scalar, t, memory_config=mem)
        return ttnn.where(c, t, scalar, memory_config=mem)

    return setup, run


def case_k_tern(fn, shapes, kind, mem=ttnn.DRAM_MEMORY_CONFIG):
    # a ternary op on tensors of one dtype: fn(a, b, c, mem) with shapes[i] None for a Python scalar in that position
    def setup(dev):
        torch.manual_seed(0)
        out = []
        for sh in shapes:
            t, dt = _rand(sh, kind)
            out.append(tile(t, dt=dt, mem=mem, dev=dev))
        return tuple(out)

    def run(dev, *ts):
        return fn(*ts, mem)

    return setup, run


CASES.update({
    # S4 beyond the no-broadcast kernel: Qwen3.6 GDN decode in Float32 in L1 (ttnn_delta_rule_ops.py:456 the decay with
    # exp(g) on the second operand, :289 the beta multiply, :414 the q scale), and the 32-bit ternary and where kernels
    "gdn_decay_mul_fp32_l1": case_k_bin(ttnn.multiply, (1, 12, 128, 128), (1, 12, 1, 1), "fp32", b_act=[ttnn.UnaryOpType.EXP]),
    "gdn_beta_mul_fp32_l1": case_k_bin(ttnn.multiply, (1, 12, 128, 128), (1, 12, 1, 1), "fp32"),
    "gdn_q_scale_fp32_l1": case_k_pyscalar(ttnn.multiply, (1, 1, 12, 128), "fp32", 128 ** -0.5),
    "mul_colb_int32_1024x1024": case_k_bin(ttnn.multiply, (1, 1, 1024, 1024), (1, 1, 1024, 1), "int32", ttnn.DRAM_MEMORY_CONFIG),
    "mul_scalarb_fp32_1024x1024": case_k_bin(ttnn.multiply, (1, 1, 1024, 1024), (1, 1, 1, 1), "fp32", ttnn.DRAM_MEMORY_CONFIG),
    "mul_pyscalar_fp32_1024x1024": case_k_pyscalar(ttnn.multiply, (1, 1, 1024, 1024), "fp32", 0.7, ttnn.DRAM_MEMORY_CONFIG),
    "add_pyscalar_int32_1024x1024": case_k_pyscalar(ttnn.add, (1, 1, 1024, 1024), "int32", 3, ttnn.DRAM_MEMORY_CONFIG),
    "where_colb_ttt_fp32": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1024), "fp32"),
    "where_rowb_ttt_fp32": case_k_where((1, 1, 1024, 1024), (1, 1, 1, 1024), (1, 1, 1024, 1024), "fp32"),
    "where_colb_tts_fp32": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), None, "fp32", scalar=0.5),
    "where_colb_tst_fp32": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), None, "fp32", scalar=0.5, scalar_true=True),
    "where_colb_tts_int32": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), None, "int32", scalar=7),
    "where_scalarb_tts_fp32": case_k_where((1, 1, 1, 1), (1, 1, 1024, 1024), None, "fp32", scalar=0.5),
    "where_tts_fp32_1024x1024": case_k_where((1, 1, 1024, 1024), (1, 1, 1024, 1024), None, "fp32", scalar=0.5),
    # Llama 3.1 8B prefill mask (models/demos/llama_3p1_8b_d_p/tt/attention.py:346): Float32 [1, 1, 256, 1] per device
    "where_tts_fp32_llama_mask": case_k_where((1, 1, 256, 1), (1, 1, 256, 1), None, "fp32", scalar=0.0),
    "where_tst_int32_1024x1024": case_k_where((1, 1, 1024, 1024), (1, 1, 1024, 1024), None, "int32", scalar=5, scalar_true=True),
    "mac_tst_fp32_1024x1024": case_k_tern(lambda a, c, mem: ttnn.mac(a, 0.7, c, memory_config=mem), [(1, 1, 1024, 1024)] * 2, "fp32"),
    "mac_tts_fp32_1024x1024": case_k_tern(lambda a, b, mem: ttnn.mac(a, b, 0.7, memory_config=mem), [(1, 1, 1024, 1024)] * 2, "fp32"),
    "lerp_tts_colb_fp32": case_k_tern(lambda a, b, mem: ttnn.lerp(a, b, 0.3, memory_config=mem), [(1, 1, 1024, 1024), (1, 1, 1024, 1)], "fp32"),
    "addcmul_fp32_1024x1024": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1024, 1024)] * 3, "fp32"),
    "addcmul_rowb_fp32": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)], "fp32"),
    "addcmul_colb_fp32": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1024)], "fp32"),
    "addcdiv_fp32_1024x1024": case_k_tern(lambda a, b, c, mem: ttnn.addcdiv(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1024, 1024)] * 3, "fp32"),
    "addcmul_rowb_int32": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=3, memory_config=mem), [(1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)], "int32"),
    "addcmul_colb_int32": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=3, memory_config=mem), [(1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1024)], "int32"),
    "addcmul_int32_1024x1024": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=3, memory_config=mem), [(1, 1, 1024, 1024)] * 3, "int32"),
    # the same kernels in bf16, whose code the change must not touch
    "mul_colb_bf16_1024x1024": case_k_bin(ttnn.multiply, (1, 1, 1024, 1024), (1, 1, 1024, 1), "bf16", ttnn.DRAM_MEMORY_CONFIG),
    "where_colb_ttt_bf16": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1024), "bf16"),
    "where_colb_tst_bf16": case_k_where((1, 1, 1024, 1), (1, 1, 1024, 1024), None, "bf16", scalar=0.5, scalar_true=True),
    "mac_tst_bf16_1024x1024": case_k_tern(lambda a, c, mem: ttnn.mac(a, 0.7, c, memory_config=mem), [(1, 1, 1024, 1024)] * 2, "bf16"),
    "lerp_tts_colb_bf16": case_k_tern(lambda a, b, mem: ttnn.lerp(a, b, 0.3, memory_config=mem), [(1, 1, 1024, 1024), (1, 1, 1024, 1)], "bf16"),
    "addcmul_bf16_1024x1024": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1024, 1024)] * 3, "bf16"),
    "addcmul_rowb_bf16": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [(1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)], "bf16"),
})


CASES.update({
    # LTX-2 DiT modulation (models/tt_dit/models/transformers/ltx/transformer_ltx.py:321, 382, 396: addcmul(shift, x, scale_p1),
    # bf16 shift and scale [1, 1, 1, D] by x [1, 1, N, D]) and Wan 2.2's gated residual (transformer_wan.py:239, addcmul(x, ff,
    # gate), gate [1, 1, 1, D]): the bf16 row-broadcast addcmul kernel (ternary_addc_ops_fpu_rowbcast.cpp)
    "ltx_mod_addcmul_2048x1024": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, memory_config=mem), [(1, 1, 1, 1024), (1, 1, 2048, 1024), (1, 1, 1, 1024)], "bf16"),
    "ltx_mod_addcmul_1024x4096": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, memory_config=mem), [(1, 1, 1, 4096), (1, 1, 1024, 4096), (1, 1, 1, 4096)], "bf16"),
    # tt_sampling's repetition penalties (models/common/sampling/tt_penalties.py:61-62): where(mask [B, vocab] bf16,
    # penalties [B, 1] bf16, 1.0), binary_ng's where broadcast kernel; batch 32, Llama 3's vocab
    "penalties_where_bf16_32x128256": case_k_where((1, 1, 32, 128256), (1, 1, 32, 1), None, "bf16", scalar=1.0),
    "wan_gate_addcmul_1024x1280": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, memory_config=mem), [(1, 1, 1024, 1280), (1, 1, 1024, 1280), (1, 1, 1, 1280)], "bf16"),
})


D32 = (1, 1, 32, 4096)
D32C = (1, 1, 32, 1)
CASES.update({
    # S4 per kernel at a size where a section is one or two tiles per core, Float32 / Int32 in L1 (each kernel edit is kept
    # only if it measures faster)
    "s4k_lerp_tts_fp32_l1": case_k_tern(lambda a, b, mem: ttnn.lerp(a, b, 0.3, memory_config=mem), [D32, D32], "fp32", _L1),
    "s4k_lerp_tts_colb_fp32_l1": case_k_tern(lambda a, b, mem: ttnn.lerp(a, b, 0.3, memory_config=mem), [D32, D32C], "fp32", _L1),
    "s4k_where_ttt_fp32_l1": case_k_where(D32, D32, D32, "fp32", _L1),
    "s4k_where_colb_ttt_fp32_l1": case_k_where(D32C, D32, D32, "fp32", _L1),
    "s4k_where_colb_tts_fp32_l1": case_k_where(D32C, D32, None, "fp32", _L1, scalar=0.5),
    "s4k_addcmul_fp32_l1": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [D32, D32, D32], "fp32", _L1),
    "s4k_addcmul_colb_fp32_l1": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=0.5, memory_config=mem), [D32C, D32, D32], "fp32", _L1),
    "s4k_addcmul_colb_int32_l1": case_k_tern(lambda a, b, c, mem: ttnn.addcmul(a, b, c, value=3, memory_config=mem), [D32C, D32, D32], "int32", _L1),
    "s4k_mul_pyscalar_fp32_l1": case_k_pyscalar(ttnn.multiply, D32, "fp32", 0.7, _L1),
})

for _t in range(4):
    CASES[f"moe_gate_b32_top8_sigmoid_s{_t}"] = case_moe_gate_shift(32, 8, True, False, _t)
    CASES[f"moe_gate_b64_top8_softmax_s{_t}"] = case_moe_gate_shift(64, 8, False, True, _t)


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=int(os.environ.get("R3_L1_SMALL", "0")))
    try:
        args = None
        for name in names:
            setup, run = CASES[name]
            # free the previous case's tensors first, so a case's L1 addresses do not depend on the case before it
            args = None
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
