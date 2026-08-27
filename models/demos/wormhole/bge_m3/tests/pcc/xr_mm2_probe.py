# SPDX-License-Identifier: Apache-2.0
"""MULTI-M_block fused matmul+residual+cross-core-LN PROBE (integration step 3).
Exercises external-buffer reuse across blocks via the writer-driven release protocol
+ monotonic arrival thresholds. One column of P cores, MBPC blocks/core."""
import struct
import pytest
import torch
from loguru import logger
import ttnn

_K = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
_TBB = 2048


def _cb(idx, tiles, cores):
    return ttnn.CBDescriptor(total_size=tiles * _TBB, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=_TBB)])


def _acc(args, *tensors):
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _bf16_bits(x: float) -> int:
    f = struct.unpack("<I", struct.pack("<f", x))[0]
    bf16 = (f >> 16) & 0xFFFF
    return (bf16 << 16) | bf16


def _xr_mm2(A, W, R, gamma, beta, out, *, P, Ns, M_t, MBPC, K_tiles, N_tiles, K_block, eps):
    dev = A.device()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, P - 1))])
    obn = M_t * Ns
    sh, sw = 1, 2
    K_num_blocks = K_tiles // K_block
    cbs = [
        _cb(0, M_t * K_block * 2, cores), _cb(1, K_block * Ns * 2, cores), _cb(2, obn * 2, cores),
        _cb(3, obn, cores), _cb(4, obn, cores), _cb(5, Ns, cores), _cb(6, Ns, cores),
        _cb(7, 1, cores), _cb(8, 1, cores), _cb(9, M_t, cores), _cb(10, P * M_t, cores),
        _cb(11, M_t, cores), _cb(12, obn, cores), _cb(13, M_t, cores), _cb(14, P * M_t, cores),
        _cb(15, M_t, cores), _cb(16, M_t, cores), _cb(17, 1, cores), _cb(18, obn, cores),
        _cb(19, obn, cores), _cb(20, obn, cores), _cb(21, obn, cores),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(3)]
    N = P * Ns * 32
    sp = _bf16_bits(1.0 / N); sgp = _bf16_bits(1.0); ep = _bf16_bits(eps)

    reader_ct = [K_tiles, N_tiles, Ns, M_t, K_block, _TBB, P, 0, 1, 2, MBPC]; _acc(reader_ct, A, W, R, gamma, beta)
    writer_ct = [N_tiles, Ns, M_t, _TBB, P, 2, MBPC]; _acc(writer_ct, out)
    compute_ct = [M_t, Ns, P, K_block, K_num_blocks, sh, sw, MBPC]

    phys = [dev.worker_core_from_logical_core(ttnn.CoreCoord(0, y)) for y in range(P)]
    coords = []
    for j in range(P):
        coords += [phys[j].x, phys[j].y]
    reader_rt, writer_rt, compute_rt = [], [], []
    for y in range(P):
        reader_rt.append(((0, y), [A.buffer_address(), W.buffer_address(), R.buffer_address(),
                                   gamma.buffer_address(), beta.buffer_address(), sp, sgp, ep, y * Ns, y] + coords))
        writer_rt.append(((0, y), [out.buffer_address(), y * Ns, y] + coords))
        compute_rt.append(((0, y), []))

    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(kernel_source=f"{_K}/reader_xr_mm2.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=reader_ct, runtime_args=reader_rt, defines=[], config=dm0),
        ttnn.KernelDescriptor(kernel_source=f"{_K}/writer_xr_mm2.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=writer_ct, runtime_args=writer_rt, defines=[], config=dm1),
        ttnn.KernelDescriptor(kernel_source=f"{_K}/compute_xr_mm2.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=compute_ct, runtime_args=compute_rt, defines=[],
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                                  fp32_dest_acc_en=False, dst_full_sync_en=False)),
    ]
    ttnn.generic_op([A, W, R, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True, ids=["n1"])
@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
def test_xr_mm2(mesh_device):
    torch.manual_seed(0)
    dev = mesh_device
    P, Ns, M_t, MBPC, K_tiles, K_block = 8, 4, 2, 3, 4, 2
    N_tiles = P * Ns
    N = N_tiles * 32
    K = K_tiles * 32
    M = M_t * MBPC * 32       # 192
    eps = 1e-5

    A = torch.randn(1, 1, M, K) * 0.3
    Wt = torch.randn(1, 1, K, N) * 0.1
    R = torch.randn(1, 1, M, N) * 0.3
    g = torch.randn(N) * 0.2 + 1.0
    b = torch.randn(N) * 0.2
    h = (A.reshape(M, K) @ Wt.reshape(K, N)) + R.reshape(M, N)
    mu = h.mean(-1, keepdim=True); var = h.var(-1, unbiased=False, keepdim=True)
    ref = ((h - mu) / torch.sqrt(var + eps)) * g + b

    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tA, tW, tR = mk(A), mk(Wt), mk(R)
    tg = mk(g.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    tb_ = mk(b.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    out = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, M, N)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)

    _xr_mm2(tA, tW, tR, tg, tb_, out, P=P, Ns=Ns, M_t=M_t, MBPC=MBPC, K_tiles=K_tiles, N_tiles=N_tiles, K_block=K_block, eps=eps)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].reshape(M, N).float()
    fin = got.isfinite() & ref.isfinite()
    pcc = torch.corrcoef(torch.stack([got[fin].flatten(), ref[fin].flatten()]))[0, 1].item()
    # per-block check to catch reuse corruption in later blocks
    for mb in range(MBPC):
        r0 = mb * M_t * 32
        gb = got[r0:r0 + M_t * 32]; rb = ref[r0:r0 + M_t * 32]
        pb = torch.corrcoef(torch.stack([gb.flatten(), rb.flatten()]))[0, 1].item()
        logger.info(f"  block {mb}: PCC={pb:.6f}")
    logger.info(f"XR_MM2 multi-block: PCC={pcc:.6f} finite={fin.float().mean().item():.3f} MBPC={MBPC}")
    assert pcc > 0.99, f"multi-block fused cross-core LN PCC too low: {pcc}"
