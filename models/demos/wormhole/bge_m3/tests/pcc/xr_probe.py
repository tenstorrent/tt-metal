# SPDX-License-Identifier: Apache-2.0
"""Standalone cross-core LN reduce PROBE. One column of P cores; LN over the full
feature dim spread across the column via all-gather of partial sum/sumsq + local
finalize. Isolates the cross-core reduction (no matmul) and gates on PCC vs torch
LayerNorm. Foundation for the fast N-split fused AttnOut+LN op."""
import struct
import pytest
import torch
from loguru import logger
import ttnn

_K = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_attn_out_ln/kernels"
_TBB = 2048  # bf16 tile bytes


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


def _xr_probe(A, gamma, beta, out, *, P, Ns, M_t, N_tiles, eps):
    dev = A.device()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, P - 1))])
    obn = M_t * Ns
    cbs = [
        _cb(0, obn, cores), _cb(2, obn, cores), _cb(5, Ns, cores), _cb(6, Ns, cores),
        _cb(7, 1, cores), _cb(8, 1, cores), _cb(9, M_t, cores), _cb(10, P * M_t, cores),
        _cb(11, M_t, cores), _cb(12, obn, cores), _cb(13, M_t, cores), _cb(14, P * M_t, cores),
        _cb(15, M_t, cores), _cb(16, M_t, cores), _cb(17, 1, cores), _cb(18, obn, cores),
        _cb(19, obn, cores), _cb(20, obn, cores),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
    N = P * Ns * 32
    scaler_packed = _bf16_bits(1.0 / N)
    scaler_g_packed = _bf16_bits(1.0)
    eps_packed = _bf16_bits(eps)

    reader_ct = [N_tiles, Ns, M_t, _TBB, P, 0, 1]; _acc(reader_ct, A, gamma, beta)
    writer_ct = [N_tiles, Ns, M_t, _TBB]; _acc(writer_ct, out)
    compute_ct = [M_t, Ns, P]

    # physical coords of the P column cores
    phys = [dev.worker_core_from_logical_core(ttnn.CoreCoord(0, y)) for y in range(P)]

    reader_rt, writer_rt, compute_rt = [], [], []
    for y in range(P):
        coords = []
        for j in range(P):
            coords += [phys[j].x, phys[j].y]
        reader_rt.append(((0, y), [A.buffer_address(), gamma.buffer_address(), beta.buffer_address(),
                                   scaler_packed, scaler_g_packed, eps_packed, y * Ns, y] + coords))
        writer_rt.append(((0, y), [out.buffer_address(), y * Ns]))
        compute_rt.append(((0, y), []))

    dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(kernel_source=f"{_K}/reader_xr.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=reader_ct, runtime_args=reader_rt, defines=[], config=dm0),
        ttnn.KernelDescriptor(kernel_source=f"{_K}/writer_xr.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=writer_ct, runtime_args=writer_rt, defines=[], config=dm1),
        ttnn.KernelDescriptor(kernel_source=f"{_K}/compute_xr.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                              core_ranges=cores, compile_time_args=compute_ct, runtime_args=compute_rt, defines=[],
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                                  fp32_dest_acc_en=False, dst_full_sync_en=False)),
    ]
    ttnn.generic_op([A, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True, ids=["n1"])
@pytest.mark.parametrize("device_params", [{"num_command_queues": 1}], indirect=True)
def test_xr(mesh_device):
    torch.manual_seed(0)
    dev = mesh_device
    P, Ns, M_t = 8, 4, 2
    N_tiles = P * Ns          # 32
    N = N_tiles * 32          # 1024
    M = M_t * 32              # 64
    eps = 1e-5

    x = torch.randn(1, 1, M, N) * 0.5 + 0.2
    g = torch.randn(N) * 0.2 + 1.0
    b = torch.randn(N) * 0.2
    mu = x.reshape(M, N).mean(-1, keepdim=True)
    var = x.reshape(M, N).var(-1, unbiased=False, keepdim=True)
    ref = ((x.reshape(M, N) - mu) / torch.sqrt(var + eps)) * g + b

    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tX = mk(x)
    tg = mk(g.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    tb_ = mk(b.reshape(1, 1, 1, N).expand(1, 1, 32, N).contiguous())
    out = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, M, N)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)

    _xr_probe(tX, tg, tb_, out, P=P, Ns=Ns, M_t=M_t, N_tiles=N_tiles, eps=eps)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))[:1].reshape(M, N).float()
    fin = got.isfinite() & ref.isfinite()
    pcc = torch.corrcoef(torch.stack([got[fin].flatten(), ref[fin].flatten()]))[0, 1].item()
    logger.info(f"XR probe: PCC={pcc:.6f} finite_frac={fin.float().mean().item():.3f} "
                f"got[0,:3]={got[0,:3].tolist()} ref[0,:3]={ref[0,:3].tolist()}")
    assert pcc > 0.99, f"cross-core LN PCC too low: {pcc}"
