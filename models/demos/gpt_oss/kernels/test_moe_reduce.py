# SPDX-License-Identifier: Apache-2.0
"""Standalone PCC test for the custom gpt-oss fused MoE expert-reduce, driven via
ttnn.generic_op (JiT kernels, no _ttnn.so rebuild).

Computes  out[t,h] = sum_e w[t,e] * down[e,t,h]  on a single core, and checks PCC
against a torch reference. (The bias term bias*sum_e w[e] is added on host in the
model-integration step; here we validate the core weighted reduce.)
"""
import torch

import ttnn

KDIR = "models/demos/gpt_oss/kernels"
E = 32  # experts (reduction dim)
T = 32  # tokens (one tile row)
H = 2880  # hidden -> Ht = H/32 output tiles
TILE = 32


def col0_score_tiles(w):
    """Build [E,1,T,32] score tensor: tile e has w[:,e] in column 0, else 0.
    COL-broadcast MAC reads column 0 of each row."""
    st = torch.zeros(E, 1, T, TILE, dtype=torch.bfloat16)
    st[:, 0, :, 0] = w.t()  # [E,T] -> column 0
    return st


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else torch.dot(a, b).item() / d


def main():
    torch.manual_seed(0)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        Ht = H // TILE
        down = torch.randn(E, 1, T, H, dtype=torch.bfloat16) * 0.1
        bias = torch.randn(H, dtype=torch.bfloat16) * 0.05
        w = torch.zeros(T, E, dtype=torch.bfloat16)
        w[:, :4] = torch.rand(T, 4, dtype=torch.bfloat16)
        # full gpt-oss tail: out = sum_e w[e]*(down[e]+bias)
        ref = torch.zeros(T, H)
        for e in range(E):
            ref += w[:, e : e + 1].float() * (down[e, 0].float() + bias.float())

        act_t = ttnn.from_torch(
            down, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sc_t = ttnn.from_torch(
            col0_score_tiles(w),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )

        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        tile_bytes = 2 * 1024  # bf16 tile

        def cb(idx, npages):
            fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
            return ttnn.CBDescriptor(total_size=npages * tile_bytes, core_ranges=core, format_descriptors=[fmt])

        cb_act, cb_sc, cb_out = 0, 1, 16
        # act CB double-buffered; scores CB holds all E resident; out double-buffered.
        cbs = [cb(cb_act, 2), cb(cb_sc, E), cb(cb_out, 2)]

        reader_ct = [cb_act, cb_sc, E, Ht]
        reader_ct += ttnn.TensorAccessorArgs(act_t).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(sc_t).get_compile_time_args()
        writer_ct = [cb_out]
        writer_ct += ttnn.TensorAccessorArgs(out_t).get_compile_time_args()
        # compute: num_output_tiles, reduction_dim_size(E), input_granularity, cb0, cb1, cbout
        compute_ct = [Ht, E, 1, cb_act, cb_sc, cb_out]

        rr = ttnn.RuntimeArgs()
        wr = ttnn.RuntimeArgs()
        rr[0][0] = [act_t.buffer_address(), sc_t.buffer_address()]
        wr[0][0] = [out_t.buffer_address(), Ht, 0]

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/moe_reduce_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=reader_ct,
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=writer_ct,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/moe_reduce_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=compute_ct,
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(),
        )

        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        out = ttnn.generic_op([act_t, sc_t, out_t], prog)
        got = ttnn.to_torch(out).float().reshape(T, H)  # = sum_e w[e]*down[e]
        # add bias term on host: bias * sum_e w[e]
        wsum = w.float().sum(dim=1, keepdim=True)  # [T,1]
        got_full = got + bias.float().unsqueeze(0) * wsum
        p_core = pcc(ref - bias.float().unsqueeze(0) * wsum, got)
        p_full = pcc(ref, got_full)
        print(f"RESULT core reduce (sum w*down) PCC = {p_core:.5f}")
        print(
            f"RESULT full tail (+bias*sum_w)  PCC = {p_full:.5f}  got.norm={got_full.norm():.2f} ref.norm={ref.norm():.2f}"
        )
        print("VERDICT:", "PASS" if p_full >= 0.99 else "FAIL")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
