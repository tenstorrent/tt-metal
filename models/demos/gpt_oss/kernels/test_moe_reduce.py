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


def _best_ncores(ht, max_cores=64):
    """Largest divisor of ht that is <= max_cores (uniform per-core tile count)."""
    best = 1
    for d in range(1, min(ht, max_cores) + 1):
        if ht % d == 0:
            best = d
    return best


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

        # Parallelize the Ht output tiles across a core grid. Pick a core count that
        # divides Ht evenly so the per-core tile count (a compile-time arg) is uniform.
        import os

        NCORES = int(os.environ.get("NCORES", "0")) or _best_ncores(Ht, max_cores=64)
        per_core = Ht // NCORES
        gx = min(8, NCORES)
        gy = (NCORES + gx - 1) // gx
        core_list = [(x, y) for y in range(gy) for x in range(gx)][:NCORES]
        # Build the CoreRangeSet from EXACTLY the assigned cores (not a bounding
        # rectangle) so no unassigned core in the rectangle gets the kernel+CBs with
        # no runtime args (that deadlocks on multi-row grids).
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in core_list])
        print(f"# Ht={Ht} NCORES={NCORES} per_core={per_core} grid={gx}x{gy}")
        tile_bytes = 2 * 1024  # bf16 tile

        def cb(idx, npages):
            fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
            return ttnn.CBDescriptor(total_size=npages * tile_bytes, core_ranges=core, format_descriptors=[fmt])

        cb_act, cb_sc, cb_out = 0, 1, 16
        # act CB holds 2*E (double-buffered E-batch); scores CB all E resident; out double-buffered.
        cbs = [cb(cb_act, 2 * E), cb(cb_sc, E), cb(cb_out, 2)]

        reader_ct = [cb_act, cb_sc, E, Ht]
        reader_ct += ttnn.TensorAccessorArgs(act_t).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(sc_t).get_compile_time_args()
        writer_ct = [cb_out]
        writer_ct += ttnn.TensorAccessorArgs(out_t).get_compile_time_args()
        # compute: num_output_tiles(per_core), reduction_dim_size(E), input_granularity=E, cb0, cb1, cbout
        # input_granularity=E => compute waits once per output tile for the whole E-batch,
        # matching the reader's batched E-read + single barrier.
        compute_ct = [per_core, E, E, cb_act, cb_sc, cb_out]

        rr = ttnn.RuntimeArgs()
        wr = ttnn.RuntimeArgs()
        for c, (x, y) in enumerate(core_list):
            st = c * per_core
            rr[x][y] = [act_t.buffer_address(), sc_t.buffer_address(), st, per_core]
            wr[x][y] = [out_t.buffer_address(), per_core, st]

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
        import time

        out = ttnn.generic_op([act_t, sc_t, out_t], prog)  # compile + first run
        ttnn.synchronize_device(dev)
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            out = ttnn.generic_op([act_t, sc_t, out_t], prog)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"RESULT timing: {ms:.4f} ms/call ({NCORES} cores)")
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
