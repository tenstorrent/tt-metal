"""matmul_block throughput by output block shape (flat expert gate/up vs down), one core: cycles a tile matmul."""
import csv, os, sys, torch, ttnn

K = "tests/ttnn/unit_tests/operations/experimental/moe_block/flat_routed_expert/kernels"
a_ = sys.argv[1].split(",")
SHAPES = [(int(a_[0]), int(a_[1]), int(a_[2]), a_[3] == "full")]
k, reps = 8, 200
dev = ttnn.open_device(device_id=0)
res = []
try:
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    for rt, ct, kcall, full in SHAPES:
        a = ttnn.from_torch(torch.randn(rt * 32, k * 32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        b = ttnn.from_torch(torch.randn(k * 32, ct * 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * pg,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)],
            )
            for i, dt, pg, n in (
                (0, ttnn.bfloat8_b, 1088, rt * k),
                (1, ttnn.bfloat4_b, 576, k * ct),
                (16, ttnn.bfloat16, 2048, 2),
            )
        ]
        rd = ttnn.RuntimeArgs()
        rd[0][0] = [a.buffer_address(), b.buffer_address()]
        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=full
        )
        ks = [
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/mm_shape_bench_reader.cpp",
                core_ranges=core,
                compile_time_args=[rt * k, k * ct]
                + ttnn.TensorAccessorArgs(a).get_compile_time_args()
                + ttnn.TensorAccessorArgs(b).get_compile_time_args(),
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{K}/mm_shape_bench_compute.cpp",
                core_ranges=core,
                compile_time_args=[rt, ct, k, reps, kcall],
                runtime_args=[],
                config=cc,
            ),
        ]
        ttnn.generic_op([a, b, o], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
        res.append((rt, ct, kcall, full))
finally:
    ttnn.close_device(dev)
for rt, ct, kcall, full in res:
    rows = list(csv.reader(open("generated/profiler/.logs/profile_log_device.csv")))
    hdr = [h.strip() for h in rows[1]]
    ix = {h: i for i, h in enumerate(hdr)}
    st = en = None
    for r in rows[2:]:
        if r[ix["zone name"]].strip() == "MM-SHAPE" and r[ix["RISC processor type"]].strip() == "TRISC_1":
            t = int(r[ix["time[cycles since reset]"]])
            if r[ix["type"]].strip() == "ZONE_START":
                st = t
            else:
                en = t
    cyc = (en - st) / (reps * k * rt * ct)
    print(
        f"rt {rt} ct {ct} {'k-call ' if kcall else 'per-step'} {'full' if full else 'half'} sync: {cyc:6.2f} cycles a tile matmul",
        flush=True,
    )
