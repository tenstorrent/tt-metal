import time, torch, ttnn, importlib.util
from models.demos.deepseek_v41_flash_d_p.tt import ckv_codec as codec

sp = importlib.util.spec_from_file_location("ref", "ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/reference.py")
ref = importlib.util.module_from_spec(sp)
sp.loader.exec_module(ref)
K = "tests/ttnn/unit_tests/operations/ds41_ckv_bfp4/kernels/"
dev = ttnn.open_device(device_id=0)
T, N = 8192, 128
g = torch.Generator().manual_seed(0)
x = torch.randn(T, 512, generator=g) * torch.pow(2.0, torch.randint(-4, 4, (T, 1), generator=g).float())
rec = ref.encode_rows(x.bfloat16())  # [T, 288] = codec format B
assert torch.equal(rec, codec.encode_b(x.bfloat16().float()))
ids = torch.randperm(T, generator=g)[:N].to(torch.int32)
core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def d(t, dt):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cbd(i, fmt, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=core,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
    )


def rt(*a):
    r = ttnn.RuntimeArgs()
    r[0][0] = list(a)
    return r


t_ids = d(ids.view(1, 1, 1, N), ttnn.uint32)
# bfp4 table: [exps | exps | nibbles] 320 B a row
tab4 = torch.cat([rec[:, :32], rec[:, :32], rec[:, 32:]], dim=1)
t_tab4 = d(tab4.view(1, 1, T, 320), ttnn.uint8)
# bf16 table (baseline): the decoded values
t_tab16 = d(ref.decode_rows(rec).bfloat16().view(1, 1, T, 512), ttnn.bfloat16)


def program(bfp4, passes, out, extra=()):
    tab = t_tab4 if bfp4 else t_tab16
    page = 320 if bfp4 else 1024
    if bfp4:
        blk, ppp, fmt, tb = N * 32 + N * 256, 1, ttnn.bfloat4_b, 576
        cbs = [cbd(0, fmt, blk, 2)]
    else:
        ppp, fmt, tb = 64, ttnn.bfloat16, 2048
        cbs = [cbd(0, fmt, 2048, 2 * ppp)]
    cbs += [cbd(1, ttnn.uint32, N * 4, 1), cbd(16, fmt, tb, 32)]
    rd_ct = (
        [0, 1, N, passes, page, int(bfp4), ppp]
        + list(ttnn.TensorAccessorArgs(tab).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(t_ids).get_compile_time_args())
    )
    wr_ct = [16, 64, passes, tb] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    ks = [
        ttnn.KernelDescriptor(
            kernel_source=K + "rows_reader.cpp",
            core_ranges=core,
            compile_time_args=rd_ct,
            runtime_args=rt(tab.buffer_address(), t_ids.buffer_address()),
            defines=[(e, "1") for e in extra if e == "PROBE_NO_GATHER"],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=K + "tiles_writer.cpp",
            core_ranges=core,
            compile_time_args=wr_ct,
            runtime_args=rt(out.buffer_address()),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=K + "rows_compute.cpp",
            core_ranges=core,
            compile_time_args=[0, 16, N, passes, ppp],
            defines=([("PROBE_BFP4", "1")] if bfp4 else []) + [(e, "1") for e in extra if e == "PROBE_NO_CONVERT"],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)


def run(bfp4, passes, extra=()):
    tb = 576 if bfp4 else 2048
    out = d(torch.zeros(1, 1, 64, tb, dtype=torch.uint8), ttnn.uint8)
    io = [t_tab4 if bfp4 else t_tab16, t_ids, out]
    desc = program(bfp4, passes, out, extra)
    ttnn.generic_op(io, desc)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    ttnn.generic_op(io, desc)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    return ttnn.to_torch(out).view(torch.uint8).reshape(64, tb), dt


got, _ = run(True, 1)
g_rows = rec[ids.long()]
exp = torch.cat([codec.rows_to_tiles(g_rows[32 * tr : 32 * tr + 32]) for tr in range(N // 32)])
bad = (got != exp).any(-1)
print(f"bfp4 rows -> tiles: {int(bad.sum())} of 64 tiles differ; bytes differing {int((got != exp).sum())}")
if bad.any():
    t = int(bad.nonzero()[0])
    print(
        "first bad tile",
        t,
        "exp[:16]",
        exp[t, :16].tolist(),
        "got[:16]",
        got[t, :16].tolist(),
        "mant exp",
        exp[t, 64:80].tolist(),
        "got",
        got[t, 64:80].tolist(),
    )
torch.save({"got": got, "exp": exp}, "/localdev/dnijemcevic/bringup/ckv_2026-10-28/probe_tiles.pt")
got16, _ = run(False, 1)
vals = ref.decode_rows(rec).bfloat16()[ids.long()].float()
dev16 = torch.cat(
    [ttnn.to_torch(ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16)).float()[None] for _ in range(0)]
    or [torch.zeros(0)]
)
for ex in ((), ("PROBE_NO_GATHER",), ("PROBE_NO_CONVERT",)):
    for P in (400,):
        _, a = run(True, P, ex)
        _, b = run(False, P, ex)
        print("variant", ex or "full")
        print(
            f"passes {P:4d}: bfp4 rows {a / P * 1e6:8.2f} us/chunk   bf16 rows + tilize {b / P * 1e6:8.2f} us/chunk  (128 rows x 512, one core)"
        )
ttnn.close_device(dev)
