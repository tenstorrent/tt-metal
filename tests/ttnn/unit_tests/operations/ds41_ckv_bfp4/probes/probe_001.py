import time, torch, ttnn, importlib.util
from pathlib import Path

sp = importlib.util.spec_from_file_location("ref", "ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/reference.py")
ref = importlib.util.module_from_spec(sp)
sp.loader.exec_module(ref)
dev = ttnn.open_device(device_id=0)
K, WIN, TOPK, H = 512, 128, 640, 64
ckc = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
)


def d(t, dt):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


g = torch.Generator().manual_seed(0)
T, Tw = 65536, 5248
x = torch.randn(T, K, generator=g)
rows = ref.encode_rows(x.bfloat16())
vals = ref.decode_rows(rows).bfloat16()
win = torch.randn(Tw, K, generator=g).bfloat16()
t_rows = d(rows.view(1, 1, T, -1), ttnn.uint8)
t_win = d(win.view(1, 1, Tw, K), ttnn.bfloat16)
t_tab = d(torch.cat([win, vals]).view(1, 1, -1, K), ttnn.bfloat16)
sink = d(torch.zeros(1, 1, 1, H).bfloat16(), ttnn.bfloat16)
for S in (160, 512):
    q = d((torch.randn(1, H, S, K, generator=g) * K**-0.5).bfloat16(), ttnn.bfloat16)
    idx = torch.empty(1, 1, S, TOPK, dtype=torch.int64)
    for s in range(S):
        idx[0, 0, s, :WIN] = torch.randperm(Tw, generator=g)[:WIN]
        idx[0, 0, s, WIN:] = torch.randperm(T, generator=g)[: TOPK - WIN]
    i4 = d(idx.to(torch.int32), ttnn.uint32)
    i16 = idx.clone()
    i16[..., WIN:] += Tw
    i16 = d(i16.to(torch.int32), ttnn.uint32)
    runs = {
        "bf16 ": lambda: ttnn.bringup.sparse_sdpa(
            q,
            t_tab,
            i16,
            K,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=128,
            compute_kernel_config=ckc,
            attention_sink=sink,
        ),
        "stock": lambda: ttnn.transformer.sparse_sdpa(
            q,
            t_tab,
            i16,
            K,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=128,
            compute_kernel_config=ckc,
            attention_sink=sink,
        ),
        "bfp4 ": lambda: ttnn.bringup.sparse_sdpa(
            q,
            t_rows,
            i4,
            K,
            kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
            scale=1.0,
            k_chunk_size=128,
            compute_kernel_config=ckc,
            attention_sink=sink,
            kv_window=t_win,
        ),
    }
    for name, f in runs.items():
        for _ in range(3):
            o = f()
            o.deallocate()
        ttnn.synchronize_device(dev)
        n = 30
        t0 = time.perf_counter()
        outs = [f() for _ in range(n)]
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / n
        for o in outs:
            o.deallocate()
        print(f"S {S:4d} H {H} {name}: {dt*1e6:8.1f} us per call")
ttnn.close_device(dev)
