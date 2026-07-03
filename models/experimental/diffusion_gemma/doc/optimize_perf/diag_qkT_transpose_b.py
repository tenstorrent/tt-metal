"""Verify ttnn.matmul(transpose_b=True) == permute(k,(0,1,3,2))+matmul (QKt), and time both."""
import ttnn, torch, time

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
B, H, C, D = 1, 8, 256, 256
q = torch.randn(B, H, C, D) * 0.1
k = torch.randn(B, H, C, D) * 0.1
mk = lambda t: ttnn.from_torch(
    t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
qt, kt = mk(q), mk(k)


def old():
    kT = ttnn.permute(kt, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = ttnn.matmul(qt, kT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kT.deallocate(True)
    return s


def new():
    return ttnn.matmul(qt, kt, transpose_b=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)


try:
    so, sn = old(), new()
    o = ttnn.to_torch(so).float()
    n = ttnn.to_torch(sn).float()
    g = (q @ k.transpose(-1, -2)).float()

    def pcc(a, b):
        a = a.flatten()
        b = b.flatten()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    print(
        f"shapes out={tuple(o.shape)}  PCC(old,new)={pcc(o,n):.6f}  PCC(new,golden)={pcc(n,g):.6f}  max|old-new|={(o-n).abs().max():.2e}  bitexact={torch.equal(o,n)}"
    )
    so.deallocate(True)
    sn.deallocate(True)
    for f, name in [(old, "old permute+matmul"), (new, "matmul transpose_b")]:
        for _ in range(3):
            x = f()
            ttnn.synchronize_device(dev)
            x.deallocate(True)
        N = 30
        t0 = time.perf_counter()
        for _ in range(N):
            x = f()
            ttnn.synchronize_device(dev)
            x.deallocate(True)
        print(f"{name:22s}: {(time.perf_counter()-t0)/N*1e3:.3f} ms/call")
except Exception as e:
    import traceback

    traceback.print_exc()
    print("FAILED:", type(e).__name__, str(e)[:200])
finally:
    ttnn.close_device(dev)
