"""add_rmsnorm_fw_rows (single-phase) vs the two-phase add_rmsnorm_fw_allcore (forced for this process via
TTML_ADD_RMSNORM_TWO_PHASE=1) and a numpy fp64 reference: outputs {sum, out, rms} and wall time (one chip)."""
import os, time

os.environ["TTML_ADD_RMSNORM_TWO_PHASE"] = "1"
import numpy as np
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
grid = dev.compute_with_storage_grid_size()
cores = grid.x * grid.y
rng = np.random.default_rng(0)


def dev_t(a):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


def bf16(a):
    return host(dev_t(a))


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def stats(name, got, ref):
    err = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
    pcc = np.corrcoef(got.ravel(), ref.ravel())[0, 1]
    return f"{name}: pcc {pcc:.6f} rel {err:.2e} max|diff| {np.abs(got - ref).max():.3g}"


eps = 1e-6
for B, T, C in [(1, 1024, 5120), (1, 2048, 5120), (1, 512, 4096), (2, 1024, 5120)]:
    res = rng.normal(0, 1, (B, 1, T, C))
    dlt = rng.normal(0, 0.5, (B, 1, T, C))
    g = rng.normal(1, 0.2, (1, 1, 1, C))
    r_t, d_t, g_t = dev_t(res), dev_t(dlt), dev_t(g)
    rows, Wt = B * T // 32, C // 32
    fits = rows <= cores
    slices = min(cores // rows, 8) if fits else 0
    st = -(-Wt // slices) if fits else 0
    s_eff = -(-Wt // st) if fits else 0
    two = ttml.ops.metal.add_rmsnorm_fw_allcore(r_t, d_t, g_t, eps)  # two-phase (env forced)
    # fp64 reference from the bf16 inputs, sum rounded to bf16 like the kernels
    rb, db, gb = bf16(res), bf16(dlt), bf16(g)
    s_ref = bf16(rb + db)
    rms_ref = np.sqrt(np.mean((rb + db) ** 2, axis=-1, keepdims=True) + eps)
    out_ref = s_ref / rms_ref * gb
    line = f"B={B} T={T} C={C} (rows {rows}, slices {s_eff} x {st} tiles): two-phase {bench(lambda: ttml.ops.metal.add_rmsnorm_fw_allcore(r_t, d_t, g_t, eps)):.0f} us"
    if fits:
        one = ttml.ops.metal.add_rmsnorm_fw_rows(r_t, d_t, g_t, s_eff, st, eps)
        line += f", rows {bench(lambda: ttml.ops.metal.add_rmsnorm_fw_rows(r_t, d_t, g_t, s_eff, st, eps)):.0f} us"
        print(line, flush=True)
        for i, name in enumerate(("sum", "out", "rms")):
            a, b = host(one[i]), host(two[i])
            print(
                "   rows vs two-phase " + stats(name, a, b) + ("  BIT-EXACT" if np.array_equal(a, b) else ""),
                flush=True,
            )
        print(
            "   rows vs fp64 "
            + stats("out", host(one[1]), out_ref)
            + " | "
            + stats("rms", host(one[2])[..., :1], rms_ref),
            flush=True,
        )
    else:
        print(line + " (rows path not applicable: rows > cores)", flush=True)
    print("   two-phase vs fp64 " + stats("out", host(two[1]), out_ref), flush=True)
ttml.autograd.AutoContext.get_instance().close_device()
