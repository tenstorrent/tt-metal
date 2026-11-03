"""Iteration 51 scoping: ttnn's experimental minimal_matmul (the DiT prefill matmul) against the committed 2D-mcast
configs on every frozen base product of the TP=4 step, in the model's operand orientation (the iteration-50 profile's
in0 | in1). Products whose in1 is a plain [K, N] operand swap directly; the transpose_b ones need the activation
transposed first (that transpose is timed too). Coarse sweep of the block configuration per shape (one chip, 20 reps,
host-timed), then the best config in alternating rounds (ABAB) against the committed call; max |diff| against the
committed output on the best config. HiFi4, fp32 accumulate everywhere."""
import itertools
import sys
import time
import numpy as np
import ttnn
import ttml
from ttml.modules.frozen_linear import (
    forward_matmul_kwargs,
    forward_transposed_matmul_kwargs,
    full_core_grid,
    matmul_kwargs_2d,
)

ttml.autograd.AutoContext.get_instance().open_device()
dev = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
rng = np.random.default_rng(51)
grid = dev.compute_with_storage_grid_size()
REPS = 20
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None


def dev_t(shape, scale=1.0):
    return ttml.autograd.Tensor.from_numpy(
        (rng.normal(0, scale, shape)).astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    ).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32)


def bench(fn, iters=REPS):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e6


def mm_config(m_blk, k_blk, n_blk, sub_h, sub_w, gx, gy):
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_blk,
        K_block_size=k_blk,
        N_block_size=n_blk,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


GRIDS = [(grid.x, grid.y), (grid.x, grid.y - 2)]
SPACE = [
    (m, k, n, sh, sw, gx, gy)
    for (gx, gy), m, k, n, (sh, sw) in itertools.product(GRIDS, (4, 8, 16), (4, 8, 16), (4, 8), ((2, 2), (1, 4)))
    if m % sh == 0 and n % sw == 0
]

# name: (M, K, N, in1 stored transposed [N, K] in the model, reference kwargs builder)
SHAPES = {
    "gate_up dX": (5184, 16384, 1024, False, lambda: matmul_kwargs_2d(5184, 16384, 1024, False)),
    "w2 fw^T": (5120, 8256, 1024, False, lambda: forward_transposed_matmul_kwargs(8256, 5120, 1024)),
    "w2 dX^T": (8256, 5120, 1024, False, lambda: matmul_kwargs_2d(8256, 5120, 1024, False)),
    "qkv fw": (1024, 5184, 1536, False, lambda: matmul_kwargs_2d(1024, 5184, 1536, False)),
    "wo dX^T": (1088, 5120, 1024, False, lambda: matmul_kwargs_2d(1088, 5120, 1024, False)),
    "lm_head fw": (1024, 5120, 32768, False, lambda: forward_matmul_kwargs(5120, 32768, 1024)),
    "gate_up fw^T": (16384, 5184, 1024, True, lambda: forward_transposed_matmul_kwargs(5184, 16384, 1024)),
    "qkv dX^T": (5184, 1536, 1024, True, lambda: matmul_kwargs_2d(5184, 1536, 1024, True)),
    "wo fw^T": (5120, 1088, 1024, True, lambda: forward_transposed_matmul_kwargs(1088, 5120, 1024)),
    "lm_head dX^T": (5120, 32768, 1024, True, lambda: matmul_kwargs_2d(5120, 32768, 1024, True)),
}

for name, (M, K, N, in1_transposed, ref_kwargs) in SHAPES.items():
    if ONLY and name not in ONLY:
        continue
    kw = ref_kwargs() or {"core_grid": full_core_grid()}
    a = dev_t((1, 1, M, K), 0.05)
    b_model = dev_t((1, 1, N, K) if in1_transposed else (1, 1, K, N), 0.05)
    b_plain = ttnn.transpose(b_model, -2, -1) if in1_transposed else b_model

    def ref():
        return ttnn.matmul(a, b_model, transpose_b=in1_transposed, compute_kernel_config=cfg, **kw)

    t_ref = bench(ref)
    t_tr = bench(lambda: ttnn.transpose(b_model, -2, -1)) if in1_transposed else 0.0
    results = []
    try:
        results.append(
            ("default", bench(lambda: ttnn.experimental.minimal_matmul(a, b_plain, compute_kernel_config=cfg)))
        )
    except Exception as e:  # noqa: BLE001
        print(f"  {name} default: {type(e).__name__} {str(e).splitlines()[0][:100]}")
    for spec in SPACE:
        conf = mm_config(*spec)
        try:
            t = bench(lambda: ttnn.experimental.minimal_matmul(a, b_plain, compute_kernel_config=cfg, config=conf))
            results.append((spec, t))
        except Exception:  # noqa: BLE001  (L1 overflow, block constraints)
            pass
    results.sort(key=lambda r: r[1])
    print(
        f"{name:13s} M={M} K={K} N={N}: committed {t_ref:7.0f} us"
        + (f" (+ in1 transpose {t_tr:4.0f})" if in1_transposed else "")
    )
    for spec, t in results[:4]:
        print(f"    minimal {str(spec):32s} {t:7.0f} us  ({t / t_ref * 100:5.1f} %)")
    best_spec, _ = results[0]
    conf = None if best_spec == "default" else mm_config(*best_spec)

    def best():
        return ttnn.experimental.minimal_matmul(a, b_plain, compute_kernel_config=cfg, config=conf)

    rounds = [(bench(ref), bench(best)) for _ in range(4)]
    r_ref = np.array([r[0] for r in rounds])
    r_best = np.array([r[1] for r in rounds])
    diff = float(np.abs(host(ref()) - host(best())).max())
    print(
        f"    ABAB committed {r_ref.mean():7.0f} ± {r_ref.std():4.0f}  minimal {r_best.mean():7.0f} ± {r_best.std():4.0f}"
        f"  -> {(r_best.mean() - r_ref.mean()) / r_ref.mean() * 100:+5.1f} %"
        + (f"  (+ transpose {t_tr:4.0f} us)" if in1_transposed else "")
        + f"; max |diff| {diff:.3e}"
    )
    sys.stdout.flush()
    for t in (a, b_model, b_plain):
        ttnn.deallocate(t)
ttml.autograd.AutoContext.get_instance().close_device()
