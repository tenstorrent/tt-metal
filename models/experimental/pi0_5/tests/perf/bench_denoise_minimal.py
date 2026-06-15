"""Denoise expert matmul A/B at M=32, INTERLEAVED scope (the in-pipeline scope).

Resolves the matmul_decode dispute: matmul_decode wins the KERNEL zone but mandates
width-sharded in0/out, so in the real denoise block (LN/RoPE/SDPA/GELU/eltwise all
interleaved) it pays I2S/S2I reshards. This bench measures the FULL device-synced
call (incl. reshards) for three kernels at the 5 denoise expert shapes:

  (a) tuned ttnn.linear   interleaved-in/out  (the production baseline)
  (b) matmul_decode       full MatmulDecodeLinear call (incl. its internal reshards)
  (c) minimal_matmul      interleaved-in/out tuned-block (no reshard tax)

Gate G1: minimal_matmul must beat tuned ttnn.linear on >=3/5 shapes at this scope.

Eager only (no trace, no tracy). Device-synced perf_counter, min-of-N.

    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_denoise_minimal.py
"""

import os
import time
import importlib.util as ilu

import torch
import ttnn

BF16 = ttnn.bfloat16
BF8 = ttnn.bfloat8_b
M = 32
NIT = int(os.environ.get("NIT", "60"))
WARM = int(os.environ.get("WARM", "10"))
PCC_GATE = 0.99

# (label, K, N) -- the 5 denoise expert matmuls at M=32.
SHAPES = [
    ("qkv", 1024, 2560),
    ("o_proj", 2048, 1024),
    ("gate", 1024, 4096),
    ("up", 1024, 4096),
    ("down", 4096, 1024),
]

# minimal_matmul block-config candidates to sweep (M_block,K_block,N_block,sub_h,sub_w).
# M=32 => 1 M-tile; keep M_block=1. Sweep K stream + N block.
MIN_CFGS = [
    (1, 8, 8, 1, 4),
    (1, 8, 4, 1, 4),
    (1, 16, 8, 1, 4),
    (1, 4, 8, 1, 2),
    (1, 32, 4, 1, 2),
    (1, 8, 2, 1, 2),
]

_MMD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "tt", "mmdecode", "matmul_decode_linear.py"
)
_spec = ilu.spec_from_file_location("matmul_decode_linear", _MMD_PATH)
_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MatmulDecodeLinear = _mod.MatmulDecodeLinear


def _pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


def _time(fn, dev):
    for _ in range(WARM):
        o = fn()
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    best = 1e18
    for _ in range(NIT):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) * 1e6)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return best


def main():
    grid = (11, 10)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    ck_lofi = ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    print(
        f"{'shape':8s} {'K':>5s} {'N':>5s} | {'linear':>9s} {'mmd_full':>9s} {'minimal':>9s} | {'min_cfg':12s} {'min/lin':>8s} {'pcc_min':>8s} verdict"
    )
    wins = 0
    for lbl, K, N in SHAPES:
        torch.manual_seed(0)
        a = torch.randn(M, K) * 0.1
        w = torch.randn(K, N) * 0.05
        ref = a.float() @ w.float()
        a_bf16 = ttnn.from_torch(a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d)
        a_bf8 = ttnn.from_torch(a, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d)
        w_bf8 = ttnn.from_torch(w, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=d)

        # (a) tuned ttnn.linear interleaved
        def lin():
            return ttnn.linear(a_bf16, w_bf8, dtype=BF16, compute_kernel_config=ck_lofi)

        t_lin = _time(lin, d)
        pcc_lin = _pcc(ref, ttnn.to_torch(lin()).reshape(M, N))

        # (b) matmul_decode full call
        try:
            mmd = MatmulDecodeLinear(d, w_bf8, weight_dtype=BF8, out_dtype=BF16, role=lbl)
            a4 = ttnn.from_torch(a.reshape(1, 1, M, K), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=d)

            def mfull():
                return mmd(a4)

            t_mmd = _time(mfull, d)
            pcc_mmd = _pcc(ref, ttnn.to_torch(mfull()).reshape(M, N))
        except Exception as e:
            t_mmd, pcc_mmd = -1, 0.0
            print(f"  [mmd err {lbl}] {repr(e)[:70]}")

        # (c) minimal_matmul sweep (interleaved in bf8 / out bf16)
        best_t, best_cfg, best_pcc = 1e18, "n/a", 0.0
        for mb, kb, nb, sh, sw in MIN_CFGS:
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            )

            def mm():
                return ttnn.experimental.minimal_matmul(
                    a_bf8,
                    w_bf8,
                    fused_activation=None,
                    config=cfg,
                    compute_kernel_config=ck_lofi,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=BF16,
                )

            try:
                o = mm()
                pcc = _pcc(ref, ttnn.to_torch(o).reshape(M, N))
                ttnn.deallocate(o)
                if pcc < PCC_GATE:
                    continue
                t = _time(mm, d)
                if t < best_t:
                    best_t, best_cfg, best_pcc = t, f"M{mb}K{kb}N{nb}/{sh}x{sw}", pcc
            except Exception:
                continue

        ratio = best_t / t_lin if best_t < 1e17 and t_lin > 0 else 9.99
        verdict = "MIN-WIN" if ratio < 0.98 else ("tie" if ratio < 1.02 else "lin-win")
        if ratio < 0.98:
            wins += 1
        mmd_s = f"{t_mmd:9.2f}" if t_mmd > 0 else "      err"
        min_s = f"{best_t:9.2f}" if best_t < 1e17 else "      n/a"
        print(
            f"{lbl:8s} {K:5d} {N:5d} | {t_lin:9.2f} {mmd_s} {min_s} | {best_cfg:12s} {ratio:8.2f} {best_pcc:8.4f} {verdict}"
        )

    print(
        f"\nGATE G1: minimal_matmul beats tuned linear on {wins}/5 shapes (need >=3)  => {'PASS' if wins >= 3 else 'FAIL'}"
    )
    print(f"METRIC g1_wins={wins}")
    ttnn.close_mesh_device(d)


if __name__ == "__main__":
    main()
