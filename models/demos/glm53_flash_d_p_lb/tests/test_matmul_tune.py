# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Matmul schedule sweep for the model's untuned per-chip matmuls (chunk 5120 on the 2x4 mesh), one chip, random
data with the model's shapes, dtypes and fidelity. Per shape: ttnn.linear with its auto config (the model today),
ttnn.experimental.minimal_matmul with its auto config, and a small grid of MinimalMatmulConfig blockings. Prints the
mean device-bound time per call (ITERS calls queued between two syncs) and the rel L2 vs the auto-config output.
GLM_MMT_SHAPES (comma list) selects shapes, GLM_MMT_ITERS (default 20)."""

import itertools
import os
import time

import pytest
import torch

import ttnn

ITERS = int(os.environ.get("GLM_MMT_ITERS", "20"))
HIFI2, HIFI4 = ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
# name: (M, K, N, in0 dtype, weight dtype, out dtype, fidelity)
SHAPES = {
    "mla_o": (640, 16384, 4096, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, HIFI2),
    "mla_qb": (640, 1536, 16384, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, HIFI2),
    "mla_kva": (5120, 4096, 512, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, HIFI2),
    "se_gate": (5120, 4096, 256, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, HIFI4),
    "se_gate_up": (5120, 4096, 512, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, HIFI4),
    "se_down": (5120, 256, 4096, ttnn.float32, ttnn.bfloat16, ttnn.float32, HIFI4),
    "se_down_bf16": (5120, 256, 4096, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, HIFI4),
    "kda_in": (2560, 4096, 8352, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, HIFI4),
    "kda_o": (2560, 2048, 4096, ttnn.float32, ttnn.bfloat16, ttnn.float32, HIFI4),
}
SEL = os.environ.get("GLM_MMT_SHAPES", ",".join(SHAPES)).split(",")
# GLM_MMT_TRUTH=1: only the auto configs, with packer_l1_acc on and off, each scored against fp32 torch math on the
# same (rounded) inputs: rel L2 and scale <dev, ref> / <ref, ref>
TRUTH = os.environ.get("GLM_MMT_TRUTH") == "1"


def _ckc(fid, l1acc=True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1acc
    )


def _timed(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(ITERS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(device)
    return (time.time() - t0) / ITERS * 1e3, out


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_matmul_tune(device):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    for name in SEL:
        M, K, N, adt, wdt, odt, fid = SHAPES[name]
        a = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=adt, layout=ttnn.TILE_LAYOUT, device=device)
        w = ttnn.from_torch(torch.randn(1, 1, K, N) / K**0.5, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
        ckc = _ckc(fid)
        mc = ttnn.DRAM_MEMORY_CONFIG
        if TRUTH:
            truth = (ttnn.to_torch(a).double().reshape(M, K) @ ttnn.to_torch(w).double().reshape(K, N)).reshape(-1)
            print(f"[mmt] == {name}: {M}x{K}x{N} {adt}x{wdt}->{odt} {fid}", flush=True)
            for l1 in (False, True):
                k = _ckc(fid, l1)
                for tag, fn in (
                    ("linear auto", lambda: ttnn.linear(a, w, dtype=odt, compute_kernel_config=k, memory_config=mc)),
                    (
                        "minimal auto",
                        lambda: ttnn.experimental.minimal_matmul(
                            a, w, dtype=odt, compute_kernel_config=k, memory_config=mc
                        ),
                    ),
                ):
                    ms, out = _timed(device, fn)
                    o = ttnn.to_torch(out).double().reshape(-1)
                    rel = float((o - truth).norm() / truth.norm())
                    sc = float((o * truth).sum() / (truth * truth).sum())
                    print(f"[mmt]   {ms:7.3f} ms  rel {rel:.3e}  scale {sc:.6f}  {tag} packer_l1_acc={l1}", flush=True)
                    ttnn.deallocate(out)
            ttnn.deallocate(a)
            ttnn.deallocate(w)
            continue
        flops = 2 * M * K * N
        res = []
        ms, ref = _timed(device, lambda: ttnn.linear(a, w, dtype=odt, compute_kernel_config=ckc, memory_config=mc))
        ref_t = ttnn.to_torch(ref).float()
        res.append(("linear auto", ms, 0.0))

        def mm(cfg):
            return lambda: ttnn.experimental.minimal_matmul(
                a, w, config=cfg, dtype=odt, compute_kernel_config=ckc, memory_config=mc
            )

        cands = [("minimal auto", None)]
        mt, nt = M // 32, N // 32
        for mb, kb, nb, (sh, sw) in itertools.product(
            (2, 4, 8), (4, 8, 16), (2, 4, 8), ((1, 2), (2, 1), (2, 2), (1, 4), (4, 1))
        ):
            if mb % sh or nb % sw or sh * sw > 4 or mb > mt or nb > nt:
                continue
            cands.append(
                (
                    f"minimal M{mb} K{kb} N{nb} sb{sh}x{sw}",
                    ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    ),
                )
            )
        for tag, cfg in cands:
            try:
                ms, out = _timed(device, mm(cfg))
                o = ttnn.to_torch(out).float()
                rel = float((o - ref_t).norm() / ref_t.norm())
                ttnn.deallocate(out)
                res.append((tag, ms, rel))
            except Exception as ex:  # a blocking may not fit L1
                res.append((tag + " FAILED " + str(ex).splitlines()[0][:60], float("inf"), float("nan")))
        res.sort(key=lambda r: r[1])
        base = next(r[1] for r in res if r[0] == "linear auto")
        print(f"[mmt] == {name}: {M}x{K}x{N} {adt}x{wdt}->{odt} {fid}; linear auto {base:.3f} ms", flush=True)
        for tag, ms, rel in res[:6] + [r for r in res if r[0] in ("linear auto", "minimal auto")]:
            print(
                f"[mmt]   {ms:7.3f} ms  {flops / ms / 1e9:6.1f} TFLOP/s  x{base / ms:4.2f}  rel {rel:.2e}  {tag}",
                flush=True,
            )
        ttnn.deallocate(a)
        ttnn.deallocate(w)
