import json
import math
import os

import pytest
import torch

import ttnn

GX = GY = 8
REPS = int(os.environ.get("FC_REPS", "12"))
WARMUP = 2
MANIFEST = os.environ.get("FC_MANIFEST", "/tmp/probe_manifest.json")
# square underperformers + their skinny same-(out,Kt) twins
SHAPES = json.loads(os.environ.get("FC_SHAPELIST_JSON", "[]")) or [
    [1024, 6144, 384],
    [6144, 4096, 64],  # square 32x12 (out384,Kt192) vs skinny 192x2
    [256, 8192, 384],
    [64, 8192, 1536],  # square 8x12  (out96,Kt256)  vs skinny 2x48
    [192, 6144, 256],
    [1536, 6144, 32],  # square 6x8   (out48,Kt192)  vs skinny 48x1
]


def percore(Mt, Nt, Kt, S, Pk):
    transpose = Mt > Nt
    y = GY // (S * Pk)
    x = S * GX
    in0 = x if transpose else y
    in1 = y if transpose else x
    return math.ceil(Mt / in0), math.ceil(Nt / in1), Kt // Pk  # M_pc, N_pc, Kt_per_band


def valid_SPk(Kt):
    return [(S, Pk) for S in (1, 2, 4, 8) for Pk in (1, 2, 4, 8) if S * Pk <= GY and Kt % Pk == 0]


def adsub(mb, nb):
    cap = 4

    def lg(v, c):
        for dd in (4, 2, 1):
            if dd <= c and v % dd == 0 and dd <= v:
                return dd
        return 1

    if nb >= mb:
        sbw = lg(nb, cap)
        sbh = lg(mb, cap // sbw)
    else:
        sbh = lg(mb, cap)
        sbw = lg(nb, cap // sbh)
    return sbh, sbw


@pytest.mark.timeout(7200)
def test_probe(device):
    device.enable_program_cache()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    manifest = []

    def measure(a, b, ref_t, rec, config=None):
        device.clear_program_cache()
        try:
            kw = {"compute_kernel_config": cc}
            if config is not None:
                kw["config"] = config
            out = ttnn.experimental.minimal_matmul(a, b, **kw)
            g = ttnn.to_torch(out).float()
            rec["pcc"] = round(torch.corrcoef(torch.stack([ref_t.flatten(), g.flatten()]))[0, 1].item(), 5)
            out.deallocate()
            for _ in range(WARMUP):
                ttnn.experimental.minimal_matmul(a, b, **kw).deallocate()
            ttnn.synchronize_device(device)
            for _ in range(REPS):
                ttnn.experimental.minimal_matmul(a, b, **kw).deallocate()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            rec["ok"] = True
        except Exception as e:
            rec["err"] = str(e).splitlines()[-1][:120]
        manifest.append(rec)
        json.dump(manifest, open(MANIFEST, "w"))

    for M, K, N in SHAPES:
        Mt, Kt, Nt = M // 32, K // 32, N // 32
        torch.manual_seed(0)
        ti = torch.randn((M, K), dtype=torch.bfloat16)
        wi = torch.randn((K, N), dtype=torch.bfloat16)
        a = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ref_t = ti.float() @ wi.float()
        base = dict(M=M, K=K, N=N, Mt=Mt, Nt=Nt, Kt=Kt)

        # default (heuristic, auto everything)
        for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
            os.environ.pop(k, None)
        measure(a, b, ref_t, {**base, "S": "auto", "Pk": "auto", "blk": "auto", "ok": False, "pcc": None})

        # JOINT sweep: for each (S,Pk), auto-blocking AND a few pinned blockings sized to that sliced region
        for S, Pk in valid_SPk(Kt):
            for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
                os.environ.pop(k, None)
            os.environ["TT_MM_NUM_SLICES"] = str(S)
            if Pk > 1:
                os.environ["TT_MM_K_SLICES"] = str(Pk)
                os.environ["TT_MM_K_FUSED"] = "1"
            Mpc, Npc, Ktb = percore(Mt, Nt, Kt, S, Pk)
            # auto-blocking for this (S,Pk)
            measure(a, b, ref_t, {**base, "S": S, "Pk": Pk, "blk": "auto", "ok": False, "pcc": None})
            # pinned blockings: mb=Mpc; nb in {Npc, Npc/2}; kb in {8, Ktb}
            nbs = sorted({Npc, max(1, (Npc + 1) // 2)})
            kbs = sorted({min(8, Ktb), Ktb})
            for nb in nbs:
                for kb in kbs:
                    if kb < 1:
                        continue
                    sbh, sbw = adsub(Mpc, nb)
                    try:
                        cfg = ttnn.MinimalMatmulConfig(
                            M_block_size=Mpc,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sbh,
                            subblock_w=sbw,
                            compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
                        )
                    except Exception:
                        continue
                    measure(
                        a,
                        b,
                        ref_t,
                        {**base, "S": S, "Pk": Pk, "blk": f"m{Mpc}n{nb}k{kb}", "ok": False, "pcc": None},
                        config=cfg,
                    )
        a.deallocate()
        b.deallocate()
        print(f"done {M}x{K}x{N}", flush=True)
    print("BENCH_DONE", flush=True)
