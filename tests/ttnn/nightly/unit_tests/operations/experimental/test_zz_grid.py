import json
import math
import os

import pytest
import torch

import ttnn

# WH characterization grid. M,N on a 16-pt geometric tile grid (ratio ~sqrt2); K coarser (11 pt). All
# measured on the DEFAULT auto path (heuristic on, no TT_MM_* env). Device kernel duration via tracy +
# per-shape profiler flush; PCC spot-checked where the torch ref is cheap. Grid configurable for BH.
GX = int(os.environ.get("FC_GRIDX", "8"))
GY = int(os.environ.get("FC_GRIDY", "8"))
REPS = int(os.environ.get("FC_REPS", "10"))
WARMUP = 2
MANIFEST = os.environ.get("FC_MANIFEST", "/tmp/grid_manifest.json")
PCC_MAC_BUDGET = float(os.environ.get("FC_PCC_BUDGET", "5e8"))  # do torch ref only below this M*K*N

M_TILES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
N_TILES = M_TILES
K_TILES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]  # 11 pt (32 .. 8192)
if os.environ.get("FC_SMOKE"):
    M_TILES = [1, 16, 256]
    N_TILES = [4, 64]
    K_TILES = [8, 192]


def predict_SPk(Mt, Nt, Kt):
    # Mirror of the factory auto-slicer (auto_S + joint K-par heuristic) so we record what it picked.
    small, big = min(Mt, Nt), max(Mt, Nt)
    S0 = 1
    if small > 0 and (small + GY - 1) // GY <= 2:
        r = math.sqrt(big / small)
        bd, c = abs(1 - r), 2
        while c <= GY:
            if abs(c - r) < bd:
                bd, S0 = abs(c - r), c
            c *= 2
    S, Pk = S0, 1
    grid_pow2 = (GY & (GY - 1)) == 0
    out = Mt * Nt
    cores = GX * GY
    if (S0 > 1 or out < cores) and grid_pow2:
        D = Kt * cores / out if out else 0
        Pk = 8 if D >= 280 else 4 if D >= 40 else 2 if D >= 20 else 1
        if Nt >= 256:
            Pk = 1
        while Pk > 1 and (GY % Pk != 0 or Kt % Pk != 0 or Kt // Pk < 8):
            Pk //= 2
        # M-padding handling (mirrors the factory): transpose puts M on cols (axis=S*GX), else rows
        # (axis=GY/(S*Pk)=1, never pads). Output-starved transpose with deep-enough K engages at S=1
        # (free column padding on idle cores); otherwise disable K-par.
        transpose = Mt > Nt

        def m_pads(pk):
            s = GY // pk
            axis = (s * GX) if transpose else (GY // (s * pk))
            return Mt % axis != 0

        if Pk > 1 and m_pads(Pk):
            pk_s1 = GY
            while pk_s1 > 1 and (Kt % pk_s1 != 0 or Kt // pk_s1 < 8):
                pk_s1 //= 2
            Pk = pk_s1 if (transpose and out < cores and pk_s1 == GY and D >= 280) else 1
        S = GY // Pk if Pk > 1 else S0
    return S, Pk


@pytest.mark.timeout(86400)
def test_grid(device):
    device.enable_program_cache()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    if os.environ.get("FC_SHAPELIST"):  # explicit [[M,K,N],...] json (e.g. re-run a failed subset)
        shapes = [tuple(s) for s in json.load(open(os.environ["FC_SHAPELIST"]))]
    else:
        shapes = [(mt * 32, kt * 32, nt * 32) for mt in M_TILES for kt in K_TILES for nt in N_TILES]
    manifest = []
    print(f"GRID sweep: {len(shapes)} shapes (grid {GX}x{GY}, reps={REPS})", flush=True)
    for idx, (M, K, N) in enumerate(shapes):
        for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
            os.environ.pop(k, None)  # pure auto path
        Mt, Kt, Nt = M // 32, K // 32, N // 32
        S, Pk = predict_SPk(Mt, Nt, Kt)
        rec = {
            "M": M,
            "K": K,
            "N": N,
            "Mt": Mt,
            "Kt": Kt,
            "Nt": Nt,
            "out_tiles": Mt * Nt,
            "pred_S": S,
            "pred_Pk": Pk,
            "macs": M * K * N,
            "ok": False,
            "pcc": None,
            "pcc_ref": None,
        }
        torch.manual_seed(0)
        ti = torch.randn((M, K), dtype=torch.bfloat16)
        wi = torch.randn((K, N), dtype=torch.bfloat16)
        a = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        device.clear_program_cache()
        try:
            out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)  # build run
            # PCC coverage: every K-par-engaged shape (the risky path) is verified against ttnn.matmul
            # (fast on-device reference, works at any size); small shapes additionally get a torch
            # ground-truth anchor. Non-K-par large shapes (the heavily-validated plain path) are skipped.
            if Pk > 1:
                ref = ttnn.matmul(a, b, compute_kernel_config=cc)
                g = ttnn.to_torch(out).float()
                rg = ttnn.to_torch(ref).float()
                rec["pcc"] = round(torch.corrcoef(torch.stack([rg.flatten(), g.flatten()]))[0, 1].item(), 5)
                rec["pcc_ref"] = "ttnn"
                ref.deallocate()
            elif rec["macs"] <= PCC_MAC_BUDGET:
                g = ttnn.to_torch(out).float()
                r = ti.float() @ wi.float()
                rec["pcc"] = round(torch.corrcoef(torch.stack([r.flatten(), g.flatten()]))[0, 1].item(), 5)
                rec["pcc_ref"] = "torch"
            out.deallocate()
            for _ in range(WARMUP):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
            ttnn.synchronize_device(device)
            for _ in range(REPS):
                ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)  # flush per shape (on-device buffer caps ~1000 ops)
            rec["ok"] = True
        except Exception as e:
            rec["err"] = str(e).splitlines()[-1][:160]
        a.deallocate()
        b.deallocate()
        manifest.append(rec)
        if idx % 25 == 0 or not rec["ok"]:
            json.dump(manifest, open(MANIFEST, "w"))
            print(
                f"  [{idx + 1}/{len(shapes)}] {M}x{K}x{N} S{S}Pk{Pk} "
                f"{'ok' if rec['ok'] else 'ERR '+rec.get('err','')} pcc={rec['pcc']}",
                flush=True,
            )
    json.dump(manifest, open(MANIFEST, "w"))
    print("BENCH_DONE", flush=True)
