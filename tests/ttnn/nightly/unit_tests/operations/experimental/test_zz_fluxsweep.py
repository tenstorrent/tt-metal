import glob
import json
import math
import os

import pytest
import torch

import ttnn

GX = GY = 8
REPS = int(os.environ.get("FC_REPS", "20"))
WARMUP = 3
MANIFEST = os.environ.get("FC_MANIFEST", "/tmp/flux_manifest.json")
SHAPES_FILE = os.environ.get("FC_SHAPES", "/localdev/cglagovich/tt-metal/tools/mm_sweep/shapes_big.txt")


def auto_S(Mt, Nt):
    mn, mx = min(Mt, Nt), max(Mt, Nt)
    if mn == 0 or (mn + GY - 1) // GY > 2:
        return 1
    r = math.sqrt(mx / mn)
    best, bd = 1, abs(1 - r)
    c = 2
    while c <= GY:
        if abs(c - r) < bd:
            bd, best = abs(c - r), c
        c *= 2
    return best


def combos_for(Mt, Kt):
    out = [(None, 1)]  # auto = pure N-slice, Pk=1
    kp, seen = [], set()
    for spk in (8, 4):
        rpg = GY // spk
        if Mt % rpg != 0:  # no M-padding
            continue
        for Pk in (2, 4, 8):
            if Pk > spk or Kt % Pk != 0:
                continue
            kp.append((spk // Pk, Pk))
    for c in kp:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:5]


def load_shapes():
    shapes = []
    for line in open(SHAPES_FILE):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        M, K, N = map(int, line.split())
        if auto_S(math.ceil(M / 32), math.ceil(N / 32)) > 1:
            shapes.append((M, K, N))
    return shapes


@pytest.mark.timeout(7200)
def test_flux_autoverify(device):
    # End-to-end check that the factory's auto K-par heuristic engages and matches the back-test.
    # Per sliced shape: time OLD (TT_MM_NO_AUTO_KPAR=1 => N-slice only, the pre-heuristic default) vs
    # NEW (no env => auto S+Pk). 24 reps each, profiler flushed per combo; parser pairs old/new.
    device.enable_program_cache()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    manifest = []
    shapes = load_shapes()
    print(f"AUTOVERIFY: {len(shapes)} sliced shapes", flush=True)
    for M, K, N in shapes:
        torch.manual_seed(0)
        ti = torch.randn((M, K), dtype=torch.bfloat16)
        wi = torch.randn((K, N), dtype=torch.bfloat16)
        a = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ref = ti.float() @ wi.float()
        for mode in ("old", "new"):
            for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
                os.environ.pop(k, None)
            if mode == "old":
                os.environ["TT_MM_NO_AUTO_KPAR"] = "1"
            rec = {
                "M": M,
                "K": K,
                "N": N,
                "S": mode,
                "Pk": 0,
                "Mt": math.ceil(M / 32),
                "Kt": math.ceil(K / 32),
                "Nt": math.ceil(N / 32),
                "out_tiles": math.ceil(M / 32) * math.ceil(N / 32),
                "ok": False,
                "pcc": None,
            }
            device.clear_program_cache()
            try:
                out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
                got = ttnn.to_torch(out).float()
                rec["pcc"] = round(torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item(), 5)
                out.deallocate()
                for _ in range(WARMUP):
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
                ttnn.synchronize_device(device)
                for _ in range(REPS):
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
                rec["ok"] = True
                print(f"  OK  {M}x{K}x{N} {mode} pcc={rec['pcc']}", flush=True)
            except Exception as e:
                rec["err"] = str(e).splitlines()[-1][:160]
                print(f"  ERR {M}x{K}x{N} {mode}: {rec['err']}", flush=True)
            manifest.append(rec)
            json.dump(manifest, open(MANIFEST, "w"), indent=1)
        a.deallocate()
        b.deallocate()
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    print("BENCH_DONE", flush=True)


@pytest.mark.timeout(7200)
def test_flux_sweep(device):
    # In-process (S,Pk) combo sweep: ONE device, program cache cleared per combo so the factory re-reads
    # TT_MM_* env. Writes a manifest; run under tracy so the ops_perf CSV captures device kernel duration.
    device.enable_program_cache()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    manifest = []
    shapes = load_shapes()
    print(f"FLUX bench: {len(shapes)} sliced shapes, reps={REPS}", flush=True)
    for M, K, N in shapes:
        Mt, Kt, Nt = math.ceil(M / 32), math.ceil(K / 32), math.ceil(N / 32)
        torch.manual_seed(0)
        ti = torch.randn((M, K), dtype=torch.bfloat16)
        wi = torch.randn((K, N), dtype=torch.bfloat16)
        a = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ref = ti.float() @ wi.float()
        for S, Pk in combos_for(Mt, Kt):
            for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED"):
                os.environ.pop(k, None)
            if S is not None:
                os.environ["TT_MM_NUM_SLICES"] = str(S)
            if Pk > 1:
                os.environ["TT_MM_K_SLICES"] = str(Pk)
                os.environ["TT_MM_K_FUSED"] = "1"
            slabel = "auto" if S is None else str(S)
            rec = {
                "M": M,
                "K": K,
                "N": N,
                "S": slabel,
                "Pk": Pk,
                "Mt": Mt,
                "Kt": Kt,
                "Nt": Nt,
                "out_tiles": Mt * Nt,
                "ok": False,
                "pcc": None,
            }
            device.clear_program_cache()
            try:
                out = ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc)
                got = ttnn.to_torch(out).float()
                res = got if got.shape[-2] == M else got.reshape(Pk, M, N).sum(0)
                rec["pcc"] = round(torch.corrcoef(torch.stack([ref.flatten(), res.flatten()]))[0, 1].item(), 5)
                out.deallocate()
                for _ in range(WARMUP):
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
                ttnn.synchronize_device(device)
                for _ in range(REPS):
                    ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=cc).deallocate()
                ttnn.synchronize_device(device)
                # Flush the on-device profiler buffer to host every combo: it caps at ~1000 ops, so a
                # whole-sweep session would overflow and drop early ops. Host accumulates all of them.
                ttnn.ReadDeviceProfiler(device)
                rec["ok"] = True
                print(f"  OK  {M}x{K}x{N} S{slabel} Pk{Pk} pcc={rec['pcc']}", flush=True)
            except Exception as e:
                rec["err"] = str(e).splitlines()[-1][:160]
                print(f"  ERR {M}x{K}x{N} S{slabel} Pk{Pk}: {rec['err']}", flush=True)
            manifest.append(rec)
            json.dump(manifest, open(MANIFEST, "w"), indent=1)
        a.deallocate()
        b.deallocate()
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    print("BENCH_DONE", flush=True)
