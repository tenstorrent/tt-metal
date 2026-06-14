import json
import math
import os

import pytest
import torch

import ttnn

# Joint (S, Pk, blocking) sweep over a BATCH of shapes (FC_SHAPELIST = json path of [[M,K,N],...]).
# Writes one manifest record per config; the orchestrator joins device times from tracy and appends to
# the persistent results JSONL. PCC (host torch) only below FC_PCC_BUDGET so big shapes stay timing-only.
GX = GY = 8
REPS = int(os.environ.get("FC_REPS", "8"))
WARMUP = 2
MANIFEST = os.environ["FC_MANIFEST"]
SHAPES = [tuple(s) for s in json.load(open(os.environ["FC_SHAPELIST"]))]
PCC_BUDGET = float(os.environ.get("FC_PCC_BUDGET", "5e8"))


def percore(Mt, Nt, S, Pk):
    transpose = Mt > Nt
    y = GY // (S * Pk)
    x = S * GX
    in0 = x if transpose else y
    in1 = y if transpose else x
    return math.ceil(Mt / in0), math.ceil(Nt / in1)


def valid_SPk(Kt):
    return [(S, Pk) for S in (1, 2, 4, 8) for Pk in (1, 2, 4, 8) if S * Pk <= GY and Kt % Pk == 0]


def adsub(mb, nb):
    cap = 4

    def lg(v, c):
        for d in (4, 2, 1):
            if d <= c and v % d == 0 and d <= v:
                return d
        return 1

    if nb >= mb:
        sbw = lg(nb, cap)
        sbh = lg(mb, cap // sbw)
    else:
        sbh = lg(mb, cap)
        sbw = lg(nb, cap // sbh)
    return sbh, sbw


def blockings(Mpc, Npc, Ktb):
    # compact per-(S,Pk) blocking grid: M whole; N in {full, half}; K in {8, per-band depth}
    nbs = sorted({Npc, max(1, (Npc + 1) // 2)})
    kbs = sorted({k for k in (8, Ktb) if 1 <= k <= Ktb})
    out = []
    for nb in nbs:
        for kb in kbs:
            sbh, sbw = adsub(Mpc, nb)
            out.append((Mpc, nb, kb, sbh, sbw))
    return out


@pytest.mark.timeout(36000)
def test_jointsweep(device):
    device.enable_program_cache()
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    manifest = []

    def run(a, b, ref, rec, config=None):
        device.clear_program_cache()
        try:
            kw = {"compute_kernel_config": cc}
            if config is not None:
                kw["config"] = config
            out = ttnn.experimental.minimal_matmul(a, b, **kw)
            if ref is not None:
                g = ttnn.to_torch(out).float()
                rec["pcc"] = round(torch.corrcoef(torch.stack([ref.flatten(), g.flatten()]))[0, 1].item(), 5)
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
        ref = (ti.float() @ wi.float()) if (M * K * N) <= PCC_BUDGET else None
        base = dict(M=M, K=K, N=N, Mt=Mt, Nt=Nt, Kt=Kt, out=Mt * Nt, orient=("T" if Mt > Nt else "N"))

        for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
            os.environ.pop(k, None)
        run(
            a,
            b,
            ref,
            {
                **base,
                "S": "auto",
                "Pk": "auto",
                "Mblk": 0,
                "Nblk": 0,
                "Kblk": 0,
                "blk": "heuristic",
                "ok": False,
                "pcc": None,
            },
        )

        for S, Pk in valid_SPk(Kt):
            for k in ("TT_MM_NUM_SLICES", "TT_MM_K_SLICES", "TT_MM_K_FUSED", "TT_MM_NO_AUTO_KPAR"):
                os.environ.pop(k, None)
            os.environ["TT_MM_NUM_SLICES"] = str(S)
            if Pk > 1:
                os.environ["TT_MM_K_SLICES"] = str(Pk)
                os.environ["TT_MM_K_FUSED"] = "1"
            Mpc, Npc = percore(Mt, Nt, S, Pk)
            Ktb = Kt // Pk
            # auto-blocking at this (S,Pk)
            run(
                a,
                b,
                ref,
                {**base, "S": S, "Pk": Pk, "Mblk": 0, "Nblk": 0, "Kblk": 0, "blk": "auto", "ok": False, "pcc": None},
            )
            # pinned blockings sized to this (S,Pk)'s per-core region
            for mb, nb, kb, sbh, sbw in blockings(Mpc, Npc, Ktb):
                try:
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sbh,
                        subblock_w=sbw,
                        compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
                    )
                except Exception:
                    continue
                run(
                    a,
                    b,
                    ref,
                    {
                        **base,
                        "S": S,
                        "Pk": Pk,
                        "Mblk": mb,
                        "Nblk": nb,
                        "Kblk": kb,
                        "sbh": sbh,
                        "sbw": sbw,
                        "blk": f"m{mb}n{nb}k{kb}",
                        "ok": False,
                        "pcc": None,
                    },
                    config=cfg,
                )
        a.deallocate()
        b.deallocate()
    print("BENCH_DONE", flush=True)
