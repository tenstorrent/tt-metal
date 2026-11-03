# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the pipeline-parallel prefill's matmul program configs (pp_prefill MM_SHAPES) on this device's grid.

Per (projection, chunk): the config PPPrefill uses today (MM_BEST if it fits the grid, else the qwen36 prefill
heuristic) is the reference; candidates are 2D multicast configs (grid gx x gy, in0 block width, the largest
output subblock) and 1D in0-multicast ones, run on one chip's shapes with random bf16 activations and bf8 / bf4
weights (as the prefill: LoFi, bf16 accumulation with packer L1 accumulation). For each candidate: the mean
device time over `iters` back-to-back runs and whether its output equals the reference's bit for bit (else the
largest difference in bf16 ulps of the reference value). Stage 1 scans the grids at one in0 block width, stage 2
tries every block width on the stage-1 leaders.

usage: python -m qwen36_resident.tests.sweep_pp_matmul --chunks 1024 --keys gdn,attn,out,gu,down --out sweep.json
"""
import argparse
import faulthandler
import json
import math
import sys
import time
from types import SimpleNamespace

import torch

import ttnn
from qwen36_resident.prefill.pp_prefill import MM_SHAPES, SMM_MAX_C, PPPrefill

TILE = 32
DTYPE = dict(gdn=ttnn.bfloat8_b, attn=ttnn.bfloat8_b, out=ttnn.bfloat8_b, gu=ttnn.bfloat4_b, down=ttnn.bfloat8_b)


def subblock(pm, pn, most=8):
    """the largest output subblock (h, w) dividing the per-core block (wider first on ties)"""
    return max(
        (
            (h, w)
            for h in range(1, most + 1)
            for w in range(1, most + 1)
            if h * w <= most and pm % h == 0 and pn % w == 0
        ),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )


def config(cand, C, N, act=None):
    kind, gx, gy, bw, sh, sw = cand
    if kind == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=C // TILE,
            per_core_N=math.ceil(N // TILE / (gx * gy)),
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=math.ceil(C // TILE / gy),
        per_core_N=math.ceil(N // TILE / gx),
        transpose_mcast=False,
        fused_activation=act,
        fuse_batch=False,
    )


def grids(C, K, N, grid):
    """(kind, gx, gy) with every grid row / column (2D) or core (1D) holding work and at least half the grid busy"""
    Mt, Nt = C // TILE, N // TILE
    out = []
    for gx in range(1, grid.x + 1):
        for gy in range(1, grid.y + 1):
            if gx * gy < grid.x * grid.y // 2:
                continue
            pm, pn = -(-Mt // gy), -(-Nt // gx)
            if (gy - 1) * pm < Mt and (gx - 1) * pn < Nt:
                out.append(("2d", gx, gy, pm, pn))
            pn1 = -(-Nt // (gx * gy))
            if (gx * gy - 1) * pn1 < Nt:
                out.append(("1d", gx, gy, Mt, pn1))
    return out


def describe(pc):
    if pc is None:
        return "auto"
    return dict(
        kind="1d" if isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig) else "2d",
        grid=(pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y),
        in0_block_w=pc.in0_block_w,
        subblock=(pc.out_subblock_h, pc.out_subblock_w),
        per_core=(pc.per_core_M, pc.per_core_N),
    )


def ulps(a, b):
    """largest |a - b| in bf16 ulps of |b| (of the larger of |b| and 1 / 64 of b's largest magnitude: tiny outputs
    are rounded at their own exponent, which would blow the ratio up)"""
    floor = b.abs().max() / 64
    e = torch.floor(torch.log2(torch.maximum(b.abs(), floor).clamp(min=1e-30)))
    return float(((a - b).abs() / torch.pow(2.0, e - 7)).max())


def sweep_sdpa(mesh, starts, iters, cands=None, dump="", ref_dump=""):
    """the prefill's chunked SDPA (pp_prefill._attn: 24 q heads x 1024 rows over a paged bf8 KV of 4 heads, block 64)
    at chunk starts `starts`: q / k chunk sizes and math fidelity against the prefill's (128, 128, HiFi4); random KV
    over the causal extent"""
    from qwen36_resident.model import HD, NKV, NQ

    C, block = 1024, 64
    rep = ttnn.ReplicateTensorToMesh(mesh)
    fid = dict(
        HiFi4=ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
        HiFi2=ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
    )
    grid = mesh.compute_with_storage_grid_size()
    if cands is None:
        cands = [(q, k, f) for q, k in ((128, 128), (128, 256), (256, 256), (256, 512), (512, 512)) for f in fid]
    # dump: save every candidate's output; ref_dump: take the reference output from another run's dump (a process with
    # other static env, e.g. TT_SDPA_GQA_MCAST=1, read once per process by the SDPA program factory)
    saved = {}
    refs = torch.load(ref_dump) if ref_dump else None
    out = []
    for cs in starts:
        blocks = (cs + C) // block
        g = torch.Generator().manual_seed(cs)
        dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        q = dev(torch.randn(1, NQ, C, HD, generator=g) * 0.3, ttnn.bfloat16)
        K = dev(torch.randn(blocks, NKV, block, HD, generator=g), ttnn.bfloat8_b)
        V = dev(torch.randn(blocks, NKV, block, HD, generator=g), ttnn.bfloat8_b)
        rm = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pt = rm(torch.arange(blocks, dtype=torch.int32).reshape(1, blocks))
        start = rm(torch.tensor([cs], dtype=torch.int32))

        def run(qc, kc, f, approx=False):
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=approx
            )
            op = lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                q, K, V, pt, chunk_start_idx_tensor=start, scale=1.0 / math.sqrt(HD), program_config=pc,
                compute_kernel_config=fid[f],
            )  # fmt: skip
            y = op()
            ttnn.synchronize_device(mesh)
            res = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
            ttnn.deallocate(y)
            t0 = time.perf_counter()
            for _ in range(iters):
                ttnn.deallocate(op())
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t0) / iters * 1e3, res

        ref_ms, ref = run(128, 128, "HiFi4")
        if refs is not None:
            if not torch.equal(refs[(cs, (128, 128, "HiFi4"))], ref):
                print(f"  NOTE: this process's (128, 128, HiFi4) differs from {ref_dump}'s", flush=True)
            ref = refs[(cs, (128, 128, "HiFi4"))]
        saved[(cs, (128, 128, "HiFi4"))] = ref
        row = dict(chunk_start=cs, ref_ms=ref_ms, cands=[])
        print(f"sdpa start {cs}: reference (128, 128, HiFi4) {ref_ms:.3f} ms", flush=True)
        for cand in cands:
            qc, kc, f = cand[:3]
            try:
                ms, res = run(qc, kc, f, len(cand) > 3 and cand[3] == "approx")
            except Exception as e:
                row["cands"].append(dict(cand=tuple(cand), error=str(e).splitlines()[0][:160]))
                continue
            saved[(cs, tuple(cand))] = res
            same = torch.equal(res, ref)
            r = dict(cand=tuple(cand), ms=ms, bit_identical=same, max_ulps=0.0 if same else ulps(res, ref))
            r["max_abs"] = float((res - ref).abs().max())
            row["cands"].append(r)
            print(
                f"  {tuple(cand)}: {ms:.3f} ms, identical {same} ({r['max_ulps']:.1f} ulps, {r['max_abs']:.2e})",
                flush=True,
            )
        out.append(row)
        for t in (q, K, V, pt, start):
            ttnn.deallocate(t)
    if dump:
        torch.save(saved, dump)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="1024,512,256")
    ap.add_argument("--keys", default="gdn,attn,out,gu,down")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--leaders", type=int, default=4)
    ap.add_argument("--sdpa", default="", help="chunk starts of an SDPA sweep (instead of the matmuls)")
    ap.add_argument("--sdpa-cands", default="", help="q:k:fidelity,... (default: the q / k / fidelity grid)")
    ap.add_argument("--sdpa-dump", default="")
    ap.add_argument("--sdpa-ref", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    # the whole 1 x 4 mesh (as the prefill opens it), every chip running the same matmul; chip 0's output is compared
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    grid = mesh.compute_with_storage_grid_size()
    mm = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    results = []
    if a.sdpa:
        try:
            cands = None
            if a.sdpa_cands:
                cands = [(int(c[0]), int(c[1]), *c[2:]) for c in (x.split(":") for x in a.sdpa_cands.split(","))]
            res = sweep_sdpa(mesh, [int(v) for v in a.sdpa.split(",")], a.iters, cands, a.sdpa_dump, a.sdpa_ref)
            json.dump(res, open(a.out, "w"), indent=1)
        finally:
            ttnn.close_mesh_device(mesh)
        return
    try:
        for C in [int(c) for c in a.chunks.split(",")]:
            for key in a.keys.split(","):
                if C <= SMM_MAX_C and key not in ("out", "down"):
                    continue  # SmallMatmul at short chunks
                K, N = MM_SHAPES[key]
                g = torch.Generator().manual_seed(C + K + N)
                x = ttnn.from_torch(
                    torch.randn(1, 1, C, K, generator=g),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                )
                w = ttnn.from_torch(
                    torch.randn(1, 1, K, N, generator=g) / math.sqrt(K),
                    dtype=DTYPE[key],
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                )
                ref_pc = PPPrefill._mm_config(SimpleNamespace(C=C, mesh=mesh, mm_cols=10), key)

                def run(pc, act=None, iters=a.iters):
                    lin = lambda: ttnn.linear(
                        x,
                        w,
                        compute_kernel_config=mm,
                        program_config=pc,
                        activation=act if pc is None else None,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                    )
                    y = lin()
                    ttnn.synchronize_device(mesh)
                    out = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                    ttnn.deallocate(y)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        ttnn.deallocate(lin())
                    ttnn.synchronize_device(mesh)
                    return (time.perf_counter() - t0) / iters * 1e3, out

                ref_ms, ref_out = run(ref_pc)
                row = dict(key=key, C=C, ref=describe(ref_pc), ref_ms=ref_ms, cands=[])
                print(f"{key} C={C}: reference {describe(ref_pc)} {ref_ms:.3f} ms", flush=True)
                Kt = K // TILE
                bws = [b for b in (2, 4, 8, 16, 32) if Kt % b == 0]
                bw0 = ref_pc.in0_block_w if ref_pc is not None and Kt % ref_pc.in0_block_w == 0 else 8

                def try_cand(cand):
                    try:
                        ms, out = run(config(cand, C, N))
                    except Exception as e:  # does not fit L1 / not a valid config
                        return dict(cand=cand, error=str(e).splitlines()[0][:160])
                    same = torch.equal(out, ref_out)
                    r = dict(cand=cand, ms=ms, bit_identical=same, max_ulps=0.0 if same else ulps(out, ref_out))
                    print(f"  {cand}: {ms:.3f} ms, identical {same} ({r['max_ulps']:.1f} ulps)", flush=True)
                    return r

                stage1 = []
                for kind, gx, gy, pm, pn in grids(C, K, N, grid):
                    stage1.append(try_cand((kind, gx, gy, bw0, *subblock(pm, pn))))
                row["cands"] += stage1
                ok = sorted((r for r in stage1 if "ms" in r), key=lambda r: r["ms"])[: a.leaders]
                for r in ok:
                    kind, gx, gy, _, _, _ = r["cand"]
                    pm, pn = (
                        (C // TILE, -(-(N // TILE) // (gx * gy)))
                        if kind == "1d"
                        else (
                            -(-(C // TILE) // gy),
                            -(-(N // TILE) // gx),
                        )
                    )
                    for bw in bws:
                        if bw != bw0:
                            row["cands"].append(try_cand((kind, gx, gy, bw, *subblock(pm, pn))))
                good = [r for r in row["cands"] if "ms" in r]
                best_id = min((r for r in good if r["bit_identical"]), key=lambda r: r["ms"], default=None)
                best_any = min(good, key=lambda r: r["ms"], default=None)
                row["best_identical"], row["best_any"] = best_id, best_any
                if key == "gu" and best_id is not None:
                    # G runs the same config with SiLU fused: it must match the reference's SiLU config too
                    silu = ttnn.UnaryOpType.SILU
                    ref_silu = PPPrefill._mm_config(SimpleNamespace(C=C, mesh=mesh, mm_cols=10), key, silu)
                    _, a_out = run(ref_silu, iters=1)
                    _, b_out = run(config(best_id["cand"], C, N, silu), iters=1)
                    row["best_identical_silu_identical"] = torch.equal(a_out, b_out)
                print(f"{key} C={C}: best identical {best_id and best_id['cand']} "
                      f"{best_id and round(best_id['ms'], 3)} ms; best any {best_any and best_any['cand']} "
                      f"{best_any and round(best_any['ms'], 3)} ms ({best_any and round(best_any['max_ulps'], 1)} ulps)",
                      flush=True)  # fmt: skip
                results.append(row)
                json.dump(results, open(a.out, "w"), indent=1)
                ttnn.deallocate(x)
                ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
