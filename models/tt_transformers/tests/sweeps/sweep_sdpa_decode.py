# SPDX-License-Identifier: Apache-2.0
"""
SDPA-decode program-config sweep — generalized CLI tool (Blackhole P150).

Give it the attention shape (--n-q-heads / --n-kv-heads / --head-dim, or a
--preset) plus a KV/context length, and it sweeps:
  - compute grid size (num cores)
  - q_chunk_size / k_chunk_size
  - exp_approx_mode (softmax exp approximation on/off)
  - compute fidelity
timing each on device and reporting the fastest.

Problem it targets (Llama-3.1-8B decode tracy profile): SdpaDecodeDeviceOperation
ran on 64 cores at ~10us avg; the model hard-codes
compute_with_storage_grid_size=(8,8) and q_chunk=k_chunk=0 (auto).

Ranked by device kernel duration (profiler). Every candidate is guarded by
try/except and the failure reason is written to the CSV.

Run with ALL THREE profiler env vars for device-kernel capture:
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd):$(pwd)/.auto MESH_DEVICE=P150
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1

Examples
  # Llama-3.1-8B (default), context 1024, full sweep
  python models/tt_transformers/tests/sweeps/sweep_sdpa_decode.py --kv-len 1024 --sweep-chunks --sweep-fid --csv sweep_sdpa.csv

  # Llama-3.2-1B preset, longer context
  python models/tt_transformers/tests/sweeps/sweep_sdpa_decode.py --preset llama3-1b --kv-len 4096 --csv sdpa_1b.csv

  # arbitrary GQA shape
  python models/tt_transformers/tests/sweeps/sweep_sdpa_decode.py --n-q-heads 28 --n-kv-heads 4 --head-dim 128 --kv-len 2048
"""
import argparse
import itertools
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_common import CSVLog, open_dev, timed_run

# Preset attention shapes: (n_q_heads, n_kv_heads, head_dim)
PRESETS = {
    "llama3-8b": (32, 8, 128),
    "llama3-70b": (64, 8, 128),
    "llama3-1b": (32, 8, 64),
    "llama3-3b": (24, 8, 128),
    "mistral-7b": (32, 8, 128),
    "qwen2-7b": (28, 4, 128),
}


# KV-cache buffer types to sweep. DRAM is the normal residence for a decode KV
# cache; L1 is included to measure the ceiling if the cache fit on-chip.
KV_MEMCFGS = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
OUT_MEMCFGS = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def build_kv(device, n_kv, kv_len, head_dim, dtype, memcfg):
    """KV cache [B=32, n_kv, kv_len, head_dim] in the requested buffer type.

    Non-paged decode SDPA expects batch in dim 0 equal to B (=32). (TT_FATAL:
    k_shape[0] == B.)
    """
    k = torch.randn(32, n_kv, kv_len, head_dim).bfloat16().float()
    v = torch.randn(32, n_kv, kv_len, head_dim).bfloat16().float()
    kt = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    vt = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    return kt, vt


def build_q(device, n_q, head_dim, dtype):
    """Decode query [1, batch(32), n_q_heads, head_dim].

    SDPA-decode expects Q as [1, B, n_q_heads, D] with D == K/V head_dim (128),
    NOT a flattened [1,1,32,n_q*head_dim]. (TT_FATAL: k_shape[-1] == D.)
    """
    q = torch.randn(1, 32, n_q, head_dim).bfloat16().float()
    return ttnn.from_torch(
        q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def grid_candidates(max_x=8, max_y=8):
    grids = []
    for y in range(1, max_y + 1):
        for x in range(1, max_x + 1):
            if x * y >= 8:  # skip trivially tiny grids
                grids.append((x, y))
    # de-dup by core count keeping a few shapes; keep all — cheap
    return grids


def main():
    ap = argparse.ArgumentParser(description="Generalized SDPA-decode program-config sweep")
    ap.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="attention shape preset (overridden by explicit --n-q-heads etc.)",
    )
    ap.add_argument("--n-q-heads", type=int, default=None, help="number of query heads")
    ap.add_argument("--n-kv-heads", type=int, default=None, help="number of KV heads (GQA)")
    ap.add_argument("--head-dim", type=int, default=None, help="per-head dimension")
    ap.add_argument("--kv-len", type=int, default=1024, help="KV/context length for decode SDPA")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sweep-chunks", action="store_true", help="also sweep q/k chunk sizes")
    ap.add_argument("--sweep-exp", action="store_true", help="also sweep exp_approx_mode on/off")
    ap.add_argument("--sweep-fid", action="store_true", help="also sweep LoFi/HiFi2/HiFi4")
    ap.add_argument("--kv-dtype", choices=["bf8", "bf16"], default="bf8")
    ap.add_argument(
        "--kv-memcfgs",
        nargs="+",
        default=["dram"],
        choices=list(KV_MEMCFGS.keys()),
        help="KV-cache buffer types to sweep (dram/l1)",
    )
    ap.add_argument(
        "--out-memcfgs",
        nargs="+",
        default=["dram"],
        choices=list(OUT_MEMCFGS.keys()),
        help="output buffer types to sweep (dram/l1)",
    )
    ap.add_argument("--csv", type=str, default="sweep_sdpa.csv")
    args = ap.parse_args()

    # Resolve shape: preset supplies defaults, explicit flags override individually.
    pq, pk, ph = PRESETS.get(args.preset or "llama3-8b")
    n_q = args.n_q_heads if args.n_q_heads is not None else pq
    n_kv = args.n_kv_heads if args.n_kv_heads is not None else pk
    head_dim = args.head_dim if args.head_dim is not None else ph

    kv_dtype = ttnn.bfloat8_b if args.kv_dtype == "bf8" else ttnn.bfloat16
    dev = open_dev()

    log = CSVLog(
        args.csv,
        [
            "n_q",
            "n_kv",
            "head_dim",
            "grid",
            "num_cores",
            "q_chunk",
            "k_chunk",
            "exp_approx",
            "fidelity",
            "kv_len",
            "kv_mem",
            "out_mem",
            "src",
            "dur_us",
            "status",
            "note",
        ],
    )
    hdr = (
        f"{'grid':>7s} {'nc':>4s} {'qch':>5s} {'kch':>5s} {'exp':>4s} {'fid':>6s} "
        f"{'kvm':>4s} {'om':>4s} {'src':>4s} {'us':>9s}"
    )
    print(f"\n===== SDPA-decode  n_q={n_q} n_kv={n_kv} hd={head_dim} kv_len={args.kv_len} =====")
    print(hdr)

    chunks = [(0, 0)]
    if args.sweep_chunks:
        chunks += [(32, 32), (64, 64), (128, 128), (0, 128), (0, 256), (0, 512)]
    exps = [False, True] if args.sweep_exp else [False]
    fids = (
        [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
        if args.sweep_fid
        else [ttnn.MathFidelity.HiFi2]
    )

    qt = build_q(dev, n_q, head_dim, ttnn.bfloat16)
    cur_pos = ttnn.from_torch(
        torch.tensor([args.kv_len - 1] * 32, dtype=torch.int32), device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    best = None
    n_ok = n_err = 0
    # KV tensors depend only on the buffer type, so build them once per kv_mem
    # (outer loop) and reuse across the program-config sweep to save rebuilds.
    for kv_mem in args.kv_memcfgs:
        kt = vt = None
        try:
            kt, vt = build_kv(dev, n_kv, args.kv_len, head_dim, kv_dtype, KV_MEMCFGS[kv_mem])
        except Exception as e:
            msg = str(e).strip().split("\n")[0][:80] or type(e).__name__
            print(f"  (skip kv_mem={kv_mem}: {msg})")
            for t in (kt, vt):
                try:
                    if t is not None:
                        t.deallocate(True)
                except Exception:
                    pass
            continue
        for (gx, gy), (qch, kch), exp_approx, fid, out_mem in itertools.product(
            grid_candidates(), chunks, exps, fids, args.out_memcfgs
        ):
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                exp_approx_mode=exp_approx,
                q_chunk_size=qch,
                k_chunk_size=kch,
            )
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=fid,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            def run():
                return ttnn.transformer.scaled_dot_product_attention_decode(
                    qt,
                    kt,
                    vt,
                    cur_pos_tensor=cur_pos,
                    program_config=pc,
                    compute_kernel_config=ckc,
                    memory_config=OUT_MEMCFGS[out_mem],
                )

            rb = [
                n_q,
                n_kv,
                head_dim,
                f"({gx},{gy})",
                gx * gy,
                qch,
                kch,
                exp_approx,
                str(fid).split(".")[-1],
                args.kv_len,
                kv_mem,
                out_mem,
            ]
            try:
                r = timed_run(dev, run, args.iters)
            except Exception as e:
                n_err += 1
                msg = str(e).strip().split("\n")[0][:80] or type(e).__name__
                log.row(rb + ["", "", "ERR", msg])
                continue
            n_ok += 1
            log.row(rb + [r["src"], f"{r['dur_ns']/1000:.2f}", "OK", ""])
            print(
                f"({gx},{gy}){'':>2s} {gx*gy:4d} {qch:5d} {kch:5d} {str(exp_approx)[0]:>4s} "
                f"{str(fid).split('.')[-1]:>6s} {kv_mem:>4s} {out_mem:>4s} {r['src']:>4s} {r['dur_ns']/1000:9.2f}"
            )
            if best is None or r["dur_ns"] < best[0]:
                best = (r["dur_ns"], gx, gy, qch, kch, exp_approx, fid, kv_mem, out_mem, r["src"])
        for t in (kt, vt):
            try:
                if t is not None:
                    t.deallocate(True)
            except Exception:
                pass

    print(f"\n  [SDPA] OK={n_ok} ERR={n_err}")
    if best:
        d, gx, gy, qch, kch, exp_approx, fid, kv_mem, out_mem, src = best
        print(
            f"  BEST: grid=({gx},{gy})={gx*gy}c q_chunk={qch} k_chunk={kch} "
            f"exp_approx={exp_approx} fid={str(fid).split('.')[-1]} kv_mem={kv_mem} out_mem={out_mem} "
            f"-> {d/1000:.2f}us [{src}]"
        )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
