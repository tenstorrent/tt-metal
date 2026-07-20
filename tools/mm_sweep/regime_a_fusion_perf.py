#!/usr/bin/env python3
# Same-config A/B perf for the regime_a_matmul epilogue fusions vs the mask-0 (no-fusion) path.
# For each (shape, config) it launches a hang-safe subprocess per variant (none|bias|act|addcmul), each
# = 1 warmup + 8 timed device-profiler iters, and reports median kernel us, per-RISC critical spans,
# delivered GB/s (logical-bytes / time, 512 GB/s convention), and the fusion latency overhead vs none.
#
# Usage:  python regime_a_fusion_perf.py                 (run the default matrix + write JSON/report)
#         python regime_a_fusion_perf.py --run M K N Ns Pk Sm kb nsb FUSE     (internal worker)
import json, os, subprocess, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import regime_a_bench as rb

ROOT = rb.ROOT
JSON = f"{HERE}/regime_a_fusion.json"
# Auto-generated perf table lives in its own file; the curated REGIME_A_FUSION_REPORT.md (correctness
# matrix, unsupported combos, etc.) is hand-maintained and must NOT be clobbered by write_report().
REPORT = f"{HERE}/REGIME_A_FUSION_PERF.md"
FUSES = ["none", "bias", "act", "addcmul", "bias_act", "bias_addcmul", "chunk2", "bias_chunk2"]

# (label, M, K, N, cfg=(Ns,Pk,Sm,kb,nsb) or None -> auto, note)
MATRIX = [
    ("256x2048x1024", 256, 2048, 1024, (1, 4, 2, 2, 4), "Sm>1 W1 (picker cfg)"),
    ("256x6144x768", 256, 6144, 768, (1, 6, 2, 4, 2), "Pk>1 Sm>1 W1 (picker cfg)"),
    ("256x15360x768", 256, 15360, 768, (1, 6, 2, 2, 3), "W>1 deep-K (picker cfg)"),
    ("32x2304x6144", 32, 2304, 6144, (2, 3, 1, 1, 6), "wide-N control"),
    ("32x6144x3072_pk1", 32, 6144, 3072, (1, 1, 1, 4, 6), "Pk=1 (no reduction)"),
]


def worker(M, K, N, Ns, Pk, Sm, kb, nsb, fuse):
    import torch
    import ttnn
    from models.common.utility_functions import comp_pcc

    try:
        os.remove(rb.BIN_CSV)
    except OSError:
        pass
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        in0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        in1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wcfg)
        cfg = None
        if Pk > 0:
            cfg = ttnn.RegimeAMatmulConfig(
                k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb
            )
        kw = dict(config=cfg)
        # Variant token set: bias / act / addcmul / chunk2 (composable). bias is applied before act;
        # act and addcmul are mutually exclusive (validated). chunk2 => output column-split into 2 chunks.
        parts = fuse.split("_")
        want_bias = "bias" in parts
        want_act = "act" in parts
        want_addcmul = "addcmul" in parts
        chunks = 2 if "chunk2" in parts else 1
        ref = (t0.float() @ t1.float())[0, 0]
        if want_bias:
            bt = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)
            kw["bias_tensor"] = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            ref = ref + bt.float().reshape(1, -1)
        if want_act:
            kw["fused_activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
            ref = torch.relu(ref)
        if want_addcmul:
            rt = torch.randn(1, 1, M, N, dtype=torch.bfloat16)
            gt = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)
            kw["fused_ternary_scalar"] = 1.0
            kw["fused_ternary_input_a"] = ttnn.from_torch(rt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            kw["fused_ternary_input_b"] = ttnn.from_torch(gt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            ref = rt.float()[0, 0] + 1.0 * ref * gt.float()[0, 0]

        def run_op():
            if chunks > 1:
                return ttnn.experimental.regime_a_matmul_split(in0, in1, chunks, -1, **kw)
            return ttnn.experimental.regime_a_matmul(in0, in1, **kw)

        out = run_op()  # warmup / compile / PCC
        if chunks > 1:
            got = torch.cat([ttnn.to_torch(ttnn.from_device(o))[0, 0] for o in out], dim=-1).float()
        else:
            got = ttnn.to_torch(ttnn.from_device(out))[0, 0].float()
        ok, pcc = comp_pcc(ref, got, 0.999)
        for _ in range(rb.ITERS):
            o = run_op()
            ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)
    runs, cores, per_risc_us, spread = rb.parse_runs_detail()
    print(
        "RESULT "
        + json.dumps(
            {
                "runs": runs,
                "pcc": float(pcc),
                "ok": bool(ok),
                "cores": cores,
                "per_risc_us": per_risc_us,
                "spread": spread,
            }
        )
    )


def launch(M, K, N, cfg, fuse):
    Ns, Pk, Sm, kb, nsb = cfg if cfg else (1, 0, 1, 1, 1)
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_HOME"] = ROOT
    env["ARCH_NAME"] = "blackhole"
    env["PYTHONPATH"] = ROOT
    args = [str(a) for a in (M, K, N, Ns, Pk, Sm, kb, nsb, fuse)]
    cmd = ["timeout", "-s", "TERM", "150", sys.executable, __file__, "--run"] + args
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=200, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            d = json.loads(line[7:])
            if d["runs"] and d["ok"]:
                med = statistics.median(d["runs"]) / rb.FREQ * 1e6
                gbps = rb.logical_bytes(M, K, N) / (statistics.median(d["runs"]) / rb.FREQ) / 1e9
                return {
                    "us": med,
                    "gbps": gbps,
                    "pcc": d["pcc"],
                    "cores": d["cores"],
                    "per_risc": d["per_risc_us"],
                    "spread": d["spread"],
                }
            return {"error": f"pcc={d.get('pcc')} ok={d.get('ok')}"}
    return {"error": r.stderr[-300:] or f"rc={r.returncode}"}


def main():
    out = {}
    for label, M, K, N, cfg, note in MATRIX:
        out[label] = {"M": M, "K": K, "N": N, "cfg": list(cfg) if cfg else None, "note": note, "variants": {}}
        for fuse in FUSES:
            res = launch(M, K, N, cfg, fuse)
            out[label]["variants"][fuse] = res
            print(
                f"{label:22} {fuse:8} {res.get('us','ERR'):>8} "
                f"{('%.1f' % res['gbps']) if 'gbps' in res else '':>7} GB/s "
                f"{res.get('error','')}",
                flush=True,
            )
    json.dump(out, open(JSON, "w"), indent=2)
    # report
    write_report(out)


def write_report(out):
    lines = [
        "# Regime-A fusion A/B perf (same-config vs mask-0)\n",
        "Median device-kernel us over 8 profiler iters; delivered GB/s = logical_bytes/time.",
        "Overhead = (fused_us - none_us)/none_us. per-RISC = critical (max-over-cores) KERNEL span:",
        "BRISC/NCRISC = the two data-movement RISCs (in1 reader + in0-ring/reduce/output writer, split",
        "across the NoCs), TRISC = compute. All three overlapping ~= total => in0/in1 overlap intact.\n",
    ]
    for label, d in out.items():
        v = d["variants"]
        base = v["none"].get("us")
        lines.append(f"\n## {label}  ({d['note']}, cfg={d['cfg']}, cores={v['none'].get('cores')})")
        lines.append("| variant | us | GB/s | overhead | PCC | per-RISC us (BRISC/NCRISC/TRISC) |")
        lines.append("|---|---|---|---|---|---|")
        for fuse in FUSES:
            r = v[fuse]
            if "us" not in r:
                lines.append(f"| {fuse} | ERR: {r.get('error','')[:40]} | | | | |")
                continue
            ov = "" if fuse == "none" or not base else f"{(r['us']-base)/base*100:+.1f}%"
            pr = r.get("per_risc") or {}
            prs = f"{pr.get('BRISC',0):.1f}/{pr.get('NCRISC',0):.1f}/{pr.get('TRISC',0):.1f}"
            lines.append(f"| {fuse} | {r['us']:.2f} | {r['gbps']:.1f} | {ov} | {r['pcc']:.5f} | {prs} |")
    open(REPORT, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {JSON} and {REPORT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        a = sys.argv[2:]
        worker(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), int(a[7]), a[8])
    else:
        main()
