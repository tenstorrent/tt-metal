#!/usr/bin/env python3
"""R1b decode table: per run and cache position, SDPA decode DEVICE KERNEL DURATION (ops report, invocation order), KV bytes, GB/s."""
import glob, os, re
import pandas as pd

BYTES = {"bfp8_b": 1088 / 1024, "bfloat16": 2.0}
rows = []
for f in sorted(glob.glob("r1b_*_ops_perf_results.csv")):
    tag = os.path.basename(f).replace("_ops_perf_results.csv", "")
    o = pd.read_csv(f, low_memory=False)
    dur = [c for c in o.columns if c.startswith("DEVICE KERNEL DURATION")][0]
    prov = open(tag + ".csv").readline()
    s = o[o["OP CODE"].astype(str).str.contains("Decode", regex=True)].reset_index(drop=True)
    ns = s[dur].astype(float).tolist()
    cores = int(s["CORE COUNT"].iloc[0]) if len(s) else -1
    env = dict(re.findall(r"(\w+)=(\S+)", prov.split("env ")[1].split(" mode")[0])) if "env " in prov else {}
    if "mla_decode_nonpaged" in tag:
        # analysis/mla_decode_latency_sweep.py form: b8 nh16 nkv1 kv_lora 512 d_rope 64, K bf16 (V = K slice, separate tensor), cur_pos = cache - 1;
        # 3 warm-ups + MLAD_ITERS 1 = 4 invocations per cache length, first discarded
        b = int(env.get("MLAD_B", 8))
        d_qk = 576
        kvd = "bfloat16"
        seqs = [int(x) for x in env["MLAD_SEQS"].split(",")]
        per = 3 + int(env.get("MLAD_ITERS", 1))
        for pi, s_ in enumerate(seqs):
            g = ns[pi * per : (pi + 1) * per]
            if len(g) < per:
                continue
            med = pd.Series(g[1:]).median()
            kvb = b * s_ * d_qk * BYTES[kvd]  # K only streamed once (latent reuse)
            rows.append(
                dict(
                    run=tag,
                    kind="MLA decode non-paged (model form: b8 nh16 kv_lora512 d_rope64 KV bf16), K bytes only",
                    batch=b,
                    cores=cores,
                    kv_dtype=kvd,
                    position=s_,
                    kv_rows=s_,
                    kv_MB=kvb / 1e6,
                    dur_ns=" / ".join(f"{x:,.0f}" for x in g),
                    median_ns=med,
                    us=med / 1000,
                    GBps=kvb / med,
                )
            )
        continue
    if "mla" in tag:
        b = int(env.get("MLAD_B", 4))
        d_qk = 576
        kvd = "bfp8_b"
        positions = [int(x) for x in env.get("MLAD_POS", "1024,4096,8192").split(",")]
        iters = 3
        for pi, pos in enumerate(positions):
            g = ns[pi * iters : (pi + 1) * iters]
            if len(g) < iters:
                continue
            med = pd.Series(g[1:]).median()
            kv_len = ((pos + 1 + 63) // 64) * 64
            kvb = b * kv_len * d_qk * BYTES[kvd]  # K only (reuse_k)
            rows.append(
                dict(
                    run=tag,
                    kind="MLA decode paged, nh128 kv_lora512 d_rope64, K only (reuse_k), 64-core Q shard",
                    batch=b,
                    cores=cores,
                    kv_dtype=kvd,
                    position=pos,
                    kv_rows=kv_len,
                    kv_MB=kvb / 1e6,
                    dur_ns=" / ".join(f"{x:,.0f}" for x in g),
                    median_ns=med,
                    us=med / 1000,
                    GBps=kvb / med,
                )
            )
    elif "nonpaged" in tag:
        b = int(env["DEC_B"])
        kvd = env["DEC_KV_DTYPE"]
        iters = int(env["DEC_ITERS"])
        seqs = [int(x) for x in env["DEC_POS"].split(",")]
        for pi, s_ in enumerate(seqs):
            g = ns[pi * iters : (pi + 1) * iters]
            if len(g) < iters:
                continue
            med = pd.Series(g[1:]).median()
            kvb = b * 8 * s_ * 128 * 2 * BYTES[kvd]
            rows.append(
                dict(
                    run=tag,
                    kind="non-paged sdpa_decode (decode_latency_sweep form), full cache attended",
                    batch=b,
                    cores=cores,
                    kv_dtype=kvd,
                    position=s_,
                    kv_rows=s_,
                    kv_MB=kvb / 1e6,
                    dur_ns=" / ".join(f"{x:,.0f}" for x in g),
                    median_ns=med,
                    us=med / 1000,
                    GBps=kvb / med,
                )
            )
    else:
        b = int(env["b"])
        kvd = env["kv_dtype"]
        pos = int(env["pos"])
        med = pd.Series(ns[1:]).median()
        kv_len = ((pos + 1 + 31) // 32) * 32
        kvb = b * 8 * kv_len * 128 * 2 * BYTES[kvd]
        rows.append(
            dict(
                run=tag,
                kind="paged sdpa_decode (tt_transformers form)",
                batch=b,
                cores=cores,
                kv_dtype=kvd,
                position=pos,
                kv_rows=kv_len,
                kv_MB=kvb / 1e6,
                dur_ns=" / ".join(f"{x:,.0f}" for x in ns),
                median_ns=med,
                us=med / 1000,
                GBps=kvb / med,
            )
        )
d = pd.DataFrame(rows)
pd.set_option("display.width", 260)
print(d.to_string(index=False))
d.to_csv("r1b_decode_table.csv", index=False)
