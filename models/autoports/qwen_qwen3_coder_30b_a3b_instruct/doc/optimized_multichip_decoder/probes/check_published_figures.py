# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert every headline figure in this stage's docs against its artifact.

The defect that has cost this model a review in all three previous stages is a
number in prose that no artifact produces. ``window.py`` makes the profile
windows re-derivable; this makes the *published* figures checkable, in one
command, with no device:

    python check_published_figures.py

Exits non-zero on mismatch. The published values are duplicated here on purpose,
because a checker that reads the document it is checking checks nothing.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
K = "DEVICE KERNEL DURATION [ns]"
bad = []


def chk(name, published, artifact):
    same = abs(published - artifact) < 5e-4
    print(f"{'OK  ' if same else 'BAD '}{name:38s} published={published} artifact={artifact}")
    if not same:
        bad.append(name)


def dev0(path):
    rs = [r for r in csv.DictReader(gzip.open(path, "rt")) if r["DEVICE ID"] == "0"]
    rs.sort(key=lambda r: int(r["HOST START TS"]))
    return rs


rows = dev0(D / "ops_perf_optimized_multichip_decode.csv.gz")
us = lambda idxs: sum(int(rows[i][K]) for i in idxs) / 1000

# --- the decode profile, device 0, rows 154-221 (see window.py) ---------------
chk("decode layer total", 362.828, us(range(154, 222)))
chk("  input_layernorm", 6.663, us([154, 155]))
chk("  attention", 60.400, us(range(156, 176)))
chk("  all-reduce after wo", 33.063, us([176, 177, 178]))
chk("  residual add (attn)", 1.969, us([179]))
chk("  post_attention_layernorm", 6.663, us([180, 181]))
chk("  router block", 71.412, us(list(range(182, 202)) + [203]))
chk("  normed shard->interleaved", 0.876, us([202]))
chk("  expert sparse_matmul pair", 82.718, us([205, 213]))
chk("  expert reshape/eltwise tail", 69.573, us([204] + list(range(206, 213)) + [214, 215, 216, 217]))
chk("  all-reduce after experts", 27.581, us([218, 219, 220]))
chk("  residual add (moe)", 1.910, us([221]))
chk("collectives", 60.644, us([176, 177, 178, 218, 219, 220]))
chk("replicated norms", 13.326, us([154, 155, 180, 181]))
chk("TopK", 26.356, us([184]))
chk("FillPad before TopK", 4.190, us([183]))
chk("router projection", 6.241, us([182]))
chk("wo projection", 8.228, us([174]))
chk("threshold tail would remove", 17.007, us(range(190, 198)))
chk("expert M-padding reshapes", 41.313, us([206, 211, 214]))

# --- the standalone sweeps, from the probe logs they were printed by ----------
# The router projection is published at its **8-core** figure because the norm
# shards over 8. The 4-core leg of the same sweep is faster standalone and is
# not what ships; three documents once published it as though it were, which is
# why both legs are asserted here and named for what they are.
nrp = (D / "probes" / "norm_router_probe.log").read_text()
for want, what in (
    ("router matmul in0 L1 wsh  8 cores        5.85 us", "router mm, 8 cores (SHIPPED)"),
    ("router matmul in0 L1 wsh  4 cores        4.30 us", "router mm, 4 cores (not shipped)"),
    ("router matmul interleaved (shipped)      24.62 us", "router mm, interleaved (before)"),
):
    ok = want in nrp
    print(f"{'OK  ' if ok else 'BAD '}{what:38s} log={want.split()[-2]} us")
    if not ok:
        bad.append(what)

# The rotary lever: measured, faster, and rejected on a cache-convention
# conflict rather than on speed. Both halves are asserted because the *reason*
# is the finding -- a later reader who sees only "3.05x" would re-adopt it.
rp = (D / "probes" / "rope_probe.log").read_text()
rl = (D / "probes" / "rope_layer_probe.log").read_text()
for want, where, what in (
    ("rotary_embedding (shipped, HF order, DRAM)     3.84 us", rp, "rotary, shipped HF op"),
    ("rotary_embedding_llama (decode, Meta, L1)     1.26 us", rp, "rotary, llama op"),
    ("PCC 1.0000000", rp, "rotary, llama == HF standalone"),
    ("attention out: max|diff| 1.221e-04  PCC 0.9999697", rl, "rotary, fresh KV cache"),
    ("attention out, primed cache: max|diff| 8.911e-02  PCC 0.1932974", rl, "rotary, primed KV cache"),
):
    ok = want in where
    print(f"{'OK  ' if ok else 'BAD '}{what:38s} log={want[-24:].strip()}")
    if not ok:
        bad.append(what)

# The ethernet-link lever, re-measured after review found the probe could not
# tell its own legs apart. Asserted as the *order control* it now is: the mean
# of each configuration, and that each reads the same at both leg positions.
lp = (D / "probes" / "links_probe.log").read_text()
links = {2: [], 1: []}
for line in lp.splitlines():
    if line.startswith("P|pass") and "num_links=" in line and "FAILED" not in line:
        n = 1 if "num_links=1" in line else 2
        t = line.split()
        links[n].append(float(t[t.index("ms") - 1]))
for n, want in ((2, 0.43400), (1, 0.42875)):
    got = sum(links[n]) / len(links[n]) if links[n] else 0.0
    chk(f"decode num_links={n} mean", want, round(got, 5))
chk("decode 1-link gain", 1.0122, round((sum(links[2]) / 6) / (sum(links[1]) / 6), 4))
ok = len(links[1]) == 6 and len(links[2]) == 6
print(f"{'OK  ' if ok else 'BAD '}{'links probe ran 6 passes each':38s} {len(links[2])} vs {len(links[1])}")
if not ok:
    bad.append("links probe pass count")

# --- the CSVs ----------------------------------------------------------------
read = lambda f: {int(r[list(r)[0]]): r for r in csv.DictReader((D / f).open())}
dec, pre = read("perf_decode.csv"), read("perf_prefill.csv")
b_dec, b_pre = read("perf_baseline_1x1_decode.csv"), read("perf_baseline_1x1_prefill.csv")
chk("traced decode ctx128", 0.4286, float(dec[128]["median_ms"]))
chk("traced decode ctx1k", 0.5254, float(dec[1024]["median_ms"]))
chk("traced decode ctx4k", 0.8667, float(dec[4096]["median_ms"]))
chk("prefill S=128 us/tok", 25.13, float(pre[128]["us_per_token"]))
chk("prefill S=2048 us/tok", 18.02, float(pre[2048]["us_per_token"]))
chk("1x1 decode ctx128", 0.5638, float(b_dec[128]["median_ms"]))
chk("1x1 prefill S=2048 us/tok", 69.28, float(b_pre[2048]["us_per_token"]))

j = json.loads((D / "perf_summary.json").read_text())
pick = lambda rs, key, val, field: [r for r in rs if r[key] == val][0][field]
chk("decode ctx128 vs 1x1", 1.315, pick(j["decode"], "context_len", 128, "speedup"))
chk("decode ctx128 vs stage 03", 1.112, round(pick(j["stage04_vs_stage03_decode"], "context_len", 128, "speedup"), 3))
chk("prefill S=2048 vs 1x1", 3.845, pick(j["prefill"], "seq_len", 2048, "speedup"))

# --- stage 03, the frozen before ---------------------------------------------
s3 = dev0(D.parent / "multichip_decoder" / "ops_perf_multichip_decode.csv.gz")
chk("stage-03 decode layer", 414.661, sum(int(s3[i][K]) for i in range(134, 198)) / 1000)
chk("stage-03 -> 04 device time", 1.143, round(414.661 / 362.828, 3))
s3_dec = read("../multichip_decoder/perf_decode.csv")
chk("stage-03 traced decode ctx128", 0.4767, float(s3_dec[128]["median_ms"]))

# --- the forward pointer stage 04 added to stage 03's README ------------------
# That paragraph restates stage-04 figures inside a frozen document, which is
# exactly where a figure drifts unnoticed: nothing else in that directory is
# ever regenerated, so nothing else re-derives it. Three of its numbers were
# wrong on review. Assert the text itself.
fwd = (D.parent / "multichip_decoder" / "README.md").read_text()
for want, what in (
    ("362.828 µs of device time", "stage-04 layer device time"),
    ("414.661 recorded here", "stage-03 layer device time"),
    ("ctx 128 is 0.4286 ms", "stage-04 traced decode ctx128"),
    ("directory's 0.4767", "stage-03 traced decode ctx128"),
):
    ok = want in fwd
    print(f"{'OK  ' if ok else 'BAD '}{'fwd ptr: ' + what:38s} text={want!r}")
    if not ok:
        bad.append("forward pointer: " + what)

print()
if bad:
    print(f"{len(bad)} MISMATCH: " + ", ".join(bad))
    sys.exit(1)
print("all published figures reconcile to their artifacts")
