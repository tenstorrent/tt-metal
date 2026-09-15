"""Single-prefill LATENCY (TTFT) for one request through the PP pipeline.

Why this is a separate script from analyze_pp.py / analyze_longctx2.py: those measure THROUGHPUT from
chunk-to-chunk intervals on the last rank, which is a steady-state rate and says nothing about how long
one prefill takes. Latency is rank 0 starting the request's FIRST chunk to the last rank finishing its
LAST chunk -- it includes pipeline fill and drain, which the interval metrics deliberately exclude.

Two things make a throughput run's numbers NOT a TTFT:
  * more than one request in flight lets request N+1's fill hide inside request N's drain, so
    tokens/wall overstates a single prefill. This script refuses to report a latency in that case.
  * PREFILL_KV_ONLY_LAST_LAYER=1 (the runner's DEFAULT) runs the final layer kv-only and skips the
    final norm + LM head, so no first token is produced at all. This script checks the runner banner
    and labels the result accordingly.
"""
import re
import sys

log = sys.argv[1]
txt = open(log, errors="replace").read()

cs = [
    (int(r), int(c), float(t), int(st), int(en))
    for r, c, t, s, st, en in re.findall(
        r"\[pp rank (\d)\] CHUNK_START c=(\d+) compute_start=([0-9.]+) slot=(-?\d+) \[(\d+),(\d+)\)", txt
    )
]
e2e = {
    int(r): (float(a), float(b))
    for r, a, b in re.findall(
        r"\[pp rank (\d)\] E2E_CLOCK first_compute_start=([0-9.]+) last_compute_end=([0-9.]+)", txt
    )
}
if not cs or not e2e:
    raise SystemExit("no CHUNK_START / E2E_CLOCK records — is this a runner log from a completed run?")

last = max(r for r, *_ in cs)
kv_only = re.search(r"PREFILL_KV_ONLY_LAST_LAYER *= *(\w+)", txt)
kv_only = (kv_only.group(1).lower() == "true") if kv_only else None
emits_token = kv_only is False

r0 = [(c, t, en) for r, c, t, st, en in cs if r == 0 and st == 0]
n_req = len(r0)
chunks_r0 = sum(1 for r, *_ in cs if r == 0)
context = max(en for *_, en in cs)

print(f"log        : {log}")
print(f"ranks      : 0..{last}   chunks/rank: {chunks_r0}   requests: {n_req}   context: {context}")
print(
    f"last layer : {'FULL (norm + LM head -> a first token IS produced)' if emits_token else 'kv-only (no norm/LM head -> NO token produced)'}"
)

if n_req != 1:
    print(
        f"\nREFUSING to report a TTFT: this run had {n_req} requests, so request N+1's pipeline fill\n"
        f"overlaps request N's drain and tokens/wall is a stream throughput, not one prefill's latency.\n"
        f"Re-run with PP_REQUESTS=1 (and PP_KV_ONLY_LAST_LAYER=0 for a true TTFT)."
    )
    raise SystemExit(1)

lat = e2e[last][1] - e2e[0][0]
label = "TTFT" if emits_token else "prefill-completion latency (NOT TTFT — no token emitted)"
print(f"\n{label}")
print(f"  context {context} tokens in {lat:.3f}s  ->  {context/lat:,.0f} tok/s")

ts = sorted(t for r, c, t, st, en in cs if r == last)
if len(ts) > 1:
    d = [b - a for a, b in zip(ts, ts[1:])]
    fill = ts[0] - e2e[0][0]
    print(
        f"  pipeline fill (rank0 start -> rank{last} first chunk): {fill*1000:.1f}ms ({fill/lat*100:.1f}% of latency)"
    )
    print(f"  per-chunk interval on rank{last}: first {d[0]*1000:.1f}ms  last {d[-1]*1000:.1f}ms")
