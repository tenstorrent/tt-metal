#!/usr/bin/env bash
# Gate for stage 02b (advisor-challenger).
#
# CRITICAL checks are things that make the result wrong or unattributable: the control, the measurement
# rule, provenance, batch/order. ADVISORY checks are completeness bookkeeping -- they print WARN and do
# not fail, because a run that found a real win must never be blocked by a missing label. Set
# CHALLENGER_STRICT=1 to promote advisories to failures.
#
# Usage: 02b-advisor-challenger.check.sh [<model_dir>]   (or MODEL_DIR=..., or HF_MODEL=...)
set -uo pipefail

MD=${1:-${MODEL_DIR:-}}
if [ -z "$MD" ] && [ -n "${HF_MODEL:-}" ]; then MD=$(echo "$HF_MODEL" | tr 'A-Z/.-' 'a-z___'); fi
if [ -z "$MD" ]; then
  echo "CRITICAL: no model dir. Pass \$1, or set MODEL_DIR=models/autoports/<md>, or HF_MODEL." >&2
  exit 1
fi
MD=${MD#models/autoports/}; MD=${MD%/}
ROOT=${TT_METAL_HOME:-$(pwd)}
D="$ROOT/models/autoports/$MD/doc/advisor_challenger"
fail=0; warn=0
err()  { echo "CRITICAL: $*" >&2; fail=1; }
adv()  { if [ "${CHALLENGER_STRICT:-0}" = 1 ]; then echo "CRITICAL: $*" >&2; fail=1
         else echo "WARN: $*" >&2; warn=$((warn+1)); fi }
ok()   { echo "  ok: $*"; }

command -v python3 >/dev/null 2>&1 || { echo "CRITICAL: python3 required" >&2; exit 1; }
[ -d "$D" ] || { echo "CRITICAL: $D does not exist -- stage produced nothing" >&2; exit 1; }
export CH_D="$D" CH_STRICT="${CHALLENGER_STRICT:-0}" CH_WANT="${CHALLENGER_DECODE_BATCH:-}"

# ---- 1. the control: frozen incumbent, median of a fixed n, policy from what executed -------------
INC="$D/incumbent.json"
if [ ! -s "$INC" ]; then
  err "no incumbent.json. Without a frozen control the delta is not a measurement."
else
  python3 - <<'PY' || fail=1
import json, os, statistics, sys
d = json.load(open(os.environ["CH_D"] + "/incumbent.json")); bad = []
r = d.get("repeats_ms")
if not isinstance(r, list) or len(r) < 5:
    bad.append("repeats_ms must have >=5 entries. The decision rule is non-overlap, whose "
               "false-positive rate is 1/C(2n,n): 5% at n=3 against 0.40% at n=5.")
m = d.get("incumbent_ms")
if not isinstance(m, (int, float)):
    bad.append("incumbent_ms missing")
elif isinstance(r, list) and len(r) >= 2 and abs(m - statistics.median(r)) > 1e-9:
    bad.append(f"incumbent_ms {m} must be median(repeats_ms) = {statistics.median(r)}. min-of-n is "
               "biased low by an amount growing with n, so cells with different n are not comparable.")
src = (d.get("shipped_policy_source") or "").lower()
if not src:
    bad.append("shipped_policy_source missing -- it must name the artifact the policy came from")
elif "constructor_default" in src:
    bad.append("shipped_policy_source cites constructor_defaults: those are the CLASS's default "
               "arguments, not the run's effective config. Source it from what EXECUTED.")
if not d.get("shipped_policy"):
    bad.append("shipped_policy missing")
for b in bad: print(f"CRITICAL: incumbent.json: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  [ "$fail" = 0 ] && ok "incumbent.json: n>=5, incumbent_ms = median, policy sourced from execution"
fi

# ---- 2. captures: one per layer kind, dtypes match shipped, batch and order agree -----------------
shopt -s nullglob
caps=("$D"/shard_advise/*/report.json)
[ ${#caps[@]} -eq 0 ] && err "no shard_advise/<layer_kind>/report.json -- one capture per LAYER KIND"
for rj in "${caps[@]}"; do
  kind=$(basename "$(dirname "$rj")"); export CH_RJ="$rj" CH_KIND="$kind"
  python3 -c "import json,os,sys; json.load(open(os.environ['CH_RJ']))" 2>/dev/null \
    || err "$kind: report.json does not parse"
  python3 - <<'PY' || fail=1
import json, os, re, sys
r = json.load(open(os.environ["CH_RJ"])); kind = os.environ["CH_KIND"]
try: i = json.load(open(os.environ["CH_D"] + "/incumbent.json"))
except Exception: i = {}
bad = []
traced, shipped = r.get("traced_weight_dtypes"), i.get("shipped_weight_dtypes")
if not traced: bad.append("report.json records no traced_weight_dtypes")
if not shipped: bad.append("incumbent.json records no shipped_weight_dtypes")
def norm(x):
    s = re.sub(r"[^a-z0-9]", "", str(x).lower()).replace("datatype", "")
    m = re.search(r"(\d+)", s); bits = m.group(1) if m else "?"
    return ("bfp" if ("bfp" in s or (s.endswith("b") and "bfloat" in s)) else "bf") + bits
if isinstance(traced, dict) and isinstance(shipped, dict):
    for role, t in traced.items():
        s2 = shipped.get(role)
        if s2 is not None and norm(t) != norm(s2):
            bad.append(f"WRONG-PRECISION CAPTURE: {role}: traced {t}, ships {s2}. Construct the "
                       "decoder with the SHIPPED POLICY, not class defaults.")
cb, ib, rb = r.get("capture_batch"), i.get("decode_batch"), i.get("requested_decode_batch")
for nm, v in (("report.json capture_batch", cb), ("incumbent.json decode_batch", ib),
              ("incumbent.json requested_decode_batch", rb)):
    if v is None: bad.append(f"{nm} missing")
if None not in (cb, ib) and int(cb) != int(ib):
    bad.append(f"capture_batch {cb} != decode_batch {ib}: advice judged at a batch it was not "
               "captured at can flip sign (+12.3% at b1 vs -8.8% at b32 on identical advice)")
if None not in (ib, rb) and int(ib) != int(rb): bad.append(f"decode_batch {ib} != requested {rb}")
want = os.environ.get("CH_WANT") or ""
if want and rb is not None and int(want) != int(rb):
    bad.append(f"orchestrator asked for DECODE_BATCH {want}, stage recorded {rb}")
ma, ca = i.get("measured_at"), r.get("captured_at")
if not ma: bad.append("incumbent.json records no measured_at")
if not ca: bad.append("report.json records no captured_at")
if ma and ca and str(ca) < str(ma):
    bad.append(f"captured_at {ca} PRECEDES measured_at {ma}: the control is contaminated")
for b in bad: print(f"CRITICAL: {kind}: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
done
[ "$fail" = 0 ] && ok "captures: per layer kind, dtypes match shipped, batch and order agree"

# ---- 3. reconciliation: tool-generated, closes at 100%, every chain resolved ----------------------
recs=("$D"/reconciliation*.json)
if [ ${#recs[@]} -eq 0 ]; then
  err "no reconciliation*.json -- no record of what was advised vs shipped"
else
  for rc in "${recs[@]}"; do
    export CH_RC="$rc"
    python3 - <<'PY' || fail=1
import json, os, sys
d = json.load(open(os.environ["CH_RC"])); nm = os.path.basename(os.environ["CH_RC"])
strict = os.environ.get("CH_STRICT") == "1"
crit, warn = [], []
if d.get("generated_by") != "advisor-challenger/scripts/reconcile.py":
    crit.append("not generated by scripts/reconcile.py. A hand-authored reconciliation is "
                "indistinguishable from a generated one; run the script and fix its input if it aborts.")
if d.get("accounting_closes_100pct") is not True:
    crit.append("accounting does not close to 100% of the measured window")
f = d.get("feasibility") or {}
fv = f.get("verdict")
if fv is None or fv == "unknown":
    crit.append("no harness noise floor: rerun reconcile.py with --incumbent incumbent.json (needs >=2 "
                "repeats_ms). Without it, a contribution of zero cannot be told apart from a cell where "
                "nothing was ever measurable.")
kept = [c for c in d.get("chains", []) if (c.get("verdict") or "").lower() == "kept"]
if fv == "not_measurable" and kept:
    crit.append(f"ceiling {f.get('ceiling_us')}us is {f.get('ceiling_vs_floor')}x the "
                f"{f.get('noise_floor_us')}us noise floor, yet {len(kept)} chain(s) are kept. A win below "
                "the floor is not attributable to the advice -- tighten the harness or report zero.")
if fv == "aggregate_only" and not any(len(c.get("ops") or []) and c.get("combined_with") for c in d.get("chains", [])):
    warn.append(f"no single chain clears the {f.get('noise_floor_us')}us floor, so chains screened alone "
                "return zero regardless of the advice. Apply the top chains together as one candidate and "
                "record it with combined_with.")
if d.get("confidence", {}).get("degraded"):
    crit.append("reconcile.py reported DEGRADED: " + "; ".join(d["confidence"].get("degraded_because", []))
                + ". The buckets and ranking are unsafe -- resolve the input before screening.")
for c in d.get("chains", []):
    v = (c.get("verdict") or "").lower()
    if v not in ("kept", "rejected", "below_threshold", "not_measurable"):
        crit.append(f"chain {c.get('chain')}: verdict {v!r} unresolved -- measure it or mark "
                    "below_threshold with its conversion value")
    elif v not in ("below_threshold", "not_measurable") and not isinstance(c.get("measured_ms"), (int, float)):
        crit.append(f"chain {c.get('chain')}: {v} with no measured_ms. A prose rejection is not a result.")
    if v in ("kept", "rejected") and not c.get("repeats_ms"):
        warn.append(f"chain {c.get('chain')}: no repeats_ms, so non-overlap cannot be rechecked")
for r in d.get("material_ops_on_le_2_cores", []):
    if not (r.get("measured_ms") or r.get("hard_error")):
        warn.append(f"{r.get('device')} runs on {r.get('shipped_cores')} core(s) at "
                    f"{r.get('share_pct')}% -- needs a measured attempt or a quoted hard error")
for r in d.get("disagreements", []):
    if r.get("bucket") == "dram_resident" and not r.get("verdict"):
        warn.append(f"{r.get('device')} dram_resident at {r.get('share_pct')}% has no verdict -- "
                    "'leave it in DRAM' is advice and de-sharding has won here")
for b in crit: print(f"CRITICAL: {nm}: {b}", file=sys.stderr)
for b in warn: print(f"{'CRITICAL' if strict else 'WARN'}: {nm}: {b}", file=sys.stderr)
sys.exit(1 if crit or (warn and strict) else 0)
PY
    [ $? -ne 0 ] && [ "${CHALLENGER_STRICT:-0}" != 1 ] && warn=$((warn+1))
  done
  [ "$fail" = 0 ] && ok "reconciliation: tool-generated, closes at 100%, chains resolved"
fi

# ---- 4. the result: non-overlap, oracle, outcome stated ------------------------------------------
FIN="$D/final.json"
if [ ! -s "$FIN" ]; then
  err "no final.json -- the contribution is unrecorded"
else
  python3 - <<'PY' || fail=1
import json, os, sys
f = json.load(open(os.environ["CH_D"] + "/final.json"))
try: i = json.load(open(os.environ["CH_D"] + "/incumbent.json"))
except Exception: i = {}
strict = os.environ.get("CH_STRICT") == "1"
crit, warn = [], []
fm = f.get("final_ms"); im = f.get("incumbent_ms", i.get("incumbent_ms"))
if not isinstance(fm, (int, float)): crit.append("final_ms missing")
if not isinstance(im, (int, float)): crit.append("incumbent_ms missing")
if not f.get("outcome"): crit.append("outcome missing -- state it (no_change / improved)")
if f.get("changed") is None: crit.append("`changed` missing -- a no-change result must be explicit")
if not f.get("oracle"): crit.append("oracle missing -- name the correctness oracle the result passed")
if f.get("oracle_passed") is not True:
    crit.append("oracle_passed is not true -- a faster decoder that fails its oracle is a regression")
if isinstance(fm, (int, float)) and isinstance(im, (int, float)) and fm > im:
    crit.append(f"final_ms {fm} > incumbent_ms {im}: this stage may not ship a slower decoder")
if f.get("changed"):
    sets = (f.get("combination") or {}).get("measured_sets")
    if not isinstance(sets, list) or not sets:
        crit.append("a change shipped with no combination.measured_sets")
    else:
        for n, s in enumerate(sets):
            if not isinstance(s.get("measured_ms"), (int, float)):
                crit.append(f"measured_sets[{n}]: no measured_ms -- best_set must be MEASURED")
            if not s.get("chains"):
                crit.append(f"measured_sets[{n}]: no `chains` -- the number is unattributable")
            if not s.get("repeats_ms"):
                warn.append(f"measured_sets[{n}]: no repeats_ms, so non-overlap cannot be checked")
        best = min((s["measured_ms"] for s in sets if isinstance(s.get("measured_ms"), (int, float))),
                   default=None)
        if best is not None and isinstance(fm, (int, float)) and fm > best:
            crit.append(f"final_ms {fm} is worse than the best measured set {best}")
    # non-overlap: every winning repeat must beat every incumbent repeat
    wr = f.get("winning_repeats_ms"); ir = i.get("repeats_ms")
    if not wr:
        warn.append("winning_repeats_ms missing -- non-overlap is the ship rule and is unverifiable")
    elif isinstance(ir, list) and ir and max(wr) >= min(ir):
        crit.append(f"NON-OVERLAP FAILED: slowest winning repeat {max(wr)} >= fastest incumbent "
                    f"repeat {min(ir)}. Overlapping distributions do not establish a win.")
    if not f.get("confirmed_fresh_process"):
        warn.append("confirmed_fresh_process not set -- cross-process variance is otherwise unmeasured")
it = f.get("iterations")
if isinstance(it, list):
    for n, e in enumerate(it):
        if not e.get("trigger"): warn.append(f"iterations[{n}]: no trigger recorded")
        if n > 0 and not e.get("reranked_from"):
            warn.append(f"iterations[{n}]: no reranked_from -- re-profile before re-ranking")
    if len(it) > 3: crit.append(f"{len(it)} iterations exceeds the cap of 3")
for b in crit: print(f"CRITICAL: final.json: {b}", file=sys.stderr)
for b in warn: print(f"{'CRITICAL' if strict else 'WARN'}: final.json: {b}", file=sys.stderr)
sys.exit(1 if crit or (warn and strict) else 0)
PY
  [ $? -ne 0 ] && [ "${CHALLENGER_STRICT:-0}" != 1 ] && warn=$((warn+1))
  [ "$fail" = 0 ] && ok "final.json: non-overlap holds, oracle passed, outcome stated"
fi

if [ "$fail" != 0 ]; then
  echo "02b-advisor-challenger gate FAILED for $MD" >&2; exit 1
fi
[ "$warn" != 0 ] && echo "02b-advisor-challenger gate PASSED for $MD with $warn advisory warning(s)"
[ "$warn" = 0 ] && echo "02b-advisor-challenger gate PASSED for $MD"
exit 0
