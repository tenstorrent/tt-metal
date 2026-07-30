#!/usr/bin/env bash
# Gate for stage 02b (advisor-challenger).
#
# Why this is strict: every failure in SHARD-ADVISOR-FINDINGS.md was a behavioural instruction that was
# satisfiable cheaply because nothing checked it. A prose rejection passed for a result; a capture built
# with class defaults passed for a capture at shipped precision; `dram_sharded_considered = 0` passed for
# "this model cannot use DS". The gate is the only thing that binds, so each check below corresponds to a
# specific documented failure, named in its message.
#
# Usage: 02b-advisor-challenger.check.sh <model_dir>
set -uo pipefail

MD=${1:?model_dir}
# The stage prompt's MODEL_DIR placeholder expands to `models/autoports/<md>`, while callers on the
# command line pass the bare `<md>`. Accept both rather than silently building
# models/autoports/models/autoports/<md> and then reporting "stage produced nothing".
MD=${MD#models/autoports/}
MD=${MD%/}
ROOT=${TT_METAL_HOME:-$(pwd)}
D="$ROOT/models/autoports/$MD/doc/advisor_challenger"
fail=0
err() { echo "CRITICAL: $*" >&2; fail=1; }
ok()  { echo "  ok: $*"; }

command -v python3 >/dev/null 2>&1 || { echo "CRITICAL: python3 required" >&2; exit 1; }

[ -d "$D" ] || { echo "CRITICAL: $D does not exist -- stage produced nothing" >&2; exit 1; }

# ---- 1. frozen incumbent, measured BEFORE any advisor artifact -------------------------------
INC="$D/incumbent.json"
if [ ! -s "$INC" ]; then
  err "no incumbent.json. The invariant final_ms <= incumbent_ms is unverifiable without a frozen
    baseline, and a baseline measured after the advisor ran is not frozen."
else
  python3 - "$INC" <<'PY' || fail=1
import json, re, sys
d = json.load(open(sys.argv[1]))
bad = []
r = d.get("repeats_ms")
if not isinstance(r, list) or len(r) < 3:
    bad.append("repeats_ms must have >=3 entries: the noise floor is what makes a tie decidable, and "
               "observed same-config spreads in this corpus were 0.2-31 us")
if not isinstance(d.get("incumbent_ms"), (int, float)):
    bad.append("incumbent_ms missing")
elif isinstance(r, list) and r and abs(d["incumbent_ms"] - min(r)) > 1e-9:
    bad.append(f"incumbent_ms {d['incumbent_ms']} must be min(repeats_ms) = {min(r)}: a challenger must "
               "beat the incumbent's BEST repeat, else you ratchet on noise")
if not isinstance(d.get("noise_floor_ms"), (int, float)):
    bad.append("noise_floor_ms missing")
src = (d.get("shipped_policy_source") or "").lower()
if not src:
    bad.append("shipped_policy_source missing -- it must name the artifact the policy came from")
elif "constructor_default" in src:
    bad.append("shipped_policy_source cites resolved_policy.constructor_defaults. Those are the CLASS'S "
               "DEFAULT ARGUMENTS, not the run's effective config -- gemma's defaults print "
               "dense_decode_dram_sharded=False on a cell whose perf CSV shows DRAM Sharded = True. "
               "Source the policy from what EXECUTED: the final tt-perf-report CSV or the selected "
               "candidate JSON.")
if not d.get("shipped_policy"):
    bad.append("shipped_policy missing")
for b in bad: print(f"CRITICAL: incumbent.json: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  [ "$fail" = 0 ] && ok "incumbent.json: >=3 repeats, incumbent_ms = best repeat, policy sourced from execution"
fi

# ---- 2. one capture per layer kind, parsing, with a matmul ------------------------------------
shopt -s nullglob
caps=("$D"/shard_advise/*/report.json)
if [ ${#caps[@]} -eq 0 ]; then
  err "no shard_advise/<layer_kind>/report.json. One capture per LAYER KIND is required: a single
    capture lets the whole search follow it to one kind -- qwen's arm ran variants on full_attention
    only while linear_attention, carrying 48 of 64 layers, got nothing but a default measurement."
fi
for rj in "${caps[@]}"; do
  kind=$(basename "$(dirname "$rj")")
  ir="$(dirname "$rj")/final_ir.mlir"
  [ -s "$rj" ] || err "$kind: report.json empty"
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$rj" 2>/dev/null \
    || err "$kind: report.json does not parse as JSON"
  [ -s "$ir" ] || err "$kind: final_ir.mlir missing. It is AUTHORITATIVE for program configs, required
    input layouts and the advisor's own reverts; report.json omits block widths and per_core_N."
  grep -q matmul "$ir" 2>/dev/null || err "$kind: final_ir.mlir contains no matmul -- capture is empty"

  # ---- 3. traced dtypes must equal the SHIPPED dtypes ----------------------------------------
  # north traced bf16 attention and shipped bfp8, so two matmuls were excluded for a dtype the model
  # never used. Cause: from_state_dict() called with no policy argument (3 of 4 scripts do this).
  python3 - "$rj" "$INC" "$kind" <<'PY' || fail=1
import json, re, sys
rep, inc, kind = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = json.load(open(rep)); i = json.load(open(inc))
except Exception as e:
    print(f"CRITICAL: {kind}: cannot read artifacts: {e}", file=sys.stderr); sys.exit(1)
traced = r.get("traced_weight_dtypes") or r.get("weight_dtypes")
if not traced:
    print(f"CRITICAL: {kind}: report.json records no traced_weight_dtypes. Without it, nobody can tell "
          "whether the capture saw the precision the model ships -- the exact defect that made north's "
          "capture exclude DS for a dtype it never used.", file=sys.stderr); sys.exit(1)
shipped = i.get("shipped_weight_dtypes")
if not shipped:
    print(f"CRITICAL: {kind}: incumbent.json records no shipped_weight_dtypes to compare against.",
          file=sys.stderr); sys.exit(1)
def norm(x):
    # The two sides spell the same dtype differently: the advisor reports `bfp_bf8`, tt-metal and the
    # candidate JSONs say `BFLOAT8_B` / `DataType.BFLOAT8_B`, and cell prose says `bfp8`. Normalise to
    # family+width. A naive chain of .replace() calls does NOT work here -- stripping "_b" out of
    # "bfp_bf8" leaves "bfpf8", which silently fails to match "bfp8" and reports a wrong-precision
    # capture on a perfectly good one.
    s = re.sub(r"[^a-z0-9]", "", str(x).lower()).replace("datatype", "")
    m = re.search(r"(\d+)", s)
    bits = m.group(1) if m else "?"
    is_block = ("bfp" in s) or (s.endswith("b") and "bfloat" in s)   # trailing _b == block float
    return ("bfp" if is_block else "bf") + bits
mism = []
if isinstance(traced, dict) and isinstance(shipped, dict):
    for role, t in traced.items():
        s = shipped.get(role)
        if s is not None and norm(t) != norm(s):
            mism.append(f"{role}: capture traced {t}, model ships {s}")
else:
    ts, ss = sorted(map(norm, traced if isinstance(traced, list) else [traced])), \
             sorted(map(norm, shipped if isinstance(shipped, list) else [shipped]))
    if ts != ss: mism.append(f"traced {ts} vs shipped {ss}")
for m in mism:
    print(f"CRITICAL: {kind}: WRONG-PRECISION CAPTURE -- {m}. Construct the decoder with the SHIPPED "
          "POLICY (advise_qwen.py is the correct template); do not call from_state_dict() with no "
          "dtype/policy argument.", file=sys.stderr)
sys.exit(1 if mism else 0)
PY

  # ---- 4. dram_sharded_considered == 0 must be classified ------------------------------------
  python3 - "$rj" "$kind" <<'PY' || fail=1
import json, re, sys
r = json.load(open(sys.argv[1])); kind = sys.argv[2]
c = r.get("dram_sharded_considered")
if c is None:
    print(f"CRITICAL: {kind}: dram_sharded_considered absent from report.json.", file=sys.stderr); sys.exit(1)
if c == 0:
    why = (r.get("dram_sharded_zero_cause") or "").lower()
    allowed = ("wrong_precision_capture", "bf16_eligibility_default")
    if why not in allowed:
        print(f"CRITICAL: {kind}: dram_sharded_considered == 0 but dram_sharded_zero_cause is "
              f"{why!r}; must be one of {allowed}. A 0 has TWO causes and they need different fixes: "
              "(a) a wrong-precision capture, or (b) the bf16-DS-off-by-default eligibility gate "
              "(bf16 DS runs at PCC 1.0000 -- policy, not capability; fix with "
              "--pipeline-options allow-bf16-dram-sharded-matmul=true). OPT-015 names only (a), which "
              "sends you hunting a capture bug that may not exist.", file=sys.stderr)
        sys.exit(1)
    if why == "bf16_eligibility_default" and not r.get("allow_bf16_dram_sharded_matmul"):
        print(f"CRITICAL: {kind}: cause is the bf16 eligibility default, so the capture must be re-run "
              "with --pipeline-options allow-bf16-dram-sharded-matmul=true before its 0 means anything.",
              file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PY
done
[ "$fail" = 0 ] && ok "captures: per layer kind, parse, contain matmul, dtypes match shipped, DS-zero classified"

# ---- 5. every disagreement carries a number ---------------------------------------------------
REC="$D/reconciliation.json"
if [ ! -s "$REC" ]; then
  err "no reconciliation.json -- there is no record of what was advised vs shipped."
else
  python3 - "$REC" <<'PY' || fail=1
import json, re, sys
d = json.load(open(sys.argv[1]))
rows = d.get("disagreements", d if isinstance(d, list) else [])
if not isinstance(rows, list):
    print("CRITICAL: reconciliation.json: disagreements must be a list", file=sys.stderr); sys.exit(1)
bad = []
for r in rows:
    op = r.get("op", "<unnamed>")
    v = (r.get("verdict") or "").lower()
    if v not in ("kept", "rejected", "below_threshold"):
        bad.append(f"{op}: verdict {v!r} not in kept|rejected|below_threshold"); continue
    if v == "below_threshold":
        if not isinstance(r.get("window_share_pct"), (int, float)):
            bad.append(f"{op}: below_threshold needs window_share_pct to justify skipping it")
        continue
    if not isinstance(r.get("measured_ms"), (int, float)):
        bad.append(f"{op}: verdict {v} with no measured_ms. A PROSE REJECTION IS NOT A RESULT -- "
                   "qwen's arm rejected the advised RMSNorm sharding in one sentence with no number, "
                   "and that advice was worth 152 us per layer.")
    if not r.get("oracle") and v == "kept":
        bad.append(f"{op}: kept with no oracle recorded -- a faster decoder that fails its correctness "
                   "oracle is a regression with a good number")
for b in bad: print(f"CRITICAL: reconciliation.json: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  [ "$fail" = 0 ] && ok "reconciliation.json: every disagreement has a number or an explicit below_threshold"
fi

# ---- 6. the invariant, and correctness of the shipped result ----------------------------------
FIN="$D/final.json"
if [ ! -s "$FIN" ]; then
  err "no final.json -- the invariant final_ms <= incumbent_ms is unproven."
else
  python3 - "$FIN" "$INC" <<'PY' || fail=1
import json, re, sys
f = json.load(open(sys.argv[1]))
try: i = json.load(open(sys.argv[2]))
except Exception: i = {}
bad = []
fm, im = f.get("final_ms"), f.get("incumbent_ms", i.get("incumbent_ms"))
floor = f.get("noise_floor_ms", i.get("noise_floor_ms", 0.0)) or 0.0
if not isinstance(fm, (int, float)): bad.append("final_ms missing")
if not isinstance(im, (int, float)): bad.append("incumbent_ms missing")
if isinstance(fm, (int, float)) and isinstance(im, (int, float)):
    if fm > im + floor:
        bad.append(f"INVARIANT VIOLATED: final_ms {fm} > incumbent_ms {im} (+floor {floor}). This stage "
                   "may not ship a slower decoder than the one it started from. Ship the incumbent "
                   "unchanged and record the no-change outcome instead.")
    elif abs(fm - im) <= floor and (f.get("changed") or f.get("shipped_change")):
        bad.append(f"final_ms {fm} vs incumbent {im} is inside the noise floor {floor}, i.e. a TIE, but "
                   "a change was shipped. TIES GO TO THE INCUMBENT -- ship it unchanged.")
if not f.get("oracle"):
    bad.append("oracle missing -- name the correctness oracle the shipped decoder passed")
if f.get("oracle_passed") is not True:
    bad.append("oracle_passed is not true -- a faster decoder that fails its oracle is a regression")
if f.get("changed") is None and f.get("shipped_change") is None:
    bad.append("neither `changed` nor `shipped_change` recorded: a NO-CHANGE outcome is a valid, "
               "publishable result but it has to be stated explicitly")
it = f.get("iterations")
if isinstance(it, list) and it:
    for n, e in enumerate(it):
        if not e.get("trigger"):
            bad.append(f"iterations[{n}]: no trigger recorded. Re-capture only after a topology rewrite "
                       "that changes an op's shape; a dtype change is not a rewrite.")
    if len(it) > 3:
        bad.append(f"{len(it)} iterations exceeds the cap of 3 -- extra captures are not free: one "
                   "wedged PCIe access needing a tt-smi reset, and qwen's second capture returned "
                   "byte-identical program configs.")
for b in bad: print(f"CRITICAL: final.json: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  [ "$fail" = 0 ] && ok "final.json: invariant holds, ties go to the incumbent, oracle passed, iterations bounded"
fi

if [ "$fail" != 0 ]; then
  echo "02b-advisor-challenger gate FAILED for $MD" >&2
  exit 1
fi
echo "02b-advisor-challenger gate PASSED for $MD"
