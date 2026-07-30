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

  # ---- 3b. BATCH: capture, incumbent and the requested batch must all agree -------------------
  # Two of the three agreeing is what hid this: a cell captured AND measured at batch 1 while the driver
  # had asked for 32, so it self-consistently answered a question nobody posed, on a model whose serving
  # batch is 32. Judging advice at a batch it was not captured at can flip its sign outright (+12.3% at
  # b1 vs -8.8% at b32 on the same advice). CHALLENGER_DECODE_BATCH is exported by the driver.
  python3 - "$rj" "$INC" "$kind" "${CHALLENGER_DECODE_BATCH:-}" <<'PY' || fail=1
import json, re, sys
rep, inc, kind, want = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
r = json.load(open(rep))
try: i = json.load(open(inc))
except Exception: i = {}
cb, ib = r.get("capture_batch"), i.get("decode_batch")
# The requested batch is read from the ARTIFACT, not the environment. An earlier version required an env
# var that only the experiment's own driver exported, which made the gate fail for anyone running this as
# an ordinary pipeline stage -- a gate that cannot pass in the pipeline is not a gate, it is a wall. The
# stage records what the prompt asked for (requested_decode_batch); the env var, when present, is an
# optional third cross-check from the orchestrator.
rb = i.get("requested_decode_batch")
bad = []
if cb is None: bad.append("report.json records no capture_batch")
if ib is None: bad.append("incumbent.json records no decode_batch")
if rb is None:
    bad.append("incumbent.json records no requested_decode_batch -- record the prompt's DECODE_BATCH so "
               "the batch actually measured can be checked against the batch asked for")
if cb is not None and ib is not None and int(cb) != int(ib):
    bad.append(f"capture_batch {cb} != incumbent decode_batch {ib}: the advice was judged at a batch it "
               "was not captured at, which can flip its sign")
for nm, v in (("incumbent decode_batch", ib), ("capture_batch", cb)):
    if v is not None and rb is not None and int(v) != int(rb):
        bad.append(f"{nm} {v} != requested_decode_batch {rb}")
if want and rb is not None and int(want) != int(rb):
    bad.append(f"orchestrator asked for DECODE_BATCH {want} but the stage recorded "
               f"requested_decode_batch {rb}")
# ORDERING: the incumbent must be frozen BEFORE the advisor runs, or `final <= incumbent` is comparing
# against a number the advisor could already have influenced. This is the only in-repo way to check it.
ma, ca = i.get("measured_at"), r.get("captured_at")
if not ma: bad.append("incumbent.json records no measured_at")
if not ca: bad.append(f"report.json records no captured_at")
if ma and ca and str(ca) < str(ma):
    bad.append(f"captured_at {ca} PRECEDES incumbent measured_at {ma}: the incumbent was not frozen "
               "before the advisor ran, so the invariant compares against a contaminated baseline")
for b in bad: print(f"CRITICAL: {kind}: BATCH/ORDER: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
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

# ---- chains are the unit of judgement, not ops ------------------------------------------------
# A chain split below the materiality bar is how the strongest advice class gets discarded unmeasured:
# one RoPE chain arrived as 9 rows of 0.46-0.97%, each under a 1% per-op bar, none measured, summing to
# 5.86% of the decode window. So: chains must exist, each material chain needs a number, and no set of
# same-chain rows may be dropped while the chain's SUM clears the threshold.
thr = d.get("threshold_pct")
chains = d.get("chains")
if not isinstance(chains, list) or not chains:
    bad.append("no `chains` array. Group advised ops into chains and threshold on the chain's SUMMED "
               "window share -- a single op resharded in isolation pays its edge conversions and only "
               "the whole L1-resident chain pays off (OPT-003).")
else:
    for c in chains:
        nm = c.get("chain", "<unnamed>")
        v = (c.get("verdict") or "").lower()
        s = c.get("summed_window_share_pct")
        if v not in ("kept", "rejected", "below_threshold"):
            bad.append(f"chain {nm}: verdict {v!r} not in kept|rejected|below_threshold"); continue
        if v == "below_threshold":
            if not isinstance(s, (int, float)):
                bad.append(f"chain {nm}: below_threshold with no summed_window_share_pct")
            elif isinstance(thr, (int, float)) and s >= thr:
                bad.append(f"chain {nm}: dropped below_threshold but its SUMMED share {s}% clears the "
                           f"{thr}% threshold -- this is the chain-shredding defect; measure it")
        elif not isinstance(c.get("measured_ms"), (int, float)):
            bad.append(f"chain {nm}: verdict {v} with no measured_ms. Measure the chain AS ONE UNIT.")
    # cross-check: rows dropped whose chain is material
    per = {}
    for r in rows:
        per.setdefault(r.get("chain", "<none>"), []).append(r)
    for nm, rs in per.items():
        tot = sum(r.get("window_share_pct") or 0.0 for r in rs)
        if isinstance(thr, (int, float)) and tot >= thr and \
           all((r.get("verdict") or "").lower() == "below_threshold" for r in rs):
            bad.append(f"chain {nm}: every one of its {len(rs)} rows dropped below_threshold while their "
                       f"sum is {tot:.3f}% >= {thr}%. Judge the chain, not the op.")
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
    # Op-level evidence for anything KEPT: an end-to-end latency delta cannot show WHERE the time moved,
    # so a kept change without a profile is a number with no mechanism behind it.
    if v == "kept" and not r.get("perf_report"):
        bad.append(f"{op}: kept with no perf_report -- every kept candidate needs its own op-level "
                   "tt-perf-report CSV referenced here, not just a latency number")
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
# ---- COMBINATION: screening alone does not find the best decoder -----------------------------------
# Chains interact. Two changes that each beat the incumbent alone can lose together, and a cumulative set
# can lose to a single change. So the shipped config must be the best MEASURED set, and it must not be
# worse than the best single change already measured -- otherwise the stage screened well and shipped badly.
comb = f.get("combination") or {}
sets = comb.get("measured_sets")
bs = f.get("best_single_ms")
kept_any = bool(f.get("changed"))
if kept_any:
    if not isinstance(sets, list) or not sets:
        bad.append("combination.measured_sets is empty while a change was shipped. Measure the cumulative "
                   "winner set (and pairwise combinations of the top chains) and record each with its ms.")
    else:
        for n, s in enumerate(sets):
            if not isinstance(s.get("measured_ms"), (int, float)):
                bad.append(f"combination.measured_sets[{n}]: no measured_ms -- best_set must be the best "
                           "MEASURED set, never an inferred one")
            if s.get("oracle_passed") is None:
                bad.append(f"combination.measured_sets[{n}]: no oracle_passed")
        best = min((s.get("measured_ms") for s in sets
                    if isinstance(s.get("measured_ms"), (int, float))), default=None)
        if best is not None and isinstance(fm, (int, float)) and fm > best + floor:
            bad.append(f"final_ms {fm} is worse than the best measured set {best} (+floor {floor}) -- "
                       "ship the best set you measured")
    if not isinstance(bs, (int, float)):
        bad.append("best_single_ms missing: without it, a combination that is worse than one of its own "
                   "members cannot be detected")
    elif isinstance(fm, (int, float)) and fm > bs + floor:
        bad.append(f"INVARIANT VIOLATED: final_ms {fm} > best_single_ms {bs} (+floor {floor}). A "
                   "combination worse than a single change already measured must not ship.")

it = f.get("iterations")
if isinstance(it, list) and it:
    for n, e in enumerate(it):
        if not e.get("trigger"):
            bad.append(f"iterations[{n}]: no trigger recorded. Re-capture only after a topology rewrite "
                       "that changes an op's shape; a dtype change is not a rewrite.")
        # every iteration after the first must re-rank from a FRESH profile: ranking once against the
        # original incumbent profile means later rounds chase a distribution that no longer exists.
        if n > 0 and not e.get("reranked_from"):
            bad.append(f"iterations[{n}]: no reranked_from. After applying winners, re-profile and re-run "
                       "reconcile.py on the NEW CSV; the original ranking is stale once the graph changes.")
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
