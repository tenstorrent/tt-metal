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
  python3 - <<'PY'
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
lc = d.get("layer_counts")
if not isinstance(lc, dict) or not lc:
    bad.append("layer_counts missing: record {<layer_kind>: <count>} for every kind, read off the model's "
               "own config (num_hidden_layers plus the kind pattern). The full-model estimate multiplies "
               "per-layer microseconds by these, so an unrecorded count means an unchecked headline.")
elif d.get("total_layers") is not None and sum(lc.values()) != d["total_layers"]:
    bad.append(f"layer_counts sum to {sum(lc.values())} but total_layers is {d['total_layers']}; "
               "every layer belongs to exactly one kind.")
src = (d.get("shipped_policy_source") or "").lower()
if not src:
    bad.append("shipped_policy_source missing -- it must name the artifact the policy came from")
elif "constructor_default" in src:
    bad.append("shipped_policy_source cites constructor_defaults: those are the CLASS's default "
               "arguments, not the run's effective config. Source it from what EXECUTED.")
if not d.get("shipped_policy"):
    bad.append("shipped_policy missing")
soft = []
if not d.get("harness_scope"):
    soft.append("harness_scope missing -- state what the harness times end to end. Unrecorded scope is why "
                "one cell's incumbent_ms is a derived per-model composite and another's is one layer.")
it = d.get("iters_per_repeat")
if not isinstance(it, int):
    soft.append("iters_per_repeat missing. A timed block averaging N>=50 replays gives a floor ~sqrt(N) "
                "tighter than single-shot timing; the corpus cell with the only material win is the one "
                "that did this. Record N so floors are comparable across cells.")
elif it < 50:
    soft.append(f"iters_per_repeat is {it}: each timed block averages too few replays to tighten the floor "
                "much. The corpus cell that reached a 0.03% floor used 50.")
for when in ("start", "end"):
    du = d.get(f"device_users_at_{when}")
    if du is None:
        soft.append(f"device_users_at_{when} not recorded. A shared host leaves no retrospective evidence of "
                    "exclusivity -- tt-smi reports board presence, not utilisation -- so it has to be sampled "
                    "while the measurement runs. Two reference cells can never be shown clean.")
    elif du.get("count"):
        soft.append(f"{du['count']} other process(es) held a device open at {when} of the control "
                    f"measurement. This number shared the device. Note that the container running "
                    "ttnn-advise maps the same device, so a capture during a timed run is one of these.")
po = d.get("process_ordinal")
if po is None:
    soft.append("process_ordinal missing: record which harness process of the session this was. The first "
                "process of a session measured a floor 60x the same configuration's in a later one -- "
                "cross-process JIT-cache warmth, which per-process warm-up cannot remove -- and the floor "
                "decides feasibility.verdict.")
elif po == 1:
    soft.append("the incumbent was measured in the FIRST harness process of the session, so its floor "
                "carries cross-process warm-up. Run once with --label warmup_discard, delete the output, "
                "then measure the control.")
wu = d.get("warmup_replays")
if not isinstance(wu, int) or wu < 10:
    bad.append(f"warmup_replays is {wu!r}: record >=10 untimed warm-up replays before the timed blocks. "
               "One corpus harness did exactly 1, and its first timed repeat then carried 73% of the "
               "reported noise floor -- a settling ramp misread as variance, which inflated the floor "
               "enough to make the whole stage look unmeasurable.")
if isinstance(r, list) and len(r) >= 3:
    full = max(r) - min(r)
    exf = max(r[1:]) - min(r[1:])
    mono = all(r[i] >= r[i + 1] for i in range(len(r) - 1))
    if full > 0 and (mono or (1 - exf / full) > 0.5):
        soft.append(f"the first timed repeat is {100 * (1 - exf / full):.0f}% of the whole spread"
                   + (" and the repeats fall monotonically" if mono else "")
                   + f", so {full * 1000:.3f}us is a settling ramp, not a noise floor (without it: "
                     f"{exf * 1000:.3f}us). Add >=10 untimed warm-up replays and re-measure. A ramp also "
                     "breaks non-overlap: a candidate timed after the incumbent in one process is warmer. "
                     "(Advisory: with few repeats this signature also occurs by chance -- P(monotone) is "
                     "1/6 at n=3 -- so it corroborates warmup_replays rather than replacing it.)")
for b in bad: print(f"CRITICAL: incumbent.json: {b}", file=sys.stderr)
for b in soft: print(f"{'CRITICAL' if os.environ.get('CH_STRICT') == '1' else 'WARN'}: incumbent.json: {b}",
                     file=sys.stderr)
sys.exit(1 if bad or (soft and os.environ.get("CH_STRICT") == "1") else (2 if soft else 0))
PY
  case $? in 0) ;; 2) warn=$((warn+1)) ;; *) fail=1 ;; esac
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
  python3 - <<'PY'
import json, os, re, socket, sys
r = json.load(open(os.environ["CH_RJ"])); kind = os.environ["CH_KIND"]
try: i = json.load(open(os.environ["CH_D"] + "/incumbent.json"))
except Exception: i = {}
bad = []; soft2 = []
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
td = os.path.join(os.path.dirname(os.environ["CH_RJ"]), "traced_dtypes.json")
prov = {}
try: prov = json.load(open(td))
except Exception: bad.append("no traced_dtypes.json beside report.json -- run the capture through "
                             "scripts/capture_template.py so what was traced is recorded independently")
ac, pin = str(prov.get("advisor_commit") or ""), str(prov.get("advisor_pin_expected") or "")
if not ac or ac.startswith("UNKNOWN"):
    bad.append(f"advisor_commit not recorded ({ac or 'absent'}). ttnn-advise does not put its version "
               "in report.json and no corpus cell recorded it anywhere, so advice from two builds is "
               "indistinguishable. Export TTMLIR_ADVISOR_HOME and re-capture.")
elif pin and not ac.startswith(pin):
    bad.append(f"advisor at {ac[:12]} but the pin is {pin}. SETUP.md pins the commit so runs are "
               "comparable; re-capture at the pin or state the deviation.")
# THE OPTIMIZER MUST NOT HAVE MOVED. That is the premise of comparing this run against the previous one:
# the toolchain may carry tracer fixes (Python, tools/ttnn-jit/) but no placement change (lib/, include/).
oc = prov.get("optimizer_files_changed_since_pin")
if oc is None:
    soft2.append("optimizer_files_changed_since_pin not recorded -- capture_template computes it as "
                 "`git diff --name-only <optimizer pin>..HEAD -- lib include`. Without it, nobody can "
                 "check the claim this comparison rests on: that the placement logic did not move.")
elif isinstance(oc, list) and oc:
    bad.append(f"the advisor toolchain changes {len(oc)} file(s) under lib/ or include/ since the "
               f"optimizer pin ({', '.join(oc[:3])}...). That is a different optimizer, so results are not "
               "comparable to the previous corpus. Either re-pin deliberately and say so, or drop them.")
# THE TOOL MUST EXIST AT THE RECORDED PATH ON THIS HOST. A contaminated corpus cell wrote documentation
# citing a TTMLIR_ADVISOR_HOME that does not exist on the machine that produced it, with plausible hashes
# and op counts. Only a byte-identity comparison caught it, days later. Three lines turn that into a stop.
tp = prov.get("tool_realpath") or prov.get("tool_path")
if not tp or not prov.get("tool_sha256") or not prov.get("host"):
    bad.append("the capture records no host fingerprint: host, tool_path/tool_realpath and tool_sha256 are "
               "required beside every captured tool output, so a capture that could not have run on this "
               "machine fails here instead of being discovered by hand later.")
elif not os.path.isfile(str(tp)):
    bad.append(f"the capture says it ran {tp}, which does not exist on this host ({socket.gethostname()}). "
               "Either the artefact came from somewhere else, or its provenance is invented.")
tm = prov.get("tracer_matches_checkout")
if tm is None:
    soft2.append("tracer_matches_checkout not recorded. advisor_commit describes the CHECKOUT, not the code "
                 "that ran: ttnn_jit is installed into the toolchain venv as a plain directory, so a git "
                 "checkout of another branch changes the commit while the imported tracer stays put.")
elif tm is not True:
    bad.append(f"the tracer that will be IMPORTED ({prov.get('tracer_imported_from')}) is not the one in the "
               f"checkout ({prov.get('tracer_checkout_path')}). The tracer decides which layers the advisor "
               "can see at all -- it is the difference between a real zero and a coverage zero -- so advice "
               "captured through a different tracer than the pinned one is not this experiment's advice. "
               "Reinstall ttnn_jit from the pinned checkout and re-capture.")
elif str(prov.get("host")) != socket.gethostname():
    soft2.append(f"the capture was produced on {prov.get('host')} and this gate runs on "
                 f"{socket.gethostname()}. Legitimate for a re-check; not for a measurement.")
if not r.get("capture_policy_source"):
    bad.append("capture_policy_source unset: nothing records that the traced decoder was built with the "
               "SHIPPED policy rather than class defaults. Dtypes are checked below; layouts and "
               "DRAM-sharding flags are not, so name the artifact the policy came from.")
if not (prov.get("capture_scope") or r.get("capture_scope")):
    bad.append("capture_scope unset: record ops_attempted, methods_substituted, env_knobs and stopped_at. "
               "Fifteen corpus captures ran 54-290 lines and five stopped at the same terminal op in four "
               "different places, from 30 ops captured down to 5, with nothing saying so -- and where a "
               "model method is substituted before tracing, the advice for that region is advice for the "
               "STAND-IN, which a reader cannot otherwise know.")
if not os.path.isfile(os.path.join(os.path.dirname(os.environ["CH_RJ"]), "final_ir.mlir")):
    bad.append("no final_ir.mlir beside report.json. It is the only artifact carrying the advised SHARD "
               "SHAPES and the full CoreRangeSet, so it is what the plan must be implemented from; "
               "report.json has neither and understates 58% of advised core counts.")
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
for b in soft2: print(f"{'CRITICAL' if os.environ.get('CH_STRICT') == '1' else 'WARN'}: {kind}: {b}",
                      file=sys.stderr)
sys.exit(1 if bad else (2 if soft2 else 0))
PY
  case $? in 0) ;; 2) warn=$((warn+1)) ;; *) fail=1 ;; esac
done
[ "$fail" = 0 ] && ok "captures: per layer kind, dtypes match shipped, batch and order agree"

# ---- 3. reconciliation: tool-generated, closes at 100%, every chain resolved ----------------------
recs=("$D"/reconciliation*.json)
if [ ${#recs[@]} -eq 0 ]; then
  err "no reconciliation*.json -- no record of what was advised vs shipped"
else
  for rc in "${recs[@]}"; do
    export CH_RC="$rc"
    python3 - <<'PY'
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
                "the floor is not attributable to the advice -- report zero with the arithmetic.")
# A ZERO BOUNDARY CEILING IS NOT A ZERO RESULT. The ceiling prices only boundary conversions the advice does
# not place, so re-gridding an op inside its L1 chain is worth 0.000us to it while measuring up to
# 236.8us/layer on hardware. Two of the corpus's three biggest wins came from cells whose ceiling said zero.
cliff = d.get("cliff_candidates") or []
unscreened = [c for c in cliff if not (c.get("measured_ms") or c.get("hard_error"))]
if unscreened:
    crit.append(f"{len(unscreened)} of {len(cliff)} cliff_candidates have neither a measured_ms nor a quoted "
                f"hard_error: {', '.join(str(c.get('op')) for c in unscreened[:4])}. These are material ops "
                "on <=2 cores where the advisor wants strictly more -- the class that produced every "
                "double-digit win in the reference corpus, and the one the boundary ceiling cannot see. "
                "Screen them, or quote the error that stops you.")
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
    if v == "kept" and c.get("oracle_passed") is not True:
        crit.append(f"chain {c.get('chain')}: kept with no oracle_passed of its own. Some ops compute the "
                    "WRONG ANSWER under particular shard specs, and a placement change is exactly what "
                    "triggers that -- so every kept candidate needs its own correctness result, not the "
                    "final winner's. Record oracle_passed and oracle_pcc per chain.")
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
    # and the mirror image: an op the advisor DECLARED unplaceable must not be screened. 41 of 54 such
    # declarations in the reference corpus were screened anyway, rediscovering the advisor's own error string.
    if r.get("advisor_unfixable") and r.get("measured_ms") is not None:
        warn.append(f"{r.get('device')} was measured although the advisor declared it unplaceable with an "
                    "exact runtime error. If you believe the declaration is wrong, disprove it with an "
                    "isolated single-op test in the advised config, not a whole-decoder measurement.")
if not (d.get("advised_plan") or {}).get("source"):
    crit.append("reconcile.py was run without --ir, so advised_plan carries no shard shapes and the advisor's "
                "plan cannot be implemented as written. Re-run with --ir shard_advise/<kind>/final_ir.mlir.")
for b in crit: print(f"CRITICAL: {nm}: {b}", file=sys.stderr)
for b in warn: print(f"{'CRITICAL' if strict else 'WARN'}: {nm}: {b}", file=sys.stderr)
sys.exit(1 if crit or (warn and strict) else (2 if warn else 0))
PY
    case $? in 0) ;; 2) warn=$((warn+1)) ;; *) fail=1 ;; esac
  done
  [ "$fail" = 0 ] && ok "reconciliation: tool-generated, closes at 100%, chains resolved"
fi

# ---- 4. the result: non-overlap, oracle, outcome stated ------------------------------------------
FIN="$D/final.json"
if [ ! -s "$FIN" ]; then
  err "no final.json -- the contribution is unrecorded"
else
  python3 - <<'PY'
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
me = f.get("model_estimate") or {}
if not all(isinstance(me.get(k), (int, float)) for k in ("before_us", "after_us", "band_us")):
    crit.append("model_estimate {before_us, after_us, band_us} missing from final.json. That is this stage's "
                "headline metric and nothing else computes it: sum reconciliation model_estimate.this_kind_us "
                "over the layer kinds for before_us, apply each kind's measured winner for after_us, and sum "
                "uncertainty_per_model_us linearly for band_us.")
elif me["before_us"] > 0 and abs(me["after_us"] - me["before_us"]) < me["band_us"] and f.get("changed"):
    warn.append(f"a change shipped whose model-level effect ({me['before_us'] - me['after_us']:.1f} us) is "
                f"inside its own uncertainty band (+/-{me['band_us']:.1f} us). Legitimate -- detection is "
                "per-layer, not per-model -- but say it plainly rather than quoting the model number alone.")
if not f.get("oracle"): crit.append("oracle missing -- name the correctness oracle the result passed")
if f.get("oracle_passed") is not True:
    crit.append("oracle_passed is not true -- a faster decoder that fails its oracle is a regression")
# WHICH ORACLE IS THE VETO. v2 asked for a differential oracle, warned in the same section that a
# differential oracle "cannot fail", and told cells to reject a candidate that "moves PCC at all". The only
# reading satisfying all three is a differential bar near 1.0 -- and a re-grid of a reduction moves the last
# few decimals BY CONSTRUCTION, so that made the highest-yield change in this stage unshippable. One cell
# wrote comp_pcc(..., 0.999999) and discarded a 13.4%-faster candidate that was MORE accurate than what it
# shipped (0.99904 vs 0.99890 against the HF reference, at a model bar of 0.995).
#
# v3: the ABSOLUTE comparison at the model's own bar is the veto; the differential one is an observation.
ok_ = str(f.get("oracle_kind") or "").lower()
if ok_ not in ("absolute", "differential"):
    crit.append("oracle_kind missing: record 'absolute' (candidate vs a reference the change cannot move) or "
                "'differential' (candidate vs the frozen incumbent). They answer different questions and only "
                "the first can veto -- a differential PCC moves in the last decimals whenever a reduction's "
                "core count changes, which is exactly the change this stage exists to find.")
bar, bar_src = f.get("oracle_pcc_bar"), f.get("oracle_bar_source")
if not isinstance(bar, (int, float)):
    crit.append("oracle_pcc_bar missing -- record the bar as a number, read from the model's own test.")
if not bar_src:
    crit.append("oracle_bar_source missing -- name the file:line the bar was read from. An invented bar is "
                "how one cell held itself to 0.999999 while every other cell in the corpus used 0.995.")
if isinstance(bar, (int, float)) and bar > 0.9999 and not f.get("oracle_bar_justification"):
    crit.append(f"oracle_pcc_bar {bar} is tighter than any model's own test bar in the reference corpus "
                "(0.995, or a recorded model-specific value). A bar that tight asks 'is this bitwise "
                "unchanged?', which no re-grid of a reduction can answer yes to. Read the bar from the "
                "model's test, or record oracle_bar_justification.")
if ok_ == "absolute" and f.get("changed") and f.get("incumbent_pcc_vs_reference") is None:
    crit.append("incumbent_pcc_vs_reference missing. The ship rule is 'within the bar AND no worse than the "
                "incumbent', which needs the incumbent scored against the same reference -- and a "
                "differential oracle cannot tell which side moved. On one corpus pair the candidate scored "
                "0.99931 and the shipped incumbent 0.98347, failing the model's own 0.995 bar.")
od = (str(f.get("oracle") or "") + " " + str(f.get("oracle_scope") or "")
      + " " + str(f.get("oracle_reference") or "")).lower()
ow = str(f.get("oracle_weights") or "").lower()
if ow not in ("real", "synthetic"):
    crit.append("oracle_weights missing: record 'real' or 'synthetic' so the correctness evidence is legible.")
if f.get("changed") and f.get("changed_precision"):
    if ow != "real":
        crit.append("this change touches dtype or fidelity, which is a PRECISION decision: it needs a "
                    "real-weight oracle. A synthetic-weight PCC does not bound the real-weight PCC, because "
                    "quantisation error depends on a weight distribution random tensors do not have.")
elif f.get("changed") and ow != "real":
    warn.append("placement-only change on a synthetic-weight oracle. Acceptable here -- state in the README "
                "that the correctness evidence is differential (same weights, candidate vs frozen incumbent) "
                "rather than absolute.")
# A differential oracle against the frozen incumbent is the RIGHT oracle for a placement change. What is
# useless is comparing an execution path against itself: one corpus cell reported eager-vs-traced-replay PCC
# of exactly 1.0, which no placement change can move.
if f.get("changed") and any(k in od for k in ("eager-vs-traced", "traced-vs-eager", "replay pcc")):
    crit.append("the oracle compares eager execution against traced replay of the same implementation, which "
                "cannot fail for a placement change (one corpus cell reported PCC exactly 1.0 this way). "
                "Compare the candidate against the frozen incumbent instead.")
if f.get("changed") and not f.get("oracle_reference"):
    warn.append("oracle_reference not named -- say what the candidate was compared against (the frozen "
                "incumbent is the expected answer for a placement change).")
for k, v in f.items():
    if k.endswith("_passed") and v is False and k != "oracle_passed":
        # If nothing shipped, a failing absolute check is usually the incumbent's pre-existing condition and
        # not this stage's defect -- one corpus cell inherited a sliding-attention PCC of 0.9889 against a
        # 0.993 bar. If something DID ship, the failure is disqualifying: you cannot tell whether the change
        # caused it.
        (crit if f.get("changed") else warn).append(
            f"{k} is false. " + ("A change shipped while a correctness check was failing, so its effect on "
                                 "that check is unknown." if f.get("changed") else
                                 "Nothing shipped, so this is most likely pre-existing in the incumbent -- "
                                 "say so explicitly and attribute it to the earlier stage."))
if isinstance(fm, (int, float)) and isinstance(im, (int, float)) and fm > im:
    crit.append(f"final_ms {fm} > incumbent_ms {im}: this stage may not ship a slower decoder")
# DID ANYONE APPLY THE ADVICE AS WRITTEN? The largest measured defect in the reference corpus: it screened
# chain by chain from the incumbent and never applied the plan whole. Where that could be measured afterwards
# the plan was worth -10.43% against the -4.88% the cell shipped, at bit-identical output, rising to -17.84%
# with the advised norm -- 3.7x. Four cells never tried it and recorded no reason, and their artefacts are
# indistinguishable from a measured rejection. So the apply-all candidate is required to exist and to resolve.
apv = f.get("advised_plan_verbatim")
if not isinstance(apv, dict):
    crit.append("advised_plan_verbatim missing. Candidate #1 is the advisor's whole plan, built from "
                "final_ir.mlir with unfixable_ops dropped. Record it with measured_ms and a verdict, or with "
                "hard_error naming the item that would not run and the single-op test that isolated it -- "
                "'not tried' must not look like 'tried and lost'.")
elif apv.get("measured_ms") is None and not apv.get("hard_error"):
    crit.append("advised_plan_verbatim has neither measured_ms nor hard_error, so it was not actually tried.")
# EVERY MEASUREMENT MUST BE ACCOUNTED FOR BY THE DECISION.
#
# This is the failure that survived from v2 into v3's first cell, twice, by two different routes. In v2 a
# cell measured 0.700120 and shipped 0.768104. In v3's first cell a candidate was measured FOUR TIMES at
# 0.543590 -- faster than the 0.550052 that shipped -- and never oracle-checked, while the two slower rungs
# both got real-weight oracles and both passed. The gate passed it because the "final_ms is worse than the
# best measured set" check reads combination.measured_sets, which held ONE of seventeen measurements.
#
# So: compare against every measurement on disk, and state the population. Scoped PER LAYER KIND, because
# measurements of different kinds have different incumbents and comparing across them is meaningless -- an
# unscoped version of this check would fire on every multi-kind cell. Disconfirmation runs are exempt by
# name: order-swap and knob-off are SUPPOSED to sit at the incumbent.
mdir = os.path.join(os.environ["CH_D"], "measurements")
kinds = {}
for k in (f.get("model_estimate") or {}).get("per_kind", {}) or {}:
    kinds[k] = None
faster, pop = [], 0
if os.path.isdir(mdir):
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".json"):
            continue
        try:
            m = json.load(open(os.path.join(mdir, fn)))
        except Exception:
            crit.append(f"measurements/{fn} does not parse. A measurement the gate cannot read is a "
                        "measurement nobody reconciled against the decision.")
            continue
        pop += 1
        med = m.get("median_ms")
        lbl = str(m.get("label") or fn)
        if not isinstance(med, (int, float)):
            continue
        if any(t in lbl.lower() for t in ("knob_off", "order_swap", "warmup", "discard", "incumbent")):
            continue                      # disconfirmation and control runs, exempt by design
        # per-kind: only compare with the shipped number when the label names the same kind
        same_kind = [k for k in kinds if k and k.split("_")[0].lower() in lbl.lower()] or None
        if isinstance(fm, (int, float)) and med < fm - 1e-9 and (same_kind or not kinds):
            faster.append((lbl, med, m.get("oracle_pcc"), m.get("verdict")))
if pop == 0 and f.get("changed"):
    warn.append("no measurements/ directory to reconcile against the decision. Every timed run belongs "
                "there, or nothing can check that the fastest candidate is the one that shipped.")
for lbl, med, pcc, verdict in faster:
    if verdict or pcc is not None:
        warn.append(f"measurements/{lbl} is faster than what shipped ({med} < {fm}) and carries "
                    f"verdict={verdict!r} pcc={pcc!r} -- state in the README why it did not ship.")
    else:
        crit.append(f"measurements/{lbl} measured {med}, FASTER than the shipped {fm}, and carries no "
                    "verdict and no oracle result. The stage's output is 'the best measured decoder': a "
                    "faster measurement must either ship or record why it did not. "
                    f"(population checked: {pop} measurement file(s).)")
if f.get("changed") and not f.get("disconfirmation"):
    warn.append("no disconfirmation recorded for a shipped change. Two cheap measurements: an ORDER SWAP "
                "(re-measure the incumbent in a fresh process AFTER the candidate -- a candidate that only "
                "wins in the later process won a warm-up) and KNOB OFF (confirm the incumbent comes back). "
                "The analysis behind this stage refuted about 1 in 6 of its own recommendations.")
if not f.get("changed") and not (f.get("could_not_do") or f.get("reachable_by_advisor")):
    warn.append("a no-change outcome with neither could_not_do[] nor reachable_by_advisor. State which "
                "kind of zero this is: nothing to find, or could not look. A blocked item is a PASSING "
                "result here -- record it rather than reaching for a number.")
if not f.get("reachable_by_advisor"):
    warn.append("reachable_by_advisor missing -- record which kinds were captured and the untraced window "
                "share, so a contribution of zero reads as 'nothing to find' rather than 'could not look'. "
                "4 of the reference corpus's 7 zeros were the second kind and nothing said so.")
if f.get("changed") and not (f.get("perf_report_incumbent") and f.get("perf_report_winner")):
    warn.append("no before/after perf report pair (perf_report_incumbent + perf_report_winner). 1 of 15 "
                "corpus cells kept one, which is why op-level verification is impossible for the rest.")
if f.get("changed"):
    sets = (f.get("combination") or {}).get("measured_sets")
    if not isinstance(sets, list) or not sets:
        crit.append("a change shipped with no combination.measured_sets")
    else:
        for n, s in enumerate(sets):
            if not isinstance(s.get("measured_ms"), (int, float)):
                crit.append(f"measured_sets[{n}]: no measured_ms -- best_set must be MEASURED")
            if s.get("oracle_passed") is not True and (s.get("set") or ["x"])[0] != "incumbent":
                crit.append(f"measured_sets[{n}]: no oracle_passed. A set that was measured but never "
                            "checked for correctness cannot be shipped or compared.")
            if not (s.get("chains") or s.get("set")):
                crit.append(f"measured_sets[{n}]: neither `chains` nor `set` -- the number is unattributable")
            if not s.get("repeats_ms"):
                warn.append(f"measured_sets[{n}]: no repeats_ms, so non-overlap cannot be checked")
            if not s.get("op_under_test"):
                warn.append(f"measured_sets[{n}]: no op_under_test {{name, incumbent_grid, candidate_grid, "
                            "legal_ladder}}. Without the incumbent's own grid two arms using the same knob "
                            "name with different defaults produce deltas that look comparable and are not.")
        if not f.get("candidate_shape_assumptions"):
            warn.append("candidate_shape_assumptions missing for a shipped change -- tile rows, "
                        "divisibility, grid shape. The corpus's largest wins are batch-pinned by "
                        "construction (one fails at batch 64 with 'Shard height 32 must match physical "
                        "height 64' and at batch 8 at build) and nothing recorded it.")
        # PRODUCTS WITHIN A KIND ARE REQUIRED, not optional: two disjoint winners' product beat every isolate
        # in both corpus cells that built one, and only 2 of 15 did.
        winners = [s for s in sets if isinstance(s.get("measured_ms"), (int, float))
                   and len(s.get("chains") or s.get("set") or []) == 1]
        if len(winners) >= 2 and not any(len(s.get("chains") or s.get("set") or []) >= 2 for s in sets):
            warn.append(f"{len(winners)} single-chain candidates were measured and no product of them was. "
                        "Once two winners touch disjoint ops their product is a required candidate -- "
                        "additivity is not predictable, and in both corpus cells that tried it the product "
                        "beat every isolate (-13.24% vs -7.60%; -2.82% vs -1.86%).")
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
sys.exit(1 if crit or (warn and strict) else (2 if warn else 0))
PY
  case $? in 0) ;; 2) warn=$((warn+1)) ;; *) fail=1 ;; esac
  [ "$fail" = 0 ] && ok "final.json: non-overlap holds, oracle passed, outcome stated"
fi

if [ "$fail" != 0 ]; then
  echo "02b-advisor-challenger gate FAILED for $MD" >&2; exit 1
fi
[ "$warn" != 0 ] && echo "02b-advisor-challenger gate PASSED for $MD with $warn advisory warning(s)"
[ "$warn" = 0 ] && echo "02b-advisor-challenger gate PASSED for $MD"
exit 0
