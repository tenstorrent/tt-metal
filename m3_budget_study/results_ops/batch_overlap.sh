#!/bin/bash
# Shared expert overlapped with the MoE dispatch (M3_MOE_OVERLAP_SHARED) and its reduce-scatter fused into the routed
# one (M3_MOE_FUSE_SHARED_RS), on the (2,4) stage-0 carve, layers 0-6 (3 dense + 4 sparse), real longbook_56320 tokens.
# Recipe and reading of the results: overlap_shared_recipe.md.
#
# Configs (CONFIGS, default "off ov ovf"): off = both knobs 0, ov = OVERLAP, ovf = OVERLAP + FUSE_RS, fuse = FUSE_RS only.
# Kinds (KINDS, default "kv kvg wall prof"), in this order:
#   smoke budget_sweep.py at W=4096, h=0, 1 warm-up (5 on a first point) + 3 timed, BUDGET_OUT_STATS=1 (not in the default)
#   kv    budget_packed.py, B=2 (W=4096), slot 0 at h=0 + slot 1 at h=16384, KV dump -> overlap/kv_<cfg>;
#         compare_kv.py vs kv_off -> overlap/kv_compare_<cfg>.txt
#   kvg   tools/profile_4x2.py on PROFILE_MESH=2x4 (chunk 5120 up to 56320 tokens): KV PCC vs the golden in the log,
#         dump -> overlap/kvg_<cfg>; profile_4x2.py --compare vs kvg_off -> overlap/kvg_compare_<cfg>.txt
#   wall  budget_sweep.py at W=4096 and W=8192, points h=0 and h=139264 (warm-up 2, 5 timed) -> overlap/budget_runs.csv
#   prof  one zone profile per config (batch_p0a.sh point w4096_h141312_prose, i.e. h=139264, PREFIX_QUIET) ->
#         profiles/ovl_<cfg>_w4096_h141312_prose; tools/overlap_check.py -> overlap/overlap_<cfg>.txt
# Fixed env: M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine, 1d fabric, bf4 experts.
#
# One device process per case, each reset first (env -u TT_VISIBLE_DEVICES tt-smi -glx_reset, inside run_budget.sh /
# run_kv_4x2.sh / run_prefill_profile.sh), under a watchdog of RUN_TIMEOUT=1200 s total and STALL_TIMEOUT=300 s
# without log output. Refuses to start a case unless .lock names owner=vmelnykov-ops-agent, or while another M3 harness
# runs. Each case appends a row (SHA + env) to runs.csv; a case whose log exists is skipped.
#
#   nohup m3_budget_study/results_ops/batch_overlap.sh > m3_budget_study/results_ops/logs/batch_overlap.out 2>&1 &
#   KINDS="kv" CONFIGS="off ov" ...   a subset;   ONLY="ovl_wall_ov_w4096" ...   single case ids;   DRY_RUN=1 lists cases
#   ID_SUFFIX=_r2 ...   appended to every case id (a repeat, e.g. CONFIGS in reverse order to check drift)
set -uo pipefail
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
export TT_METAL_HOME
LOCK="$RES/.lock"
LOCK_OWNER=vmelnykov-ops-agent
GOLDEN=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
PROSE="$GOLDEN/longbook_56320/metadata.json"
PY="$TT_METAL_HOME/python_env/bin/python"
OUT="$RES/overlap"
CONFIGS="${CONFIGS:-off ov ovf}"
KINDS="${KINDS:-kv kvg wall prof}"
mkdir -p "$OUT" "$RES/logs"

export BUDGET_RESULTS="$RES" BUDGET_LOCK="$LOCK" BUDGET_LOCK_OWNER="$LOCK_OWNER"
export BUDGET_COLLECT_CSV="$OUT/budget_runs.csv" BUDGET_STAGES=4 BUDGET_STAGE=0 BUDGET_LAYER_IDS=0,1,2,3,4,5,6
export BUDGET_TOKENS="$PROSE" M3_FABRIC=1d M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128
export M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1 EXPERT_DTYPE=bf4
export LOAD_TIMEOUT=900 STALL_TIMEOUT="${STALL_TIMEOUT:-300}" RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"
unset M3_MOE_OVERLAP_SHARED M3_MOE_FUSE_SHARED_RS M3_MOE_OVERLAP_DISPATCH_ROWS M3_MOE_LOAD_STATS

knobs () {  # config -> the two knob assignments
  case "$1" in
    off) echo "M3_MOE_OVERLAP_SHARED=0 M3_MOE_FUSE_SHARED_RS=0" ;;
    ov) echo "M3_MOE_OVERLAP_SHARED=1 M3_MOE_FUSE_SHARED_RS=0" ;;
    ovf) echo "M3_MOE_OVERLAP_SHARED=1 M3_MOE_FUSE_SHARED_RS=1" ;;
    fuse) echo "M3_MOE_OVERLAP_SHARED=0 M3_MOE_FUSE_SHARED_RS=1" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

# id | kind | config | extra env (space-separated K=V)
CASES=()
SFX="${ID_SUFFIX:-}"
for kind in $KINDS; do
  for cfg in $CONFIGS; do
    case "$kind" in
      smoke) CASES+=("ovl_smoke_${cfg}${SFX}|wall|$cfg|HARNESS=budget_sweep.py BUDGET_W=4096 BUDGET_POINTS=0:4096 BUDGET_WARMUP=1 BUDGET_ITERS=3 BUDGET_OUT_STATS=1") ;;
      kv) CASES+=("ovl_kv_${cfg}${SFX}|kv|$cfg|HARNESS=budget_packed.py BUDGET_B=2 BUDGET_COMPOS=G=0:2048,16384:2048 BUDGET_DUMP_KV=$OUT/kv_$cfg$SFX BUDGET_WARMUP=1 BUDGET_ITERS=1") ;;
      kvg) CASES+=("ovl_kvg_${cfg}${SFX}|kvg|$cfg|PROFILE_MESH=2x4 PROFILE_STAGE=0 PROFILE_NUM_LAYERS=7 PROFILE_CHUNK=5120 PROFILE_CACHE=51200 PREFILL_TRACE_DIR=$GOLDEN/longbook_56320 PROFILE_KV_PCC=1 PROFILE_KV_DUMP=$OUT/kvg_$cfg$SFX") ;;
      wall) for W in 4096 8192; do
              CASES+=("ovl_wall_${cfg}_w${W}${SFX}|wall|$cfg|HARNESS=budget_sweep.py BUDGET_W=$W BUDGET_POINTS=0:$W,139264:$W")
            done ;;
      prof) CASES+=("ovl_${cfg}_w4096_h141312_prose|prof|$cfg|") ;;
      *) echo "unknown kind $kind"; exit 1 ;;
    esac
  done
done

other_harness () {
  pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tools/profile_4x2\.py|tracy-capture' | grep -v "^$$ " || true
}

post () {  # kind cfg id: the compare / overlap readout of one finished case
  local kind="$1" cfg="$2" id="$3"
  case "$kind" in
    kv) if [ "$cfg$SFX" != off ] && [ -f "$OUT/kv_off/slot0.pt" ] && [ -f "$OUT/kv_$cfg$SFX/slot0.pt" ]; then
          COMPARE_SP=2 "$PY" "$TT_METAL_HOME/m3_budget_study/compare_kv.py" "$OUT/kv_off" "$OUT/kv_$cfg$SFX" 0:2048 1:18432 \
            > "$OUT/kv_compare_$cfg$SFX.txt" 2>&1; tail -1 "$OUT/kv_compare_$cfg$SFX.txt"
        fi ;;
    kvg) grep -E 'KV PCC vs golden' "$RES/logs/$id.log" | tail -1
         if [ "$cfg$SFX" != off ] && [ -f "$OUT/kvg_off/kv.pt" ] && [ -f "$OUT/kvg_$cfg$SFX/kv.pt" ]; then
           "$PY" "$RES/tools/profile_4x2.py" --compare "$OUT/kvg_off" "$OUT/kvg_$cfg$SFX" > "$OUT/kvg_compare_$cfg$SFX.txt" 2>&1
           tail -1 "$OUT/kvg_compare_$cfg$SFX.txt"
         fi ;;
    wall) grep -E '"kind": "point"' "$RES/logs/$id.log" | sed 's/^RESULT //' ;;
    prof) local csv; csv="$(find "$RES/profiles/$id" -name 'ops_perf_results_*.csv' 2>/dev/null | sort | tail -1)"
          if [ -n "$csv" ]; then
            "$PY" "$RES/tools/overlap_check.py" "$csv" --json "$OUT/overlap_$cfg.json" > "$OUT/overlap_$cfg.txt" 2>&1
            cat "$OUT/overlap_$cfg.txt"
          fi ;;
  esac
}

for spec in "${CASES[@]}"; do
  IFS='|' read -r id kind cfg extra <<< "$spec"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $id "* ]]; then continue; fi
  log="$RES/logs/$id.log"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "[ovl] dry-run $id ($kind): $(knobs "$cfg") $extra"; continue; fi
  if [ -e "$log" ]; then echo "[ovl] $id: $log exists, skipping"; continue; fi
  if ! grep -q "owner=$LOCK_OWNER" "$LOCK" 2>/dev/null; then echo "[ovl] $LOCK does not name owner=$LOCK_OWNER; stopping before $id"; exit 1; fi
  busy="$(other_harness)"
  if [ -n "$busy" ]; then echo "[ovl] another harness is running, stopping before $id:"$'\n'"$busy"; exit 1; fi
  kv=(); IFS=' ' read -r -a kv <<< "$(knobs "$cfg") $extra"
  start=$(date -u +%FT%TZ)
  echo "[ovl] $start START $id ($kind, $cfg)"
  case "$kind" in
    kv|wall)
      env "${kv[@]}" RUN_ID="$id" EXP=OVL LAYER_SET=L0_6 "$TT_METAL_HOME/m3_budget_study/run_budget.sh"
      status="$(grep -oE '^STATUS=[A-Z_]+' "$log" | tail -1 | cut -d= -f2)" ;;
    kvg)
      env "${kv[@]}" RUN_ID="$id" RUN_TIMEOUT="$RUN_TIMEOUT" STALL_TIMEOUT="$STALL_TIMEOUT" "$RES/tools/run_kv_4x2.sh"
      status="$(grep -oE '\[run_kv\] [^ ]+ status=[^ ]+' "$log" | tail -1 | sed 's/.*status=//')" ;;
    prof)
      # batch_p0a.sh runs the point under its own lock / busy check, reset and watchdog; its log is logs/$id.log.
      env "${kv[@]}" PREFIX="ovl_$cfg" ONLY=w4096_h141312_prose RUNS_CSV="$OUT/profile_runs.csv" \
        PER_OP_CSV="$OUT/per_op.csv" "$RES/batch_p0a.sh"
      if grep -q '\[zone-prof\] DONE' "$log" 2>/dev/null; then status=OK; else status=FAIL; fi ;;
  esac
  end=$(date -u +%FT%TZ)
  echo "[ovl] $end END $id status=${status:-?}"
  [ "${status:-}" = OK ] && post "$kind" "$cfg" "$id"
  envf="$RES/logs/$id.env"
  envline="$(grep -vE '^(run_id|git_sha|dirty|date)=' "$envf" 2>/dev/null | sort | tr '\n' ' ' | sed 's/ $//')"
  sha="$(grep '^git_sha=' "$envf" 2>/dev/null | cut -d= -f2)"
  dirty="$(git -C "$TT_METAL_HOME" status --porcelain -uno | awk '{print $2}' | tr '\n' ' ' | sed 's/ $//')"
  [ -n "$dirty" ] && sha="$sha dirty($(echo "$dirty" | wc -w): $dirty)"
  "$PY" - "$RES/runs.csv" "$id" "$start" "$end" "$sha" "$envline" \
    "KINDS=$kind CONFIGS=$cfg ONLY=$id m3_budget_study/results_ops/batch_overlap.sh" "STATUS=${status:-?}" \
    "m3_budget_study/results_ops/logs/$id.log" "OVL $kind $cfg: $(knobs "$cfg")" <<'PY'
import csv, sys
out, rid, start, end, sha, env, cmd, status, log, notes = sys.argv[1:]
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([rid, "OVL", start, end, sha, env, cmd, status, log, notes])
PY
done
echo "[ovl] $(date -u +%FT%TZ) done; results in $OUT"
