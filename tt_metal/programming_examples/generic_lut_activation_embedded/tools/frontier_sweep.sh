#!/bin/bash
# =============================================================================
# frontier_sweep.sh — EXHAUSTIVE ULP-vs-runtime frontier sweep (shardable).
#
# Measures, for every coefficient CSV in the fitter corpus (or a chosen subset),
# the on-silicon (MaxULP, runtime µs, compiles?) triple — the raw
# matrix for a ULP-vs-runtime Pareto scatter. Each config is run through the
# embedded flow (run_csv.sh): JIT-compile + Tracy-timed run over the activation's
# JSON deployment domain. Compile failures (register overflow on high-degree /
# compose exponent_alu) are recorded as compiles=0 — NOT crashes — thanks to
# run_csv's surfaced stderr.
#
# SHARDING (4-chip QuietBox): the device-exclusivity rule is PER CHIP, so run N
# instances, one per chip. BUT the embedded flow regenerates a SHARED adhoc.cpp,
# so each worker MUST run in its OWN tt-metal checkout. Local dispatch handles
# that by creating/reusing detached worktrees:
#
#   frontier_sweep.sh --dispatch-local 4 --precision bf16 --fresh
#
# It writes canonical shard CSVs back into the invoking checkout's
# results/frontier/<precision>/data/csv/ directory and configures missing
# per-worktree build_Release directories. Manual shard dispatch is still
# supported with --shard/--num-shards/--out/--cache. Use --fresh to replace
# shard CSVs; omit it to resume.
#
# TT_VISIBLE_DEVICES=C restricts UMD to physical chip C; the binary's device_id=0
# then maps to it. Each worker needs a distinct --out and --cache.
#
# RESUMABLE: re-running appends only configs not already in --out (keyed by CSV
# basename + precision), so a crash/restart picks up where it left off.
#
# Env: TT_METAL_HOME (auto-detected), TT_POLY_FIT_DIR (auto-detected).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$HERE/.." && pwd)"

# --- self-detect tt-metal + fitter (same contract as regenerate_table2.sh) ---
if [[ -z "${TT_METAL_HOME:-}" || ! -d "${TT_METAL_HOME}/tt_metal" ]]; then
  for c in "${TT_METAL_HOME:-}" "$(cd "$WORK_DIR/../../.." && pwd)" "/localdev/$USER/tt-metal" "$HOME/tt-metal"; do
    [[ -n "$c" && -x "$c/tt_metal/programming_examples/generic_lut_activation_embedded/run_csv.sh" ]] && { TT_METAL_HOME="$(cd "$c" && pwd)"; break; }
  done
fi
if [[ -z "${TT_POLY_FIT_DIR:-}" ]]; then
  for c in "$TT_METAL_HOME/../tt-polynomial-fitter" "$HOME/tt-polynomial-fitter" "$HOME/workspace/tt-polynomial-fitter" "/localdev/$USER/tt-polynomial-fitter"; do
    [[ -d "$c/data/coefficients" ]] && { TT_POLY_FIT_DIR="$(cd "$c" && pwd)"; break; }
  done
fi
FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
export TT_METAL_HOME TT_POLY_FIT_DIR="$FIT_DIR"
RUN_CSV="$TT_METAL_HOME/tt_metal/programming_examples/generic_lut_activation_embedded/run_csv.sh"
COEFFS="$FIT_DIR/data/coefficients"
ACTS="$FIT_DIR/activations"

SHARD=0 NUM_SHARDS=1 OUT="" CACHE="" FILTER="" PRECISION="bf16" PER_CFG_TIMEOUT=240 RUN_DIR="" FRESH=0 GBDT_OUT=""
DISPATCH_LOCAL=0 WORKTREE_PREFIX=""
while [[ $# -gt 0 ]]; do case "$1" in
  --shard) SHARD="$2"; shift 2 ;;
  --num-shards) NUM_SHARDS="$2"; shift 2 ;;
  --dispatch-local) DISPATCH_LOCAL="$2"; shift 2 ;;
  --worktree-prefix) WORKTREE_PREFIX="$2"; shift 2 ;;
  --run-dir) RUN_DIR="$2"; shift 2 ;;
  --out) OUT="$2"; shift 2 ;;
  --gbdt-out) GBDT_OUT="$2"; shift 2 ;;
  --cache) CACHE="$2"; shift 2 ;;
  --fresh) FRESH=1; shift ;;
  --activations) FILTER="$2"; shift 2 ;;   # comma or space separated; default = all
  --precision|-p) PRECISION="$2"; shift 2 ;;
  --timeout) PER_CFG_TIMEOUT="$2"; shift 2 ;;
  -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
  *) echo "Unknown arg: $1" >&2; exit 1 ;;
esac; done
case "$PRECISION" in
  bf16) PRECISIONS=(bf16) ;;
  fp32) PRECISIONS=(fp32) ;;
  both) PRECISIONS=(bf16 fp32) ;;
  *) echo "ERROR: --precision must be bf16, fp32, or both" >&2; exit 1 ;;
esac

dispatch_local() {
  local n="$1"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --dispatch-local must be a positive integer" >&2; exit 1; }
  [[ -z "$OUT" ]] || { echo "ERROR: --dispatch-local writes per-shard outputs; use --run-dir, not --out" >&2; exit 1; }
  [[ -x "$RUN_CSV" ]] || { echo "ERROR: run_csv.sh not found at $RUN_CSV (set TT_METAL_HOME)" >&2; exit 1; }
  [[ -d "$COEFFS" ]] || { echo "ERROR: corpus not found at $COEFFS (set TT_POLY_FIT_DIR)" >&2; exit 1; }

  local repo_root parent base wt script out cache log_dir log pid rc failures=0
  local -a worker_args
  repo_root="$(cd "$TT_METAL_HOME" && pwd)"
  parent="$(dirname "$repo_root")"
  base="$(basename "$repo_root")"
  [[ -n "$WORKTREE_PREFIX" ]] || WORKTREE_PREFIX="$parent/${base}-chip"

  if [[ -n "$RUN_DIR" ]]; then
    log_dir="$RUN_DIR/logs/dispatch"
    mkdir -p "$RUN_DIR/data/csv" "$log_dir"
  else
    log_dir="$WORK_DIR/results/frontier/${PRECISION}/logs/dispatch"
    mkdir -p "$WORK_DIR/results/frontier/${PRECISION}/data/csv" "$log_dir"
  fi

  echo "frontier_sweep dispatch: ${n} local worktrees, precision=${PRECISION}, filter='${FILTER:-all}'" >&2
  for ((chip=0; chip<n; chip++)); do
    wt="${WORKTREE_PREFIX}${chip}"
    if [[ ! -e "$wt/.git" ]]; then
      git -C "$repo_root" worktree add --detach "$wt" HEAD >&2
    fi
    script="$wt/tt_metal/programming_examples/generic_lut_activation_embedded/tools/frontier_sweep.sh"
    [[ -x "$script" ]] || { echo "ERROR: worktree sweep script missing/executable: $script" >&2; exit 1; }
    if [[ ! -f "$wt/build_Release/build.ninja" ]]; then
      echo "  chip ${chip}: configuring missing build_Release in ${wt}" >&2
      (
        cd "$wt" &&
        git submodule update --init --recursive &&
        ./build_metal.sh \
          --build-programming-examples \
          --configure-only \
          --without-python-bindings \
          --disable-warnings-as-errors \
          --disable-unity-builds \
          --enable-ccache \
          --cpm-source-cache "$repo_root/.cpmcache"
      ) >"$log_dir/configure_chip${chip}.log" 2>&1 || {
        echo "ERROR: failed to configure ${wt}/build_Release; see $log_dir/configure_chip${chip}.log" >&2
        exit 1
      }
    fi
    if [[ ! -x "$wt/build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc" ]]; then
      echo "  chip ${chip}: prebuilding adhoc target in ${wt}" >&2
      ninja -C "$wt/build_Release" programming_examples_generic_lut_activation_embedded_adhoc \
        >"$log_dir/prebuild_chip${chip}.log" 2>&1 || {
        echo "ERROR: failed to prebuild ${wt}; see $log_dir/prebuild_chip${chip}.log" >&2
        exit 1
      }
    fi
  done

  pids=()
  for ((chip=0; chip<n; chip++)); do
    wt="${WORKTREE_PREFIX}${chip}"
    script="$wt/tt_metal/programming_examples/generic_lut_activation_embedded/tools/frontier_sweep.sh"
    if [[ -n "$RUN_DIR" ]]; then
      out="$RUN_DIR/data/csv/frontier_chip${chip}.csv"
    else
      out="$WORK_DIR/results/frontier/${PRECISION}/data/csv/frontier_chip${chip}.csv"
    fi
    cache="${CACHE:-/tmp/tt-metal-cache-frontier}-${PRECISION}-${chip}"
    log="$log_dir/frontier_chip${chip}.log"
    worker_args=(
      --shard "$chip"
      --num-shards "$n"
      --precision "$PRECISION"
      --timeout "$PER_CFG_TIMEOUT"
      --out "$out"
      --cache "$cache"
    )
    [[ "$FRESH" -eq 0 ]] || worker_args+=(--fresh)
    [[ -z "$FILTER" ]] || worker_args+=(--activations "$FILTER")
    (
      cd "$wt" &&
      TT_VISIBLE_DEVICES="$chip" \
      TT_METAL_HOME="$wt" \
      TT_POLY_FIT_DIR="$FIT_DIR" \
      bash "$script" "${worker_args[@]}"
    ) >"$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "  chip ${chip}: pid=${pid} worktree=${wt} out=${out} cache=${cache} log=${log}" >&2
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=$?
      echo "frontier_sweep dispatch: worker pid=${pid} failed with rc=${rc}" >&2
      failures=$((failures+1))
    fi
  done
  [[ "$failures" -eq 0 ]] || exit 1
  echo "frontier_sweep dispatch DONE: ${n} workers completed" >&2
  local gbdt_out
  if [[ -n "$GBDT_OUT" ]]; then
    gbdt_out="$GBDT_OUT"
  elif [[ -n "$RUN_DIR" ]]; then
    gbdt_out="$RUN_DIR/data/csv/frontier_gbdt_training.tsv"
  else
    gbdt_out="$WORK_DIR/results/frontier/${PRECISION}/data/csv/frontier_gbdt_training.tsv"
  fi
  local -a gbdt_shards
  local gbdt_root
  gbdt_root="$([[ -n "$RUN_DIR" ]] && printf '%s' "$RUN_DIR" || printf '%s' "$WORK_DIR/results/frontier/${PRECISION}")"
  gbdt_shards=()
  for ((chip=0; chip<n; chip++)); do
    gbdt_shards+=("$gbdt_root/data/csv/frontier_chip${chip}.csv")
  done
  python3 "$WORK_DIR/tools/frontier_gbdt_dataset.py" \
    "${gbdt_shards[@]}" \
    --coeff-dir "$COEFFS" \
    --out "$gbdt_out" >&2
  echo "frontier_sweep dispatch GBDT table -> ${gbdt_out}" >&2
}

if [[ "$DISPATCH_LOCAL" -gt 0 ]]; then
  dispatch_local "$DISPATCH_LOCAL"
  exit 0
fi

if [[ -n "$RUN_DIR" && -z "$OUT" ]]; then
  OUT="$RUN_DIR/data/csv/frontier_chip${SHARD}.csv"
fi
FILTER="${FILTER//,/ }"
[[ -z "$OUT" ]] && OUT="$WORK_DIR/results/frontier/${PRECISION}/data/csv/frontier_chip${SHARD}.csv"
[[ -z "$CACHE" ]] && CACHE="/tmp/tt-metal-cache-frontier-${SHARD}"
[[ -x "$RUN_CSV" ]] || { echo "ERROR: run_csv.sh not found at $RUN_CSV (set TT_METAL_HOME)" >&2; exit 1; }
[[ -d "$COEFFS" ]] || { echo "ERROR: corpus not found at $COEFFS (set TT_POLY_FIT_DIR)" >&2; exit 1; }

build_worklist() {
  PYTHONPATH="$FIT_DIR${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 - "$COEFFS" "$FILTER" <<'PY'
import sys
from pathlib import Path

try:
    from ttpoly.spec.csv_io import parse_csv_artifact
except ImportError as exc:
    raise SystemExit(
        "ERROR: tt-polynomial-fitter must expose ttpoly.spec.csv_io.parse_csv_artifact(path)"
    ) from exc

coeffs = Path(sys.argv[1])
filters = {token for token in sys.argv[2].split() if token}
required = ("activation", "eval_method", "degree", "depth", "metric")

for path in sorted(coeffs.glob("*.csv")):
    identity = parse_csv_artifact(path)
    missing = [key for key in required if identity.get(key) in (None, "")]
    if missing:
        raise SystemExit(
            "ERROR: ttpoly parse_csv_artifact(path) must return structured artifact "
            f"identity field(s) {', '.join(missing)} for {path}"
        )

    activation = str(identity["activation"])
    if filters and activation not in filters:
        continue

    degree = str(identity["degree"]).replace("/", "d")
    print("\t".join([
        str(path),
        activation,
        str(identity["eval_method"]),
        degree,
        str(identity["depth"]),
        str(identity["metric"]),
    ]))
PY
}

# build the work list (filtered + sharded + deterministic order)
mapfile -t WORK < <(build_worklist)

export TT_METAL_CACHE="$CACHE"   # per-worker JIT cache (isolation)
mkdir -p "$(dirname "$OUT")"
if [[ "$FRESH" -eq 1 || ! -f "$OUT" ]]; then
  echo "csv,activation,method,degree,segments,metric,precision,bf16_maxulp,runtime_us,compiles,range,target_runtime_us,target_runtime_ns,target_cycles_1350mhz,tile_count,range_min,range_max,range_width,num_degree,den_degree,is_rational,is_polynomial,is_lowering,has_range_reduction,coeff_count,nonzero_coeff_count,max_abs_coeff,sum_abs_coeff,avg_effective_degree,max_effective_degree" > "$OUT"
fi

echo "frontier_sweep: shard $SHARD/$NUM_SHARDS, $(( ${#WORK[@]} )) candidate configs (precision=$PRECISION, filter='${FILTER:-all}'), chip TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-unset}, cache=$CACHE, out=$OUT" >&2

i=-1; done_n=0; new_n=0
for work_item in "${WORK[@]}"; do
  i=$((i+1)); [[ $(( i % NUM_SHARDS )) -eq "$SHARD" ]] || continue
  IFS=$'\t' read -r f act method deg segs metric <<< "$work_item"
  base="$(basename "$f")"
  # JSON deployment domain (the harness's source of truth)
  read -r lo hi < <(/usr/bin/python3 -c "import json,sys; d=(json.load(open('$ACTS/$act.json')).get('domain') or {}); print(d.get('min',''), d.get('max',''))" 2>/dev/null)
  RA=(); [[ "$lo" =~ ^-?[0-9] && "$hi" =~ ^-?[0-9] ]] && RA=(--range-min "$lo" --range-max "$hi")
  for prec in "${PRECISIONS[@]}"; do
    grep -q "^${base},[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,${prec}," "$OUT" 2>/dev/null && { done_n=$((done_n+1)); continue; }   # resume: skip done
    out=$(timeout "$PER_CFG_TIMEOUT" bash "$RUN_CSV" "$f" --activation "$act" --precision "$prec" --tiles 256 "${RA[@]}" 2>&1)
    row=$(echo "$out" | grep -aE '^(256_tiles|custom_256t)' | tail -1)
    if [[ -n "$row" ]]; then
      ulp=$(echo "$row" | awk '{print $5}')
      us=$(echo "$row" | awk '{print $7}' | sed 's/µs//')
      ok=1
      if [[ ! "$ulp" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ || ! "$us" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        ulp="-"
        us="-"
        ok=0
      fi
    else
      ulp="-"; us="-"; ok=0
      if [[ -n "$RUN_DIR" ]]; then
        fail_dir="$RUN_DIR/logs/failure/shard${SHARD}"
      else
        fail_dir="$(dirname "$OUT")/failure_logs/shard${SHARD}"
      fi
      mkdir -p "$fail_dir"
      printf '%s\n' "$out" > "$fail_dir/${prec}_${base}.log"
    fi
    gbdt_features=$(PYTHONPATH="$FIT_DIR${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 - "$f" "$us" "$ok" "$lo" "$hi" <<'PY'
import csv
import math
import sys

try:
    from ttpoly.spec.csv_io import parse_csv_artifact
except Exception:  # pragma: no cover - frontier worker prints the hard failure elsewhere.
    parse_csv_artifact = None

path, runtime_us_s, ok_s, lo_s, hi_s = sys.argv[1:6]

def fnum(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except Exception:
        return None

identity = parse_csv_artifact(path) if parse_csv_artifact else {}
num_degree = identity.get("num_degree", "")
den_degree = identity.get("den_degree", "")
approx_type = str(identity.get("approximation_type", "")).lower()
rr = str(identity.get("range_reduction_method", "") or "none").lower()
method = str(identity.get("eval_method", "") or "")

coeff_count = nonzero = 0
sum_abs = 0.0
max_abs = 0.0
effective = []
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    coeff_cols = [
        c for c in fields
        if (len(c) > 1 and c[0] in ("c", "n", "d") and c[1:].isdigit())
    ]
    for row in reader:
        if str(row.get("segment_id", "")).upper() == "METADATA":
            continue
        seg_eff = 0
        for col in coeff_cols:
            v = fnum(row.get(col))
            if v is None:
                continue
            coeff_count += 1
            av = abs(v)
            sum_abs += av
            max_abs = max(max_abs, av)
            if av != 0.0:
                nonzero += 1
                try:
                    seg_eff = max(seg_eff, int(col[1:]))
                except ValueError:
                    pass
        effective.append(seg_eff)

runtime_us = fnum(runtime_us_s)
target_us = runtime_us if ok_s == "1" and runtime_us is not None else ""
lo = fnum(lo_s)
hi = fnum(hi_s)
width = (hi - lo) if lo is not None and hi is not None else ""
avg_eff = (sum(effective) / len(effective)) if effective else ""
max_eff = max(effective) if effective else ""
is_rational = 1 if (approx_type == "rational" or den_degree not in ("", None, 0, "0")) else 0
is_poly = 1 if not is_rational else 0
is_lowering = 1 if method in {"identity", "clamped_affine", "threshold_identity", "algebraic_lowering"} else 0
has_rr = 0 if rr in {"", "none"} else 1

vals = [
    target_us,
    (target_us * 1000.0) if target_us != "" else "",
    (target_us * 1350.0) if target_us != "" else "",
    256,
    lo if lo is not None else "",
    hi if hi is not None else "",
    width,
    num_degree,
    den_degree,
    is_rational,
    is_poly,
    is_lowering,
    has_rr,
    coeff_count,
    nonzero,
    max_abs,
    sum_abs,
    avg_eff,
    max_eff,
]
print(",".join("" if v == "" else f"{float(v):.12g}" if isinstance(v, float) else str(v) for v in vals))
PY
)
    echo "${base},${act},${method},${deg},${segs},${metric},${prec},${ulp},${us},${ok},\"[${lo},${hi}]\",${gbdt_features}" >> "$OUT"
    new_n=$((new_n+1))
    [[ $(( new_n % 25 )) -eq 0 ]] && echo "  [shard $SHARD] $new_n new (${done_n} resumed-skip) — last: $act $base precision=$prec ulp=$ulp us=$us ok=$ok" >&2
  done
done
echo "frontier_sweep shard $SHARD DONE: $new_n measured, $done_n already-present. -> $OUT" >&2
