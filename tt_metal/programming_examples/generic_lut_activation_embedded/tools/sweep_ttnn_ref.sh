#!/usr/bin/env bash
# Generate a TTNN native reference CSV for a dtype without depending on generated
# coefficient CSVs. The activation set comes from --activations, --frontier-csv,
# --best-csv, or the fitter activation JSON directory, and each activation's
# range comes from its activation JSON domain.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
WORK_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation_embedded"
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
VENV="$REPO_ROOT/python_env/bin/activate"
SYSPY="/usr/bin/python3"

source "$WORK_DIR/profiler_helpers.sh"

PRECISION="fp32"; TILES=256; OUT=""; BEST_CSV=""; FRONTIER_CSV=""; ACT_FILTER=""; RUN_DIR=""
SHARD=0; NUM_SHARDS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precision|-p)   PRECISION="$2"; shift 2 ;;
    --tiles|-t)       TILES="$2"; shift 2 ;;
    --run-dir)        RUN_DIR="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    --best-csv)       BEST_CSV="$2"; shift 2 ;;
    --frontier-csv)   FRONTIER_CSV="$2"; shift 2 ;;
    --activations|-a) ACT_FILTER="$2"; shift 2 ;;
    --shard)          SHARD="$2"; shift 2 ;;
    --num-shards)     NUM_SHARDS="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "$PRECISION" == "bf16" || "$PRECISION" == "fp32" ]] || { echo "ERROR: --precision must be bf16 or fp32" >&2; exit 1; }
[[ "$NUM_SHARDS" -lt 1 || "$SHARD" -lt 0 || "$SHARD" -ge "$NUM_SHARDS" ]] && { echo "ERROR: invalid shard $SHARD/$NUM_SHARDS" >&2; exit 1; }
[[ -f "$VENV" ]] || { echo "ERROR: python_env missing: $VENV" >&2; exit 1; }
[[ -f "$ACCURACY_SCRIPT" ]] || { echo "ERROR: extract_accuracy.py missing; set TT_POLY_FIT_DIR" >&2; exit 1; }
if [[ -n "$RUN_DIR" && -z "$OUT" ]]; then
  OUT="$RUN_DIR/data/csv/ttnn_ref_chip${SHARD}.csv"
fi
[[ -z "$OUT" ]] && OUT="$WORK_DIR/results/frontier/${PRECISION}/data/csv/ttnn_ref_chip${SHARD}.csv"
mkdir -p "$(dirname "$OUT")"
if [[ ! -f "$OUT" ]]; then
  echo "activation,dtype,ttnn_maxulp,ttnn_meanulp,ttnn_us,ttnn_pure,ttnn_ml,range_min,range_max,tiles,status" > "$OUT"
fi

WL="$(mktemp /tmp/ttnn_ref_wl_XXXX.txt)"
NATIVE_PY="$(mktemp /tmp/ttnn_ref_native_XXXX.py)"
trap 'rm -f "$WL" "$NATIVE_PY"' EXIT

python3 - "$TT_POLY_FIT_DIR" "$PRECISION" "$ACT_FILTER" "$BEST_CSV" "$FRONTIER_CSV" > "$WL" <<'PY'
import csv, glob, json, os, sys
fit, precision, act_filter, best_csv, frontier_csv = sys.argv[1:6]
acts = []
if act_filter:
    acts = [a for a in act_filter.split(",") if a]
elif frontier_csv:
    seen = set()
    for path in glob.glob(frontier_csv) or [frontier_csv]:
        with open(path) as f:
            for row in csv.DictReader(f):
                if (row.get("dtype") or row.get("precision") or "").lower() != precision:
                    continue
                act = row.get("activation")
                if act and act not in seen:
                    seen.add(act); acts.append(act)
elif best_csv:
    with open(best_csv) as f:
        for row in csv.DictReader(f):
            if row.get("precision") == precision:
                acts.append(row["activation"])
else:
    for path in sorted(glob.glob(os.path.join(fit, "activations", "*.json"))):
        acts.append(os.path.splitext(os.path.basename(path))[0])
seen = set()
for act in acts:
    if act in seen:
        continue
    seen.add(act)
    jpath = os.path.join(fit, "activations", f"{act}.json")
    if not os.path.exists(jpath):
        print(f"{act},,,,", file=sys.stderr)
        continue
    cfg = json.load(open(jpath))
    domain = (cfg.get("domain") or {})
    lo, hi = domain.get("min"), domain.get("max")
    if lo is None or hi is None:
        print(f"{act},,,,", file=sys.stderr)
        continue
    op_name = cfg.get("op_name") or act
    p = (cfg.get("target_parameters") or {}).get("p", "")
    print(f"{act},{op_name},{p},{lo},{hi}")
PY

cat > "$NATIVE_PY" <<'PY'
import sys, torch, ttnn, numpy as np
act, op_name, prec, lo, hi, out_csv, out_npz, tiles = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7], int(sys.argv[8])
dev = ttnn.open_device(device_id=0)
try:
    fn = getattr(ttnn, op_name)
    is_bf16 = prec == "bf16"
    dt_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32
    dt_t = torch.bfloat16 if is_bf16 else torch.float32
    if is_bf16:
        bits = np.arange(65536, dtype=np.uint16)
        vals = np.frombuffer((bits.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
        m = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
        x = torch.from_numpy(np.sort(vals[m])).bfloat16()
    else:
        x = torch.linspace(lo, hi, 262144, dtype=torch.float32)
    n = len(x)
    if n:
        pad = ((n + 1023) // 1024) * 1024
        xp = torch.zeros(pad, dtype=dt_t)
        xp[:n] = x
        xt = ttnn.from_torch(xp.reshape(1, 1, 1, -1), device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
        hw = ttnn.to_torch(fn(xt)).squeeze().float().numpy()[:n].astype(np.float32)
        xn = x.float().numpy().astype(np.float32)
    else:
        xn = np.array([], dtype=np.float32)
        hw = np.array([], dtype=np.float32)
    if out_csv:
        with open(out_csv, "w") as f:
            f.write("input,output\n")
            for i in range(n):
                f.write(f"{xn[i]},{hw[i]}\n")
    if out_npz:
        np.savez_compressed(out_npz, input=xn, output=hw)
    width = 32 * tiles
    xt_t = ttnn.from_torch(torch.rand(1, 1, 32, width, dtype=dt_t) * (hi - lo) + lo, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
    for _ in range(3):
        fn(xt_t)
    ttnn.synchronize_device(dev)
    fn(xt_t)
    ttnn.synchronize_device(dev)
finally:
    ttnn.close_device(dev)
PY

total="$(grep -c . "$WL")"
echo "=== ttnn-ref sweep precision=$PRECISION shard=$SHARD/$NUM_SHARDS total=$total chip=${TT_VISIBLE_DEVICES:-unset} ==="
n=0
while IFS=, read -r act op_name target_p lo hi; do
  [[ -z "$act" ]] && continue
  idx="$n"; n=$((n + 1))
  [[ $(( idx % NUM_SHARDS )) -eq "$SHARD" ]] || continue
  if grep -q "^${act},${PRECISION}," "$OUT" 2>/dev/null; then
    continue
  fi
  if [[ -z "$lo" || -z "$hi" ]]; then
    printf '%s,%s,,,,,,%s,%s,%s,%s\n' "$act" "$PRECISION" "$lo" "$hi" "$TILES" "no_domain" >> "$OUT"
    continue
  fi
  if [[ "$op_name" == "multigammaln" && -n "$target_p" && "$target_p" != "4" ]]; then
    printf '%s,%s,,,,,,%s,%s,%s,no_ttnn_ref_p_%s\n' "$act" "$PRECISION" "$lo" "$hi" "$TILES" "$target_p" >> "$OUT"
    echo "$act $PRECISION no_ttnn_ref_p_$target_p" >&2
    continue
  fi
  if [[ -n "$RUN_DIR" ]]; then
    dump="$RUN_DIR/data/dumps/ttnn/${PRECISION}/${act}/ttnn.npz"
    mkdir -p "$(dirname "$dump")"
    acc_dump="$RUN_DIR/data/tmp/ttnn_ref_${act}_chip${SHARD}.csv"
  else
    dump="$WORK_DIR/results/frontier/${PRECISION}/data/tmp/ttnn_ref_${act}_chip${SHARD}.csv"
    acc_dump="$dump"
  fi
  mkdir -p "$(dirname "$acc_dump")" "$(dirname "$dump")"
  prof_dir="$WORK_DIR/results/frontier/${PRECISION}/data/tmp/profiler/ttnn_ref_${act}_chip${SHARD}"
  rm -rf "$prof_dir"; mkdir -p "$prof_dir"
  set +e
  ( source "$VENV"; TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$prof_dir" \
      python3 "$NATIVE_PY" "$act" "$op_name" "$PRECISION" "$lo" "$hi" "$acc_dump" "$([[ -n "$RUN_DIR" ]] && echo "$dump")" "$TILES" >/dev/null 2>&1 )
  rc=$?
  set +e
  if [[ "$rc" -ne 0 || ! -s "$dump" ]]; then
    printf '%s,%s,,,,,,%s,%s,%s,native_fail_rc_%s\n' "$act" "$PRECISION" "$lo" "$hi" "$TILES" "$rc" >> "$OUT"
    echo "$act $PRECISION native_fail_rc_$rc" >&2
    continue
  fi
  acc="$("$SYSPY" "$ACCURACY_SCRIPT" "$act" "$acc_dump" 2>/dev/null | tail -1)"
  maxulp="$(echo "$acc" | cut -d, -f5)"
  meanulp="$(echo "$acc" | cut -d, -f6)"
  pure="$(echo "$acc" | cut -d, -f9)"
  ml="$(echo "$acc" | cut -d, -f11)"
  pcsv="$prof_dir/.logs/profile_log_device.csv"
  us=""
  [[ -f "$pcsv" ]] && us="$(extract_profiler_compute_time "$pcsv" "$WORK_DIR")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,pass\n' \
    "$act" "$PRECISION" "$maxulp" "$meanulp" "$us" "$pure" "$ml" "$lo" "$hi" "$TILES" >> "$OUT"
  echo "$act $PRECISION MaxULP=$maxulp us=$us"
done < "$WL"
