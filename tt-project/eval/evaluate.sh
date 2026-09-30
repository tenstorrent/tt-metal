#!/usr/bin/env bash
# One-shot quality check of a candidate run against a reference run (both from gen_seeds.sh).
#
#   evaluate.sh REF_DIR CAND_DIR [OUT_DIR]     (FULL=1 for full VBench, SEEDS=0,1 to restrict)
#
# Writes OUT_DIR/compare.json, OUT_DIR/vbench_{ref,cand}.json, OUT_DIR/review_sheet.png and
# OUT_DIR/review_grid.mp4. VBench scores are cached per run dir, so the reference is scored once.
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n '2,8p' "$0"; exit 2; }
ref=$1 cand=$2 out=${3:-$2/eval}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py=/home/smarton/fasth3/tt-metal/.venv-eval/bin/python
mkdir -p "$out"
seeds=${SEEDS:+--seeds $SEEDS}
mode=partial; vb_args=()
[[ -n "${FULL:-}" ]] && { mode=full; vb_args+=(--full); }

"$py" "$here/compare.py" "$ref" "$cand" $seeds --json "$out/compare.json"
for side in ref cand; do
  dir=${!side}
  cache=$dir/vbench_${mode}.json
  if [[ ! -f $cache || -n "${SEEDS:-}" || -n $(find "$dir" -maxdepth 1 -name 'seed*.mp4' -newer "$cache") ]]; then
    videos=("$dir")
    [[ -n "${SEEDS:-}" ]] && videos=($(for s in ${SEEDS//,/ }; do echo "$dir/seed$s.mp4"; done)) && cache=$out/vbench_${mode}_${side}.json
    "$py" "$here/vbench_eval.py" "${videos[@]}" "${vb_args[@]}" --out "${TMPDIR:-/tmp}/fasth3_vbench" --json "$cache" \
      2>&1 | grep -E "^[a-z_]+ +[0-9.]+ |^total|^skip|Error" || true
  fi
  cp "$cache" "$out/vbench_$side.json"
done
"$py" - "$out/vbench_ref.json" "$out/vbench_cand.json" <<'PY'
import json, sys
ref, cand = (json.load(open(p)) for p in sys.argv[1:])
print(f"{'dimension':<24} {'ref':>8} {'cand':>8} {'delta':>8}")
for dim, score in cand["dims"].items():
    if dim in ref["dims"]:
        print(f"{dim:<24} {ref['dims'][dim]:8.4f} {score:8.4f} {score - ref['dims'][dim]:+8.4f}")
PY
"$py" "$here/review.py" "$ref" "$cand" --names ref,cand $seeds --out "$out/review"
