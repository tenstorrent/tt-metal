#!/usr/bin/env bash
# Generate a Qwen3-TTS single-block Tracy report (the qwen3_tts_n300_blocks.txt format).
#
# One Tracy capture per -k selector, repeated -n times; the report kept for each block is
# the run whose device-time total is the median. Repeats matter on N300: the CCL ops
# (reduce_scatter / all_gather) swing ~2x run to run at these payload sizes, so a single
# capture per block is not safe to compare against another single capture.
#
#   ./qwen3_tts_block_report.sh                                  # N300, 3 runs
#   ./qwen3_tts_block_report.sh -m N150 -o n150_blocks.txt
#   ./qwen3_tts_block_report.sh -b qwen3_tts_n300_blocks.txt     # add a vs-baseline column
#   ./qwen3_tts_block_report.sh -A -d /tmp/qwen3_blocks          # re-assemble, no capture
#
# Run from the tt-metal repo root. ~40 s per capture, so the default is ~16 minutes.
set -euo pipefail

MESH=N300
RUNS=3
OUT=""
BASELINE=""
WORK=""
ASSEMBLE_ONLY=0

usage() { sed -n '2,14p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

while getopts "m:n:o:b:d:Ah" opt; do
  case $opt in
    m) MESH=$OPTARG ;;
    n) RUNS=$OPTARG ;;
    o) OUT=$OPTARG ;;
    b) BASELINE=$OPTARG ;;
    d) WORK=$OPTARG ;;
    A) ASSEMBLE_ONLY=1 ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

: "${OUT:=qwen3_tts_$(echo "$MESH" | tr 'A-Z' 'a-z')_blocks.txt}"
: "${WORK:=generated/qwen3_tts_block_report/$(echo "$MESH" | tr 'A-Z' 'a-z')}"

TEST=models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py
[[ -f $TEST ]] || { echo "error: run this from the tt-metal repo root ($TEST not found)" >&2; exit 1; }

# -k selectors, in report order. Use the full test name: -k talker_layer_prefill would
# match all three prefill buckets and put three windows in one capture.
SELECTORS=(
  test_talker_layer_prefill_32
  test_talker_layer_prefill_64
  test_talker_layer_prefill_128
  test_talker_layer_decode
  test_cp_layer_prefill
  test_cp_layer_decode
  test_speaker_tdnn
  test_speaker_block
)

mkdir -p "$WORK"

if [[ $ASSEMBLE_ONLY -eq 0 ]]; then
  [[ -n ${VIRTUAL_ENV:-} ]] || { echo "error: activate the venv first: source python_env/bin/activate" >&2; exit 1; }
  export TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
  export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
  export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
  export MESH_DEVICE="$MESH"

  total=$(( ${#SELECTORS[@]} * RUNS )); done_n=0
  for sel in "${SELECTORS[@]}"; do
    for ((i = 1; i <= RUNS; i++)); do
      done_n=$((done_n + 1))
      printf '[%2d/%2d] %s run %d ... ' "$done_n" "$total" "$sel" "$i"
      if ! python -m tracy -p -v -r -m pytest -s -q "$TEST" -k "$sel" \
             > "$WORK/run_${sel}_$i.log" 2>&1; then
        echo "FAILED (see $WORK/run_${sel}_$i.log)"; exit 1
      fi
      csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
      cp "$csv" "$WORK/${sel}_$i.csv"
      tt-perf-report --start-signpost start --end-signpost stop "$csv" \
        > "$WORK/${sel}_$i.rpt" 2>/dev/null
      echo "$(grep -E '^ +100\.0 %' "$WORK/${sel}_$i.rpt" | head -1 | grep -oE '[0-9,]+ μs' | head -1)"
    done
  done
fi

MESH="$MESH" RUNS="$RUNS" OUT="$OUT" WORK="$WORK" BASELINE="$BASELINE" \
SELECTORS="${SELECTORS[*]}" python3 - <<'PY'
import os, pathlib, re, statistics

MESH, WORK, OUT = os.environ["MESH"], pathlib.Path(os.environ["WORK"]), pathlib.Path(os.environ["OUT"])
RUNS, BASELINE = int(os.environ["RUNS"]), os.environ.get("BASELINE", "")
SELECTORS = os.environ["SELECTORS"].split()

TITLES = {
    "test_talker_layer_prefill_32": "Talker DecoderLayer prefill seq=32 (DRAM-sharded QKV, M=1 tile)",
    "test_talker_layer_prefill_64": "Talker DecoderLayer prefill seq=64 (interleaved QKV, M=2 tiles; JA demo bucket)",
    "test_talker_layer_prefill_128": "Talker DecoderLayer prefill seq=128 (interleaved QKV, M=4 tiles)",
    "test_talker_layer_decode": "Talker DecoderLayer decode seq=1",
    "test_cp_layer_prefill": "CodePredictor layer prefill seq=2",
    "test_cp_layer_decode": "CodePredictor layer decode seq=1 (start_pos=2)",
    "test_speaker_tdnn": "SpeakerEncoder entry TDNN 128→512, T=384 (host conv + device relu)",
    "test_speaker_block": "SpeakerEncoder SERes2Net block_idx=1, T=384",
}
SUMMARY = re.compile(r"^\s+100\.0 %.*$", re.M)

def totals(sel):
    out = []
    for i in range(1, RUNS + 1):
        f = WORK / f"{sel}_{i}.rpt"
        if not f.exists():
            continue
        m = SUMMARY.search(f.read_text())
        if m:
            n = re.findall(r"([\d,]+) μs", m.group(0))
            if n:
                out.append((int(n[0].replace(",", "")), i))
    return out

# Optional baseline: pull each block's window total out of a previous report's Window totals.
base = {}
if BASELINE and pathlib.Path(BASELINE).exists():
    txt = pathlib.Path(BASELINE).read_text()
    for sel in SELECTORS:
        i = txt.find(f"  -k {sel}\n")
        if i < 0:
            continue
        m = SUMMARY.search(txt, i)
        if m and m.start() - i < 400:
            n = re.findall(r"([\d,]+) μs", m.group(0))
            if n:
                base[sel] = int(n[0].replace(",", ""))

mesh_line = "ttnn.open_mesh_device (1, 2), Fabric 1D" if MESH == "N300" else "ttnn.open_mesh_device (1, 1)"
L = [f"Qwen3-TTS single-block Tracy reports — MESH_DEVICE={MESH}"
     + (" (1x2, TP=2)" if MESH == "N300" else " (1x1, TP=1)") + "\n",
     "=" * 88 + "\n\n",
     "Source: models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py\n",
     f"Mesh:   MESH_DEVICE={MESH} → {mesh_line}\n",
     "Arch:   wormhole, 64 worker cores per chip\n",
     f"Note:   {RUNS} Tracy captures per -k; the report shown per block is the median-total run.\n",
     "        All run totals are listed so the spread is visible — N300 CCL ops swing ~2x\n",
     "        run to run, so a single capture is not safe to compare against another.\n",
     "        Device time is kernel time; large op-to-op gaps are host/dispatch (no Metal trace).\n",
     "        Speaker TDNN uses host F.conv1d; the window is mostly ttnn.relu.\n",
     f"        Generated by qwen3_tts_block_report.sh; raw captures in {WORK}/\n\n",
     "Window totals" + ("  (median of %d captures)" % RUNS if RUNS > 1 else "") + "\n",
     "-" * 88 + "\n"]

picked = []
for sel in SELECTORS:
    vals = totals(sel)
    L.append(TITLES.get(sel, sel) + "\n")
    L.append(f"  -k {sel}\n")
    if not vals:
        L.append("  (capture missing)\n\n")
        continue
    only = sorted(v for v, _ in vals)
    med = statistics.median_low(only)
    run = next(i for v, i in vals if v == med)
    picked.append((sel, run))
    L.append(SUMMARY.search((WORK / f"{sel}_{run}.rpt").read_text()).group(0).rstrip() + "\n")
    if RUNS > 1:
        line = "  runs: " + ", ".join(f"{v} μs" for v in only) + f"   median {med} μs"
        if sel in base:
            d = med - base[sel]
            line += f"   vs baseline {base[sel]} μs  ->  {d:+d} μs ({100.0*d/base[sel]:+.1f} %)"
        L.append(line + "\n")
    L.append("\n")

L.append("\n")
for sel, run in picked:
    L += ["=" * 88 + "\n", TITLES.get(sel, sel) + "\n",
          f"-k {sel}   MESH_DEVICE={MESH}"
          + (f"   (median capture, run {run} of {RUNS})\n" if RUNS > 1 else "\n"),
          "=" * 88 + "\n",
          (WORK / f"{sel}_{run}.rpt").read_text().rstrip() + "\n\n\n"]

OUT.write_text("".join(L))
print(f"\nwrote {OUT}  ({len(L)} chunks, {OUT.stat().st_size} bytes)")
PY
