#!/usr/bin/env bash
# Generate the Qwen3-TTS prefill / decode per-op perf reports.
#
# One Tracy capture per window (see test_qwen3_tts_perf_report.py), each replaying
# the Metal trace the demo replays exactly once between `start` / `stop` signposts.
# Per window it writes, under models/demos/qwen3_tts/ops_list/perf_report/<window>/:
#
#   run.log             the pytest/tracy run, including the ms/replay median
#   ops.csv             the raw ops_perf_results CSV (every column, every op)
#   tt-perf-report.txt  tt-perf-report over the signpost window
#   ops_list.md         full per-op list + rollups (qwen3_tts_perf_report_opslist.py)
#
# and a summary.md tying the windows together.
#
#   ./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh              # all windows
#   ./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh -w prefill_demo,decode_frame
#   ./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh -m N150
#   ./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh -A          # re-assemble only
#
# Run from the tt-metal repo root with the venv active. ~2-4 min per window.
set -euo pipefail

MESH=${MESH_DEVICE:-N300}
WINDOWS=""
OUTDIR=""
ASSEMBLE_ONLY=0
# The profiler's DRAM buffer holds this many programs. Default is 1000; the AR frame
# is ~4200 device ops, and past the budget the device DROPS markers and the CSV comes
# back partial with no error. Keep it well above the largest window.
OP_SUPPORT_COUNT=${QWEN3_TTS_OP_SUPPORT_COUNT:-20000}

usage() { sed -n '2,20p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

while getopts "m:w:o:c:Ah" opt; do
  case $opt in
    m) MESH=$OPTARG ;;
    w) WINDOWS=$OPTARG ;;
    o) OUTDIR=$OPTARG ;;
    c) OP_SUPPORT_COUNT=$OPTARG ;;
    A) ASSEMBLE_ONLY=1 ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

TEST=models/demos/qwen3_tts/tests/test_qwen3_tts_perf_report.py
[[ -f $TEST ]] || { echo "error: run this from the tt-metal repo root ($TEST not found)" >&2; exit 1; }
: "${OUTDIR:=models/demos/qwen3_tts/ops_list/perf_report}"

# window name -> pytest -k selector
ALL_WINDOWS=(prefill_demo prefill_32 prefill_64 prefill_128 decode_talker decode_cp decode_frame)
selector_for() {
  case $1 in
    prefill_demo) echo "test_prefill[demo]" ;;
    prefill_32)   echo "test_prefill[32]" ;;
    prefill_64)   echo "test_prefill[64]" ;;
    prefill_128)  echo "test_prefill[128]" ;;
    decode_talker) echo "test_decode_talker" ;;
    decode_cp)     echo "test_decode_cp" ;;
    decode_frame)  echo "test_decode_frame" ;;
    *) echo "" ;;
  esac
}

if [[ -n $WINDOWS ]]; then
  IFS=',' read -r -a SELECTED <<< "$WINDOWS"
else
  SELECTED=("${ALL_WINDOWS[@]}")
fi
for w in "${SELECTED[@]}"; do
  [[ -n $(selector_for "$w") ]] || { echo "error: unknown window '$w' (have: ${ALL_WINDOWS[*]})" >&2; exit 1; }
done

mkdir -p "$OUTDIR"

if [[ $ASSEMBLE_ONLY -eq 0 ]]; then
  [[ -n ${VIRTUAL_ENV:-} ]] || { echo "error: activate the venv first: source python_env/bin/activate" >&2; exit 1; }
  export TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
  export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
  export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
  export MESH_DEVICE="$MESH"

  n=0
  for w in "${SELECTED[@]}"; do
    n=$((n + 1))
    sel=$(selector_for "$w")
    dir="$OUTDIR/$w"
    mkdir -p "$dir"
    printf '[%d/%d] %-14s %-22s ... ' "$n" "${#SELECTED[@]}" "$w" "$sel"

    # Pass 1, no profiler: wall clock, median of 10 traced replays. Separate on
    # purpose — under the device profiler every replay writes markers for every op on
    # every core, and ten of them bury the post-processing in gigabytes, for a number
    # this pass gets in 25 s.
    if ! python -m pytest -s -q "$TEST" -k "$sel" > "$dir/timing.log" 2>&1; then
      echo "FAILED timing pass (see $dir/timing.log)"; exit 1
    fi

    # Pass 2, under Tracy: exactly one replay between the signposts.
    if ! python -m tracy -p -v -r --op-support-count "$OP_SUPPORT_COUNT" \
           -m pytest -s -q "$TEST" -k "$sel" > "$dir/run.log" 2>&1; then
      echo "FAILED (see $dir/run.log)"; exit 1
    fi

    # A run that overflowed the profiler's DRAM buffer produces a PARTIAL csv and no
    # error. Refuse to publish that as a report.
    if grep -q "Profiler DRAM buffers were full" "$dir/run.log"; then
      echo "FAILED: profiler dropped markers — raise -c above $OP_SUPPORT_COUNT"; exit 1
    fi

    csv=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
    cp "$csv" "$dir/ops.csv"
    # tt-perf-report is a nice-to-have second view and is NOT allowed to fail the run:
    # it is a separately installed tool that trails the repo. It currently dies with
    # "Unknown math fidelity: HiFi3" on any window touching the CodePredictor, whose
    # matmuls default to HiFi3. ops_list.md is the primary artifact and reads the same CSV.
    if ! tt-perf-report --start-signpost start --end-signpost stop --no-stacked-report \
           "$dir/ops.csv" > "$dir/tt-perf-report.txt" 2>&1; then
      echo -n "(tt-perf-report failed: $(tail -1 "$dir/tt-perf-report.txt")) "
    fi
    # A capture with no signposts IS fatal — it means the window never opened.
    if grep -q "No signposts found" "$dir/tt-perf-report.txt"; then
      echo "FAILED: no signposts in the capture (see $dir/tt-perf-report.txt)"; exit 1
    fi
    OPSLIST=models/demos/qwen3_tts/tests/qwen3_tts_perf_report_opslist.py
    python3 "$OPSLIST" --window "$w" --json "$dir/totals.json" "$dir/ops.csv" > "$dir/ops_list.md"
    # decode_frame carries inner signposts; split the frame into its two halves too.
    if [[ $w == decode_frame ]]; then
      python3 "$OPSLIST" --window "decode_frame: CP half" \
        --start cp_frame_start --end cp_frame_stop \
        --json "$dir/totals_cp.json" "$dir/ops.csv" > "$dir/ops_list_cp.md"
      python3 "$OPSLIST" --window "decode_frame: Talker half" \
        --start talker_decode_start --end talker_decode_stop \
        --json "$dir/totals_talker.json" "$dir/ops.csv" > "$dir/ops_list_talker.md"
    fi
    grep -m1 '\[perf_report\]' "$dir/timing.log" | sed 's/^\[perf_report\] //' || echo "(no ms line)"
  done
fi

OUTDIR="$OUTDIR" MESH="$MESH" WINDOWS="${SELECTED[*]}" python3 - <<'PYSUM'
import json, os, pathlib, re

out = pathlib.Path(os.environ["OUTDIR"])
mesh = os.environ["MESH"]
windows = os.environ["WINDOWS"].split()
ms_re = re.compile(r"\[perf_report\] .*?([\d.]+) ms \(median of (\d+) traced replays\)")

lines = [
    f"# Qwen3-TTS prefill / decode perf report — MESH_DEVICE={mesh}\n\n",
    "Each window is one Tracy capture of a single Metal-trace replay — the same trace\n"
    "the demo replays — between `start` / `stop` signposts. Wall clock comes from a\n"
    "separate unprofiled pass, median of 10 replays.\n\n",
    "| window | wall clock | device kernel time | op-to-op gap | ops | report |\n",
    "|---|--:|--:|--:|--:|---|\n",
]
for w in windows:
    d = out / w
    ms = dev = gapms = nops = "-"
    tj = d / "totals.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        dev, gapms, nops = f"{t['device_ms']:.2f} ms", f"{t['gap_ms']:.2f} ms", str(t["ops"])
    log = d / "timing.log"
    if log.exists():
        m = ms_re.search(log.read_text())
        if m:
            ms = f"{float(m.group(1)):.2f} ms"
    lines.append(f"| `{w}` | {ms} | {dev} | {gapms} | {nops} | [{w}/ops_list.md]({w}/ops_list.md) |\n")

lines += [
    "\n## Reading it\n\n",
    "- **Quote the wall clock**, not device + gap. The two columns come from different\n"
    "  passes: wall clock is unprofiled, while device and gap are measured with the\n"
    "  device profiler writing a marker per op per core per RISC, which costs real time\n"
    "  on the chip. Device + gap therefore runs a little ABOVE the unprofiled wall clock\n"
    "  (Talker decode: 17.6 ms against 16.0 ms). Use device time to rank ops and to A/B\n"
    "  against another capture; use wall clock for the ms/frame you report.\n",
    "- Under TP the CSV holds one row per chip per op; `ops` is the merged per-chip op\n"
    "  count, and each op's time is the max across chips.\n",
    "- `decode_frame` = `decode_cp` + `decode_talker`; it carries inner signposts\n"
    "  (`cp_frame_start`/`cp_frame_stop`, `talker_decode_start`/`talker_decode_stop`) so\n"
    "  both halves can be sliced out of that one capture:\n\n"
    "  ```\n"
    "  python3 models/demos/qwen3_tts/tests/qwen3_tts_perf_report_opslist.py \\\n"
    "    --window cp_frame --start cp_frame_start --end cp_frame_stop \\\n"
    "    models/demos/qwen3_tts/ops_list/perf_report/decode_frame/ops.csv\n"
    "  ```\n",
    "- One AR frame runs the CodePredictor 15 times against 1 Talker decode — compare\n"
    "  per-frame cost, never per-layer cost.\n",
]
(out / "summary.md").write_text("".join(lines))
print(f"\nwrote {out}/summary.md")
print("".join(lines[2:4 + len(windows)]))
PYSUM
