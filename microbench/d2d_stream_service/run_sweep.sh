#!/usr/bin/env bash
#
# Per-config D2D microbenchmark sweep runner (throughput and/or latency).
#
# Each config runs in its OWN process. Per-process isolation is robust to the flaky
# first-run eth init and to any single config wedging the board, and keeps one failure
# from poisoning the rest. (The in-process submesh-reuse fix — D2D_BENCH_REUSE_SUBMESH,
# default ON — also lets the whole sweep run in a single process; this script stays
# per-process for robustness + per-config status reporting.)
#
# Reports per-config status (PASS / HANG / INIT_ABORT / ERROR) + metrics. A HANG wedges an
# ethernet core, so the board is reset (tt-smi -r) after one; flaky init aborts (EXIT 134)
# are retried once after a reset.
#
# Env overrides:
#   MODE      throughput | latency | both        (default: both)
#   BIN OUTDIR TIMEOUT WARMUP ITERS RESET_CMD CHECK_DATA
#   SIZES     throughput size_index list          (default "0 1 2 3 4")
#   STAGES    latency num_stages list             (default "2 4 8")
#   MDS       metadata_bytes list                 (default "0 12")
#   LEASES    lease list                          (default "0 1")
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

BIN=${BIN:-./build_Release/test/tt_metal/distributed/d2d_stream_benchmarks}
OUTDIR=${OUTDIR:-d2d_sweep_results}
TIMEOUT=${TIMEOUT:-260}
MODE=${MODE:-both}
SIZES=${SIZES:-"0 1 2 3 4"}
STAGES=${STAGES:-"2 4 8"}
MDS=${MDS:-"0 12"}
LEASES=${LEASES:-"0 1"}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
CHECK_DATA=${CHECK_DATA:-1}     # throughput only; verifies receiver == sender pattern
RESET_CMD=${RESET_CMD:-"tt-smi -r"}

mkdir -p "$OUTDIR"
if [ ! -x "$BIN" ]; then
    echo "ERROR: benchmark binary not found at $BIN (build it first)" >&2
    exit 1
fi

# Merge per-config JSON files matching $1 glob into a Google-Benchmark CSV,
# then run format_d2d_csv.py to produce a clean sorted CSV at $2.
# Silently skips if no JSON files are found or python3 is unavailable.
format_csv() {
    local glob_pattern="$1" out_csv="$2"
    local raw_csv="${out_csv%.csv}_raw.csv"
    local fmt_script
    fmt_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/format_d2d_csv.py"
    # shellcheck disable=SC2086  # intentional glob expansion of $glob_pattern
    python3 - "$raw_csv" $glob_pattern <<'PYEOF'
import json, csv, glob as _glob, sys
out = sys.argv[1]
files = sorted(set(f for a in sys.argv[2:] for f in (_glob.glob(a) or [a] if '*' in a else [a])))
BASE = ["name","iterations","real_time","cpu_time","time_unit",
        "bytes_per_second","items_per_second","label","error_occurred","error_message"]
rows, extra_cols = [], set()
for f in files:
    try:
        data = json.load(open(f))
        for b in data.get("benchmarks", []):
            if b.get("run_type") == "iteration":
                rows.append(b)
                extra_cols.update(k for k in b if k not in BASE)
    except Exception:
        pass
if not rows:
    sys.exit(0)
fns = BASE + sorted(extra_cols)
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=fns, extrasaction="ignore")
w.writeheader()
for r in rows:
    w.writerow({k: r.get(k, "") for k in fns})
PYEOF
    [ -f "$raw_csv" ] || return
    python3 "$fmt_script" "$raw_csv" "$out_csv" && rm -f "$raw_csv" && echo "  -> $out_csv"
}

reset_board() {
    echo "  -> resetting board ($RESET_CMD)"
    timeout 180 $RESET_CMD >/dev/null 2>&1 || echo "  -> WARN: reset returned non-zero"
}

# Best-effort scalar counter extraction from a google-benchmark JSON (no jq dependency).
json_counter() {  # $1=file $2=key
    grep -oE "\"$2\": *[0-9.eE+-]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9.eE+-]+$"
}

# Run one config in its own process (retry once on flaky init), echo the gtest-style status.
# $1=filter $2=json $3=log ; returns status in REPLY_STATUS, exit code in REPLY_EC.
run_cfg() {
    local filter="$1" json="$2" log="$3" ec
    rm -f "$json"
    for attempt in 1 2; do
        D2D_BENCH_WARMUP="$WARMUP" D2D_BENCH_TPUT_ITERS="$ITERS" D2D_BENCH_CHECK_DATA="$CHECK_DATA" \
            timeout "$TIMEOUT" "$BIN" --benchmark_filter="$filter" \
            --benchmark_out="$json" --benchmark_out_format=json > "$log" 2>&1
        ec=$?
        [ "$ec" -ne 134 ] && break
        echo "  (init abort, attempt $attempt; resetting + retrying)"
        reset_board
    done
    REPLY_EC=$ec
    case "$ec" in
        0)   REPLY_STATUS="PASS" ;;
        124) REPLY_STATUS="HANG" ;;
        134) REPLY_STATUS="INIT_ABORT" ;;
        *)   REPLY_STATUS="ERROR" ;;
    esac
    [ "$REPLY_STATUS" = "HANG" ] && reset_board    # unwedge before the next config
}

run_throughput_sweep() {
    local summary="$OUTDIR/summary_throughput.csv"
    echo "size_index,metadata_bytes,lease,status,gbps,payload_bytes,data_ok,exit_code" > "$summary"
    echo "== THROUGHPUT =="
    printf "%-9s %-4s %-5s | %-11s %-12s %-7s %s\n" "size_idx" "md" "lease" "status" "gbps" "data_ok" "exit"
    printf -- "----------------------------------------------------------------------\n"
    for s in $SIZES; do for md in $MDS; do for lease in $LEASES; do
        name="tput_s${s}_md${md}_l${lease}"
        run_cfg "BM_D2DStreamThroughput/size_index:${s}/metadata_bytes:${md}/lease:${lease}" \
                "$OUTDIR/$name.json" "$OUTDIR/$name.log"
        gbps="" ; payload="" ; dataok=""
        if [ "$REPLY_STATUS" = "PASS" ]; then
            gbps="$(json_counter "$OUTDIR/$name.json" throughput_gbps)"
            payload="$(json_counter "$OUTDIR/$name.json" payload_bytes)"
            dataok="$(json_counter "$OUTDIR/$name.json" data_ok)"
        fi
        printf "%-9s %-4s %-5s | %-11s %-12s %-7s %s\n" "$s" "$md" "$lease" "$REPLY_STATUS" "${gbps:-NA}" "${dataok:-NA}" "$REPLY_EC"
        echo "${s},${md},${lease},${REPLY_STATUS},${gbps:-NA},${payload:-NA},${dataok:-NA},${REPLY_EC}" >> "$summary"
    done; done; done
    echo "  -> $summary"
    format_csv "$OUTDIR/tput_*.json" "$OUTDIR/summary_throughput_formatted.csv"
}

run_latency_sweep() {
    local summary="$OUTDIR/summary_latency.csv"
    echo "num_stages,metadata_bytes,lease,status,per_hop_us,total_avg_us,p50_us,p99_us,exit_code" > "$summary"
    echo "== LATENCY =="
    printf "%-9s %-4s %-5s | %-11s %-11s %-11s %s\n" "n_stages" "md" "lease" "status" "per_hop_us" "total_us" "exit"
    printf -- "----------------------------------------------------------------------\n"
    for ns in $STAGES; do for md in $MDS; do for lease in $LEASES; do
        name="lat_ns${ns}_md${md}_l${lease}"
        run_cfg "BM_D2DStreamLatency/num_stages:${ns}/metadata_bytes:${md}/lease:${lease}" \
                "$OUTDIR/$name.json" "$OUTDIR/$name.log"
        ph="" ; tavg="" ; p50="" ; p99=""
        if [ "$REPLY_STATUS" = "PASS" ]; then
            ph="$(json_counter "$OUTDIR/$name.json" per_hop_simple_us)"
            tavg="$(json_counter "$OUTDIR/$name.json" total_avg_us)"
            p50="$(json_counter "$OUTDIR/$name.json" total_p50_us)"
            p99="$(json_counter "$OUTDIR/$name.json" total_p99_us)"
        fi
        printf "%-9s %-4s %-5s | %-11s %-11s %-11s %s\n" "$ns" "$md" "$lease" "$REPLY_STATUS" "${ph:-NA}" "${tavg:-NA}" "$REPLY_EC"
        echo "${ns},${md},${lease},${REPLY_STATUS},${ph:-NA},${tavg:-NA},${p50:-NA},${p99:-NA},${REPLY_EC}" >> "$summary"
    done; done; done
    echo "  -> $summary"
    format_csv "$OUTDIR/lat_*.json" "$OUTDIR/summary_latency_formatted.csv"
}

case "$MODE" in
    throughput) run_throughput_sweep ;;
    latency)    run_latency_sweep ;;
    both)       run_throughput_sweep; echo; run_latency_sweep ;;
    *) echo "ERROR: MODE must be throughput|latency|both (got '$MODE')" >&2; exit 1 ;;
esac

echo
echo "Done. Per-config JSON + logs in $OUTDIR/"
