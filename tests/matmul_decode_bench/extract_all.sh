#!/usr/bin/env bash
# Self-contained: PYTHONPATH = the fork ONLY (no tt_symbiote).
set -u
FORK=/home/ttuser/salnahari/tt-metal-matmul_decode
BENCH=$FORK/tests/matmul_decode_bench
PY=$FORK/python_env/bin/python
EXTRACTOR=$BENCH/extract_perf.py
CSVDIR=$BENCH/results/chunked_sweep_nopatch_csvs
export PYTHONPATH=$FORK
OUT=$CSVDIR/kernel_table_data.tsv
printf "tag\tstatus\top\tkernel_us\tcalls_fwd\treps\tspread\n" > "$OUT"
for log in "$CSVDIR"/raw/*.log; do
  tag=$(basename "$log" .log)
  status=$(grep -hoE ": (TIMED|INVALID-RUN|NOFIT-BUILD|PCC-FAIL)" "$log" | tail -1 | sed 's/: //')
  [ -z "$status" ] && status="UNKNOWN"
  if [ "$status" = "TIMED" ]; then
    case "$tag" in *native*) op=native;; *) op=mmd;; esac
    csv="$CSVDIR/$tag.csv"
    line=$(EXTRACT_MODE=mmsweep METRIC=KERNEL MMSWEEP_OP=$op "$PY" "$EXTRACTOR" "$csv" 2>/dev/null | grep -E "us/fwd=" | tail -1)
    us=$(echo "$line"     | grep -oE "us/fwd=[0-9.]+"     | head -1 | cut -d= -f2)
    calls=$(echo "$line"  | grep -oE "calls/fwd=[0-9.]+"  | head -1 | cut -d= -f2)
    reps=$(echo "$line"   | grep -oE "reps=[0-9]+"        | head -1 | cut -d= -f2)
    spread=$(echo "$line" | grep -oE "spread=[0-9.]+%"    | head -1 | cut -d= -f2)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$status" "$op" "${us:-NA}" "${calls:-NA}" "${reps:-NA}" "${spread:-NA}" >> "$OUT"
  else
    printf "%s\t%s\t-\t-\t-\t-\t-\n" "$tag" "$status" >> "$OUT"
  fi
done
echo "wrote $OUT"
echo "TIMED: $(awk -F'\t' '$2=="TIMED"' "$OUT"|wc -l)  with µs: $(awk -F'\t' '$2=="TIMED"&&$4!="NA"' "$OUT"|wc -l)"
