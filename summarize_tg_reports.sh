#!/usr/bin/env bash
# Summarizes sweep_perf_reports_tg_2/ into a table: sub_h, sub_w, num_buffers, time_us, tflops, flops_pct

OUTDIR="${1:-sweep_perf_reports_tg_opus}"

printf "%-6s  %-6s  %-20s  %10s  %12s  %10s\n" \
    "sub_h" "sub_w" "num_buffers" "time_us" "tflops" "flops_pct"
printf "%-6s  %-6s  %-20s  %10s  %12s  %10s\n" \
    "------" "------" "--------------------" "----------" "------------" "----------"

for f in "$OUTDIR"/sub_h=*-sub_w=*-num_buffers_per_channel=*.txt; do
    fname=$(basename "$f" .txt)

    sub_h=$(echo "$fname"    | grep -oP 'sub_h=\K[^-]+')
    sub_w=$(echo "$fname"    | grep -oP 'sub_w=\K[^-]+')
    num_buf=$(echo "$fname"  | grep -oP 'num_buffers_per_channel=\K.*')

    line=$(grep "MinimalMatmulStridedReduceScatterAsync" "$f" | grep "μs" | head -1)
    if [[ -z "$line" ]]; then
        printf "%-6s  %-6s  %-20s  %10s\n" "$sub_h" "$sub_w" "$num_buf" "NO DATA"
        continue
    fi

    time_us=$(echo "$line" | grep -oP '[\d,]+(?= μs)' | head -1 | tr -d ',')
    tflops=$(echo "$line"   | grep -oP '[\d.]+(?= TFLOPs)')
    flops_pct=$(echo "$line" | grep -oP '[\d.]+(?= %)' | tail -1)

    printf "%-6s  %-6s  %-20s  %10s  %12s  %9s%%\n" \
        "$sub_h" "$sub_w" "$num_buf" "${time_us}" "${tflops:-N/A}" "${flops_pct:-N/A}"
done | sort -k4 -n
