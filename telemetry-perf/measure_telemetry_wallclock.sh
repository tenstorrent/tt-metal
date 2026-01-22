#!/bin/bash
# Measure telemetry performance impact using wall-clock timing
# This bypasses all Python import issues by just timing actual workloads

set -e

TELEMETRY_BIN="/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
FSD_PATH="/localdev/kkfernandez/fsd.textproto"
TT_METAL_HOME="/localdev/kkfernandez/tt-metal"

# Pick a simple workload - any command that uses the devices
# Option 1: Use tt-smi itself as the workload
WORKLOAD_CMD="tt-smi -s"
WORKLOAD_NAME="tt-smi snapshot"

# Option 2: If you have a simple test script, use that instead
# WORKLOAD_CMD="python3 your_test.py"
# WORKLOAD_NAME="your test"

echo "================================================================================"
echo "TELEMETRY PERFORMANCE IMPACT - WALL CLOCK TIMING"
echo "================================================================================"
echo ""
echo "Workload: $WORKLOAD_NAME"
echo "Command: $WORKLOAD_CMD"
echo ""

# Function to run workload and measure time
run_test() {
    local telemetry_enabled=$1
    local polling_interval=$2
    local run_number=$3

    if [ "$telemetry_enabled" = "true" ]; then
        # Start telemetry
        $TELEMETRY_BIN \
            --polling-interval "$polling_interval" \
            --port 7070 \
            --fsd "$FSD_PATH" \
            > /dev/null 2>&1 &
        local telem_pid=$!
        sleep 2  # Let telemetry initialize
    fi

    # Run workload and measure time
    local start=$(date +%s.%N)
    $WORKLOAD_CMD > /dev/null 2>&1
    local end=$(date +%s.%N)

    if [ "$telemetry_enabled" = "true" ]; then
        # Stop telemetry
        kill $telem_pid 2>/dev/null || true
        wait $telem_pid 2>/dev/null || true
        sleep 1  # Cool down
    fi

    # Calculate duration in milliseconds
    local duration=$(echo "($end - $start) * 1000" | bc)
    echo "$duration"
}

# Arrays to store results
declare -a baseline_times
declare -a tel_5s_times
declare -a tel_1s_times
declare -a tel_100ms_times
declare -a tel_10ms_times

N_RUNS=5

echo "Running tests with $N_RUNS iterations each..."
echo ""

# Baseline (no telemetry)
echo "=== BASELINE (No Telemetry) ==="
for i in $(seq 1 $N_RUNS); do
    time=$(run_test false "" $i)
    baseline_times+=($time)
    printf "  Run %d: %8.2f ms\n" $i $time
done
echo ""

# Telemetry at 5s
echo "=== TELEMETRY: 5s polling ==="
for i in $(seq 1 $N_RUNS); do
    time=$(run_test true "5s" $i)
    tel_5s_times+=($time)
    printf "  Run %d: %8.2f ms\n" $i $time
done
echo ""

# Telemetry at 1s
echo "=== TELEMETRY: 1s polling ==="
for i in $(seq 1 $N_RUNS); do
    time=$(run_test true "1s" $i)
    tel_1s_times+=($time)
    printf "  Run %d: %8.2f ms\n" $i $time
done
echo ""

# Telemetry at 100ms
echo "=== TELEMETRY: 100ms polling ==="
for i in $(seq 1 $N_RUNS); do
    time=$(run_test true "100ms" $i)
    tel_100ms_times+=($time)
    printf "  Run %d: %8.2f ms\n" $i $time
done
echo ""

# Telemetry at 10ms
echo "=== TELEMETRY: 10ms polling ==="
for i in $(seq 1 $N_RUNS); do
    time=$(run_test true "10ms" $i)
    tel_10ms_times+=($time)
    printf "  Run %d: %8.2f ms\n" $i $time
done
echo ""

# Calculate statistics
calc_stats() {
    local -n arr=$1
    local sum=0
    local count=${#arr[@]}

    # Calculate mean
    for val in "${arr[@]}"; do
        sum=$(echo "$sum + $val" | bc)
    done
    local mean=$(echo "scale=2; $sum / $count" | bc)

    # Calculate std dev
    local var_sum=0
    for val in "${arr[@]}"; do
        local diff=$(echo "$val - $mean" | bc)
        local sq=$(echo "$diff * $diff" | bc)
        var_sum=$(echo "$var_sum + $sq" | bc)
    done
    local variance=$(echo "scale=2; $var_sum / $count" | bc)
    local std=$(echo "scale=2; sqrt($variance)" | bc)

    echo "$mean $std"
}

# Calculate statistics for each configuration
read baseline_mean baseline_std <<< $(calc_stats baseline_times)
read tel_5s_mean tel_5s_std <<< $(calc_stats tel_5s_times)
read tel_1s_mean tel_1s_std <<< $(calc_stats tel_1s_times)
read tel_100ms_mean tel_100ms_std <<< $(calc_stats tel_100ms_times)
read tel_10ms_mean tel_10ms_std <<< $(calc_stats tel_10ms_times)

# Calculate overhead
calc_overhead() {
    local baseline=$1
    local with_telemetry=$2
    echo "scale=2; (($with_telemetry - $baseline) / $baseline) * 100" | bc
}

overhead_5s=$(calc_overhead $baseline_mean $tel_5s_mean)
overhead_1s=$(calc_overhead $baseline_mean $tel_1s_mean)
overhead_100ms=$(calc_overhead $baseline_mean $tel_100ms_mean)
overhead_10ms=$(calc_overhead $baseline_mean $tel_10ms_mean)

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
printf "%-20s %10s ± %6s ms   %10s\n" "Configuration" "Mean" "StdDev" "Overhead"
echo "--------------------------------------------------------------------------------"
printf "%-20s %10.2f ± %6.2f ms   %10s\n" "Baseline" $baseline_mean $baseline_std "---"
printf "%-20s %10.2f ± %6.2f ms   %9.2f%%\n" "Telemetry (5s)" $tel_5s_mean $tel_5s_std $overhead_5s
printf "%-20s %10.2f ± %6.2f ms   %9.2f%%\n" "Telemetry (1s)" $tel_1s_mean $tel_1s_std $overhead_1s
printf "%-20s %10.2f ± %6.2f ms   %9.2f%%\n" "Telemetry (100ms)" $tel_100ms_mean $tel_100ms_std $overhead_100ms
printf "%-20s %10.2f ± %6.2f ms   %9.2f%%\n" "Telemetry (10ms)" $tel_10ms_mean $tel_10ms_std $overhead_10ms
echo ""
echo "================================================================================"

# Save results to JSON
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="/tmp/telemetry_wallclock_results_${TIMESTAMP}.json"

cat > "$RESULTS_FILE" << EOF
{
  "workload": "$WORKLOAD_NAME",
  "timestamp": "$TIMESTAMP",
  "n_runs": $N_RUNS,
  "results": {
    "baseline": {
      "mean_ms": $baseline_mean,
      "std_ms": $baseline_std,
      "times": [$(IFS=,; echo "${baseline_times[*]}")]
    },
    "telemetry_5s": {
      "mean_ms": $tel_5s_mean,
      "std_ms": $tel_5s_std,
      "overhead_pct": $overhead_5s,
      "times": [$(IFS=,; echo "${tel_5s_times[*]}")]
    },
    "telemetry_1s": {
      "mean_ms": $tel_1s_mean,
      "std_ms": $tel_1s_std,
      "overhead_pct": $overhead_1s,
      "times": [$(IFS=,; echo "${tel_1s_times[*]}")]
    },
    "telemetry_100ms": {
      "mean_ms": $tel_100ms_mean,
      "std_ms": $tel_100ms_std,
      "overhead_pct": $overhead_100ms,
      "times": [$(IFS=,; echo "${tel_100ms_times[*]}")]
    },
    "telemetry_10ms": {
      "mean_ms": $tel_10ms_mean,
      "std_ms": $tel_10ms_std,
      "overhead_pct": $overhead_10ms,
      "times": [$(IFS=,; echo "${tel_10ms_times[*]}")]
    }
  }
}
EOF

echo "Results saved to: $RESULTS_FILE"
echo ""
