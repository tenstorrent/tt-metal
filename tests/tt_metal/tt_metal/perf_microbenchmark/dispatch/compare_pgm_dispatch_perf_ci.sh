#!/bin/bash

LOG_FILE1="$TT_METAL_HOME/tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/pgm_dispatch_golden.log"
LOG_FILE2="results.log"

# Run the pgm dispatch sweep with trace mode
cd $TT_METAL_HOME
./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/sweep_pgm_dispatch.sh --trace | tee log
./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/filt_pgm_dispatch.pl log > $LOG_FILE2

THRESHOLD=4 # Percentage difference threshold

# Check if log files exist
if [[ ! -f "$LOG_FILE1" || ! -f "$LOG_FILE2" ]]; then
    echo "Error: One or both log files do not exist."
    exit 1
fi

# Read and compare values from the log files
line_count=0
exit_code=0
while IFS= read -r line1 && IFS= read -r line2 <&3; do
    # Convert commas to newlines to handle both formats
    values1=($(echo "$line1" | tr ',' '\n'))
    values2=($(echo "$line2" | tr ',' '\n'))

    # Iterate through values
    for i in "${!values1[@]}"; do
        value1="${values1[$i]}"
        value2="${values2[$i]}"

        # Check if both values are numeric
        if [[ -z "$value1" || -z "$value2" || ! "$value1" =~ ^[0-9]+(\.[0-9]+)?$ || ! "$value2" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Got invalid numeric value in output, check if all pgm dispatch tests ran properly."
            cat $LOG_FILE2
            exit 1
        fi
        if (( $(echo "$value2 < $value1" | bc -l) )); then
          echo "Line $line_count test $i got $value2 which is lower than expected $value1, consider updating $LOG_FILE1"
        fi
        # Calculate the percentage difference
        if (( $(echo "$value1 != 0" | bc -l) )); then
            percentage_diff=$(echo "scale=2; 100 * (($value2 - $value1) / $value1)" | bc)
        else
            continue
        fi

        # Check if the percentage difference exceeds the threshold
        if (( $(echo "$percentage_diff > $THRESHOLD" | bc -l) )); then
            echo "Error: Line $line_count test $i expected value $value1 but got $value2"
            exit_code=1
        fi
    done
    line_count=$((line_count+1))
done < "$LOG_FILE1" 3< "$LOG_FILE2"

exit $exit_code
