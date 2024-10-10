#! /usr/bin/env bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

PROFILER_SCRIPTS_ROOT=$TT_METAL_HOME/tt_metal/tools/profiler
PROFILER_TEST_SCRIPTS_ROOT=$TT_METAL_HOME/tests/tt_metal/tools/profiler
PROFILER_ARTIFACTS_DIR=$TT_METAL_HOME/generated/profiler
if [[ "$TT_METAL_PROFILER_DIR" ]]; then
    PROFILER_ARTIFACTS_DIR=$TT_METAL_PROFILER_DIR
fi
PROFILER_OUTPUT_DIR=$PROFILER_ARTIFACTS_DIR/reports

remove_default_log_locations(){
    rm -rf $PROFILER_ARTIFACTS_DIR
    echo "Removed all profiler artifacts"
}

verify_perf_line_count_floor(){
    csvLog=$1
    LINE_COUNT=$2

    lineCount=$(cat $csvLog | wc | awk  'NR == 1 { print $1 }')

    if [[ ! $lineCount =~ ^[0-9]+$ ]]; then
        echo "Value for line count was $lineCount, not a number !" 1>&2
        exit 1
    fi

    if (( lineCount < LINE_COUNT )); then
        echo "Value for line count was $lineCount, not higher than $LINE_COUNT" 1>&2
        exit 1
    fi

    echo "Value for line count was in correct range" 1>&2
}

verify_perf_line_count(){
    csvLog=$1
    LINE_COUNT=$2

    lineCount=$(cat $csvLog | wc | awk  'NR == 1 { print $1 }')

    if [[ ! $lineCount =~ ^[0-9]+$ ]]; then
        echo "Value for line count was $lineCount, not a number !" 1>&2
        exit 1
    fi

    if (( lineCount != LINE_COUNT )); then
        echo "Value for line count was $lineCount, not equal to $LINE_COUNT" 1>&2
        exit 1
    fi

    echo "Value for line count was correct" 1>&2
}

verify_perf_column(){
    csvLog=$1
    column=$2
    LOWER_BOUND=$3
    UPPER_BOUND=$4

    header=$(cat $csvLog |awk -v column="$column"  -F, 'NR == 1 { print $column }')
    res=$(cat $csvLog |awk -v column="$column"  -F, 'NR == 2 { print $column }')

    if [[ ! $res =~ ^[0-9]+$ ]]; then
        echo "Value for $header was $res, not a number !" 1>&2
        exit 1
    fi

    if (( res > UPPER_BOUND )) || (( res < LOWER_BOUND )); then
        echo "Value for $header was $res, not between $UPPER_BOUND, $LOWER_BOUND" 1>&2
        exit 1
    fi

    echo "Value for $header was within range" 1>&2
}
