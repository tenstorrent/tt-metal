#!/usr/bin/env bash

ARTEFACTS_DIR="${RUNNER_TEMP:-/tmp}/tt-llk-build"
COVERAGE_INFO="${ARTEFACTS_DIR}/merged_coverage.info"

if [ ! -f "$COVERAGE_INFO" ]; then
    echo "Error: Coverage file $COVERAGE_INFO not found." >&2
    echo "Ensure coverage artefcats was generated!" >&2
    exit 1
fi
genhtml "$COVERAGE_INFO" --output-directory ../../coverage_report
