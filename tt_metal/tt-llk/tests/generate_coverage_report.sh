#!/usr/bin/env bash

if [ ! -f "/tmp/tt-llk-build/merged_coverage.info" ]; then
    echo "Error: Coverage file /tmp/tt-llk-build/merged_coverage.info not found." >&2
    echo "Ensure coverage artefcats was generated!" >&2
    exit 1
fi
genhtml /tmp/tt-llk-build/merged_coverage.info --output-directory ../../coverage_report
