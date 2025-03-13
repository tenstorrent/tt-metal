#!/bin/bash

set -eo pipefail

echo "Checking for weka..."
sleep 3

if [ ! -d "/mnt/MLPerf/ccache" ]; then
  echo "::error title=mlperf-not-mounted::Weka is not mounted. Please let the infra team know."
  exit 1
fi
