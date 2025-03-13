#!/bin/bash

set -eo pipefail

if [ ! -d "/mnt/MLPerf/ccache" ]; then
  echo "::error title=mlperf-not-mounted::Weka is not mounted. Please let the infra team know."
  exit 1
fi
