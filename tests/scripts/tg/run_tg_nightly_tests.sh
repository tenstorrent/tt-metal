#!/bin/bash

run_tg_nightly_ccl_tests() {
  # Record the start time
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_nightly_ccl_tests"

  pytest tests/nightly/tg/ccl

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_tests $duration seconds to complete"
}
