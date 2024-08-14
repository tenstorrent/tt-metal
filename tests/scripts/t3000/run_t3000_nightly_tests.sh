#/bin/bash

run_t3000_ccl_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ccl_tests"

  # Falcon40B prefill 60 layer end to end with 10 loops; we need 8x8 grid size
  pytest -n auto tests/nightly/t3000/ccl --timeout=180 ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ccl_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}
