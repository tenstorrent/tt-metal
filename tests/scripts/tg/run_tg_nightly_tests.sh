#/bin/bash

run_tg_llama3_70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_tg_llama3_70b_tests"

  # Falcon40B prefill 60 layer end to end with 10 loops; we need 8x8 grid size
  pytest tests/nightly/tg/models/demos/tg/llama3_70b ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_tg_llama3_70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}
