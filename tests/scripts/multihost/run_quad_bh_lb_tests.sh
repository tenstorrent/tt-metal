#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_quad_bh_lb_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_quad_bh_lb_unit_tests"

  # tcp flags are default for tt-run
  local mpi_args="--hostfile /etc/mpirun/hostfile"
  local mpirun_args="$mpi_args --mca btl_tcp_if_exclude docker0,lo"
  local rank_binding_quad_bh_lb="tests/tt_metal/distributed/config/quad_bh_lb_rank_bindings.yaml"

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation  --print-connectivity --send-traffic --hard-fail ; fail+=$?

  echo "LOG_METAL: Running health checks via tt-run"
  tt-run --rank-binding "$rank_binding_quad_bh_lb" --mpi-args "$mpi_args" ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  tt-run --rank-binding "$rank_binding_quad_bh_lb" --mpi-args "$mpi_args" ./build/tools/scaleout/run_cluster_validation  --print-connectivity --send-traffic --hard-fail ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_quad_bh_lb_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_bh_lb_demo_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_quad_bh_lb_demo_tests"

  # tcp flags are default for tt-run
  local mpi_args="--hostfile /etc/mpirun/hostfile"
  local mpirun_args="$mpi_args --mca btl_tcp_if_exclude docker0,lo"
  local rank_binding_quad_bh_lb="tests/tt_metal/distributed/config/quad_bh_lb_rank_bindings.yaml"

  tt-run --rank-binding "$rank_binding_quad_bh_lb" --mpi-args "$mpi_args" \
    bash -c "export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct && \
             export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct && \
             pytest models/tt_transformers/demo/simple_text_demo.py -k 'performance and batch-1' --data_parallel 8  --timeout 600" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_quad_bh_lb_demo_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_quad_bh_lb_tests() {
  run_quad_bh_lb_unit_tests
  run_quad_bh_lb_demo_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_quad_bh_lb_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
