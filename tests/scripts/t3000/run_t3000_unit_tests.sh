#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_t3000_ttmetal_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttmetal_tests"
  ./build/test/tt_metal/distributed/distributed_unit_tests

  echo "LOG_METAL: Testing TT_METAL_VISIBLE_DEVICES functionality"
  ./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh ; fail+=$?

  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="DeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="DeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="DeviceFixture.ActiveEthKernelsDirectRingGatherAllChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="DeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_ERISC_IRAM=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueMultiDevice*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleDevice*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_ERISC_IRAM=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQMultiDevice*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter="DPrintFixture.*:WatcherFixture.*" ; fail+=$?

  # Programming examples
  ./build/programming_examples/distributed/distributed_program_dispatch
  ./build/programming_examples/distributed/distributed_buffer_rw
  ./build/programming_examples/distributed/distributed_eltwise_add
  ./build/programming_examples/distributed/distributed_trace_and_events

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttfabric_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttfabric_tests"
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3kCustomMeshGraphControlPlaneTests*
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3k*MeshGraphFabric2DDynamicTests*

  # originally were in TT-NN, now promoted to TT-Metal (Fabric)
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*WorkerFabricEdmDatapath*:*EdmFabric*"
  # Instantiate a 1x8 Mesh on a T3K with 2D Fabric
  TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.yaml ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*Fabric2DFixture.TestUnicast*"

  # TODO (issue: #24335) disabled slow dispatch tests for now, need to re-evaluate if need to add in a different pool
  #TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

  # these tests cover mux fixture as well
  TT_METAL_FABRIC_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  TT_METAL_FABRIC_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3k*MeshGraphFabric2DDynamicTests*

  TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
  TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_at_least_2x2_mesh.yaml
  TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttnn_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttnn_tests"
  ./build/test/ttnn/unit_tests_ttnn
  ./build/test/ttnn/unit_tests_ttnn_tensor
  ./build/test/ttnn/unit_tests_ttnn_ccl
  ./build/test/ttnn/unit_tests_ttnn_ccl_multi_tensor
  ./build/test/ttnn/unit_tests_ttnn_ccl_ops
  ./build/test/ttnn/unit_tests_ttnn_accessor
  ./build/test/ttnn/test_ccl_multi_cq_multi_device
  pytest tests/ttnn/unit_tests/test_multi_device_trace.py ; fail+=$?
  pytest tests/ttnn/unit_tests/test_multi_device_events.py ; fail+=$?
  pytest tests/ttnn/unit_tests/operations/test_prefetcher.py::test_run_prefetcher_post_commit_multi_device ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_async.py ; fail+=$?
  pytest tests/ttnn/distributed/test_tensor_parallel_example_T3000.py ; fail+=$?
  pytest tests/ttnn/distributed/test_data_parallel_example.py ; fail+=$?
  pytest tests/ttnn/distributed/test_hybrid_data_tensor_parallel_example_T3000.py ; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttnn_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tt_metal_multiprocess_tests() {
  local mpi_args="--allow-run-as-root --tag-output"

  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2.yaml
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/multi_host_fabric_tests
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2.yaml

  # Big-Mesh 2x4 Regression tests
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter="*BigMeshDualRankTest2x4*"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="*MeshWorkloadTest*"
}

run_t3000_ttnn_multiprocess_tests() {
  local mpi_args="--allow-run-as-root --tag-output"

  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2

  # Big-Mesh 2x4 Regression tests
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/unit_tests_ttnn --gtest_filter="*LaunchOperation*"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/distributed/test_data_parallel_example.py
}

run_t3000_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py ; fail+=$?
  #pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  pytest -n auto models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_1_layer_t3000.py ; fail+=$?


  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3-small_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3-small_tests"

  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/

  # Run all Llama3 tests for 1B, 3B and 8B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
    LLAMA_DIR=$llama_dir pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3-small_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-11b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-11b_tests"

  # Llama3.2-11B weights
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-11b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.1-70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.1-70b_tests"

  # Llama3.1-70B weights
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/

  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  LLAMA_DIR=$llama70b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.1-70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-90b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-90b_tests"

  # Llama3.2-90B weights
  # use repacked weights to shorten unit test time by loading only the necessary weights
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/repacked/

  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-90b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}


run_t3000_mistral_tests() {

  echo "LOG_METAL: Running run_t3000_mistral_unit_tests"

  tt_cache_path="/mnt/MLPerf/tt_dnn-models/Mistral/TT_CACHE/Mistral-7B-Instruct-v0.3"
  hf_model="/mnt/MLPerf/tt_dnn-models/Mistral/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"

  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_attention.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_attention_prefill.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_embedding.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_mlp.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_rms_norm.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_decoder.py
  TT_CACHE_PATH=$tt_cache_path HF_MODEL=$hf_model pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py

}

run_t3000_llama3.2-11b-vision_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-11b-vision_unit_tests"

  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-11b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests"

  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  # Use MESH_DEVICE env variable to run on an N300 mesh
  mesh_device=N300

  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  MESH_DEVICE=$mesh_device LLAMA_DIR=$llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-90b-vision_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-90b-vision_unit_tests"

  # use repacked weights to shorten unit test time by loading only the necessary weights
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/repacked/

  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  LLAMA_DIR=$llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-90b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_attention.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_rms_norm.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_embedding.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_decoder.py ; fail+=$?
  # Mixtral prefill tests
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp_prefill.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe_prefill.py ; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_grok_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_grok_tests"

  pytest -n auto models/experimental/grok/tests/test_grok_rms_norm.py ; fail+=$?
  pytest -n auto models/experimental/grok/tests/test_grok_attention.py ; fail+=$?
  pytest -n auto models/experimental/grok/tests/test_grok_mlp.py --timeout=500; fail+=$?
  pytest -n auto models/experimental/grok/tests/test_grok_moe.py --timeout=600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_grok_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_unet_shallow_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_unet_shallow_tests"

  pytest -n auto models/experimental/functional_unet/tests/test_unet_multi_device.py; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_unet_shallow_tests took $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_qwen25_vl_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  # install qwen25_vl requirements
  pip install -r models/demos/qwen25_vl/reference/requirements.txt

  # export PYTEST_ADDOPTS for concise pytest output
  export PYTEST_ADDOPTS="--tb=short"

  qwen25_vl_32b=/mnt/MLPerf/tt_dnn-models/qwen/Qwen2.5-VL-32B-Instruct/
  qwen25_vl_72b=/mnt/MLPerf/tt_dnn-models/qwen/Qwen2.5-VL-72B-Instruct/

  for qwen_dir in "$qwen25_vl_32b" "$qwen25_vl_72b"; do
    # test_mlp.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_mlp.py --timeout 400 || fail=1
    echo "LOG_METAL: Unit tests in test_mlp.py for $qwen_dir on T3K completed"
    # test_rms_norm.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_rms_norm.py --timeout 180 || fail=1
    echo "LOG_METAL: Unit tests in test_rms_norm.py for $qwen_dir on T3K completed"
    # test_vision_attention.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_vision_attention.py --timeout 180 || fail=1
    echo "LOG_METAL: Unit tests in test_vision_attention.py for $qwen_dir on T3K completed"
    # test_vision_block.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_vision_block.py --timeout 600 || fail=1
    echo "LOG_METAL: Unit tests in test_vision_block.py for $qwen_dir on T3K completed"
    # test_patch_merger.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_patch_merger.py --timeout 180 || fail=1
    echo "LOG_METAL: Unit tests in test_patch_merger.py for $qwen_dir on T3K completed"
    # test_model.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_model.py -k two_layers --timeout 180 || fail=1
    echo "LOG_METAL: Unit tests in test_model.py for $qwen_dir on T3K completed"
    # test_wrapped_model.py
    MESH_DEVICE=T3K HF_MODEL=$qwen_dir pytest -n auto models/demos/qwen25_vl/tests/test_wrapped_model.py -k two_layers --timeout 180 || fail=1
    echo "LOG_METAL: Unit tests in test_wrapped_model.py for $qwen_dir on T3K completed"
  done

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run ttmetal tests
  run_t3000_ttmetal_tests

  # Run ttfabric tests
  run_t3000_ttfabric_tests

  # Run ttnn tests
  run_t3000_ttnn_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run llama3-small (1B, 3B, 8B) tests
  run_t3000_llama3-small_tests

  # Run llama3.2-11B tests
  run_t3000_llama3.2-11b_tests

  # Run llama3.1-70B tests
  run_t3000_llama3.1-70b_tests

  # Run llama3.2-90B tests
  run_t3000_llama3.2-90b_tests

  # Run llama3.2-11B-vision tests
  run_t3000_llama3.2-11b-vision_unit_tests

  # Run mistral tests
  run_t3000_mistral_tests

  # Run llama3.2-11B-vision tests on spoofed N300 mesh
  run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests

  # Run llama3.2-90B-vision tests
  run_t3000_llama3.2-90b-vision_unit_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run grok tests
  run_t3000_grok_tests

  # Run unet shallow tests
  run_t3000_unet_shallow_tests
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

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
