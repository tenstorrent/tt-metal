#!/bin/bash
set -eo pipefail

TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache

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

  echo "LOG_METAL: Testing TT_VISIBLE_DEVICES functionality"
  ./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh ; fail+=$?

  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectRingGatherAllChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter="MeshDeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_ERISC_IRAM=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueMultiDevice*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleDevice*Fixture.*" ; fail+=$?
  TT_METAL_ENABLE_ERISC_IRAM=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQMultiDevice*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter="DPrintMeshFixture.*:MeshWatcherFixture.*" ; fail+=$?

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

  # Offline test for Cluster Validation Tool
  ./build/tools/scaleout/run_cluster_validation --global-descriptor-path tools/tests/scaleout/global_system_descriptors/proto/4_lb_superpod_physical_desc.textproto --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto --deployment-descriptor-path tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto --print-connectivity --hard-fail

  # these tests cover mux fixture as well
  TT_METAL_FABRIC_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  TT_METAL_FABRIC_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"

  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=T3k*MeshGraphFabric2DDynamicTests*

  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_at_least_2x2_mesh.yaml
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml

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
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_strict_connection_multi_process_rank_bindings.yaml  ./build/test/tt_metal/multi_host_fabric_tests
  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2.yaml

  # Big-Mesh 2x4 Regression tests
  # Tests are disabled for now due to ND hangs
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter="*BigMeshDualRankTest2x4*"
  #tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter="*MeshWorkloadTest*"
}

run_t3000_ttnn_multiprocess_tests() {
  local mpi_args="--allow-run-as-root --tag-output"

  tt-run --mpi-args "$mpi_args" --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2

  # Big-Mesh 2x4 Regression tests
  local mesh2x4_rank_binding="tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" build/test/ttnn/unit_tests_ttnn --gtest_filter="*LaunchOperation*"
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/distributed/test_data_parallel_example.py
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/unit_tests/operations/ccl/test_new_all_broadcast.py::test_all_broadcast_sharded_2x4
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_t3000.py::test_all_to_all_combine_no_trace_submesh
  tt-run --mpi-args "$mpi_args" --rank-binding "$mesh2x4_rank_binding" pytest -svv "tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py::test_point_to_point[silicon_arch_name=wormhole_b0-dtype=torch.bfloat16-shape_coords=((1, 1, 1, 16), ((0, 0), (0, 1)))-tile-mesh_device=(2, 4)-device_params={'fabric_config': <FabricConfig.FABRIC_1D: 1>}]"
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

run_t3000_gemma3-small_tests() {
  pytest models/demos/gemma3/tests/test_ci_dispatch.py -k "27b"
}

run_t3000_llama3-small_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3-small_tests"

  # Llama3.2-1B
  llama1b=meta-llama/Llama-3.2-1B-Instruct
  # Llama3.2-3B
  llama3b=meta-llama/Llama-3.2-3B-Instruct
  # Llama3.1-8B
  llama8b=meta-llama/Llama-3.1-8B-Instruct

  # Run all Llama3 tests for 1B, 3B and 8B weights
  for hf_model in "$llama1b" "$llama3b" "$llama8b"; do
    tt_cache=$TT_CACHE_HOME/$hf_model
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
    HF_MODEL=$hf_model TT_CACHE_PATH=$tt_cache pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $hf_model completed"
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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b

  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

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
  llama70b=meta-llama/Llama-3.1-70B-Instruct
  tt_cache_llama70b=$TT_CACHE_HOME/$llama70b

  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  HF_MODEL=$llama70b TT_CACHE_PATH=$tt_cache_llama70b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

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
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/$llama90b

  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_attention.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_attention_prefill.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_embedding.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_mlp.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_rms_norm.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_decoder.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py ; fail+=$?

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

  hf_model=mistralai/Mistral-7B-Instruct-v0.3
  tt_cache_path=$TT_CACHE_HOME/$hf_model

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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b

  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

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
  llama11b=meta-llama/Llama-3.2-11B-Vision-Instruct
  tt_cache_llama11b=$TT_CACHE_HOME/$llama11b
  # Use MESH_DEVICE env variable to run on an N300 mesh
  mesh_device=N300

  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  MESH_DEVICE=$mesh_device HF_MODEL=$llama11b TT_CACHE_PATH=$tt_cache_llama11b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

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
  llama90b=meta-llama/Llama-3.2-90B-Vision-Instruct
  tt_cache_llama90b=$TT_CACHE_HOME/$llama90b

  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_image_block.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  HF_MODEL=$llama90b TT_CACHE_PATH=$tt_cache_llama90b pytest -n auto models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

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

  mixtral8x7=mistralai/Mixtral-8x7B-v0.1
  tt_cache_mixtral8x7=$TT_CACHE_HOME/$mixtral8x7

  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_rms_norm.py --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_mlp.py --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_moe.py --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_decoder.py --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_decoder_prefill.py --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 CI=true pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_model.py::test_model_inference[wormhole_b0-device_params0-8-performance-256-1-page_params0-paged_attention-quick] --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 CI=true pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_model.py::test_model_inference[wormhole_b0-device_params0-8-performance-256-1-page_params0-default_attention-quick] --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 CI=true pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_model_prefill.py::test_model_inference[wormhole_b0-device_params0-1layer-performance-max128k-4k-page_params0-paged_attention-8] --timeout=720 ; fail+=$?
  HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 CI=true pytest -n auto models/tt_transformers/tests/mixtral/test_mixtral_model_prefill.py::test_model_inference[wormhole_b0-device_params0-1layer-performance-max128k-4k-page_params0-default_attention-8] --timeout=720 ; fail+=$?

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
  start_time=$(date +%s)

  # install qwen25_vl requirements
  pip install -r models/demos/qwen25_vl/requirements.txt

  # export PYTEST_ADDOPTS for concise pytest output
  export PYTEST_ADDOPTS="--tb=short"

  # Qwen2.5-VL-72B provides good enough coverage for other model variants -- 3B, 32B
  qwen25_vl_72b=Qwen/Qwen2.5-VL-72B-Instruct
  tt_cache_72b=$TT_CACHE_HOME/$qwen25_vl_72b

  # run unit tests
  MESH_DEVICE=T3K HF_MODEL=$qwen25_vl_72b TT_CACHE_PATH=$tt_cache_72b pytest models/demos/qwen25_vl/tests/ --ignore=models/demos/qwen25_vl/tests/test_ci_dispatch.py --ignore=models/demos/qwen25_vl/tests/conftest.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: Unit tests for $qwen25_vl_72b on T3K completed in $duration seconds"
}

run_t3000_ccl_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ccl_tests"

  # all gather: 1 ring, 1 line, 1 2d, 1 sharded should be covered
  # width sharded to interleaved case using linear
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_sharded_to_interleaved[wormhole_b0-fabric_linear-check-i2s_shape0-1-layout0-ag_input_dtype0-mesh_device0]
  # 10 iteration trace test with fabric ring (dit_shape now in test_ttnn_all_gather, no barrier parameters)
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_ttnn_all_gather[wormhole_b0-fabric_ring-perf-mem_config_input0-mem_config_ag0-dit_shape-1link-mesh_device0]
  # 2D fabric case – hanging on main? tracking with issue #30250
  # pytest -n auto tests/nightly/t3000/2d_ccl/test_minimal_all_gather_async.py::test_all_gather_async_training_shapes[wormhole_b0-fabric_2d_dynamic_linear-check-mem_config_input0-mem_config_ag0-tt_training_test_one-mesh_device0-1link]
  # training shapes
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_all_gather_async.py::test_all_gather_async_training_shapes[wormhole_b0-fabric_linear-check-mem_config_input0-mem_config_ag0-tt_training_test_four-mesh_device0-1link]

  # reduce scatter: 1 ring, 1 line, 1 2d, 1 sharded should be covered
  # sharded intermediate case with cluster axis 1
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_minimal_async_linear_sharded
  # composite case
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_training_shapes[wormhole_b0-fabric_linear-random-check-mem_config_input0-mem_config_rs0-tt_training_test_one-mesh_device0-1link]
  # long trace test on dim=1 with ring, currently hanging when run in the suite even though it passes when run in isolation
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async[wormhole_b0-fabric_ring-barrier_without_persistent_buffers-random-perf-mem_config_input0-mem_config_rs0-scatter_dim_1_test_one-1link-mesh_device0]
  # long running dim = 3 trace test without barrier and with persistent buffers
  pytest -n auto tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async[wormhole_b0-fabric_ring-no_barrier_with_persistent_buffers-random-perf-mem_config_input0-mem_config_rs0-padded_dim_2_test_one-1link-mesh_device0]

  # all reduce: 1 test should be enough
  # 4 chip test with bfloat8_b
  pytest -n auto tests/nightly/t3000/ccl/test_all_reduce.py::test_ring_all_reduce_post_commit[wormhole_b0-True-device_params0-math_op0-DRAM-bfloat8_b-layout0-2x4x2048x32-4-1]

  # p2p: 1 test should be enough
  # trace test with device delay
  pytest -n auto tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point.py::test_point_to_point_with_device_delay -k tile

  # all broadcast: row major + tile test
  # both rm and tile test are called here
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_new_all_broadcast.py::test_all_broadcast_trace

  # all to all dispatch: 1 test for 2d and 1 for 1d linear should be enough
  # fabric 1d linear test on cluster axis 0 as other CCL tests aren't testing on this axis
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_t3000.py::test_all_to_all_dispatch_trace[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-num_links=MAX_LINKS-dram-dram-s128-hidden_size=7168-select_experts_k=8-experts_per_device=8-batches_per_device=8-cluster_axis_0-2x4_grid-trace_mode=True-fabric_1d_linear]
  # fabric 2d test on cluster axis 1
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_t3000.py::test_all_to_all_dispatch_no_trace[silicon_arch_name=wormhole_b0-dram_output-dram_input-dtype=DataType.BFLOAT16-num_links=MAX_LINKS-b1s3-hidden_size=7168-select_experts_k=8-experts_per_device=8-cluster_col-2x4_grid-trace_mode=False-fabric_2d]

  # all to all combine: 1 test for 1d ring and 1 for 2d should be enough
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_t3000.py::test_all_to_all_combine_no_trace[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-topology=None-dram-dram-num_iters=2-scheme=random-local_reduce=True-seq=2-hidden_size=7000-select_experts_k=8-experts_per_device=8-batches_per_device=8-fabric_1d_ring_axis_1]
  # fabric 2d test on cluster axis 0
  pytest -n auto tests/ttnn/unit_tests/operations/ccl/test_all_to_all_combine_t3000.py::test_all_to_all_combine_no_trace[silicon_arch_name=wormhole_b0-dtype=DataType.BFLOAT16-topology=None-dram-dram-num_iters=2-scheme=random-local_reduce=True-seq=2-hidden_size=7000-select_experts_k=8-experts_per_device=8-batches_per_device=8-fabric_2d_axis_0]

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ccl_tests $duration seconds to complete"
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
