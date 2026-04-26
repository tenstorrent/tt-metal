#!/usr/bin/env bash
# Requires TT_METAL_HOME. Replace <YOUR_BH_GALAXY_HOST> below with the same hostname as rank 64 in
# blitz_decode_pipeline_rank_file_superpod. See README.md.

cd "${TT_METAL_HOME}"

export HOSTSP=bh-glx-c01u02:4,bh-glx-c01u08:4,bh-glx-c02u02:4,bh-glx-c02u08:4,bh-glx-c03u02:4,bh-glx-c03u08:4,bh-glx-c04u02:4,bh-glx-c04u08:4,bh-glx-c07u02:4,bh-glx-c07u08:4,bh-glx-c08u02:4,bh-glx-c08u08:4,bh-glx-c09u02:4,bh-glx-c09u08:4,bh-glx-c10u02:4,bh-glx-c10u08:4,bh-glx-c05u02:1

# 1 prefill rank + 64 decode stages; KV only prefill → decode rank 0 (see example_disaggregated_prefill_decode_cross_context.cpp)
tt-run \
  --mock-cluster-rank-binding "${TT_METAL_HOME}/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/mock_bh_6u_65_rank_prefill_decode_cluster_desc_mapping.yaml" \
  --rank-bindings-mapping "${TT_METAL_HOME}/tests/tt_metal/distributed/config/disaggregated_prefill_decode_1_prefill_64_decode_rank_bindings_mapping.yaml" \
  --mpi-args "--allow-run-as-root --oversubscribe" \
  "${TT_METAL_HOME}/build/programming_examples/distributed/example_disaggregated_prefill_decode_cross_context"

# Optional: fabric control plane (hardware rankfile; fill HOSTSP)
# TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
#   --mpi-args "--map-by rankfile:file=${TT_METAL_HOME}/aisle_c_testfiles/blitz_decode_pipeline_rank_file_superpod --bind-to hwt:overload-allowed --host ${HOSTSP} --tag-output --allow-run-as-root --oversubscribe" \
#   --rank-bindings-mapping "${TT_METAL_HOME}/aisle_c_testfiles/disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml" \
#   "${TT_METAL_HOME}/build/test/tt_metal/tt_fabric/fabric_unit_tests" \
#   --gtest_filter='ControlPlaneFixture.TestControlPlaneInitNoMGD'
