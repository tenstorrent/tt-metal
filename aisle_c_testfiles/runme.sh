#!/usr/bin/env bash
# Requires TT_METAL_HOME. Replace <YOUR_BH_GALAXY_HOST> below with the same hostname as rank 64 in
# blitz_decode_pipeline_rank_file_superpod. See README.md.

cd "${TT_METAL_HOME}"

export HOSTSP=bh-glx-c01u02:4,bh-glx-c01u08:4,bh-glx-c02u02:4,bh-glx-c02u08:4,bh-glx-c03u02:4,bh-glx-c03u08:4,bh-glx-c04u02:4,bh-glx-c04u08:4,bh-glx-c07u02:4,bh-glx-c07u08:4,bh-glx-c08u02:4,bh-glx-c08u08:4,bh-glx-c09u02:4,bh-glx-c09u08:4,bh-glx-c10u02:4,bh-glx-c10u08:4,bh-glx-c05u02:1

export HOSTS=bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08,bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08,bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08,bh-glx-c07u02,bh-glx-c07u08,bh-glx-c08u02,bh-glx-c08u08,bh-glx-c09u02,bh-glx-c09u08,bh-glx-c10u02,bh-glx-c10u08

mpirun --host "${HOSTS}" tt-smi -glx_reset

mpirun-ulfm --host "${HOSTS}" --mca btl_tcp_if_include ens5f0np0 --tag-output ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path /data/scaleout_configs/SP5/cabling_descriptor.textproto --deployment-descriptor-path /data/scaleout_configs/SP5/deployment_descriptor.textproto --send-traffic --num-iterations 5

# 1 prefill rank + 64 decode stages; KV only prefill → decode rank 0 (see example_disaggregated_prefill_decode_cross_context.cpp).
# Hardware: rankfile + HOSTSP; aisle host-to-host rank-bindings mapping (see README.md).
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
  --tcp-interface ens5f0np0 \
  --mpi-args "--map-by rankfile:file=aisle_c_testfiles/blitz_decode_pipeline_rank_file_superpod --bind-to hwt:overload-allowed --host ${HOSTSP} --tag-output" \
  --rank-bindings-mapping aisle_c_testfiles/disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml \
  "${TT_METAL_HOME}/build/programming_examples/distributed/example_disaggregated_prefill_decode_cross_context"
