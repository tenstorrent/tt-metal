#!/usr/bin/env bash
# Requires TT_METAL_HOME. Replace <YOUR_BH_GALAXY_HOST> below with the same hostname as rank 64 in
# blitz_decode_pipeline_rank_file_superpod. See README.md.

cd "${TT_METAL_HOME}"

export HOSTSP=bh-glx-d03u08:4,bh-glx-d03u02:4,bh-glx-d04u02:4,bh-glx-d04u08:4,bh-glx-d06u08:4,bh-glx-d06u02:4,bh-glx-d05u08:4,bh-glx-d05u02:4,bh-glx-d07u02:4,bh-glx-d07u08:4,bh-glx-d08u02:4,bh-glx-d08u08:4,bh-glx-d09u02:4,bh-glx-d09u08:4,bh-glx-d10u02:4,bh-glx-d10u08:4,<YOUR_BH_GALAXY_HOST>:1

# 1 prefill rank + 64 decode stages; KV only prefill → decode rank 0 (see example_disaggregated_prefill_decode_cross_context.cpp).
# Hardware: rankfile + HOSTSP; aisle host-to-host rank-bindings mapping (see README.md).
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
  --tcp-interface ens5f0np0 \
  --mpi-args "--map-by rankfile:file=aisle_d_testfiles/blitz_decode_pipeline_rank_file_superpod --bind-to hwt:overload-allowed --host ${HOSTSP} --tag-output" \
  --rank-bindings-mapping aisle_d_testfiles/disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml \
  "${TT_METAL_HOME}/build/programming_examples/distributed/example_disaggregated_prefill_decode_cross_context"
