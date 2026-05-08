#!/usr/bin/env bash
# Requires TT_METAL_HOME. Replace <YOUR_BH_GALAXY_HOST> below with the same hostname as rank 64 in
# blitz_decode_pipeline_rank_file_superpod. See README.md.

cd "${TT_METAL_HOME}"

export HOSTSP=bh-glx-120-a02u08:4,bh-glx-120-a02u02:4,bh-glx-120-a03u02:4,bh-glx-120-a03u08:4,bh-glx-120-a04u02:4,bh-glx-120-a04u08:4,bh-glx-120-a05u02:4,bh-glx-120-a05u08:4,bh-glx-120-a06u02:4,bh-glx-120-a06u08:4,bh-glx-120-a07u02:4,bh-glx-120-a07u08:4,bh-glx-120-a08u02:4,bh-glx-120-a08u08:4,bh-glx-120-a09u02:4,bh-glx-120-a09u08:4,<YOUR_BH_GALAXY_HOST>:1

# 1 prefill rank + 64 decode stages; KV only prefill → decode rank 0 (see example_disaggregated_prefill_decode_cross_context.cpp).
# Hardware: rankfile + HOSTSP; aisle host-to-host rank-bindings mapping (see README.md).
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
  --tcp-interface ens5f0np0 \
  --mpi-args "--map-by rankfile:file=aisle_a_testfiles/blitz_decode_pipeline_rank_file_superpod --bind-to hwt:overload-allowed --host ${HOSTSP} --tag-output" \
  --rank-bindings-mapping aisle_a_testfiles/disaggregated_prefill_decode_host_2_host_rank_bindings_mapping.yaml \
  "${TT_METAL_HOME}/build/programming_examples/distributed/example_disaggregated_prefill_decode_cross_context"
