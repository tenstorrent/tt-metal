## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the job, the metric and the allowed paths.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: the shapes, the routing (uniform top-k, seed 0), and the ROW_MAJOR bf16 dispatch buffer with bf4 ND-sharded weights.
- unified_routed_expert_ffn.cpp / device_operation.cpp / types.hpp: the op entry, validation, and the fixed 11x8 grid (kCoreGridX/Y).
- device/unified_routed_expert_ffn_program_factory.cpp: CB sizing and the L1 fitter (per_core_M_max=4, or 8 for kimi-k3), sender placement (weight sender row gx%8, x sender col (gy+1)%11), and the runtime-arg layouts.
- kernels/adaptive_chunk.hpp: chunk / per_core_M picker hard-wired to kGridY=8, plus the count-0 skip contract.
- kernels/compute/fused_swiglu.cpp: per-expert loop, UNPACK->MATH/PACK mailbox count handshake, matmul with h=1 subblocks, and per-K-block tilize of x on every core.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: counts bcast; per-K-block x stick reads plus row mcast; gate weight read plus column mcast (handed to the writer when IN1_WRITER_MCAST); down weight split; activated row mcast with a rotating sender.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT/DOWN_SPLIT weight reads, the NoC-1 gate/up multicast, and the output drain.
- tt-llk blackhole ckernel.h: the mailbox is a blocking hardware FIFO, so I keep one write per worked expert.
- $DREAM_HOME/routed-expert-prefill/reports/r01-b01-a01 + eval/summary.md: this node's result (used for the effective-bandwidth numbers in reflection.md).
- $DREAM_HOME/routed-expert-prefill/reports/baseline_1: per-shape timings; all RISCs busy for the whole kernel; 110 available worker cores, 88 used; 1350 MHz.

## Nodes consulted
- None. This is the first node of the campaign.

## Docs / external references
- /tmp/rexp_b01/sim.py (scratch, not committed): replays the test's routing to get per-expert tile counts and the MAC-row gain for 1/2/4 groups.
