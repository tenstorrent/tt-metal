## Files read
- `agent_orch/WORKER.md`, `agent_orch/campaigns/routed-expert-prefill/campaign.yaml`: job, metric, allowed paths, gate.
- `history.md`, plus every node's proposal.md / reflection.md (dumped from the tags into /tmp/r04b04/*.md).
  Round 3 read in full, rounds 1-2 by reflection.
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: the gate/up loop (steps 1-3 per K-block),
  the row-shared x owner multicast, the TILE x sender, the phase/expert/chunk structure.
  - Receivers ack both streams at the top of block kb, then wait both valids.
  - So block kb+1's acks on BOTH streams wait for both streams' block kb.
  - The x owner of kb+1 multicasts only in its own iteration kb+1, after its own w(kb).
- `kernels/dataflow/unified_routed_expert_ffn_writer.cpp`: the IN1_WRITER_MCAST gate/up multicast.
  It does `in1_ready.wait(R); set(0)`, multicast, `in1_valid` set_multicast, per block, serial with
  its own up read.
- `kernels/compute/fused_swiglu.cpp`: the matmul_phase / fused GU loops, to confirm the compute side
  consumes x / gate / up per K-block and is untouched by a reader-only change.
- `kernels/adaptive_chunk.hpp`: `x_owner_col` / `x_owner_slots`. Owners rotate over the row's
  non-weight-sender columns, so consecutive blocks have different owners and an owner's blocks are ≥5
  apart.
- LLK, to rule the compute side in or out first:
  - `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`, `llk_math_matmul.h`,
    `common/inc/ckernel_template.h`, `cmath_common.h` (set_dst_write_addr);
  - `tt_metal/hw/inc/api/compute/matmul.h` (matmul_block, the dynamic-throttle wrapper),
    `sentinel/compute_kernel_sentinel.h` (state_configure is a no-op when disabled).
  - Disassembled TRISC1 (local compile, /tmp/r04b04/sz): the unthrottled matmul_block issues only
    `add; sw (SETC16); ttmop` per tile, plus ~10 instructions per call. So MATH RISC issue is not
    the ~31 cycles/mult.
- `tt_metal/hw/inc/api/semaphore.h`, `api/dataflow/circular_buffer.h`, `dataflow_api.h`:
  `Semaphore::value()` (fresh read), `pages_reservable_at_back` / `pages_available_at_front`
  (non-blocking).
- `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md`, `tests/ttnn/unit_tests/benchmarks/test_benchmark.py`:
  - BH reaches 87-89% of LoFi peak with bf8 / bf4 and the same matmul_block LLK, so the hardware and
    LLK can stream at ~18 cycles/tile.
  - So the ~31 cycles/mult measured here is environmental (data delivery), not the engine.

## Nodes consulted
- r03-b02-a03 (root): FPU 16.1 cycles/mult when busy and ~50% duty; the FW throttle is never engaged.
- r03-b01-a04, r03-b02-a04, r03-b03-a04: operand swaps and reuse modes leave cyc/mult at ~31.
  The engine is not the limit; don't touch compute.
- r03-b03-a02: DM wait counters. Receivers wait on x ~2x as long as on weights (dsv3 202 / 100 µs).
  All-NoC-0 reads are bad.
- r03-b04-a02: the gate/up block period has 3-8 µs that is neither the sender's DRAM read nor matmul.
  x reads are never late.
- r03-b04-a04: down read issued before the receivers' acks is a real DN saving. Its next #2 suggests
  decoupling the x ack.
- r01-b03-a01, r01-b04-a01: sender-side read-ahead lost 5-9%. Hypothesis: lock-step grid coupling.
  Never tested directly until this node.
- r02-b02-a02, r03-b02-a02: VCs and x-owner placement are neutral, so not repeated.

## Docs / external references
- None beyond the in-repo headers above.

## Tooling
- /tmp/r04b04/cc/build.sh (adapted from r03-b03-a02's): local NCRISC compile of the reader for the
  row-major / TILE / legacy-up / bias variants, with or without the profiler, plus a size check.
  Reader text +1.4 KB with the profiler (13.0 → 14.6 KB), +1.3 KB without.
- /tmp/r04b04/sz/kc.sh: local TRISC compile + disassembly.
