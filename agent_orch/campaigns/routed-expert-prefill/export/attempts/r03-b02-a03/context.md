## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — the job, rules, all 40 nodes' index.
- kernels/adaptive_chunk.hpp — layouts (10/5x2/mixed), two-phase planner, cost model. kimi-k3's model rejects a mixed phase B (2820 vs 0.95 x 2846). The measured tail matches the model: 3 groups run the 6th expert in ~154 µs.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — GU and DN per-K-block handshakes. The receiver acks block kb+1 only after valid(kb). The weight read of kb+1 starts only after all receivers ack.
- kernels/compute/fused_swiglu.cpp — GU K-loop (1 x w subblocks, reuse_a), last-block PACK pipeline, multiply_phase (SiLU), down matmul_phase, phase zones RE-GU/RE-ACT/RE-DN on TRISC_1.
- unified_routed_expert_ffn_program_factory.cpp — subblock picks (gu_sub_w = largest divisor of per_core_N_gu <= 8), L1 fit, ring depths.
- tt_metal/hw/inc/api/compute/matmul.h — matmul_block reads the ARC FW scratch word each call; odd -> llk_math_matmul<MM_THROTTLE_MAX=5>.
- tt-llk llk_math_matmul.h — throttle level 5 caps throughput at 33%. reuse_a = ct >= rt, so in1 is streamed into SrcA once per tile-mult whenever rt = 1.
- tt-llk llk_unpack_AB_matmul.h — operand A -> SrcB, B -> SrcA. With rt = 1 the bf4 weight tile is unpacked per mult.
- tt_metal/tools/profiler/perf_counters.hpp, hw/inc/internal/tt-1xx/blackhole/hw_counters.h, tensix.h — counter banks, counter_sel values, cntl/out register addresses, start/stop/read protocol.
- tech_reports/GEMM_FLOPS — LoFi ideal 16 cycles per tile; BH big GEMMs reach up to 96%.
- tech_reports/PerfCounters/perf-counters.md — counter semantics (FPU util, SrcA/B valid waits, unpacker busy).
- tests/.../test_routed_expert_prefill.py + model configs — per-chip expert counts and tokens per expert. mistral/mm3: 4 experts x 5 tiles. gpt-oss: 4 x 8. dsv3/glm: 8 x 5. kimi-k3: 28 x ~3.

## Nodes consulted
- r03-b02-a02 (parent) — x-owner move was neutral. Its next #2 asked for exactly this kind of measurement.
- r03-b02-a01, r03-b01-a01/a02 — kernel config buffer size limit. That is why the instrumentation is profiler-only and size-checked (+1.5 KB TRISC1 under the profiler, 0 without).
- r03-b04-a01/a02 — NCRISC zones record; TRISC_0 zones do not. Gate read is short; block period is 9-16 µs.
- r03-b03-a01/a02 — DM counters (re_prof.hpp pattern: TS_DATA two counters per record). Every weight stream must stay split across NoCs.
- r02-b02-a03 — the op is bound by aggregate DRAM bandwidth on the big shapes; look-ahead only redistributes it.
- r02-b04-a03 — 1-row layout, kimi-k3 -2.3% (DRAM-bound at ~408 GB/s).
- r01-b02-a04, r01-b03-a03, r01-b04-a03 — subblock shape / per-call overhead / deeper rings don't move the ~30-40 cycles per tile-mult.
- r01-b01-a02 — per-core tilize removal worth ~5% on gpt-oss.
- r02-b04-a04 — SiLU body (25 instr); its PACK port is worth ~1 µs per expert by my estimate.

## Own analysis of the parent's profile (reports/r03-b02-a02)
Scripts are in /tmp/r03b02a03/:
- zones.py: per-shape phase sums.
- units.py: per-row, per-unit GU/DN.
- skew.py / col.py: within-row GU-end skew. On kimi-k3 col 10 finishes ~11 µs early; that is benign.
- bw.py: chip DRAM throughput timeline. kimi-k3 steady 400-433 GB/s; mistral 335-354; mm3 ~375.
- lot.py / phb.py / call.py: per-call group lottery. dsv3 phase-B groups start anywhere in 268-367 µs; one call had one group 85 µs slow.
- maxcore2.py / rowcol.py: which core is critical. gpt-oss col 10 has no N work but owns x blocks on group row 1.

## Docs / external references
- None outside the repo.
