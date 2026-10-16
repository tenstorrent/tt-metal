# r02-b01-a04: combine r02-b01-a03's PACK-thread pipelined activation with r02-b03-a03's lineage (mixed-height layouts + single-round two-phase plan + op-local short SFPU bodies), running the short SFPU bodies on PACK

## Motivation
Two orthogonal wins exist on two branches of round 2 and have never been combined:
- r02-b01-a03 (this parent, 1.5275): the gate/up activation's SFPU op runs on the PACK thread, pipelined
  per LW-subblock into the last gate/up K-block, so it overlaps the next subblock's last-block matmul on
  MATH. Measured: minimax-m3 -5.2%, gpt-oss -3.2%, glm -2.3%, kimi-k2.7 -1.5% vs its parent.
- r02-b03-a03 (best, 1.5435) on top of r02-b03-a02: the schedule (r02-b04-a01's mixed-height 3+3+2+2
  layouts, worth -18% on gpt-oss, plus the two-phase plan when the leftover fits one phase-B round,
  worth -3..-4% on kimi-k2.7 / dsv4-pro) and op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU SFPU
  bodies with hoisted constants and replayed unrolled loops (gpt-oss -4.8%, minimax-m3 -4.2%).
The parent's reflection names this combination as next step #1. Neither branch touched the other's
code: b01-a03 changed the GU phase of fused_swiglu.cpp, b03 changed adaptive_chunk.hpp, the host
topology sets, the dataflow kernels' plan walk, the expert mailbox and the SFPU op bodies.

## Mechanism
- Apply the r01-b04-a04 -> r02-b03-a03 code diff onto this parent (3-way). adaptive_chunk.hpp, reader,
  writer and program factory apply cleanly; fused_swiglu.cpp conflicts are resolved by hand:
  - take b03's plan_launch / per-expert (count, group_rows) mailbox and per-expert chunk_M;
  - keep b01-a03's PACK-thread pipeline; replace its PACK_ACT_* stock calls by b03-a03's op-local
    `re_act::binary_act` / `binary_act_init` (binary_act_sfpu.h included for TRISC_PACK instead of
    TRISC_MATH). The replay buffer is per thread; on PACK it is otherwise used only by the fast-tilize
    pack init, which re-records it on every owned x block (InitAndUninit), so overwriting it from the
    SFPU between tilizes is harmless.
- Keep r02-b01-a02's `start` fold and the RE-* MATH zones (inert).
Files: unified_routed_expert_ffn/device/kernels/{adaptive_chunk.hpp, compute/fused_swiglu.cpp,
compute/binary_act_sfpu.h, dataflow/*_reader.cpp, dataflow/*_writer.cpp},
device/unified_routed_expert_ffn_program_factory.cpp.

## Why this is not a repeat
No node combines the PACK-thread activation pipeline with the b03 schedule or the short SFPU bodies.
r02-b03-a02 was a combination of two schedules; this combines the schedule line with the compute-kernel
line. Numerics: both pieces were measured bit-identical; the pipeline only moves where the same sfpi
function runs.

## Expected effect and risk
Expected ~1.55-1.56: the schedule gains carry over as in r02-b03-a02/a03; the pipeline adds
-2..-5% on gpt-oss / minimax-m3 (less than on b01 since the shorter SFPU body now hides less), ~-1..-2% on
glm / kimi-k2.7 (SiLU overlap). Risks: replayed SFPU on PACK colliding with pack replay use (hang or
garbage -> PCC fail), JIT errors from the merge, a plan/mailbox mismatch between the parent's
start-fold reader/writer and b03's plan walk (hang). Judge by score vs r02-b03-a03 and per shape.
