## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, metric.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (all) — where each weight sender starts a
  gate/up and a down block (main path + r04-b02-a01's look-aheads), which NoC each RISC uses, where linked
  multicasts are open (the r07-b02-a01 deadlock rule), the counts broadcast (core (0,0) is the counts reader
  and every core waits for its multicast before planning).
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp (grep) — its weight reads are released only by
  the reader's up_go / down_go, so a gate in the reader holds both RISCs.
- device/kernels/adaptive_chunk.hpp (all) — layouts, group/row mapping, plan_launch, phase A / B.
- device/unified_routed_expert_ffn_program_factory.cpp (CB / semaphore / define / reader-arg sections) —
  16 semaphores max and 15 in use (so the board is a new CB c_21, not a semaphore), L1 margin 48 KB for small
  CBs, reader defines, my_nt_gu = gx, counts multicast rectangle start = logical (0,0).
- device/kernels/compute/re_hwprof.hpp — profiler-only MATH counters, made opt-in (r07-b03-a01's change).
- tt_metal/hw/inc/api/dataflow/dataflow_api.h — noc_semaphore_inc (non-posted atomic, only the atomic
  barrier waits for it), noc_async_read one-packet template, barriers invalidate the BH L1 cache.
- tt_metal/impl/kernels/kernel_types.cpp — reader = RISCV_1 on the preferred DRAM-read NoC (NoC 0 on BH).

## Nodes consulted
- Every node r01-r07 (history.md, all reflections; proposals of the ones below).
- r07-b02-a01 (parent code) — gradient now deterministic; per-row ends; linked-multicast deadlock rule.
- r07-b04-a01 — idle top group: the topmost ACTIVE group becomes the slow one; "cure has to be dynamic".
- r06-b02-a01 — static start stagger: top group can run much faster when others yield (mm3 259 -> 210 µs GU),
  but a static offset rotates the loser and costs aggregate rate.
- r06-b01-a01 — row NoC tilt: top rows are slowed by the other rows' traffic, not their own split.
- r01-b01-a03 — the only pacing attempt: balanced to ~1% but +30-50% from an 87-way non-posted multicast per
  block on the column-0 core inside its write barriers; this node's design avoids exactly that.
- r07-b03-a01 — RE_HWPROF opt-in frees ~1 KB TRISC1; kernel-config sizes (70656 B limit, k2.7 near it).
- r07-b01-a01 — plan-once + tie pass (likely a sibling's port; this node does not touch the planner).

## Measurements made for this node
- reports/r07-b02-a01 profile_log_device.csv with /tmp/r06b01a01/zones.py: per-row RE-GU/RE-DN for mistral,
  mm3, dsv3, kimi-k3 (tables in proposal.md).
- Local sizes (/tmp/r08b04/sz/b.sh, mistral-like CT args, profiler build): reader 18944 -> 19908 B (+964);
  TRISC1 (/tmp/r08b04/kc.sh) 14184 -> 13164 B (-1020) from the RE_HWPROF opt-in. Variants (TILE x, bias,
  legacy up, no writer multicast, non-profiler) all compile.

## Docs / external references
- none beyond the tt-metal sources above.
