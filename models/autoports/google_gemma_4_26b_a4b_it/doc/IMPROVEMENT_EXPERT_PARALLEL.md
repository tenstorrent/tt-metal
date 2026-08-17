# The implementation change most likely to move this model: expert-parallel MoE

Operator analysis, 2026-08-17. This is about improving the port, not the harness.

## The problem, measured

This port has the **worst TP4 efficiency in the corpus**: 26.0–29.2 % across both
layer kinds and both batch sizes, i.e. a 1.04–1.17× speedup on four devices.
Stage 05's collective re-tune recovered only 6.66 % / 11.12 %, well below the
15–32 % that collective-bound starters recovered elsewhere, because **this model
is not collective-bound — it is compute-granularity-bound.**

The mesh plan shows why:

| role | global | per device (TP4) |
|---|---:|---:|
| dense intermediate | 2112 (padded 2176) | 544 |
| **expert intermediate** | **704 (padded 768)** | **192** |

A 26 B model with only **2,816 hidden** puts its parameters in **128 experts**,
not in width. Splitting each expert's 704-wide intermediate four ways leaves a
**192-wide matmul per device** — far too narrow to fill a Blackhole core grid.
Sharding harder cannot fix that; it is the cause.

## The change

Stop tensor-fracturing the experts. Use **expert parallelism (EP=4)**: 32 of the
128 experts resident per chip, each selected expert's intermediate kept **whole at
704** rather than cut to 192. Router and top-8 mask stay replicated as now; the
per-layer hidden all-reduce over expert output is replaced by dispatch/combine
around the sparse expert call.

Width per device goes **192 → 704, a 3.7× wider matmul**, which is exactly the
axis the profile says is starved.

## Why this is worth measuring rather than assuming

`doc/multichip_decoder/mesh_plan.md` **rejected EP=4 analytically, not by
measurement**:

> "Rejected: EP=4 with 32 experts/chip as the first path. Top-8 tokens would need
> dispatch/combine collectives around each sparse expert call; at batch 1 the
> communication and imbalance risk dominate, while TP fractures every selected
> expert's DRAM traffic uniformly."

That is the same reasoning, and the same outcome, as a documented mistake in this
fleet. From the QB2 cross-model study on **gpt-oss-20b**, the MoE port on this
same 1×4 Blackhole mesh:

> "The TP2 run dropped the compiler's expert-parallelism *analytically* and
> tensor-fractured; the TP4 rerun **implemented both and measured them**, and
> **EP4 (eight whole experts/rank, the compiler's original choice) won
> decisively** (decode 0.599 vs TP-fracture 0.656 ms; prefill 26.7 vs 39.8). So
> the compiler's expert-parallel prior was **right**, and the analytical TP2 drop
> was the inferior call — vindicating the 'measure the EP baseline before
> overriding it' recommendation."

So the corpus already contains the experiment, the verdict, and an explicit
recommendation not to do what this port did. And the case here is **stronger than
gpt-oss's**, because gpt-oss had 32 experts top-4 with a wider intermediate,
whereas this model's TP4 split lands at 192 columns — the narrowest expert matmul
in the fleet, and the reason its efficiency is the lowest.

The rejection's own premise is also testable rather than self-evident:
"communication and imbalance risk dominate **at batch 1**". gpt-oss measured that
trade and EP still won; and this port's headline serving profile is batch 1, where
its efficiency is 26–28 %.

## Expected effect, stated honestly

gpt-oss gained **8.7 % decode and 33 % prefill** from EP over TP-fracture. Do not
assume the same number here: the layer counts, expert counts and routing differ.
What is defensible is the direction — the starved axis is per-device matmul width,
EP widens it 3.7×, and the one measured comparison on this mesh favoured EP.

Two secondary effects to watch, both real:
- **Load imbalance.** Top-8 of 128 experts over 4 chips means a token's selected
  experts are unevenly distributed; per-step cost becomes the max over chips, not
  the mean. gpt-oss carried the same risk at top-4 of 32 and still won.
- **Weight residency.** EP stores 32 whole experts per chip instead of a quarter
  of all 128; total per-chip expert bytes are unchanged, so the memory envelope
  and the 262,144-token context contract should hold. Verify against
  `doc/context_contract.json` rather than assuming.

## How to test it without a rewrite

The decode path already dispatches only the top-8 selected experts through
`ttnn.sparse_matmul`, so the sparse execution machinery exists; what changes is
**which slice of the expert tensors each device holds** and the collective around
the call. Suggested order:

1. Measure the current TP-fractured layer as the control at B1 and B32 — the
   numbers are in `doc/optimized_multichip_decoder/README.md` (sliding 1.070 ms,
   full 1.114 ms at B1).
2. Build the EP variant at the **decoder-layer** level only, as stage 04 did, and
   compare like-for-like at the same batches with the same precision policy.
3. Keep the replicated router and the existing top-8 mask so the only variable is
   expert placement.

If EP wins at the layer level, the full-model gain follows the same 25 sliding + 5
full structure, and it is the largest single lever available on this port: a
26–29 % efficiency starting point is 2.5× off the fleet's best (Qwen3.6-27B's
linear layers reach 89.3 % at B32).

## What this is not

This does not touch the GPQA accuracy blocker, which is a separate and
still-unresolved question — see `HANDOFF.md`. Performance and that defect are
independent workstreams; do not let one gate the other.
