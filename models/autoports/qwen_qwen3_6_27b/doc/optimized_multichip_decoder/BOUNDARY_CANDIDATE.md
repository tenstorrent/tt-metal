# Untested boundary candidate: full-norm hidden-sharded handoff

Operator note added between stage 05 and stage 06. Nothing here is measured on
this model; it records a candidate with cross-model evidence so the
optimized-full-model stage inherits it as a named target instead of rediscovering
it.

## What stage 05 selected, and what it tested

Stage 05 kept the **replicated BF16 hidden-5120 layer boundary**. The one
alternative it measured was the **coherent fractured residual**, which removes
the inter-layer collective entirely and was 3.82% slower across a traced
linear->full stack (5.332280 vs 5.136247 ms), so replicated was retained.

Removing the inter-layer collective requires each rank to normalise its own
shard, i.e. a **distributed norm**. That is the weaker of the two sharded
designs.

## The variant that was not measured

The design Qwen3-32B and Qwen2.5-Coder-32B both shipped is different: keep an
all-gather so the norm sees the full hidden vector, then hand off **sharded** to
the next consumer.

    RS (row-parallel output) -> AG -> full RMSNorm -> sharded stacked handoff

It does not remove the collective; it changes what crosses the boundary. On
Qwen2.5-Coder-32B a fair traced comparison put it ~9% ahead of replicated
(0.792 vs 0.870 ms).

`work_log.md` lists this family as the "highest-priority coherent-family retry"
("RS carried through residual/norm into next consumer") but no stage-05 artifact
records it being built or timed.

## Why it is worth a measurement here specifically

- Both models that shipped it are **64 layers**, the same depth as this one, and
  both are dense TP4 1x4 on the same lane.
- The cross-model finding is explicit that the winning sharded boundary remains
  an untested-better lever for every model that rejected **only** the fractured
  variant. With this stage, that list becomes six models.

## Why the expected value is uncertain, stated honestly

On this model the collectives are a smaller share of decode than on the two
Qwens: linear B32 TP4 is 4.4330 ms/layer of which two recurrent-state matmuls
are ~434 us each, so the boundary is not the dominant term. The ~9% seen
elsewhere should not be assumed here. It needs a fair traced comparison at B1
and B32 on a real stacked linear->full boundary, not an isolated probe -- the
stage-05 log notes a prior probe "gathered at the next projection and did not
carry the layout across a real stacked layer", which is exactly the unfair
comparison to avoid.

## Owner

Stage 07 (optimized-full-model) already owns residual layouts, collectives and
host boundaries across the assembled model, and is the natural place to measure
this. If it wins there, it changes the inter-layer contract for all 64 layers.
