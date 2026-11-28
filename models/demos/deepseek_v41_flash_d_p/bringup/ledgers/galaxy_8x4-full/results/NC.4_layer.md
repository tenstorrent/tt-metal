# NC.4: the MoE block's prepared programs (not ported)

Stopped before any code: none of the three programs runs on the shipped model path (`MOE_BLOCK_SRC=block`, the
default since owner 2026-11-18). Evidence and the decision: BREADCRUMBS.md, section "NC.4"; known_issues DS41-401.

| program | where it runs | calls a chunk (shipped path) | host us before / after | host ms a chunk saved |
|---|---|---|---|---|
| tail_a2a.TailA2A (via tt/prepared_tail.ShapeKeyedTailA2A) | MOE_BLOCK_SRC=carried only | 0 | n/a (not ported) | 0 |
| fused_head (carried FusedHead) | MOE_BLOCK_SRC=carried only | 0 | n/a (not ported) | 0 |
| tail_engine.TailEngine | a latent model only (K3); DSV41_FLASH has none | 0 | n/a (not ported) | 0 |

On the shipped path the same steps already run as native C++ ops of `ttnn.bringup.moe_block` (moe_head; moe_tail's
tail all-to-all / tail engine plans in moe_tail/device/moe_tail_plan.cpp), or as stock ops under OWN_ROWS.
