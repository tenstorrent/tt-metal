<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model bring-up kit

Everything needed to drive a new model bring-up the way one was actually done: recipe, log
templates, the threshold primitive, the citation verifier, and the landmines.

**Start at [`BRINGUP_RECIPE.md`](BRINGUP_RECIPE.md).** Read it end to end before writing code. It is
written to be executed by an agent or a person, phase by phase, each phase ending in a gate with a
measured number and a raw log.

## What is here

| Path | What it is |
|---|---|
| `BRINGUP_RECIPE.md` | The recipe. Phases P0-P8, P10, P9; the logging protocol; how to set a threshold; the gate index; the failure playbook. |
| `LANDMINES.md` | Traps found the hard way, grouped by how they fail. The silent ones are the expensive ones. |
| `templates/` | The five log files a bring-up keeps, empty, with their required fields. |
| `examples/noise_floor.py` | The threshold primitive. Copy it into your package's test helpers. |
| `examples/verify_citations.py` | Machine-checks every `path:line` in your code and docs. |
| `examples/module_test_vs_ref.py` | The shape of a gate test: identical weights both sides, computed floor, negative control. |

## What is NOT here, on purpose

No model implementation. The recipe points at the two prefill packages already in this tree —
`models/demos/gpt_oss_d_p/` and `models/demos/minimax_m3/` — as the structural templates for
`MeshConfig`, `CCLManager`, a dense MLP with its collective tail, attention, and a block-cyclic KV
cache. Those are the real reference; duplicating them here would only let them drift.

## The three rules that carry the most weight

1. **Every judgement call is logged**, with a falsifier and a blast radius. If a reviewer cannot
   reconstruct *why* a number was chosen from the logs alone, the logging failed.
2. **Gate on the gap to a computed noise floor**, never on a PCC copied from another
   implementation — its reference may share the device's own rounding. `examples/noise_floor.py`
   and recipe section 2.
3. **A gate with no raw log did not happen.**

## Provenance

Distilled from a Llama-3.1-8B prefill bring-up on a 4x8 Blackhole Galaxy: 11 phases, 30+ gates, 169
tests, a full-model KV cache PCC of min K 0.99789 / V 0.99134 against an fp32 golden, and three
race-free runs producing one hash. The numbers quoted throughout the recipe and `LANDMINES.md` are
from that run and were measured, not estimated.
