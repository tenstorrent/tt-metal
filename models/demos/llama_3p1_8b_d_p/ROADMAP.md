<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Shared prefill integration milestones

Follow [Adding a prefill model](../common/prefill/docs/ADDING_A_PREFILL_MODEL.md).
Each milestone needs its stated evidence before completion.

| Milestone | Acceptance |
|---|---|
| Isolated branch | Original disaggregated and model-PR branches unchanged |
| Compact tests | Retain model numerics; use ordinary runtime/table tests and common runner coverage |
| Reference traces | Two distinct 2K passages; FP32 HF K/V for all 32 layers |
| Address table | All K/V configs, slots, layers and pages match independent live tensors; protobuf roundtrip preserves owners and addresses |
| Shared runner/producer | H2D requests fill two slots; independent full-prefix golden PCC passes; runner exits cleanly |
| Standard launcher | `run_multirank_pcc.sh llama31 sc1` passes the per-rank verdict gate |
| Larger capacities | List 4K–64K follow-up checks after 2K passes |

See [commands and pass criteria](docs/runner-integration.md). Saved historical model
results remain in `docs/validation-2k.md` and `docs/performance-prefill.md`.
New shared-runner results must be recorded separately; host checks alone are not
hardware acceptance.
