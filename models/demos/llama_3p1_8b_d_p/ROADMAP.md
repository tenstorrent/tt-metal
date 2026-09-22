<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Shared prefill integration milestones

Follow [Adding a prefill model](../common/prefill/docs/ADDING_A_PREFILL_MODEL.md).
The current acceptance configuration is SC1, 2K capacity and two slots.

| Milestone | Status | Evidence |
|---|---|---|
| Isolated branch | Done | Original disaggregated and model-PR branches preserved |
| Compact tests | Done | Custom bridge harness removed; numerical model coverage preserved |
| Reference traces | Done | Two distinct 2K passages; independent FP32 HF K/V for all 32 layers |
| Address table | Done — passed | 3 tests; exact synthetic page mapping, protobuf and real writer readback |
| Shared runner/producer | Done — passed | Both slots complete; all-layer golden PCC exceeds 0.99; clean shutdown |
| Standard launcher | Done — passed | `run_multirank_pcc.sh llama31 sc1`: exit 0, 1/1 rank verdict |
| Host regressions | Done — passed | 29 compact host checks |
| Larger-capacity checklist | Done | [4K–64K plan](docs/runner-capacity-plan.md) |
| Larger-capacity execution | To do | Separate shared-runner acceptance at 4K, 8K, 16K, 32K and 64K |

See [commands and pass criteria](docs/runner-integration.md) and the
[recorded SC1 results](docs/runner-sc1-validation.md). Saved historical model
results remain in `docs/validation-2k.md` and `docs/performance-prefill.md`.
