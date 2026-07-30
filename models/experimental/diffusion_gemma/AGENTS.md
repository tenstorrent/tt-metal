# DiffusionGemma bring-up — agent guide

Status: current — terse working context for agents starting on this branch.
Owns: the no-shared-edits rule and the project conventions.
See also: [plan + execution contract](plan.md) (read first) · [refuted list](doc/REFUTED.md) · [decision fidelity](doc/decision_fidelity/README.md) · [perf hub](doc/optimize_perf/README.md) · [serving hub](doc/vllm_integration/README.md)

**Read [plan.md](plan.md) before inferring any default, number or flag from anywhere else in this
tree.** It carries the current execution contract, the environment recipe and the shipped defaults;
everything dated is evidence, not guidance.

## What it is

- HF [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
  (released 2026-06-11, Apache-2.0), transformers class `DiffusionGemmaForBlockDiffusion`,
  `model_type=diffusion_gemma`. Multimodal input; bring up **text-first**. Fine-tuned from
  `google/gemma-4-26B-A4B`, so only the generation procedure plus a few weights differ.
- Work branch `diffusion-gemma-function` (earlier work on `zni/diffusion-gemma-bringup`); tracking
  issue tenstorrent/tt-metal#47452, label `DiffusionGemma`.
- Model config, the three-phase generation procedure and the reuse/"do NOT rebuild" list:
  [plan.md §1–§2](plan.md).

## The no-shared-edits rule

**Never edit `models/demos/gemma4/` or any other shared directory.** All DiffusionGemma fixes stay
inside `models/experimental/diffusion_gemma/`. When a feature appears to need a gemma4 change, **copy
the file into `diffusion_gemma/` and edit the copy** (`tt/concat_moe.py`, `tt/chunked_prefill.py` and
`tt/commit_decode.py` all exist for this reason). `check_no_shared_gemma4_edits.sh` is the gate,
evaluated against the `origin/main` merge-base; the branch currently carries two inherited violations
flagged for owner action — see [decision fidelity](doc/decision_fidelity/README.md), and the related
"backbone untouched" contradiction in [plan.md §6](plan.md#6-open-items-and-contradictions).

## Where the rest lives

- Denoise mask geometry, the `attn_mask` ⊥ `sliding_window_size` rule, the causal-only gemma4
  chunked-prefill workaround, and the W2b long-context result: [plan.md §3](plan.md).
- QB2 environment, hardware recovery and checkpoint paths: [plan.md §5](plan.md).
- Launch flags, the block-granular emission contract and the served-context ceiling:
  [serving hub](doc/vllm_integration/README.md).
- TT-plugin constraints (spec-decode hard-blocked, no scheduler chunked prefill, phase-based
  continuous batching, APC force-disabled for sliding-window models, model-owns-forward/attention/KV):
  [vllm_native_plan](doc/vllm_integration/vllm_native_plan.md). Scheduler chunked prefill is a
  *different* thing from DG-local model-side chunked/ragged prefill — name which path a benchmark used.
- The bf16 decision floor and the GPQA measurement traps:
  [decision fidelity](doc/decision_fidelity/README.md).

## Gotchas

- **Determinism:** token-for-token PCC vs torch needs the torch run's exact Gumbel noise **and**
  renoise ids injected; on-device RNG will not bit-match ([plan.md §7](plan.md)).
- gemma4 has **no entropy computation** — the entropy harness is net-new, and the PCC harness must
  validate diffusion *decisions* (entropy, argmax, accept mask), not just logits.
- A clean short-prompt causal PCC does **not** de-risk the 256K QB2 fit for the diffusion path: the
  budget must also cover per-step canvas K/V scratch and the non-causal long-context mask buffers
  ([QB2 memory budget](QB2_MEMORY_BUDGET.md)).

## Conventions

- Commit messages must **NOT** include a `Co-Authored-By` trailer.
- Commit and push after each meaningful verified batch; do not accumulate a large uncommitted pile.
  (bhqb is set up for interactive Claude Code: `~/.config/claude/env` holds `ANTHROPIC_API_KEY`,
  sourced from `.zshrc`.)
- Do not skip device tests by default for device-facing changes; when a skip is genuinely
  unavoidable, record the reason.
