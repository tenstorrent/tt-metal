# GPT-OSS 20B — what is being optimized (read this first)

> ⚠️ **This directory contains TWO different lineages.** Don't assume any file here is "the current
> optimized model" — most of `tt/` and `doc/` is an **older, superseded experiment**. The current
> optimize work lives in a different repo/branch (see below).

## TL;DR

- **CURRENT** prettify → optimize work optimizes the **prettified multichip model** and lives on
  **ttnn-models branch `mvasiljevic/gpt-oss-optimize`** (github `svuckovicTT/ttnn-models`,
  `openai/gpt-oss-20b/model/graph_0/model_ttnn.py`). It is *driven* by the Codex pipeline defined here in
  **`.agents/prompts/alchemy_optimize/`** (this tt-metal repo), but the model it edits/produces is in
  ttnn-models, not in `tt/` here.
- **OLD / superseded (2026-07-25):** `tt/functional_decoder.py`, `tt/optimized_decoder.py`, and
  `doc/{functional,optimized}_decoder/` are the earlier single-device autoport experiment — a decoder
  ported from the EmitPy package, then optimized *from that functional_decoder*. Kept for history only.

## The two lineages

### 1. CURRENT — prettify → optimize on the prettified multichip model  ✅ active

- **Input:** the *prettified* model from the project-alchemy prettify pipeline — a real 4-chip
  (tensor-parallel, 1×4 mesh) gpt-oss whose flat codegen forward was refactored into labeled **per-kind
  layer classes** (`ModelTTNN`, `GptOssAttention`, `GptOssMLP`, `GptOssDecoderLayer`) in
  `…/graph_0/model_ttnn.py`. It is self-contained (imports only `ttnn/torch/params/consteval`) and does
  **not** derive from `functional_decoder`.
- **Optimize:** the 3-stage Codex `multigoal` in **`.agents/prompts/alchemy_optimize/`**:
  `01-graph-fusing → 02-optimize-per-kind → 03-optimize-full-model`. It is model-agnostic — stage 2
  *discovers* the per-kind block classes and optimizes each. Every stage is gated by a **full-model PCC
  check** (`_full_model_check.sh`; PCC ≥ 0.98 vs the CPU golden, also reports traced TPS) — i.e. "check
  the full model at every stage", so multichip is never reinvented and every change is verified against
  the whole model rather than a cropped layer.
- **Where the optimized model + per-stage commits land:** ttnn-models branch
  **`mvasiljevic/gpt-oss-optimize`** (each passing stage is committed there with its PCC + TPS).
- **Results so far** (correctness-gated, real qb2 traced run):

  | stage | full-model PCC | trace TPS |
  |---|---:|---:|
  | baseline (prettified) | 0.999743 | 730 |
  | graph-fusing | 0.999778 | 858 (+17%) |
  | per-kind optimize | 0.998841 | 1076 (+47% vs baseline) |
  | optimize-full-model | (terminal stage) | — |

### 2. OLD — autoport `functional_decoder` → `optimized_decoder`  🗄️ superseded (2026-07-25)

- A **single-device** (1×1 Blackhole) decoder ported from the EmitPy package to
  `tt/functional_decoder.py`, then `tt/optimized_decoder.py` optimized **from** it
  (`from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import …`).
- This is the earlier "functional_decoder → optimize" approach (single-device, no multichip/full-model).
  It is **not** the prettified-model pipeline and is **not** what the current run optimizes. Retained only
  as worklog/reference: `doc/functional_decoder/`, `doc/optimized_decoder/`.

## Correctness oracle (both)

Full-model PCC vs a CPU golden is the correctness oracle. In the current pipeline it is the runner-side
gate after every stage, so an optimization is only kept if the whole model still matches the golden.
