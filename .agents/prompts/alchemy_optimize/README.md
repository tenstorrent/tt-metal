# alchemy_optimize — the Codex "optimize" half of the prettify → optimize pipeline

3-stage `multigoal` that optimizes the **prettified multichip model** (the labeled per-kind layer
classes produced by project-alchemy prettify), gating on a **full-model PCC check after every stage**.

- **Input / output model is NOT in this repo.** It's the prettified
  `openai/gpt-oss-20b/model/graph_0/model_ttnn.py` in **ttnn-models**, and each passing stage is committed
  to ttnn-models branch **`mvasiljevic/gpt-oss-optimize`** (github `svuckovicTT/ttnn-models`). This repo
  only holds the pipeline *definition*.
- Full picture, provenance (vs the old autoport `functional_decoder`/`optimized_decoder`), and results:
  `models/autoports/openai_gpt_oss_20b/README.md`.

## Stages (each `<stage>.txt` + gate `<stage>.check.sh`)
1. `01-graph-fusing` — `$graph-fusing` topology pass.
2. `02-optimize-per-kind` — **discovers** the per-kind block classes and optimizes each separately
   (`$optimize`/`$shard-advise`/`$datatype-sweep`). Model-agnostic.
3. `03-optimize-full-model` — integrate + chain L1 residual stream + traced TPS (terminal stage).

`_full_model_check.sh` runs the prettified model's `main.py` (PCC ≥ 0.98 vs CPU golden + trace TPS) and,
on pass, best-effort commits+pushes the model to the ttnn-models optimize branch.

## Run
```bash
python3 .agents/scripts/multigoal \
  --repo <tt-metal> --codex-bin ~/.local/bin/codex --codex-home ~/.codex \
  --sandbox danger-full-access --approval-policy never \
  --replace MODEL_DIR=<…/ttnn-models/openai/gpt-oss-20b/model/graph_0> \
  .agents/prompts/alchemy_optimize/01-graph-fusing.txt \
  .agents/prompts/alchemy_optimize/02-optimize-per-kind.txt \
  .agents/prompts/alchemy_optimize/03-optimize-full-model.txt
```
