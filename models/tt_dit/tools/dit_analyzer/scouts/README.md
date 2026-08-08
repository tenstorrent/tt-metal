# scouts — whole-pipeline entry points & verify-loop toolkit

Hand-written drivers that assemble a whole DiT-family pipeline under the metadata shim and
report collective redundancies. They stand in for running the real `pipeline_*.generate()`
(see `../GALAXY_PLAN.md` → "Where the code lives" for why, and the blockers writeup).

| file | what it does |
|---|---|
| `scout_h3_pipeline.py` | The whole MiniMax-H3 pipeline as a connected DAG — encoder → DiT → {video VAE, audio VAE}. `python3 scout_h3_pipeline.py [2x4\|4x8\|prod]`. Prints the ranked report. |
| `render_full.py` | Re-renders the same graph with *all* findings (not just the top 8). |
| `trace_dit.py` | Builds the DiT stage standalone and dumps a finding class with full proofs — the per-stage trace template. |
| `probe_348.py` | Example of tracing one finding (the output-head `participant_shrink`) to real-vs-artifact before spending device time — the verify-loop in miniature. |

> **These need the H3 model code.** They import the DiT / VAE / audio-VAE, which live on the
> MiniMax-H3 integration branch, **not** on the analyzer branch. Run them from a tree that has
> both the tool and the models — see `../GALAXY_PLAN.md`. The conform harnesses one level up
> (`../conform_encoder.py`, `../conform_dit_heads.py`) do **not** need this — they run from the
> analyzer branch alone.
