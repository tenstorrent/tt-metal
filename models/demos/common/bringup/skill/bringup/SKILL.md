---
name: bringup
description: Start, watch or resume a gated model bring-up (prefill on Tenstorrent) with the framework in models/demos/common/bringup. Use when someone wants to bring up a new model, asks how a bring-up run is going, or wants to resume or fork one.
---

# /bringup: intake for the gated bring-up framework

You talk to the person, write the model spec, get it approved, and hand over to the orchestrator. You do not implement
anything yourself. Framework docs: `models/demos/common/bringup/README.md`.

## Start a new bring-up

1. Ask for what you cannot find out yourself, in one message: the model (HF id), the target (context length and chunk
   size, e.g. 55k in 5k chunks), the ladder rungs if they have a preference, and a layer subset if the model will not fit.
   Propose defaults: a first rung of 2 chunks at a small chunk size with full dumps, a mid rung, the last chunk after a
   golden prefix, then the full target.
2. Check on the spot and report:
   - the checkpoint: `python -c "from huggingface_hub import snapshot_download as s; print(s('<hf_id>', local_files_only=True))"`
     or its size on the hub if it is not local; it goes to `/localdev/$USER/bringup/<model>/hf/`;
   - the box: `ls /dev/tenstorrent | wc -l` chips and the architecture (`tt-smi -ls`; never `tt-smi -r`);
   - the config: layers, hidden size, heads, KV heads, head_dim, experts, attention types per layer, from `config.json`
     (the text config for multimodal checkpoints);
   - what the repo already has for this model family (grep `models/demos` and the repo map).
3. Scaffold: `python -m models.demos.common.bringup new --model <slug> --hf-id <hf_id>`. Then fill
   `models/demos/<slug>/bringup/spec.yaml` from the template: `num_layers`, `block_types` (one entry per distinct block,
   with every layer exactly once and a representative layer), `state.tensors`, `box`, `target`, `ladder`, and
   `checkpoint.expect` / `checkpoint.config` from the index and config.
4. Validate: `python -m models.demos.common.bringup validate --spec <spec> --ledger-only` and
   `python -c "from models.demos.common.bringup.core.spec import Spec; print(Spec.load('<spec>').validate())"`.
5. Show the spec to the person. Wait for an explicit yes. Then record it:
   `python -m models.demos.common.bringup approve intake --spec <spec>`.
6. Write the first tasks: `python -m models.demos.common.bringup.plan.ledger_gen --spec <spec> --early --write`
   (R, G, B and PL.0; PL.0 adds the component, swap, ladder, contract and perf tasks once the reference exists).
7. Launch. The orchestrator is a long-running process that starts one `claude -p` per step. Give the person the
   command to run in their own terminal (or with `!` in this session), and do not run it in the background yourself:
   `python -m models.demos.common.bringup.orchestrator run --spec <spec>`

## Report on a run

`python -m models.demos.common.bringup status --spec <spec>` and the `waiting` / `reason` fields in
`<bringup_dir>/state.json`. Summarize: passed, running, stopped (with the reason), waiting for a person (with what to do).
The dashboard: `python -m models.demos.common.bringup.dashboard.export --spec <spec>`.

## Resume, rerun, fork

- After a stop: fix or decide, then `python -m models.demos.common.bringup.orchestrator resume --spec <spec>`.
- Approvals: `python -m models.demos.common.bringup approve plan|perf --spec <spec>`.
- `python -m models.demos.common.bringup rerun --from <id> --spec <spec>`;
  `python -m models.demos.common.bringup fork --from <id> --name <run> --spec <spec>`;
  `python -m models.demos.common.bringup compare --spec <specA> --other <specB>`.
