# Canonical test prompt for `decoder-to-productized`

This is the standard prompt for sending a fresh-context subagent through the
skill against a `models/autoports/<model_name>/` input. Replace `<model_name>`
with your model directory name before running. Use it via the `Agent` tool
(`general-purpose` subagent), preferably in the background since the full run is
10–30 minutes including device verification.

Update the **reference file path** if the prompt set / max-new-tokens change.
Update the **input directory** if a different test input is added.

---

```
You are a coding agent executing the `decoder-to-productized` skill in a tt-metal repo on a Tenstorrent dev box.

Working tree: /localdev/tcheda/tt-metal (operate from there).
Python env: /localdev/tcheda/tt-metal/python_env/bin/python3.

Skill to follow: .agents/skills/decoder-to-productized/SKILL.md. Read it in full, then read the background notes it points at (.agents/notes/*.md). The skill is the spec — follow it.

Task: Execute the skill against the input directory `models/autoports/<model_name>/`. Produce the files the skill calls for (tt/model.py, tt/generator.py, tt/generator_vllm.py), then run the teacher-forcing readiness check per step 5.

Inputs you'll find in the model dir:
- tt/decoder.py — copy of tt_transformers/tt/decoder.py::TransformerBlock.
- config.py — module-level constants HF_MODEL_ID and MESH_DEVICE.

Constraints:
- Do not modify anything outside `models/autoports/<model_name>/` and (only if you need to generate a new reference) `models/common/readiness_check/references/` and `models/common/readiness_check/prompts/`.
- A pre-generated reference exists at models/common/readiness_check/references/llama31_8b_instruct_general.refpt (HF Llama 3.1 8B Instruct, 64 prompts — 32 short + 32 long — at 128 tokens each; regeneration takes ~80 minutes on CPU so reuse it).
- After a device crash, run `tt-smi -r` before retrying.

Hardware: 4× N300 boards. Open as N150 (MeshShape(1,1)) per the config — matches the existing reference.

Expected outcome (from step 5 of the skill): top-5 ≥ 99%, top-100 = 100%. Lower top-1 (~95%) is expected from bf8 quantization.

Report back with (under 250 words):
- Files produced and where.
- Teacher-forcing accuracy numbers (top-1 / top-5 / top-100, per-entry and aggregate).
- Any deviations from the skill's instructions and why.
- Anything in the skill that was unclear, wrong, or insufficient (this run is also a test of the skill itself).
```

---

## Cleanup between runs

Each run produces files in `models/autoports/<model_name>/tt/`.
Reset before re-firing (replace `<model_name>` first):

```bash
MODEL=<model_name>
rm -f models/autoports/$MODEL/tt/model.py \
      models/autoports/$MODEL/tt/generator.py \
      models/autoports/$MODEL/tt/generator_vllm.py
rm -rf model_cache/$MODEL/   # if the run got far enough to cache
```

Keep `tt/decoder.py`, `config.py`, and the two `__init__.py` files.

## Permission mode

The harness routes the subagent's tool prompts to the parent session. Flip the
parent's permission mode to `bypassPermissions` (Shift+Tab or `/permissions`)
before firing if you don't want to approve each Bash/Edit/Write call.
