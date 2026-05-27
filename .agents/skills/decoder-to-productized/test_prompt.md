# Canonical test prompt for `decoder-to-productized`

This is a compact prompt for sending a fresh-context subagent through the
productization stage against a `models/autoports/<model_name>/` input. Replace
`<model_name>` with your model directory name before running.

Update the **reference file path** if the prompt set / max-new-tokens change.
Update the **input directory** if a different test input is added.

---

```
You are a coding agent productizing a ported TTNN decoder in a tt-metal repo on a Tenstorrent dev box.

Working tree: /localdev/tcheda/tt-metal (operate from there).
Python env: /localdev/tcheda/tt-metal/python_env/bin/python3.

Context to read: .agents/notes/model-bringup-mission.md, .agents/skills/decoder-to-productized/SKILL.md, and the background notes it points at. Use them as guidance, then make the implementation choices the codebase and model require.

Task: Productize `models/autoports/<model_name>/`. Produce `tt/model.py`, `tt/generator.py`, and `tt/generator_vllm.py`, then run the readiness checks.

Inputs you'll find in the model dir:
- tt/decoder.py — copy of tt_transformers/tt/decoder.py::TransformerBlock.
- config.py — module-level constants HF_MODEL_ID and MESH_DEVICE.

Constraints:
- Do not modify anything outside `models/autoports/<model_name>/` and (only if you need to generate a new reference) `models/common/readiness_check/references/` and `models/common/readiness_check/prompts/`.
- A pre-generated reference exists at models/common/readiness_check/references/llama31_8b_instruct_general.refpt (HF Llama 3.1 8B Instruct, 64 prompts — 32 short + 32 long — at 128 tokens each; regeneration takes ~80 minutes on CPU so reuse it).
- After a device crash, run `tt-smi -r` before retrying.

Hardware: 4× N300 boards. Open as N150 (MeshShape(1,1)) per the config — matches the existing reference.

Expected outcome for this fixture: top-5 ≥ 99%, top-100 = 100%. Lower top-1 (~95%) is expected from bf8 quantization.

Report back with:
- Files produced and where.
- Readiness accuracy numbers (top-1 / top-5 / top-100, per-entry and aggregate).
- Important implementation choices, blockers, or deferred vLLM work.
- Anything in the skill that was unclear, wrong, or insufficient.
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
