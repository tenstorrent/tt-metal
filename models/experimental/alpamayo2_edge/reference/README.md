# Alpamayo 2 Edge reference: validation against the official stack

Every TT PCC test in this package uses the pure-torch reference in this directory as its golden.
`validate_official.py` shows that the reference matches the official model: HF transformers 5.17 plus
the NVlabs `alpamayo2_edge` package. The check runs once, or again after a reference change.

## Official venv

The official stack needs Python 3.12 and transformers 5.17. The repo stays on transformers 5.12.1, so
the official stack gets its own venv outside the repo.

```bash
cd /proj_sw/user_dev/ctr-mdukic/venvs
uv python install 3.12
uv venv --python 3.12 alpamayo-official
uv pip install --python alpamayo-official/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0 torchvision==0.23.0
uv pip install --python alpamayo-official/bin/python transformers==5.17.0 \
    "alpamayo2-edge @ git+https://github.com/NVlabs/alpamayo2-edge@02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e"
```

Pinned: Python 3.12.16, torch 2.8.0+cpu, torchvision 0.23.0+cpu, transformers 5.17.0, tokenizers 0.23.3,
and `alpamayo2_edge` at commit `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e`. Both stacks load checkpoint
snapshot `b92b6aa326bd3e2235db28f248ad5a30b5890e64`.

## Running

Both steps are CPU-only. They import torch, so they run through the device MCP job runner and take about
5 minutes each. The dump goes to `/localdev/ctr-mdukic/alpamayo_cache/official/` and is never committed.

```bash
# official venv: synthetic_sample(0) through the NVlabs helper and sample_trajectories_from_data, fp32 / SDPA
HF_HOME=/localdev/ctr-mdukic/hf_data HF_HUB_OFFLINE=1 \
  /proj_sw/user_dev/ctr-mdukic/venvs/alpamayo-official/bin/python \
  models/experimental/alpamayo2_edge/reference/validate_official.py dump --out /localdev/ctr-mdukic/alpamayo_cache/official

# repo venv: rebuild the inputs, run ReferenceModel on the official CoTs and noise, compare
HF_HOME=/localdev/ctr-mdukic/hf_data HF_HUB_OFFLINE=1 python_env/bin/python \
  models/experimental/alpamayo2_edge/reference/validate_official.py compare --dump /localdev/ctr-mdukic/alpamayo_cache/official
```

`dump` runs the official model twice and records each run with hooks:

- **sampled:** the official sampler with seed 0, using upstream defaults (temperature 0.6, top-p 0.98,
  256 new tokens, 10 Euler steps). Its CoT, cut at the anchor by `make_forced_cot`, is
  `<NOMINAL>\n<|traj_future_start|>`, which is 6 tokens.
- **long_forced:** a 128-token CoT forced through the same official path by a logits processor. It is
  127 tokens of fixed driving text plus the anchor. It checks decode positions and expert attention over
  a long CoT.

`compare` teacher-forces each CoT through `ReferenceModel.decode`. It runs the expert on the official
noise, and on each official Euler-step input for the per-step velocities. Inputs must match bitwise.
Each tensor must have PCC ≥ 0.99999. The script also reports max-abs error and max-rel error, which is
max-abs divided by the official tensor's max magnitude. The two stacks are not bitwise equal because they
order ops differently: the reference uses eager matmul attention and the official stack uses SDPA.

## Results

Inputs are bitwise equal, ints exact and pixel values exact: `input_ids`, `attention_mask`,
`mm_token_type_ids`, `patches` vs `pixel_values`, `image_grid_thw`, `position_ids` and `rope_deltas`,
plus the ego history. S = 3256 tokens and N = 11568 patches, as recorded in Task 4.

| tensor | case | 1 − PCC | max abs | max rel |
|---|---|---|---|---|
| vision encoder (post_layernorm) [11568, 1152] | – | 1.8e-13 | 4.7e-4 | 1.0e-5 |
| vision tower + projector [2892, 2048] | – | 7.8e-13 | 2.1e-5 | 3.4e-6 |
| prefill last logits [1, 135168] | – | 9.6e-13 | 4.8e-6 | 3.2e-7 |
| forced-CoT logits, all steps | sampled (6) | 8.6e-14 | 1.5e-5 | 5.5e-7 |
| forced-CoT logits, all steps | long_forced (128) | 9.0e-13 | 1.3e-5 | 8.1e-7 |
| forced-CoT logits, worst step | long_forced (step 54) | 3.3e-12 | 1.1e-5 | 8.8e-7 |
| K layer 0 / 27 (3384 rows) | long_forced | 2.3e-14 / 2.2e-13 | 1.0e-5 / 1.2e-4 | 2.1e-6 / 5.1e-6 |
| V layer 0 / 27 (3384 rows) | long_forced | 5.1e-13 / 6.3e-13 | 1.1e-6 / 5.5e-5 | 1.2e-6 / 8.1e-6 |
| expert velocity, step 0 (noise) | long_forced | 6.4e-15 | 4.8e-7 | 1.4e-7 |
| expert velocity, worst Euler step (9) | long_forced | 1.5e-14 | 5.4e-7 | 1.6e-7 |
| actions after 10 Euler steps [1, 64, 2] | sampled / long_forced | 2.0e-14 / 1.7e-13 | 8.9e-8 / 7.5e-8 | 3.1e-7 / 5.9e-7 |
| xyz [1, 64, 3] | sampled / long_forced | 2.2e-16 / 2.2e-16 | 5.4e-7 / 3.8e-6 m | 8.6e-9 / 5.9e-8 |
| rot [1, 64, 3, 3] | sampled / long_forced | 0 / 2.2e-16 | 1.9e-8 / 1.1e-8 | 1.9e-8 / 1.1e-8 |

The CoT prefix length (prompt + CoT up to the anchor) matches in both cases: 3262 and 3384. Each
`compare` run writes the full table to `compare.json` next to the dump.
