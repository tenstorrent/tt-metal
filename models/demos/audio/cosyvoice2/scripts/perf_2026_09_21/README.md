# Performance and accuracy scripts, 2026-09-20/21 session

Scripts behind the numbers in `BRINGUP_STATUS.md` (top section). They were written as
one-off measurements against the real checkpoints on an N150, copied here so they
survive the session. They are not tests and are not run in CI.

Setup: run with `/opt/venv/bin/python` from a directory other than the repo root, with
`PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal`.
Set `COSYVOICE2_SCRATCH` (checkpoint-derived cache dir, default `/tmp/cosyvoice2_stage1_eval`)
and `OUT_DIR` (results and wavs). Wrap device jobs in `timeout -s KILL N` so a hung card
cannot stall the session (recover with `tt-smi -r`).

| Script | What it measures |
|---|---|
| `gen_tokens7.py` | Generates and saves speech tokens for the seven everyday sentences (`tokens7.json`) |
| `torch_ref_mel7.py` | Pure-torch flow mel for those tokens (ground truth for the listening arms) |
| `arm_run.py` | One listening arm (`ARM=R/A/B/C/D`; `COSYVOICE2_FLOW_SDPA`, `COSYVOICE2_FLOW_MATMUL_CC`, `FLOW_DTYPE` select the variant): mel error vs torch, WER, speaker similarity, wavs |
| `rtf_warm.py` | Warm end-to-end RTF with the current defaults: LLM/flow/HiFT split, first request vs three warm repeats, WER, memory |
| `rtf_seven.py` | Same harness over all seven sentences (older, fp32 flow) |
| `est_profile.py` | Traced-vs-eager time of one estimator call at the sentence's T (needs `VARIANT`, `est_tokens.json`; scratch monkeypatches prepare conv weights and keep the time embedding on device) |
| `est_profile_T.py` | Same, at a chosen T (`NTOK` tokens): the traced-ms-versus-T sweep (about 7.8 + 0.083*T ms) |
| `dtype_probe.py` | Records the dtype of every weight and activation per component |
| `cfg_dram_check.py` | Whether `config_tensors_in_dram` keeps L1_SMALL flat for each conv type (run with the env var 0 and 1) |
| `spk_torch_check.py` | Speaker similarity of the 4.36 s sentence through a fully torch pipeline, plus real same-speaker clip baselines |
| `steps_check.py` | Mel error of the torch flow at 8/6/5/4/3 Euler steps against 10 |

Caveats: scripts that read `tokens7.json`, `est_tokens.json` or per-arm outputs expect the
earlier scripts to have run first with the same `OUT_DIR`. `est_profile*.py` also assume
the traced-solver pieces are NOT in the repo (the scratch patches are the point).
