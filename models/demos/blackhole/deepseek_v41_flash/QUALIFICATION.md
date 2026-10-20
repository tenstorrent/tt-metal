# DeepSeek-V4.1-Flash requalification checklist

Run on every candidate image / runtime spec (B=16 / 262,144 and B=4 / 1M) before release. Each step must pass; do not skip the probe: GSM8K and perf did not catch the bug below.

1. **Boot** from the shipped runtime spec (`assets/runtime_spec_*_vision.json`): `/health` 200, no TT_THROW in the server log.
2. **4,934-token tool-call probe, mandatory** (`tests/data/probe_t22_k3_ids.json`, raw ids to `/v1/completions`, greedy, `max_tokens 80`, `skip_special_tokens false`; script: `evals_gpqa_tau3/scripts/probe_load.py <url> <ids.json> 15 3`): 3 probes idle, 3 while 15 other requests decode, 3 idle again. Every output must be the call `get_user_details(user_id=omar_rossi_1241)` (one distinct output). Why: in the B=16 / 262,144 build the decode index-key slab was re-allocated by the slice_write export and the answer was wrong in every slot (0/9) while GSM8K was only 2-3 points lower and short prompts were fine.
3. **GSM8K** (`assets/run_gsm8k_eval.sh`): B=16 about 94-97 %, B=4 about 96-97 %.
4. **Short-prompt perf** (`vllm bench serve`, random prompts, 128 output tokens, 3 repeats): the TTFT / TPOT table of `RELEASE_NOTES.md` (16 x 1,024, 16 x 8,192, 4 x 1,024, 4 x 8,192).
5. **Load stability**: 16 concurrent 8,192-token requests (3 repeats) without a device timeout.
6. **Device regression tests** (32-chip galaxy): `tests/test_b16_slot_regression.py` (production-like adapter, every slot, spec and plain), `tests/test_key_export_inplace.py` (key slab export is in place, 3 layers, cheap); CPU: `tests/test_key_export_default.py`.
