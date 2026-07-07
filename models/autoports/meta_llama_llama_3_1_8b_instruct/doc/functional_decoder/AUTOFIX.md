# AutoFix Report

## Starting Evidence

- Fresh AutoDebug report: `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/AUTODEBUG.md`.
- Original failing gate:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

- Observed result: skipped because `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")` receives a gated-repo `401`.
- Synthetic target-shape prefill is passing at seq 16, 17, and 64 with PCC >= 0.998, so the remaining blocker is specifically the required real-weight PCC evidence.

## Hypothesis Experiments

- Hypothesis: a default HF cache or token exists but the test is not using it correctly.
  Experiment: AutoDebug checked HF cache environment variables, token env vars, `/home/mvasiljevic/.cache/huggingface`, and `huggingface_hub.try_to_load_from_cache` for config/index/shard names.
  Result: no token/cache override was present and no cached snapshot files were found.
  Verdict: refuted.
  Evidence artifact(s): `AUTODEBUG.md`.
  Fix, if any: added explicit `LLAMA31_8B_INSTRUCT_HF_PATH` support for a future canonical local HF checkout.
  Verification: `pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short` now skips before opening a TT mesh and names the unblock variable.

- Hypothesis: the forge-generated tree contains or points to a local real checkpoint.
  Experiment: AutoDebug inspected `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/{params.py,model_pt.py,consteval.py,model_ttnn.py}` and searched for canonical weight shards.
  Result: `model_pt.py` and `params.py` reload the same gated HF repo; the generated tree contains code/IR/logs, not weights.
  Verdict: refuted.
  Evidence artifact(s): `AUTODEBUG.md`.
  Fix, if any: none.
  Verification: no canonical shards found.

- Hypothesis: repo-local `models/tt_transformers/model_params/Llama-3.1-8B-Instruct` is enough for real-weight evidence.
  Experiment: inspected `config.json`, `params.json`, and `performance_decoder_config.json`.
  Result: these are config mirrors only and contain no tensor data.
  Verdict: refuted.
  Evidence artifact(s): `AUTODEBUG.md`.
  Fix, if any: none.
  Verification: config mirror is used only as a synthetic-test fallback shape source.

- Hypothesis: `/localdev/hmijatovic/model_cache/meta-llama/Llama-3.1-8B-Instruct/N300` provides usable real weights.
  Experiment: inspected filenames and probed representative `.tensorbin` metadata.
  Result: files are TTNN fused/sharded/quantized/layout-specific caches such as `wqkv_sharded_2d_dtype_BFLOAT8_B_layout_TILE.tensorbin`, not canonical HF weights and not sufficient to instantiate the HF reference layer.
  Verdict: refuted for this gate.
  Evidence artifact(s): `AUTODEBUG.md`; exploratory probe recorded in `work_log.md`.
  Fix, if any: none.
  Verification: representative `.tensorbin` tensors were distributed/cache-specific and not directly convertible to canonical HF state dict tensors.

- Hypothesis: canonical Llama shards exist elsewhere under `/localdev`.
  Experiment:

```bash
find /localdev -path '*Llama-3.1-8B-Instruct*' \( -name 'model.safetensors.index.json' -o -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'consolidated*.pth' \) -print 2>/dev/null | head -100
```

  Result: no output.
  Verdict: refuted for searched local paths.
  Evidence artifact(s): command output in session; `AUTODEBUG.md`.
  Fix, if any: none.
  Verification: no canonical HF or raw Meta checkpoint file was found.

## Final Status

- Initial AutoFix status was blocked by external credential/data availability.
- On resume, `HF_TOKEN` access became available. The required real-weight single-layer PCC gate passed:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

- Result: passed in 93.20s with PCC `0.9999980443319924`.
- For environments without HF authentication, set `LLAMA31_8B_INSTRUCT_HF_PATH` to a canonical local HF checkout containing `config.json`, `model.safetensors.index.json`, and all referenced shards.
