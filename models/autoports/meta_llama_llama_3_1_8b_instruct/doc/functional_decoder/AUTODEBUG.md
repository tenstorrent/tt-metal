# AutoDebug: real-weight functional decoder evidence gap

## Starting Evidence

Problem command:

```bash
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py::test_real_weight_single_layer_prefill_matches_hf --tb=short
```

The in-tree evidence records this as skipped because `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")` returned a gated-repo `401`:

- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/context_contract.json:33-35`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/work_log.md:76-82`
- `models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/README.md:31-34`

I did not rerun that pytest command because the `mesh_device` fixture in `test_functional_decoder.py:66-72` opens a TT mesh device before the test body. This diagnosis stayed host-only.

Relevant test facts:

- `test_functional_decoder.py:155-160` calls `AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16).eval()` and skips on any exception.
- The test does not try `local_files_only=True`, an explicit local path, or an environment override.
- `functional_decoder.py:73-85` accepts canonical HF/layer-local keys such as `model.layers.0.self_attn.q_proj.weight`.
- `functional_decoder.py:163-205` builds TT weights from that canonical `state_dict`.

## Hypotheses For Local Real Weights

1. A default Hugging Face cache may already contain a canonical snapshot for `meta-llama/Llama-3.1-8B-Instruct`.

Verdict: refuted. `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_HUB_CACHE`, and `XDG_CACHE_HOME` are unset. `/home/mvasiljevic/.cache/huggingface` exists but has no `hub` directory and no model snapshot. `huggingface_hub.try_to_load_from_cache` returned `None` for `config.json`, `model.safetensors.index.json`, `model-00001-of-00004.safetensors`, and `pytorch_model.bin.index.json`.

2. The generated forge/TTNN artifacts may embed or point to real HF weights.

Verdict: refuted. `/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_pt.py:28-29` reloads the gated HF model by repo id. `params.py:373-390` also calls `model_pt.load_pytorch_model()` and maps the live model state dict into TTNN tensors. The generated tree has code/IR/logs only; no canonical `.safetensors`, `.bin`, or `.pth` weight shards were present.

3. Repo-local model params under `models/tt_transformers/model_params/Llama-3.1-8B-Instruct` may be enough.

Verdict: refuted. These files are shape/config mirrors (`config.json`, `params.json`, `performance_decoder_config.json`) and contain no tensor data or checkpoint path. They are useful for config fallback, not real-weight PCC evidence.

4. `/localdev/hmijatovic/model_cache/meta-llama/Llama-3.1-8B-Instruct/N300` may provide usable real weights.

Verdict: refuted for this stage gate. It contains real-model-derived `.tensorbin` TT transformer caches, but they are preprocessed TTNN tensors, not canonical HF weights. Filenames show fused, sharded, layout-encoded, and quantized tensors, for example:

```text
/localdev/hmijatovic/model_cache/meta-llama/Llama-3.1-8B-Instruct/N300/tensor_cache_instruct_bfp8/layers.0.attention.wqkv_sharded_2d_dtype_BFLOAT8_B_layout_TILE.tensorbin
/localdev/hmijatovic/model_cache/meta-llama/Llama-3.1-8B-Instruct/N300/tensor_cache_instruct_bfp8/layers.0.feed_forward.w1_sharded_dtype_BFLOAT4_B_layout_TILE.tensorbin
/localdev/hmijatovic/model_cache/meta-llama/Llama-3.1-8B-Instruct/N300/tensor_cache_instruct_bfp8/tok_embeddings.weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin
```

That cache does not provide the HF reference model needed by `test_real_weight_single_layer_prefill_matches_hf`, and converting it back would require a mesh/layout-aware composer plus de-fusion/de-sharding/de-quantization. It would also not be a canonical HF BF16/FP checkpoint for a strict HF-vs-TT PCC gate.

5. Some other local `/localdev/*` HF or raw Meta checkpoint may exist.

Verdict: refuted by file search. I found many copied `params.json`/`config.json` mirrors and TT runtime artifacts, but no canonical Llama `model.safetensors`, `model.safetensors.index.json`, `model-*.safetensors`, `pytorch_model*.bin`, `pytorch_model*.bin.index.json`, or `consolidated*.pth` checkpoint for this model.

## Experiments And Results

Host-only commands used:

```bash
nl -ba models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
nl -ba models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py
nl -ba /localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/{params.py,model_pt.py,consteval.py,model_ttnn.py}
```

Result: the test and generated artifacts all ultimately require canonical HF model weights loaded from `meta-llama/Llama-3.1-8B-Instruct`; no alternate local source is wired in.

```bash
find /localdev -type d -name 'models--meta-llama--Llama-3.1-8B-Instruct'
find /localdev -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name '*.bin.index.json' -o -name '*.safetensors.index.json' \) -path '*Llama-3.1-8B-Instruct*'
find /localdev -type f \( -name 'model.safetensors.index.json' -o -name 'model.safetensors' -o -name 'model-*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'pytorch_model*.bin.index.json' \) | grep -Ei 'llama|meta'
```

Result: no HF cache snapshot directory and no canonical Llama HF safetensors/bin shards. The only `pytorch_model.bin` hits were unrelated Whisper assets.

```bash
find /localdev -type f \( -name 'consolidated*.pth' -o -name 'params.json' -o -name 'checklist.chk' \) | grep -Ei 'llama|meta'
```

Result: only small `params.json` mirrors for TT model configs; no raw Meta `consolidated*.pth` checkpoint.

```bash
find /localdev/hmijatovic/model_cache -maxdepth 6 -type f -printf '%p\t%s bytes\n'
```

Result: found TTNN `.tensorbin` model caches for Llama-3.1-8B-Instruct under `N300/tensor_cache_instruct_bfp8`, including layer-0 attention/MLP/norm tensors and full-model embedding/lm-head cache files. These are not canonical HF weight shards.

```bash
python - <<'PY'
import os
for k in ['HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','HF_HUB_CACHE','XDG_CACHE_HOME']:
    print(k, os.environ.get(k))
print('HF_TOKEN set?', bool(os.environ.get('HF_TOKEN')))
print('HUGGING_FACE_HUB_TOKEN set?', bool(os.environ.get('HUGGING_FACE_HUB_TOKEN')))
PY
```

Result: all cache override variables and token variables are unset in this process.

```bash
python - <<'PY'
from huggingface_hub import try_to_load_from_cache
repo = 'meta-llama/Llama-3.1-8B-Instruct'
for name in ['config.json', 'model.safetensors.index.json', 'model-00001-of-00004.safetensors', 'pytorch_model.bin.index.json']:
    print(name, try_to_load_from_cache(repo, name))
PY
```

Result: all returned `None`.

## Verdict

Verified: no viable local canonical HF weight path exists in the inspected `/localdev/*` paths or the default user HF cache.

Verified: the generated forge/TTNN artifacts do not contain a local checkpoint fallback; they reload the same gated HF model.

Verified: the only Llama real-weight-looking local data is TTNN `.tensorbin` cache data under `/localdev/hmijatovic/model_cache/.../tensor_cache_instruct_bfp8`, but it is preprocessed, fused/sharded, quantized, layout-specific, and insufficient to instantiate the HF reference layer required by the current real-weight PCC test.

Refuted: repo-local config mirrors are enough to satisfy the real-weight evidence gate.

Uncertain: whether another non-`/localdev` shared storage mount has canonical weights. I did not search outside `/localdev`, `/home/mvasiljevic/.cache/huggingface`, and the provided/generated paths.

## Recommendation For Main Agent

Treat this as an external credential/data blocker, not an implementation blocker.

Do not count the current skipped test as satisfying the stage goal. Do not use the `.tensorbin` cache as replacement evidence for the real-weight HF-vs-TT PCC gate.

To unblock, provide either:

- HF credentials with access to `meta-llama/Llama-3.1-8B-Instruct`, or
- a canonical local HF checkout containing `config.json` plus `model.safetensors.index.json` and all referenced `model-*.safetensors` shards.

Minimal future code/test change once such a path exists: make the real-weight test load from an explicit local path override, for example `LLAMA31_8B_INSTRUCT_HF_PATH`, and call:

```python
model = AutoModelForCausalLM.from_pretrained(local_path_or_hf_id, torch_dtype=torch.bfloat16, local_files_only=Path(local_path_or_hf_id).is_dir()).eval()
```

Then use the same `model.state_dict()` path already exercised by `FunctionalDecoder.from_state_dict`.
