# EmbeddingGemma 2 (text tower)

TTNN implementation of the text tower of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on one Blackhole P150a:
token embedding, projection-only per-layer inputs (PLE), 24 bidirectional encoder layers, final
norm, 512->768 projection, masked mean pooling and L2 normalization. The output matches
`SentenceTransformer("google/embeddinggemma-2").encode(text)` (`[Transformer, Pooling(mean), Normalize]`).

Status: stage E1 (functional text encoder). The vision and audio towers (`tt/vision`, `tt/audio`)
and the multimodal merge come in later stages.

## Setup

Run from the tt-metal root inside `python_env`. The HF reference needs transformers >= 5.20
(`models/embedding_gemma2`); `python_env` pins 5.12.1, so the tests put an overlay directory first
on `PYTHONPATH`. The overlay holds only `transformers==5.20.2`, `huggingface_hub==1.33.0` and
`tokenizers==0.23.3`, installed with `--no-deps --target` (python_env is unchanged):

```bash
uv pip install --python python_env/bin/python --no-deps --target /local/ttuser/gtobar/venvs/tf_overlay \
    transformers==5.20.2 "huggingface_hub>=1.31,<2" tokenizers==0.23.3

export TT_METAL_HOME=$PWD PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD
export TT_VISIBLE_DEVICES=0 HF_MODEL=google/embeddinggemma-2
```

The overlay reference was checked against a separate python 3.11 / transformers 5.21.0 venv: the
two `EmbeddingGemma2Model` outputs on a 2-prompt padded batch have max abs difference 0.0 (fp32).
The tests compute the reference live (no stored goldens); the HF model code is used unmodified.

TTNN weights are cached under `~/.cache/ttnn/models/embedding_gemma2/<snapshot>/`
(override with `EMBEDDING_GEMMA2_CACHE`).

## Use

```python
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder

enc = TextEncoder.from_pretrained(device, "google/embeddinggemma-2", max_seq_len=8192)
batch = tokenizer(["task: search result | query: northern lights"], padding=True, return_tensors="pt")
emb = enc.encode(batch["input_ids"], batch["attention_mask"])               # eager, [B, 768], unit norm

enc.capture_all([(1, 32), (8, 512)])          # every (batch, bucket) the session will replay
emb = enc.encode(batch["input_ids"], batch["attention_mask"], traced=True)  # trace replay
```

Open the device with a trace region (`trace_region_size=128 MiB` is what the tests use).
Logical lengths run in the smallest power-of-two bucket >= L (32 ... 32768); `capture_all`
explains why every trace is captured in one call.

## Layout

| path | content |
|---|---|
| `tt/config.py` | config dataclasses built from the HF config; buckets; compute configs |
| `tt/embedding.py` | `TextEmbedding` (scaled token embedding) |
| `tt/per_layer_input.py` | `PerLayerInput` (PLE projection + norm), `PerLayerInputBlock` (PLE gate in each layer) |
| `tt/rms_norm.py` | `RMSNorm` (weight used as-is; `with_scale=False` for v_norm) |
| `tt/rope.py` | `RotaryTables` (cos/sin per layer type) |
| `tt/masks.py` | `AttentionMasks` (band + key-padding additive masks) |
| `tt/attention.py` | `Attention` (sliding: hd 256 / 2 KV; full: hd 512 / 1 KV) |
| `tt/mlp.py` | `MLP` (gated gelu-tanh) |
| `tt/encoder_layer.py` | `EncoderLayer` (sandwich norms, PLE block, layer_scalar) |
| `tt/pooler.py` | `Pooler` (masked mean + L2 normalize) |
| `tt/text_model.py` | `TextModel` (device-only forward) |
| `tt/text_encoder.py` | `TextEncoder` (host: bucketing, upload, trace capture/replay, readback) |
| `reference/hf_reference.py` | HF fp32 reference loader + activation capture |
| `tests/pcc/` | PCC tests per layer kind, full stack, trace, determinism, fallback audit |
| `tests/perf/perf.py` | traced latency + Tracy forward |
| `demo/demo.py` | retrieval demo |
| `doc/` | `context_contract.json`, `functional_text/` (stage report, work log, perf artifacts) |

Every module has a config dataclass, a `from_state_dict(...)` loader that writes the ttnn weight
cache, and a `forward` without host work. Contracts (shapes, layouts, dtypes, mask semantics) are
in each module docstring and in `doc/functional_text/README.md`.

## Tests

```bash
pytest models/demos/blackhole/embedding_gemma2/tests/pcc -q          # all PCC / trace / audit tests
pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py -k traced -s
pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s
```

## Out of scope (decoder-only items of the bring-up skill)

This model is an encoder that returns one embedding per input. It has no paged KV cache, decode
step, page table, current-position tensor or sampling, so none is built. Chunked prefill is not
needed either: the whole sequence runs in one pass per bucket, up to 32768 tokens.

## Limitations

See `doc/functional_text/README.md` (dense O(S^2) masks, power-of-two buckets, supported context
32768 of the advertised 262144).
