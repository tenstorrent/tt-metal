# d1-3B decision model on Wormhole (N150)

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) is a **decision model**: you give it a state (text,
JSON, or images) and typed questions, and it returns calibrated probabilities per option in **one forward pass
with zero generated tokens**. It is post-trained from LFM2.5-VL-3B (LFM2 hybrid ShortConv/attention backbone +
SigLIP2 NaFlex vision), so this port reuses [`models/demos/multimodal/lfm25_vl`](../lfm25_vl) and adds only the
3B config, the answer-slot decision head, and a prefill-only traced path.

## Platforms

Single Wormhole chip: **N150** (also runs on one chip of an N300). Batch 1, prefill only.

## Question types

| `type` | `criteria` | answer |
|---|---|---|
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` | `noul`: P(yes) |
| `choice` | `{name: description}` | `choice`, `confidence`, `probabilities` |
| `score` (2–10 ordered levels) | list of level descriptions | `score` (expected level), `confidence`, `probabilities`, `legend` |

## How it works

1. **Prompt** — rendered by d1's own `prompt.py` from the checkpoint (byte-exact; the readout was trained on
   that exact text). No chat template.
2. **Prefill** — token (and image) embeddings are right-padded to a bucket (128/256/512/1024/2048/4096) and run
   through the 30-layer hybrid backbone as one trace per bucket. Every operator is causal, so padding never
   reaches the answer slot. The last real row is selected with a one-hot matmul and the final norm applied.
3. **Readout** — d1 ships no calibration, so a decision is a plain softmax over each option's best token at the
   answer slot. The LM head is tied to the embedding, so only the option tokens' logits are computed (a few
   dot products on host), not the 128K-vocab projection. This is exact vs the full log-softmax (verified 1e-6).

## Setup

```bash
pip install -r models/demos/multimodal/d1_3b/requirements.txt
hf download LiquidAI/d1-3B --revision 051bcc464b01b9f92942b364d9586b0ef5912432
```

## Run

```bash
export MESH_DEVICE=N150

# Demo
python models/demos/multimodal/d1_3b/demo/demo.py

# Host-side: prompt rendering + readout vs the HF reference (no device)
pytest models/demos/multimodal/d1_3b/tests/test_decision.py

# On device: full-model PCC + decision accuracy vs the fp32 reference
pytest models/demos/multimodal/d1_3b/tests/test_model.py -s

# Latency per bucket
pytest models/demos/multimodal/d1_3b/tests/test_perf.py -s
```

### Regenerating the golden reference

The committed `tests/reference/decision_reference.json` is the HF model's fp32 output
(`trust_remote_code=True`). It needs `transformers>=5.14` (d1's remote code) and is best run in a separate venv:

```bash
python models/demos/multimodal/d1_3b/tests/generate_reference.py --dtype float32
python models/demos/multimodal/d1_3b/tests/generate_reference.py --dtype bfloat16  # the precision floor for Tier D
```

Large per-question tensors (hidden states, full logits, pixel inputs) go to `$D1_SIDECAR_DIR`
(`~/.cache/d1_3b/reference` by default), not the repo.

## API

```python
from models.demos.multimodal.d1_3b.tt.d1 import TtD1
import ttnn
from models.demos.multimodal.d1_3b.tt.model import L1_SMALL_SIZE, TRACE_REGION_SIZE

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE)
d1 = TtD1.from_pretrained(mesh)
d1.system_one("I was charged twice, please refund one.", {"refund": {"type": "noul", "instructions": "Refund request?"}})
# {"answers": {"refund": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 39, "output_tokens": 0}}
```

## Accuracy and performance

See [PERF.md](PERF.md).

## Known limitations

- **Prefill only.** Decision models never decode; there is no KV-cache / decode path.
- **Images** go through the SigLIP2 tower on device (reusing lfm25_vl); the `TtD1` convenience API currently
  takes text states only, while `test_model` validates the image decisions against goldens.
- **32K context** is a stretch goal; buckets cap at 4096 tokens in v1.
