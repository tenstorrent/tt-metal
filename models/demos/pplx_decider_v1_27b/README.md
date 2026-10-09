# pplx-decider-v1-27b on Blackhole

- A decision model fine-tuned from Qwen3.8-27B.
- Returns calibrated probabilities over 1-255 options.
- Question types: `choice`, `noul` (yes-no), `score`.
- Text only on TT.

## How it runs on qwen36

| Part                   | Source                                   | Name                                                                    |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| Model args             | Reused from `models/demos/blackhole/qwen36` | `Qwen36ModelArgs`                                                    |
| Model                  | Reused                                   | `Qwen36Model`                                                           |
| State dict remap       | Reused                                   | `remap_qwen36_state_dict`                                               |
| Prefill                | Reused                                   | `prefill_traced_chunked`, `prefill_tp`                                  |
| Mesh test setup        | Reused                                   | `parametrize_mesh_tp`                                                   |
| Checkpoint key rename  | pplx-specific (this dir)                 | `tt/weight_mapping.py`                                                  |
| Synthetic LM head      | pplx-specific                            | Readout rows at decision token ids, zeros elsewhere                     |
| MTP                    | pplx-specific                            | `enable_mtp=False`                                                      |
| Prompt format          | pplx-specific                            | `tt/decision.py`                                                        |
| Answer structure       | pplx-specific                            | `answer()`                                                              |

- The synthetic head works because the readout is applied after the final norm, like the LM head.

## Requirements

- 1x4 Blackhole mesh: `MESH_DEVICE=P150x4` (for example 2x P300).
- Host RAM: ~60 GB for the bf16 weight load.
- Host RAM: ~155 GB for the fp32 CPU reference generator.
- Disk: ~52 GB weights plus ~31 GB TT tensor cache.

## Setup

- Download the checkpoint, or set `HF_MODEL` to a local directory.

```bash
hf download perplexity-ai/pplx-decider-v1-27b
export MESH_DEVICE=P150x4
export TT_CACHE_PATH=$HOME/tt_cache/pplx_decider_v1_27b
```

| Variable         | Meaning                                                            |
| ---------------- | ------------------------------------------------------------------ |
| `HF_MODEL`       | Hub id or local dir. Default: `perplexity-ai/pplx-decider-v1-27b`. |
| `MESH_DEVICE`    | Mesh shape. Use `P150x4`.                                          |
| `TT_CACHE_PATH`  | TT tensor cache dir. Use a pplx-specific dir.                      |
| `HF_HUB_OFFLINE` | Set to `1` to skip hub lookups.                                    |

- The cache path does not include the model name.
- Sharing it with another Qwen checkpoint loads the wrong weights.
- If `TT_CACHE_PATH` is unset, the cache goes inside the checkpoint dir.

## Run

Demo (choice, noul, score samples):

```bash
pytest models/demos/pplx_decider_v1_27b/demo/demo.py
```

Device accuracy test (TT vs CPU reference):

```bash
pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
```

Batch-1 latency sweep, eager vs traced (512 to 4096 tokens):

```bash
pytest models/demos/pplx_decider_v1_27b/tests/test_perf.py
```

CPU tests:

```bash
pytest models/demos/pplx_decider_v1_27b/tests/test_decision.py
pytest models/demos/pplx_decider_v1_27b/tests/test_weight_mapping.py
```

Regenerate the CPU fp32 reference:

```bash
python models/demos/pplx_decider_v1_27b/tests/generate_reference.py \
    --hf-model perplexity-ai/pplx-decider-v1-27b \
    --output models/demos/pplx_decider_v1_27b/tests/reference/decision_reference.json
```

- All flags are optional.
- Defaults: `--hf-model` is `$HF_MODEL`, `--threads` is the CPU count.

## Python usage

```python
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

decider = PplxDecider.from_pretrained(mesh_device)  # mesh_device: 1x4 mesh
decider.capture_prefill_trace()  # optional; speeds up prompts of 2048 tokens or more
question = {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {
        "billing": "Charges and refunds",
        "technical_support": "Integration errors",
        "sales": "Questions about buying a product",
    },
}
result = decider.predict("My Stripe integration keeps failing.", question)
```

- Open the mesh with `trace_region_size` >= `TRACE_REGION_SIZE` (1 GiB) before capturing the trace.

Returned dict for `choice`:

| Key             | Value                                      |
| --------------- | ------------------------------------------ |
| `type`          | `"choice"`                                 |
| `probabilities` | Option key to probability                  |
| `choice`        | Key of the most likely option              |
| `confidence`    | Float in 0 to 1, rescaled from top prob    |

- `noul` returns `{"type": "noul", "noul": p_true}`.
- `score` returns `probabilities`, `legend`, `score` and `confidence`.

## Results

- Compares TT output with the fp32 CPU reference from `tests/generate_reference.py`.

Accuracy vs the fp32 CPU reference (`tests/test_model.py`, 1x4 Blackhole, bfp8 weights):

| Example | Tokens | Top-1 match | PCC (chunked / oracle / traced) | Max prob diff (chunked / oracle / traced) |
| --- | --- | --- | --- | --- |
| choice | 110 | yes | 0.99875 / 0.99844 / 0.99875 | 0.0004 / 0.0017 / 0.0004 |
| noul | 95 | yes | 0.99929 / 0.99921 / 0.99929 | 0.0009 / 0.0017 / 0.0009 |
| score | 101 | yes | 0.99900 / 0.99881 / 0.99900 | 0.0082 / 0.0057 / 0.0082 |
| long_license | 2437 | yes | 0.99843 / 0.99817 / 0.99860 | 0.0008 / 0.0011 / 0.0012 |

- Chunked: `prefill_traced_chunked`, the path `PplxDecider` uses.
- Oracle: `prefill_tp`, a single stateless prefill.
- The same input twice gives bit-identical logits.
- Traced: `prefill_traced_chunked` after `capture_prefill_trace()`.

Performance (1x4 Blackhole, batch 1):

| Metric | Value |
| --- | --- |
| Model load, warm tensor cache | ~11 s |
| Model load, cold (builds ~31 GB cache) | ~3 min |

Batch-1 decision latency (ms, p50, warm, 1x4 Blackhole):

| Tokens | Eager | Traced |
| --- | --- | --- |
| 512 | 155 | 157 |
| 1024 | 189 | 186 |
| 2048 | 288 | 252 |
| 3072 | 480 | 435 |
| 4096 | 579 | 504 |

- Below about 1k tokens the eager path is bound by host dispatch (~155 ms floor).
- 512 and 1024 tokens run the masked bucket eagerly, so traced equals eager.
- Trace capture also compiles every masked bucket, so the first call at a new bucket size does not pay a compile (up to ~17 s).

## Limitations

- Text only; vision weights are not loaded.
- Batch 1.
- Prefill only; no decode or sampling.
- Trace covers full 2048-token chunks only; shorter prompts and the tail run eagerly.
- Max 8192 tokens.
- The full-vocab synthetic lm_head costs ~635 MB DRAM per device.
- Logits are bf16 on device.
