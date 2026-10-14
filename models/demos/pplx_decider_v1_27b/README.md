# pplx-decider-v1-27b on Blackhole

- A decision model fine-tuned from Qwen3.8-27B.
- Returns calibrated probabilities over 1-255 options.
- Question types: `choice`, `noul` (yes-no), `score`.
- Text only on TT.

## How it runs on qwen36

| Part                   | Source                                      | Name                                                                    |
| ---------------------- | ------------------------------------------- | ----------------------------------------------------------------------- |
| Model args             | Reused from `models/demos/blackhole/qwen36` | `Qwen36ModelArgs`                                                       |
| Model                  | Reused                                      | `Qwen36Model`                                                           |
| State dict remap       | Reused                                      | `remap_qwen36_state_dict`                                               |
| Prefill                | Reused                                      | `prefill_traced_chunked`, `prefill_tp`                                  |
| Sequence parallel (2D) | Reused                                      | `SeqParallel` in `qwen36/tt/sp.py`                                      |
| Checkpoint key rename  | pplx-specific (this dir)                    | `tt/weight_mapping.py`                                                  |
| Bucket prefill traces  | pplx-specific                               | `tt/bucket_trace.py`: one prefill + readout trace per length bucket     |
| Prefill tuning         | pplx-specific                               | `tt/prefill_tuning.py`: SDPA, AGMM and reduce-scatter configs by length |
| Synthetic LM head      | pplx-specific                               | Readout rows at decision token ids, zeros elsewhere                     |
| MTP                    | pplx-specific                               | `enable_mtp=False`                                                      |
| Prompt format          | pplx-specific                               | `tt/decision.py`                                                        |
| Answer structure       | pplx-specific                               | `answer()`                                                              |
| Mesh test setup        | pplx-specific                               | `parametrize_mesh_traced`, `model_mesh` in `tests/common.py`            |

- The synthetic head works because the readout is applied after the final norm, like the LM head.

## Meshes

| Mesh                  | Layout                                        | Example hardware                  |
| --------------------- | --------------------------------------------- | --------------------------------- |
| 4x8 (32 chips)        | TP8 along each row, prompt split over 4 rows  | Blackhole Galaxy                  |
| 1x8                   | TP8                                           | Galaxy submesh                    |
| 1x4                   | TP4                                           | QuietBox 2, or a Galaxy submesh   |

- On 4x8 every row holds the full TP8 model, and each row prefills a contiguous quarter of the prompt bucket.
- Attention gathers K/V across the rows; GDN layers prefill each row's slice independently and stitch the recurrent state across rows with `ttnn.experimental.kda.chain_affine_transforms` (`qwen36/tt/sp.py`).

## Requirements

- A Blackhole mesh from the table above. For 1xN hosts set `MESH_DEVICE=P150x4` (for example 2x P300) or `MESH_DEVICE=P150x8`.
- Host RAM: ~60 GB for the bf16 weight load.
- Host RAM: ~155 GB for the fp32 CPU reference generator.
- Disk: ~52 GB weights plus ~31 GB TT tensor cache per mesh shape.

## Setup

- Download the checkpoint, or set `HF_MODEL` to a local directory.

```bash
hf download perplexity-ai/pplx-decider-v1-27b
export TT_CACHE_PATH=$HOME/tt_cache/pplx_decider_v1_27b
export MESH_DEVICE=P150x4  # 1xN hosts only; leave it unset for a 4x8 Galaxy run
```

| Variable           | Meaning                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| `HF_MODEL`         | Hub id or local dir. Default: `perplexity-ai/pplx-decider-v1-27b`.              |
| `MESH_DEVICE`      | 1xN mesh shape: `P150x4` or `P150x8`.                                           |
| `TT_CACHE_PATH`    | TT tensor cache dir. Use a pplx-specific dir.                                   |
| `HF_HUB_OFFLINE`   | Set to `1` to skip hub lookups.                                                 |
| `PPLX_PARENT_MESH` | Galaxy only: the full mesh the tests open, `4x8` or `8x4`.                      |
| `PPLX_SUBMESH`     | Galaxy only: the mesh the model runs on, `4x8` (the parent), `1x8` or `1x4`.    |

- The cache path does not include the model name.
- Sharing it with another Qwen checkpoint loads the wrong weights.
- If `TT_CACHE_PATH` is unset, the cache goes inside the checkpoint dir.
- Tensor caches carry the mesh shape (for example `tensor_cache_bfp8_mesh4x8`), so each mesh shape builds its own.

## Run

Demo (choice, noul, score samples):

```bash
pytest models/demos/pplx_decider_v1_27b/demo/demo.py
```

Device accuracy test (TT vs CPU reference):

```bash
pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
```

Batch-1 latency sweep, eager vs traced (256 to 4096 tokens):

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

### On a Galaxy

- Open the full mesh and run on it or on a submesh of it: a bare 1x8 or 1x4 open fails the fabric handshake, and `TT_VISIBLE_DEVICES` breaks the fabric.
- `tests/common.py` does this for the demo and the tests from `PPLX_PARENT_MESH` and `PPLX_SUBMESH`.

```bash
unset TT_VISIBLE_DEVICES MESH_DEVICE
export PPLX_PARENT_MESH=4x8 PPLX_SUBMESH=4x8   # SP4 x TP8 on all 32 chips
pytest models/demos/pplx_decider_v1_27b/demo/demo.py
pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
pytest models/demos/pplx_decider_v1_27b/tests/test_perf.py

# TP8 on a 1x8 submesh, TP4 on a 1x4 submesh
MESH_DEVICE=P150x8 PPLX_PARENT_MESH=4x8 PPLX_SUBMESH=1x8 pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
MESH_DEVICE=P150x4 PPLX_PARENT_MESH=8x4 PPLX_SUBMESH=1x4 pytest models/demos/pplx_decider_v1_27b/tests/test_model.py
```

- On 4x8, `test_model.py` runs the traced path only and `test_perf.py` times the traced prefill only: the chunk and oracle paths are 1xN-only.
- `PPLX_TEST_PATHS` (comma list of `chunked`, `oracle`, `traced`, `traced_chunk`) picks the paths `test_model.py` runs.

## Python usage

```python
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

decider = PplxDecider.from_pretrained(mesh_device)  # mesh_device: a 1x4, 1x8 or 4x8 mesh
decider.capture_prefill_trace()  # one trace per length bucket up to 4096 tokens; required on 4x8
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

- Open the mesh with `trace_region_size` >= `TRACE_REGION_SIZE` (1 GiB) before capturing the traces.
- On a Galaxy open the full mesh with `FABRIC_1D_RING`, as `tests/common.py` does.

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
- Measured on a Blackhole Galaxy; 1x8 and 1x4 are submeshes of it.

Accuracy vs the fp32 CPU reference (`tests/test_model.py`, bfp8 weights, traced path):

| Example      | Tokens | PCC 4x8 | PCC 1x8 | PCC 1x4 |
| ------------ | ------ | ------- | ------- | ------- |
| choice       | 110    | 0.99881 | 0.99882 | 0.99890 |
| noul         | 95     | 0.99936 | 0.99930 | 0.99940 |
| score        | 101    | 0.99921 | 0.99911 | 0.99910 |
| long_license | 2437   | 0.99862 | 0.99846 | 0.99861 |

- Top-1 matches the reference on every example and mesh; max probability diff is 0.0024.
- On 1xN the chunked, oracle and traced_chunk paths also match top-1, with PCC 0.9984 to 0.9993.
- Traced: the bucket traces after `capture_prefill_trace()`, the path `PplxDecider` uses.
- Chunked: `prefill_traced_chunked` with no trace captured.
- Traced_chunk: the 2048-token chunk trace that serves prompts above 4096 tokens on 1xN.
- Oracle: `prefill_tp`, a single stateless prefill.
- The same input twice gives bit-identical logits.

Batch-1 decision latency (ms, p50, warm, traced, `tests/test_perf.py`):

| Tokens | 4x8 SP4 x TP8 | 1x8 TP8 | 1x4 TP4 |
| ------ | ------------- | ------- | ------- |
| 256    | 35.9          | 41.1    | 61.2    |
| 512    | 41.2          | 52.3    | 71.4    |
| 1024   | 49.9          | 72.8    | 110.1   |
| 2048   | 62.9          | 130.0   | 195.2   |
| 3072   | 77.7          | 198.0   | 331.8   |
| 4096   | 92.7          | 270.6   | 440.8   |

- Without traces (1xN only) the same prompts take 220 to 510 ms, bound by host dispatch.
- A QuietBox 2 (`P150x4`) has not been re-measured since the bucket traces.

| Metric                                  | Value     |
| --------------------------------------- | --------- |
| Model load, warm tensor cache           | ~11-13 s  |
| Model load, cold (builds ~31 GB cache)  | ~3-4 min  |
| `capture_prefill_trace()`, 4x8          | ~20-35 s  |

## Tuning switches

- The defaults are the tuned settings.

| Variable                 | Default             | Effect                                                                       |
| ------------------------ | ------------------- | ---------------------------------------------------------------------------- |
| `PPLX_SP`                | `1`                 | 2D mesh: split each bucket's tokens over the mesh rows; `0` replicates rows  |
| `PPLX_SP_MIN`            | `256`               | Smallest bucket that uses sequence parallel                                  |
| `PPLX_BUCKETS`           | `256,512,...,4096`  | Comma list of bucket sizes, multiples of 128                                 |
| `PPLX_BUCKET_TRACE`      | `1`                 | `0`: only the 2048-token chunk trace (1xN)                                   |
| `PPLX_RS_BF8`            | `0`                 | Reduce-scatters move bf8 partials; faster from ~1.5k tokens, calibration effect not settled |
| `QWEN36_SP_GDN_FOLD_MAX` | `256`               | SP GDN: one folded delta-rule call for row slices up to this length; `0` never folds |

## Limitations

- Text only; vision weights are not loaded.
- Batch 1.
- Prefill only; no decode or sampling.
- Bucket traces cover prompts up to 4096 tokens.
- On 1xN longer prompts, up to `max_seq_len` (default 8192), use the 2048-token chunk trace, and the tail runs eagerly.
- On 4x8 prompts above 4096 tokens raise `NotImplementedError`, and every prompt needs `capture_prefill_trace()` first.
- The full-vocab synthetic lm_head costs ~635 MB DRAM per device.
- Logits are bf16 on device.
