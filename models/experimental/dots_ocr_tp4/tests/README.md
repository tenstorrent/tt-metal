# dots.ocr — end-to-end & TP-scheme tests (`tt_symbiote`)

All tests need a **`(1, 4)` mesh (4 devices)**
and are verified on Blackhole QB2 — set `MESH_DEVICE=P150x4`.

> The per-module PCC unit tests for the `dots_ocr_tp4` building blocks are documented
> separately [below](#dotsocr-tp4--per-module-unit-tests).

| # | Test | Scope | Checks against | Scheme options |
|---|---|---|---|---|
| 1a | `test_dots_ocr_decode_one_layer_l1_boundaries` | One decoder layer, decode (1 tok) | HF reference layer (PCC) | `row` (default) / `col_parallel` |
| 1b | `test_dots_ocr_decode_full_decoder_l1_boundaries` | Full 28-layer decoder, decode (1 tok) | HF reference stack (PCC) | `row` (default) / `col_parallel` |
| 1c | `test_dots_ocr_vision` | Full vision + text pipeline on a demo image | Generated text output | `row` (default) / `col_parallel` |
| 2 | `test_dots_ocr_attention_tp4` | Decode attention, TP=4 | HF reference (PCC) | `k_parallel` / `n_parallel` |
| 3 | `test_dots_ocr_mlp_tp4` | Decode MLP, TP=4 | Torch reference (PCC) | `row` / `col_fused_shardout` / `col_fused` / `col` |

## 1. `test_dots_ocr.py`

### 1a. Single decoder layer (decode, PCC)

Runs one decoder layer in decode mode for one token and PCC-checks it against the HF
reference layer.

```bash
# row parallel (default)
MESH_DEVICE=P150x4 pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_decode_one_layer_l1_boundaries

# col parallel (use 'both' to run row + col)
MESH_DEVICE=P150x4 DOTS_OCR_DECODE_ONE_LAYER_TP_SCHEMES=col_parallel pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_decode_one_layer_l1_boundaries
```

### 1b. Full 28-layer decoder (decode, PCC)

Same as 1a but across all 28 decoder layers; PCC-checks the final stack output for one
token. Catches error that accumulates layer-to-layer.

```bash
# row parallel (default)
MESH_DEVICE=P150x4 pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_decode_full_decoder_l1_boundaries

# col parallel (use 'both' to run row + col)
MESH_DEVICE=P150x4 DOTS_OCR_DECODE_FULL_DECODER_TP_SCHEMES=col_parallel pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_decode_full_decoder_l1_boundaries
```

### 1c. Full vision + text pipeline (end-to-end)

Runs the full vision + text pipeline on a demo image and prints the generated text.

```bash
# col parallel
MESH_DEVICE=P150x4 TT_SYMBIOTE_RUN_MODE=TRACED DOTS_OCR_TP_DECODE_SCHEME=col_parallel pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr.py::test_dots_ocr_vision
```


## 2. `test_dots_ocr_attention_tp4.py`

Unit test for the decode attention at TP=4; PCC-checks the final decode token against the
HF reference. Runs both `k_parallel` and `n_parallel`
QKV schemes.

```bash
MESH_DEVICE=P150x4 pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr_attention_tp4.py
```

## 3. `test_dots_ocr_mlp_tp4.py`

Unit test for the MLP at TP=4 (prefill and decode), PCC-checked against a torch reference.
Covers the `row`, `col_fused_shardout`, `col_fused`, and `col` variants.

```bash
MESH_DEVICE=P150x4 pytest -s \
    models/experimental/tt_symbiote/tests/test_dots_ocr_mlp_tp4.py
```

---

# dots.ocr TP4 — per-module unit tests

Module-level PCC unit tests for the dots.ocr **TP4** (4-chip tensor-parallel) building
blocks. Each test builds one TTNN module from a reference (`from_torch` →
`set_device`/`preprocess`/`move` as needed), runs its `forward` on the mesh, and
compares the result to a numerical reference via PCC. They isolate a single module so
a regression (wrong shard ordering, dropped collective, bad RoPE, mis-tuned matmul)
shows up at the smallest boundary.

Two designs are covered:

- **Text decoder (replicated-hidden TP4)** — the hidden stream is the full `H=1536`
  replicated on every chip; attention is head-sharded (3 Q-heads + 1 KV-head/chip,
  fused per-chip `[Q3|K1|V1]`), MLP is column/row-parallel, and the only collective is
  a final `all_reduce`. Reference weights are **random** (no model download).
- **Vision tower (N/K-shard, Blackhole-only)** — `TTNNDotsVisionMLPTP4BH` /
  `TTNNDotsVisionAttentionTP4BH` from `tt_symbiote`, hardware-swept BH kernels with a
  Ring `all_reduce`. Reference weights are the **real pretrained dots.ocr** weights.

| Test file | Test(s) | Module under test | Reference | Input (per chip) | PCC gate |
|---|---|---|---|---|---|
| `test_attention.py` | `test_dots_ocr_attention_tp4` | `DotsOCRAttentionTP4` (text GQA prefill) | `TorchGQAAttention` (half-half RoPE, causal GQA 12h/2kv) | `[1, 2816, 1536]` bf16 | **0.99** |
| `test_mlp.py` | `test_dots_ocr_mlp_tp4` | `DotsOCRMLPTP4` (text SwiGLU) | `TorchSwiGLUMLP` (`silu(gate)*up → down`) | `[1, 2816, 1536]` bf16 | **0.98** |
| `../../tt_symbiote/tests/test_vision_mlp_attn_tp4_module.py` | `test_vision_mlp_tp4_module`, `test_vision_attn_tp4_module` | `TTNNDotsVisionMLPTP4BH`, `TTNNDotsVisionAttentionTP4BH` | HF vision MLP / qkv+o_proj + 2D-RoPE float ref | `[1, 1, 11264, 1536]` bf8 | **0.95** (≈0.998 / 0.983 obs.) |

## What each test validates

### `test_attention.py` — text GQA attention
Builds `DotsOCRAttentionTP4.from_torch(mesh, config, torch_attn)` and PCCs the
replicated prefill output vs a float `TorchGQAAttention`. Exercises the per-chip fused
`[Q3|K1|V1]` QKV projection, half-half RoPE, GQA head sharding (KV head 0 on chips 0–1,
head 1 on chips 2–3), SDPA, and the row-parallel `o_proj` + `all_reduce`. Lower PCC here
points at a sharding/RoPE/collective bug.

### `test_mlp.py` — text SwiGLU MLP
Builds `DotsOCRMLPTP4.from_torch(mesh, config, torch_mlp)` and PCCs vs a float
`TorchSwiGLUMLP`. Exercises the fused column-parallel gate+up matmul, `silu(gate)*up`,
the row-parallel `down` matmul + `all_reduce`, and the production low-precision recipe
(BFP4 gate/up + down, BFP8 activations) — hence the slightly looser 0.98 gate.

### `test_vision_mlp_attn_tp4_module.py` — vision MLP + attention (Blackhole TP4)
Builds the real `TTNNDotsVision{MLP,Attention}TP4BH` modules end-to-end and PCCs **every
one of the 4 TP device shards** against the HF reference (both modules reduce to a
replicated full-hidden output, so all shards must agree). The MLP path is N-shard
fc1/fc3 → K-shard fc2 + Ring `all_reduce`; the attention path uses L1 activations, the
swept QKV/o_proj/SDPA program configs, and the device 2D-RoPE tables. Distinct from the
`*_n_k_shard_*` tests, which exercise the raw sharded matmul ops rather than the module.

## Running

```bash
# Text modules (random weights, no download; default mesh = 1x4 on this host)
MESH_DEVICE=P150x4 pytest -s models/experimental/dots_ocr_tp4/tests/test_attention.py
MESH_DEVICE=P150x4 pytest -s models/experimental/dots_ocr_tp4/tests/test_mlp.py

# Vision modules (Blackhole P150x4 / P300x2 only; loads pretrained dots.ocr weights)
MESH_DEVICE=P150x4 pytest -s models/experimental/tt_symbiote/tests/test_vision_mlp_attn_tp4_module.py
```

## Notes / requirements

- **Mesh**: text tests resolve the mesh from `MESH_DEVICE` via
  [`tests/common.py`](common.py) (`resolve_mesh_shape`, `device_params`) and default to
  the 4-chip `1x4` ring; the design targets 4 chips (`num_heads % 4 == 0`,
  `intermediate % 4 == 0`). The vision tests are gated to **Blackhole** `P150x4`/`P300x2`
  (`run_for_blackhole`, swept kernels) and skip otherwise.
- **Weights**: text tests build random torch references in-process (no network). The
  vision tests load `rednote-hilab/dots.ocr` via `from_pretrained` — set
  `DOTS_OCR_MODEL_PATH` to a local checkpoint or rely on the HF cache; they `skip` if
  weights are unavailable.
- **Shape**: the text seq_len is `2816` (the OCR vision-token bucket); the vision
  modules use the fixed `S=11264` (grid `88x128`) bucket the BH kernels are swept for.
- **Comparison**: text outputs are read back replicated via `from_replicated_to_torch`;
  vision outputs are checked per-device via `ttnn.get_device_tensors`.
