# ACE-Step v1.5 (Torch ref + TTNN)

This folder provides:

- `torch_ref/`: PyTorch reference implementation
- `ttnn_impl/`: TTNN implementation with one-to-one module mapping
- `tests/`: per-module PCC validation (Torch vs TTNN)

## Mandatory constraints (enforced by design)

- **TTNN device purity**: TTNN modules must not call PyTorch ops inside their `forward()`; the only allowed transfers are:
  - Host → device at the start of the run (inputs + weights)
  - Device → host at the end (final outputs for PCC comparison)
- **One-to-one mapping**: every Torch module has a TTNN equivalent.

## Attention modes (`attention_impl`)

`AceConfig.attention_impl` / `AceConfigTTNN.attention_impl`:

| Value | Torch module | TTNN module |
|-------|----------------|-------------|
| `"explicit"` | `MultiHeadSelfAttention` | `MultiHeadSelfAttentionTTNN` (matmul + softmax + causal mask) |
| `"sdpa"` | `MultiHeadSelfAttentionSDPA` (`F.scaled_dot_product_attention`) | `MultiHeadSelfAttentionSDPATTNN` (`ttnn.transformer.scaled_dot_product_attention`) |

Use the same string on both configs so `TransformerBlock` / `TransformerBlockTTNN` stay aligned.

TTNN SDPA rejects TILE tensors whose **logical** head dimension is smaller than the tile padding on that axis (e.g. head_dim 16 inside 32-wide tiles). The TTNN module zero-pads Q/K/V along head_dim to the next multiple of **32** before SDPA and slices the output back to the real `d_head`, with **`scale = 1/sqrt(d_head)`** so scaling still matches PyTorch.

## Layout

```
ace_step_v1_5/
  torch_ref/
  ttnn_impl/
  tests/
```

## Running tests

From repo/workspace root:

```bash
python -m pytest models/demos/ace_step_v1_5/tests \
  --confcutdir=models/demos/ace_step_v1_5/tests -q
```

If you have TT hardware/runtime, set:

```bash
export MESH_DEVICE=N150   # or N300 / T3K
```

## Notes

- PCC threshold in tests is set to `>= -0.9` per request (very lenient). You can tighten it later.
- The `mesh_device` pytest fixture is **session-scoped** (one `open_mesh_device` per process). Opening and closing a mesh around every test exhausts Metal context IDs and can trigger invalid `context_id` / teardown crashes on distributed meshes.

## TTNN demo (`ttnn_impl/full_pipeline.py`) — weights default to **Base**

The TTNN entrypoint defaults to Hugging Face **`ACE-Step/acestep-v15-base`** when you omit a local checkpoint:

```bash
cd /home/ubuntu/proj_sdk/tt-metal
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --out-npy /tmp/ace_features.npy
```

Override weights explicitly:

```bash
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --checkpoint-safetensors /path/to/model.safetensors \
  --out-npy /tmp/ace_features.npy
```

Turbo lives under the umbrella repo ``ACE-Step/Ace-Step1.5``; pick it with ``--hf-subfolder``:

```bash
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --hf-repo-id ACE-Step/Ace-Step1.5 \
  --hf-subfolder acestep-v15-turbo \
  --out-npy /tmp/out_turbo.npy
```

### HF parity note (`ttnn_impl/full_pipeline.py` / `ttnn_impl/dit_decoder_core.py`)

This TTNN path implements the **DiT decoder stack** (patch embed → conditioned transformer layers → output head) but is **not** a guaranteed byte-for-byte match to HF ``modeling_acestep_v15_*.py``. Documented gaps include **RoPE**, **per-layer sliding-window attention masks**, and **runtime vs lookup timestep embeddings**. See the docstring on ``AceStepV15TTNNPipeline``.

## Torch demo (HF weights + deterministic output signature)

This demo downloads an ACE-Step 1.5 checkpoint from Hugging Face, extracts the **DiT output head**
weights (`norm_out`, `scale_shift_table`, `proj_out`), runs a small forward pass, and prints:

- the **full snapshot path** on disk
- the inferred **state_dict prefix** for the output head inside the checkpoint
- output **shape** and a small numeric signature (`mean/std/first8`)

```bash
cd /home/ubuntu/proj_sdk/tt-metal
python3 -m models.demos.ace_step_v1_5.torch_ref.hf_output_head_demo \
  --repo-id "ACE-Step/Ace-Step1.5" \
  --subfolder "acestep-v15-turbo" \
  --seed 0 --batch 1 --original-seq-len 257 --noise-std 1.0
```
