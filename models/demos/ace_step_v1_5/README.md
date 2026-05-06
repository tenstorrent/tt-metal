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
