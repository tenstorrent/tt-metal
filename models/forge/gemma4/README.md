# Gemma4-31B-it on tt-metal (1×4 Blackhole mesh)

Inference implementation of `google/gemma-4-31B-it` for a (1, 4) Blackhole
device mesh. Loads HF weights directly, runs prefill and decode through
a single `Gemma4ForCausalLM` instance, and samples next tokens via a
`Generator` that does one prefill followed by N decode steps with
persistent KV caches.

## Architecture

Gemma4-31B-it is a 30B-parameter decoder-only transformer with 60
layers in two flavors:

- **Sliding-window attention** (50 layers): GQA with 8 Q heads and 4 KV
  heads per shard, head_dim = 256, 256-token sliding window.
- **Full attention** (10 layers — every 6th layer plus L59): same head
  layout, no window cutoff.

Each layer is the standard pre-norm Gemma block:
`RMSNorm → Attention → RMSNorm → residual → RMSNorm → FeedForward → RMSNorm → residual → ×layer_scalar`.
Layer 59 (terminal) folds its `layer_scalar` into the LM head's
`last_layer_scalar`.

## File layout

| File | Role |
|---|---|
| `model.py` | `Gemma4ForCausalLM` — top-level orchestrator, dispatches `mode='prefill'\|'decode'` to `_call_prefill` / `_call_decode`. |
| `attention.py` | `Attention` — fused QKV matmul, q/k_norm, RoPE, KV cache write, SDPA, o_proj. Both layer types in one class. |
| `decoder_layer.py` | `SlidingDecoderLayer` / `FullDecoderLayer` — composition of Attention + FeedForward + four RMSNorms. |
| `feed_forward.py` | `FeedForward` — gate-up-down MLP with GeLU. |
| `rms_norm.py` | `RMSNorm` — pre-layer normalization. |
| `prelude.py` | Mode-specific bookkeeping ops that run once per forward pass before the layer loop (RoPE caches, position helpers, mask scaffolding). |
| `lm_head.py` | `LMHead` — final RMSNorm + tied embedding matmul + softcap. |
| `scaled_embedding.py` | Token embedding lookup with the Gemma4 normalization scale. |
| `rope.py` | `RoPESetup` — builds per-mode inv_freq tables for the preludes. |
| `caches.py` | `Gemma4Caches` — preallocated K/V buffers (50 sliding @ `[1,4,256,256]`, 10 full @ `[1,1,256,512]`), plus `reset()`. |
| `generator.py` | `Generator` — `generate()` (fast prefill+decode) and `slow_generate()` (prefill-loop fallback). |
| `weights.py` | HF state-dict loading + scalar/RoPE constant builders. |
| `runtime_inputs.py` | Synthesizes the per-call non-KV runtime inputs (token IDs, scalar 1.0, ones helper at slot 26). |
| `layer_table.py` | `LAYER_TABLE` — per-layer type and historical input slot indices. |
| `config.py` | Gemma4 hyperparameter wrapper. |
| `utils.py` | `DeviceGetter` — shared mesh device singleton. |
| `demo.py` | End-to-end CLI demo. |
| `test_prefill.py` / `test_decode.py` | PCC gates against `reference_logits/{prefill,decode}.pt`. |

## Setup

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
export PYTHONPATH=$(pwd):$(pwd)/models/forge
```

Hardware: 4 Blackhole devices arranged as a (1, 4) mesh.

## Running the demo

```bash
python -m gemma4.demo "What is your favorite city?"
```

Defaults: `--seq-len 128`, `--max-new-tokens 32`. The demo:

1. Tokenizes the prompt with Gemma's chat template.
2. Loads HF weights and builds the model.
3. Calls `Generator.generate(prompt_ids, max_new_tokens=...)`:
   - One prefill pass writes the prompt's K/V into the caches at rows
     `0..len(prompt)-1`.
   - Each decode step writes the new K/V at row `current_pos` via
     `paged_update_cache`, reads the cache via
     `scaled_dot_product_attention_decode(... sliding_window_size=256)`,
     and samples greedily.

```bash
# Other examples
python -m gemma4.demo "Why is the sky blue?" --max-new-tokens 50
python -m gemma4.demo "Tell me a story" --seq-len 200 --max-new-tokens 100
```

`seq_len` must be greater than the prompt token count and ≤ 256 (the
KV cache row count). Generation is capped at
`min(max_new_tokens, seq_len - prompt_len)`.

## Running PCC tests

```bash
pytest models/forge/gemma4/test_prefill.py -v -s
pytest models/forge/gemma4/test_decode.py  -v -s
```

Both tests build the model with `seq_len=19` (the codegen-baked length
the reference logits at `reference_logits/{prefill,decode}.pt` were
captured at), run a single forward pass, and assert
`PCC > 0.99` against the reference. Current values:

- `prefill PCC ≈ 0.999464` (argmax = `2205` "As")
- `decode  PCC ≈ 0.999524` (argmax = `240017`)

Each test takes ~60 s on the (1, 4) mesh.

## Implementation notes

### Single model instance, two modes

`Gemma4ForCausalLM.from_state_dict(hf, mesh_device, *, seq_len=128)`
builds one instance with both decode and prefill scaffolding.
`__call__(input, *, mode, current_pos=0)` dispatches to
`_call_decode` or `_call_prefill`. Layers (`SlidingDecoderLayer` /
`FullDecoderLayer`) take `is_decode` per call and route to
`_decode_body` / `_prefill_body` internally.

### KV caches

`Gemma4Caches` holds 60 K and 60 V tensors, one per layer (sliding
shape `[1, 4, 256, 256]`, full shape `[1, 1, 256, 512]`, both
BFLOAT16 TILE replicated). Each layer holds direct refs to its K/V
slice. `Generator.reset()` (which calls `model.reset_kv_caches()`)
zeros every cache and re-attaches the fresh refs to layers; the
generator owns this between independent sessions.

### Cache writes

- **Prefill** (sliding and full): `ttnn.fill_cache(k_cache, K, 0)`
  writes `seq_len` rows at the start of the cache. Same for V.
- **Decode** (sliding and full):
  `ttnn.experimental.paged_update_cache(k_cache, K, update_idxs_tensor=current_pos)`
  writes a single row at `current_pos`. The K/V tensors are sharded
  to L1 `[32, 256]` (sliding) or `[32, 512]` (full) before the call.

### SDPA

- **Sliding decode** (post-rewrite):
  `ttnn.transformer.scaled_dot_product_attention_decode(Q, k_cache, v_cache, cur_pos_tensor=current_pos, scale=1.0, sliding_window_size=256, ...)`.
  The op handles GQA (8 Q heads over 4 KV heads), causal masking,
  and the window cutoff. Mirrors `tt_transformers/tt/attention.py`.
- **Sliding prefill** and **full** (decode + prefill): manual matmul
  SDPA inherited from the codegen. Functionally correct; could be
  replaced with `ttnn.transformer.scaled_dot_product_attention[_decode]`
  for further simplification.

`scale=1.0` matches the legacy graph: Gemma3-style attention relies
on RMSNorm-controlled Q/K magnitudes and skips the standard
`1/sqrt(head_dim)` factor. Passing the textbook scale gives
gibberish.

### Position scalar plumbing

`current_pos` flows through the model as two related tensors injected
in `Gemma4ForCausalLM.__call__`:

- **Slot 0** (`var_0`) gets `current_pos`. `SlidingPreludeDecode`
  reads it directly (`fp32(var_0) * sliding_inv_freq`) to build the
  RoPE cos/sin caches, so Q and the new K end up with RoPE(`current_pos`).
  It also flows into `FullPreludeDecode` as `ttnn_reshape_18`, where
  the full-attention position mask (`var_0 >= arange(0..255)`) admits
  positions `0..current_pos` — including the new K just written at
  row `current_pos`.
- **Per-layer pos slots** (`runtime_inputs[2]` for L0..L58) also get
  `current_pos`. They feed `paged_update_cache(update_idxs=…)` and
  `scaled_dot_product_attention_decode(cur_pos_tensor=…)`.

Both slots receive the same value (`current_pos`); the orchestration's
`ttnn_add_0 = var_0 + var_185(=1)` is computed for op-graph completeness
but is only consumed by the now-unused sliding position-helper subgraph.
At `current_pos=0` everything collapses to 0, matching the codegen
reference.

### Generator

```python
generator = gemma4.Generator(model, tokenizer, seq_len=128)
ids = generator.generate(prompt_ids, max_new_tokens=32)
```

`generate()` is the fast `O(seq_len + N)` path. `slow_generate()` is
the prefill-loop fallback (each step pads `prompt + tokens-so-far` to
`seq_len` and re-runs prefill); useful for cross-checking the fast
path or for cases where decode-side state diverges.

## Known issues

- **Sliding prefill body** and **full attention bodies** still use the
  codegen-derived manual matmul SDPA (~1500 lines combined). Functional
  and PCC-passing, but a future cleanup could replace them with
  `ttnn.transformer.scaled_dot_product_attention[_decode]` to match
  the canonical pattern used in `_sliding_decode`.

## Reference

`models/tt_transformers/tt/attention.py` is the canonical tt-metal
attention pattern this implementation increasingly mirrors. The
sliding-decode rewrite tracks lines 630-820 there;
`paged_update_cache` and `scaled_dot_product_attention_decode` use
the same kwargs and shape conventions.
