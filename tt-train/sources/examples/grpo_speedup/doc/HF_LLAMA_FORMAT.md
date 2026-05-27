# HuggingFace Llama state-dict format

Reference for what a HuggingFace `transformers` Llama checkpoint looks like
on disk and in memory, so that `ttml` can export its own trained Llama
weights back into the HF format and have them be loadable by
`AutoModelForCausalLM.from_pretrained(...)`.

The concrete listing below was produced by
[`dump_llama_keys.py`](./dump_llama_keys.py) against
`meta-llama/Llama-3.2-1B-Instruct`. The structural rules apply to every
Llama 3 / 3.1 / 3.2 / 3.3 decoder (only counts and dims change).

---

## 1. Model configuration → tensor counts

For a Llama-3-style decoder with `L` layers, hidden size `H`, intermediate
size `I`, `n_heads` attention heads, `n_kv_heads` grouped-query heads, head
dim `D = H / n_heads`, and vocab size `V`, the HF state dict has:

```
3 global tensors
+ L blocks × 9 tensors per block
= 3 + 9·L
```

| Model                    | L  | H    | I     | n_heads | n_kv_heads | D   | V       | total keys |
|--------------------------|----|------|-------|---------|------------|-----|---------|------------|
| Llama-3.2-1B-Instruct    | 16 | 2048 | 8192  | 32      | 8          | 64  | 128 256 | 147        |
| Llama-3.2-3B-Instruct    | 28 | 3072 | 8192  | 24      | 8          | 128 | 128 256 | 255        |
| Llama-3.1-8B-Instruct    | 32 | 4096 | 14336 | 32      | 8          | 128 | 128 256 | 291        |

`n_kv_heads · D` is the K / V projection width (e.g. `8 · 64 = 512` for 1B).

---

## 2. HF naming convention (Llama-3.2-1B-Instruct)

All linear weights follow PyTorch's `nn.Linear` layout: shape is
`(out_features, in_features)` and the forward pass is
`y = x @ W.T + b` (no biases on the Llama linears).

### 2.1 Global tensors (3)

| HF key                       | shape         | semantics                          |
|------------------------------|---------------|------------------------------------|
| `model.embed_tokens.weight`  | `(V, H)`      | input token embedding lookup table |
| `model.norm.weight`          | `(H,)`        | final RMSNorm gain (γ)             |
| `lm_head.weight`             | `(V, H)`      | output projection                  |

For Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct
**`tie_word_embeddings = true`** in `config.json`. The safetensors files
on the Hub only contain `model.embed_tokens.weight`; HF re-uses the same
storage for `lm_head` on load. `transformers` *will* return both keys
from `model.state_dict()` (they share the same tensor), but the on-disk
representation has only one. See [§4 Export checklist](#4-export-checklist).

### 2.2 Per-block tensors (9 per `model.layers.{i}`)

Replace `{i}` with the block index `0 … L-1`.

| HF key                                                 | shape                  | role                          |
|--------------------------------------------------------|------------------------|-------------------------------|
| `model.layers.{i}.input_layernorm.weight`              | `(H,)`                 | pre-attention RMSNorm γ       |
| `model.layers.{i}.self_attn.q_proj.weight`             | `(n_heads · D, H)`     | Q projection (e.g. `2048×2048`) |
| `model.layers.{i}.self_attn.k_proj.weight`             | `(n_kv_heads · D, H)`  | K projection (e.g. `512×2048`)  |
| `model.layers.{i}.self_attn.v_proj.weight`             | `(n_kv_heads · D, H)`  | V projection (e.g. `512×2048`)  |
| `model.layers.{i}.self_attn.o_proj.weight`             | `(H, n_heads · D)`     | attention output projection   |
| `model.layers.{i}.post_attention_layernorm.weight`     | `(H,)`                 | pre-MLP RMSNorm γ             |
| `model.layers.{i}.mlp.gate_proj.weight`                | `(I, H)`               | SwiGLU gate (W1)              |
| `model.layers.{i}.mlp.up_proj.weight`                  | `(I, H)`               | SwiGLU up   (W3)              |
| `model.layers.{i}.mlp.down_proj.weight`                | `(H, I)`               | SwiGLU down (W2)              |

### 2.3 Full key list for Llama-3.2-1B-Instruct (147 keys)

Output of `dump_llama_keys.py`, verbatim:

```text
model.embed_tokens.weight                        shape=(128256, 2048)
model.layers.0.self_attn.q_proj.weight           shape=(2048, 2048)
model.layers.0.self_attn.k_proj.weight           shape=(512, 2048)
model.layers.0.self_attn.v_proj.weight           shape=(512, 2048)
model.layers.0.self_attn.o_proj.weight           shape=(2048, 2048)
model.layers.0.mlp.gate_proj.weight              shape=(8192, 2048)
model.layers.0.mlp.up_proj.weight                shape=(8192, 2048)
model.layers.0.mlp.down_proj.weight              shape=(2048, 8192)
model.layers.0.input_layernorm.weight            shape=(2048,)
model.layers.0.post_attention_layernorm.weight   shape=(2048,)
...  (layers 1 .. 15 repeat the 9-tensor pattern above)  ...
model.layers.15.self_attn.q_proj.weight          shape=(2048, 2048)
model.layers.15.self_attn.k_proj.weight          shape=(512, 2048)
model.layers.15.self_attn.v_proj.weight          shape=(512, 2048)
model.layers.15.self_attn.o_proj.weight          shape=(2048, 2048)
model.layers.15.mlp.gate_proj.weight             shape=(8192, 2048)
model.layers.15.mlp.up_proj.weight               shape=(8192, 2048)
model.layers.15.mlp.down_proj.weight             shape=(2048, 8192)
model.layers.15.input_layernorm.weight           shape=(2048,)
model.layers.15.post_attention_layernorm.weight  shape=(2048,)
model.norm.weight                                shape=(2048,)
lm_head.weight                                   shape=(128256, 2048)
```

---

## 3. ttml ↔ HF mapping

ttml's `LlamaCompositeKV` exposes its parameters as 4-D tensors
`(1, 1, rows, cols)` (matrices) or `(1, 1, 1, n)` (vectors). The leading
two unit dims are ttml's storage convention and must be squeezed for HF.

### 3.1 Global tensors

| ttml key (live)                | ttml shape              | HF key                        | HF shape         | transform                                 |
|--------------------------------|-------------------------|-------------------------------|------------------|-------------------------------------------|
| `Llama/fc/weight`              | `(1, 1, V, H)`          | `model.embed_tokens.weight`   | `(V, H)`         | `.squeeze((0,1))`                         |
| `Llama/fc/weight` *(same)*     | `(1, 1, V, H)`          | `lm_head.weight`              | `(V, H)`         | `.squeeze((0,1))` — tied; see [§4](#4-export-checklist) |
| `Llama/ln_fc/gamma`            | `(1, 1, 1, H)`          | `model.norm.weight`           | `(H,)`           | `.squeeze()`                              |

> **Weight tying.** The ttml config
> [`llama3_2_1B.yaml`](../../../configs/model_configs/llama3_2_1B.yaml)
> sets `weight_tying: enabled`, so ttml has **no** separate
> `tok_embeddings`/`embed_tokens` parameter — `Llama/fc/weight` is used
> both as the LM head and (transposed) as the input embedding lookup.

### 3.2 Per-block tensors

For block `i`:

| ttml key                                          | ttml shape                                                        | HF key                                                  | HF shape                  | transform                                                              |
|---------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------|---------------------------|------------------------------------------------------------------------|
| `Llama/blocks/{i}/attention_norm/gamma`           | `(1, 1, 1, H)`                                                    | `model.layers.{i}.input_layernorm.weight`               | `(H,)`                    | `.squeeze()`                                                           |
| `Llama/blocks/{i}/attention/q_linear/weight`      | `(1, 1, n_heads·D, H)`                                            | `model.layers.{i}.self_attn.q_proj.weight`              | `(n_heads·D, H)`          | `.squeeze((0,1))` — see [§3.3](#33-qk-weight-layout-rope-convention)    |
| `Llama/blocks/{i}/attention/kv_linear/weight`     | `(1, 1, 2·n_kv_heads·D, H)`<br>e.g. `(1, 1, 1024, 2048)` for 1B   | **split row-wise** into:<br>`model.layers.{i}.self_attn.k_proj.weight` (first `n_kv_heads·D` rows)<br>`model.layers.{i}.self_attn.v_proj.weight` (last `n_kv_heads·D` rows) | `(n_kv_heads·D, H)` each | `.squeeze((0,1))`, then `torch.split(W, n_kv_heads·D, dim=0)`           |
| `Llama/blocks/{i}/attention/out_linear/weight`    | `(1, 1, H, n_heads·D)`                                            | `model.layers.{i}.self_attn.o_proj.weight`              | `(H, n_heads·D)`          | `.squeeze((0,1))`                                                      |
| `Llama/blocks/{i}/mlp_norm/gamma`                 | `(1, 1, 1, H)`                                                    | `model.layers.{i}.post_attention_layernorm.weight`      | `(H,)`                    | `.squeeze()`                                                           |
| `Llama/blocks/{i}/mlp/w1/weight`                  | `(1, 1, I, H)`                                                    | `model.layers.{i}.mlp.gate_proj.weight`                 | `(I, H)`                  | `.squeeze((0,1))`                                                      |
| `Llama/blocks/{i}/mlp/w3/weight`                  | `(1, 1, I, H)`                                                    | `model.layers.{i}.mlp.up_proj.weight`                   | `(I, H)`                  | `.squeeze((0,1))`                                                      |
| `Llama/blocks/{i}/mlp/w2/weight`                  | `(1, 1, H, I)`                                                    | `model.layers.{i}.mlp.down_proj.weight`                 | `(H, I)`                  | `.squeeze((0,1))`                                                      |

Naming gotchas:

* ttml MLP uses Meta's `w1`/`w2`/`w3` (`w1` = gate, `w3` = up, `w2` =
  down). HF uses semantic names (`gate_proj`, `up_proj`, `down_proj`).
  Do **not** swap `w2` and `w3`.
* ttml fuses K and V into a single `kv_linear`. HF stores them as
  separate `k_proj` and `v_proj`. The split is row-wise (output axis):
  rows `[0 : n_kv_heads·D)` → K, rows `[n_kv_heads·D : 2·n_kv_heads·D)` → V.
  Match the order ttml used when *building* `kv_linear` — i.e. K rows
  first, V rows second.

### 3.3 Q/K weight layout & RoPE convention

This is the one subtle correctness gotcha.

* **HF** applies rotary on the *complex* halves of `head_dim`: it
  interprets the head vector as two halves
  `[h0..h_{D/2-1}, h_{D/2}..h_{D-1}]` and rotates `(h_i, h_{i+D/2})`
  pairs.
* **Meta's reference code** applies rotary on *interleaved* pairs
  `(h_0, h_1), (h_2, h_3), …`. Converting an HF Q/K row vector into the
  Meta layout requires a per-head permutation —
  [`convert_hf_qkv_to_meta_format`](../../../../models/tt_transformers/tt/load_checkpoints.py)
  in `tt-transformers` does exactly that, and
  [`convert_hf_to_meta_no_qkv_permute`](../../../../models/tt_transformers/tt/load_checkpoints.py)
  is the opt-out used when the model keeps HF-style RoPE.

**Action item for ttml's exporter:** know which RoPE convention ttml
trained with and either pass the Q/K rows through unchanged (ttml keeps
HF-style RoPE) or apply the inverse of `convert_hf_qkv_to_meta_format`
(ttml uses Meta-style RoPE). The current `LlamaCompositeKV` follows the
tt-transformers conventions; cross-check against the loader path used
during *training* before exporting.

The `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and all RMSNorm
gammas need **no** permutation — only Q and K do, and only if the RoPE
layout differs.

---

## 4. Export checklist

When writing a ttml-state-dict → HF safetensors exporter, the work is:

1. **Rename** every ttml key to its HF counterpart using the table in
   [§3.2](#32-per-block-tensors).
2. **Squeeze** the two leading unit dims:
   `tensor.reshape(tensor.shape[2:])` for matrices,
   `tensor.reshape(-1)` for vectors.
3. **Split `kv_linear`** row-wise into `k_proj` and `v_proj` (each
   `n_kv_heads · D` rows tall).
4. **Apply the Q/K permutation** *only if* ttml trained with Meta-style
   RoPE (see [§3.3](#33-qk-weight-layout-rope-convention)). For
   Llama-3.2-1B as configured in
   [`llama3_2_1B.yaml`](../../../configs/model_configs/llama3_2_1B.yaml),
   inspect the training loader to decide.
5. **Resolve weight tying.** ttml has only one `Llama/fc/weight`.
   * **Preferred:** copy it to `model.embed_tokens.weight` only, and set
     `tie_word_embeddings: true` in the exported `config.json` so HF
     ties at load (matches Meta's released Llama-3.2-1B checkpoint).
   * **Alternative:** also write `lm_head.weight` pointing to the same
     buffer; safetensors will deduplicate via shared storage.
6. **Cast & save.** HF distributes Llama-3.2 weights as `bfloat16`
   (`torch_dtype: bfloat16` in `config.json`). Cast accordingly before
   `safetensors.torch.save_file(...)`.
7. **Write companion files** so `AutoModelForCausalLM.from_pretrained`
   succeeds:
   * `config.json`  — copy from the upstream model or re-emit using
     `transformers.LlamaConfig(...).save_pretrained(...)`. Required
     fields: `hidden_size`, `intermediate_size`, `num_hidden_layers`,
     `num_attention_heads`, `num_key_value_heads`, `vocab_size`,
     `max_position_embeddings`, `rope_theta`, `rope_scaling`,
     `tie_word_embeddings`, `torch_dtype`, `architectures:
     ["LlamaForCausalLM"]`.
   * `generation_config.json` — copy from upstream for chat behaviour.
   * `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
     — copy from upstream; ttml does not modify the tokenizer.
   * `model.safetensors[.index.json]` — produced by the save call. For
     1B everything fits in one file; for >5GB use sharded save.

### Minimal exporter sketch

```python
import torch
from safetensors.torch import save_file

def ttml_state_to_hf(ttml_params: dict[str, torch.Tensor],
                    *, n_kv_heads: int, head_dim: int,
                    permute_qk: bool) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    out["model.embed_tokens.weight"] = ttml_params["Llama/fc/weight"].reshape(-1, ttml_params["Llama/fc/weight"].shape[-1])
    out["model.norm.weight"]         = ttml_params["Llama/ln_fc/gamma"].reshape(-1)

    block_keys = sorted({k.split("/")[2] for k in ttml_params if k.startswith("Llama/blocks/")}, key=int)
    for i in block_keys:
        p = f"Llama/blocks/{i}"
        out[f"model.layers.{i}.input_layernorm.weight"]          = ttml_params[f"{p}/attention_norm/gamma"].reshape(-1)
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = ttml_params[f"{p}/mlp_norm/gamma"].reshape(-1)

        q = ttml_params[f"{p}/attention/q_linear/weight"].reshape(*ttml_params[f"{p}/attention/q_linear/weight"].shape[-2:])
        kv = ttml_params[f"{p}/attention/kv_linear/weight"].reshape(*ttml_params[f"{p}/attention/kv_linear/weight"].shape[-2:])
        o  = ttml_params[f"{p}/attention/out_linear/weight"].reshape(*ttml_params[f"{p}/attention/out_linear/weight"].shape[-2:])
        k, v = torch.split(kv, n_kv_heads * head_dim, dim=0)

        if permute_qk:
            # inverse of convert_hf_qkv_to_meta_format; see §3.3
            q = _meta_to_hf_qk(q, head_dim)
            k = _meta_to_hf_qk(k, head_dim)

        out[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = o

        out[f"model.layers.{i}.mlp.gate_proj.weight"] = ttml_params[f"{p}/mlp/w1/weight"].reshape(*ttml_params[f"{p}/mlp/w1/weight"].shape[-2:])
        out[f"model.layers.{i}.mlp.up_proj.weight"]   = ttml_params[f"{p}/mlp/w3/weight"].reshape(*ttml_params[f"{p}/mlp/w3/weight"].shape[-2:])
        out[f"model.layers.{i}.mlp.down_proj.weight"] = ttml_params[f"{p}/mlp/w2/weight"].reshape(*ttml_params[f"{p}/mlp/w2/weight"].shape[-2:])

    return out
```

`_meta_to_hf_qk` is the inverse of the row-reshuffle in
`convert_hf_qkv_to_meta_format`; skip it if ttml uses HF-style RoPE.

---

## 5. Verifying an exported checkpoint

```bash
# (a) sanity-check key set
python3 tt-train/sources/examples/grpo_speedup/dump_llama_keys.py \
    | tee /tmp/hf_keys.txt   # baseline from upstream

# (b) load the export and compare
python3 - <<'PY'
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("/path/to/exported/dir")
for k, v in m.state_dict().items():
    print(f"{k:55s} {tuple(v.shape)}")
PY
```

The key sets and shapes must match Section [§2.3](#23-full-key-list-for-llama-32-1b-instruct-147-keys)
exactly. Then run a forward-pass equivalence test (HF logits vs ttml
logits on the same prompt) — see
[`gen_hf_ttt.py`](./gen_hf_ttt.py) and
[`compare_ttt_dummy.py`](./compare_ttt_dummy.py) for harnesses that
already line up tokenisation, sampling RNG, and decode settings across
HF / ttml / tt-transformers.

---

## References

* HF reference implementation: `transformers.models.llama.modeling_llama`
  ([huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)).
* `convert_hf_to_meta` / `map_hf_to_meta_keys` /
  `convert_hf_qkv_to_meta_format`:
  [`models/tt_transformers/tt/load_checkpoints.py`](../../../../models/tt_transformers/tt/load_checkpoints.py).
* ttml model definition: `LlamaCompositeKV` in
  [`tt-train/sources/examples/grpo_speedup/utils/llama_overrides.py`](./utils/llama_overrides.py).
* Live key dump script: [`dump_llama_keys.py`](./dump_llama_keys.py).
