# Llama weight transfer (ttml ↔ tt-transformers)

Reference for two related dict formats:

1. The HuggingFace `transformers` Llama state-dict (the format on the
   HF Hub).
2. The on-device dict that
   [`LlamaCompositeKV.weights_ref_hf_dict()`](../sources/examples/grpo_remote_rollout/utils/llama_overrides.py)
   produces for transfer to
   [`tt_transformers.tt.model.Transformer.update_weights`](../../models/tt_transformers/tt/model.py).
   This same dict is also the wire format consumed by the cross-rank
   [`WeightBridge`](../sources/examples/grpo_remote_rollout/utils/weight_bridge.py),
   which ships it from a ttml rank to a tt-transformers rank over MPI.
   It is what the GRPO BoolQ example
   ([`tt-train/sources/examples/grpo_remote_rollout/boolq/`](../sources/examples/grpo_remote_rollout/boolq/))
   uses to push the freshly-updated policy to the inference worker on
   every training step.

The structural rules apply to every Llama 3 / 3.1 / 3.2 / 3.3 decoder
(only counts and dims change). The full key listing in §2.3 is for
`meta-llama/Llama-3.2-1B-Instruct`.

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
representation has only one. The in-process transfer format in §3 fans
the single ttml tensor out into both HF keys (see [§3.4 Subtleties](#34-subtleties)).

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

Verbatim listing for `meta-llama/Llama-3.2-1B-Instruct`:

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

## 3. Transfer format: `ttml → tt-transformers`

[`LlamaCompositeKV.weights_ref_hf_dict()`](../sources/examples/grpo_remote_rollout/utils/llama_overrides.py)
returns a `dict[str, ttnn.Tensor]` that is the **wire format** between
ttml and tt-transformers'
[`Transformer.update_weights(hf_state_dict, hf_rope=False)`](../../models/tt_transformers/tt/model.py).
Everything in this section is the contract — both the in-process call
and the cross-rank bridge enforce it.

### 3.1 Universal tensor properties

Every value in the returned dict has the same per-tensor representation:

| property        | value                                          | notes                                                                  |
|-----------------|------------------------------------------------|------------------------------------------------------------------------|
| Python type     | `ttnn.Tensor`                                  | on-device handle, no host download                                     |
| `dtype`         | `ttnn.bfloat16`                                | matches ttml's storage; consumer recasts if its destination differs    |
| `layout`        | `ttnn.TILE_LAYOUT`                             | matches ttml's storage; `_inplace_copy` re-layouts only on mismatch    |
| `memory_config` | `ttnn.DRAM_MEMORY_CONFIG` (interleaved)        | consumer re-shards if its destination is L1 or width-sharded           |
| mesh placement  | fully replicated across every mesh axis        | DDP supported (e.g. `[1, 2]`); TP / CP / sharded weights not — see [§3.4](#34-subtleties) |
| rank / shape    | 4D, `(1, 1, *, *)`                             | HF Linear / embedding / gamma shape wrapped in two leading unit dims   |

The 4D shape convention is the consumer's contract — every leaf
`.update()` method in tt-transformers (`Attention`, `MLP`, `RMSNorm`,
`LMHead`, `Embedding`) accepts 4D explicitly and adapts dtype / layout /
memcfg inside `_inplace_copy`.

### 3.2 Key set

Identical to HF safetensors (§2): **`3 + 9·L`** HF dot-keys. For
Llama-3.2-1B-Instruct that is exactly the 147 keys listed in
[§2.3](#23-full-key-list-for-llama-32-1b-instruct-147-keys).

### 3.3 Shape per key

`L = num_hidden_layers`, `H = hidden_size`, `I = intermediate_size`,
`V = vocab_size`, `n_h = num_attention_heads`,
`n_kv = num_key_value_heads`, `D = H / n_h`.

| HF key                                                | 4D shape              |
|-------------------------------------------------------|-----------------------|
| `model.embed_tokens.weight`                           | `(1, 1, V, H)`        |
| `lm_head.weight`                                      | `(1, 1, V, H)`        |
| `model.norm.weight`                                   | `(1, 1, 1, H)`        |
| `model.layers.{i}.input_layernorm.weight`             | `(1, 1, 1, H)`        |
| `model.layers.{i}.post_attention_layernorm.weight`    | `(1, 1, 1, H)`        |
| `model.layers.{i}.self_attn.q_proj.weight`            | `(1, 1, n_h·D, H)`    |
| `model.layers.{i}.self_attn.k_proj.weight`            | `(1, 1, n_kv·D, H)`   |
| `model.layers.{i}.self_attn.v_proj.weight`            | `(1, 1, n_kv·D, H)`   |
| `model.layers.{i}.self_attn.o_proj.weight`            | `(1, 1, H, n_h·D)`    |
| `model.layers.{i}.mlp.gate_proj.weight`               | `(1, 1, I, H)`        |
| `model.layers.{i}.mlp.up_proj.weight`                 | `(1, 1, I, H)`        |
| `model.layers.{i}.mlp.down_proj.weight`               | `(1, 1, H, I)`        |

Strip the two leading unit dims and you get the HF Linear `(out, in)` /
embedding `(V, H)` / gamma `(H,)` shapes from §2 exactly.

### 3.4 Subtleties

- **Tied embeddings.** Llama-3.2-1B-Instruct has
  `tie_word_embeddings = true`. ttml stores a single `Llama/fc/weight`.
  The exporter exposes the same `ttnn.Tensor` handle under **both**
  `model.embed_tokens.weight` and `lm_head.weight` — the consumer's
  strict dispatcher requires both keys. Aliasing is safe because
  `update_weights` does a per-key `ttnn.copy` into separate destination
  buffers and never aliases the source.

- **K / V split.** ttml fuses K and V into a single `kv_linear/weight`
  of shape `(1, 1, 2·n_kv·D, H)` with K rows first (rows
  `[0, n_kv·D)`) and V rows second. The exporter splits these into
  `k_proj.weight` and `v_proj.weight` via two `ttnn.slice` calls. These
  slices are *newly allocated* tensors
  (~`2·n_kv·D·H·sizeof(bf16)·L` bytes total; ~64 MB for
  Llama-3.2-1B-Instruct) and can be freed immediately after
  `update_weights` returns.

- **Q / K row order (RoPE convention).** ttml's
  `GroupedQueryAttentionCompositeKV` applies RoPE on Meta-style
  interleaved-pair rotary `(h_0, h_1), (h_2, h_3), …`;
  `safetensors_loader._unpermute_proj_rows` has already converted
  HF → Meta on load. tt-transformers' default for Llama-3.2-1B
  (`use_hf_rope = False`) uses the same Meta convention. So Q / K rows
  in the exported dict are already in the consumer's order; pass
  `hf_rope = False` to `update_weights`. The exporter does **not**
  apply any row permutation. If a future consumer is built with
  `use_hf_rope = True`, the caller is responsible for permuting Q / K
  on host before re-uploading (the on-device HF ↔ Meta permutation in
  `Attention.update` currently raises `NotImplementedError`).

- **Aliasing & lifetime.** Apart from the per-layer K / V slices, every
  value in the dict is a *handle* into ttml's live parameter store. Do
  not mutate ttml's parameters between the call to `weights_ref_hf_dict`
  and the call to `update_weights`. After `update_weights` returns the
  K / V slices may be deallocated; the rest are owned by ttml and stay
  live as long as the ttml model lives.

- **Replicated parameters only.** Every value in the dict must be
  fully replicated across every mesh axis (`PlacementReplicate`
  everywhere) — enforced at runtime by
  `WeightBridge._validate_source_tensor` for the cross-rank path. This
  is independent of the *sender's* mesh shape: a DDP-only `[1, N]`
  mesh (e.g. `mesh_shape: [1, 2], enable_ddp: true` from
  [`grpo_boolq_llama_1b_ddp_2dev.yaml`](../configs/training_configs/grpo_boolq_llama_1b_ddp_2dev.yaml))
  produces replicated weights and is supported. TP / CP / sharded
  weights are **not**: callers would need to compose shards back to a
  replicated view on host before exporting (not yet implemented).

  On the receiving side, `tt_transformers.Attention.update` currently
  raises `NotImplementedError` when `num_devices_per_group > 1`, so the
  receiver mesh has to keep per-tensor weights on a single device group
  (typically a `[1, 1]` submesh on the tt-transformers side). The
  asymmetric `[1, 2] → [1, 1]` cross-rank configuration used by the
  GRPO BoolQ example satisfies both constraints simultaneously: the
  bridge ships replicated weights over a single `(0,0) → (0,0)` socket
  connection, and the receiver lands them on its lone chip.

### 3.5 Usage

In-process (ttml model and tt-transformers model on the same Python
process and the same mesh):

```python
hf_dict = ttml_model.weights_ref_hf_dict()
try:
    ttt_model.update_weights(hf_dict, hf_rope=False)
finally:
    del hf_dict  # frees the per-layer k_proj / v_proj slices
```

Cross-rank, via the `WeightBridge` (one process per rank, separate
meshes; the BoolQ GRPO example uses this path on every step). The
transport is created explicitly and threaded through the RPC client:

```python
from utils.mpi_rollout import MPIRolloutClient
from utils.weight_bridge import HostWeightBridge, TTML_RANK, TTT_RANK

# ttml rank (sender)
bridge = HostWeightBridge.init_sender(mesh=ttml_mesh, peer_rank=TTT_RANK)
client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)   # ctor calls bridge.connect()
client.send_weights(ttml_model.weights_ref_hf_dict())

# ttt rank (receiver) — typical setup inside MPIRolloutServer.serve_forever:
bridge = HostWeightBridge.init_receiver(
    mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=worker.submeshes
)
# server responds to OP_REQUEST_TRANSFER by calling bridge.receive_weights()
for submesh_dict in bridge.receive_weights():
    ttt_model.update_weights(submesh_dict, hf_rope=False)
```

`MPIRolloutClient.__init__` takes `peer_rank=` and `bridge=` — see
[`mpi_rollout.py`](../sources/examples/grpo_remote_rollout/utils/mpi_rollout.py)
and the working construction in
[`boolq_training_example.py`](../sources/examples/grpo_remote_rollout/boolq/boolq_training_example.py).

End-to-end smoke tests:
- Weight transfer (in-process, per submesh):
  [`test_weight_transfer.py`](../tests/python/grpo_remote_rollout/weight_transfer/test_weight_transfer.py).
- `WeightBridge` cross-rank transport:
  [`test_weight_bridge.py`](../tests/python/grpo_remote_rollout/weight_bridge/test_weight_bridge.py).

---

## References

* HF reference implementation: `transformers.models.llama.modeling_llama`
  ([huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)).
* ttml model definition: `LlamaCompositeKV` in
  [`grpo_remote_rollout/utils/llama_overrides.py`](../sources/examples/grpo_remote_rollout/utils/llama_overrides.py).
* tt-transformers dispatcher:
  [`Transformer.update_weights`](../../models/tt_transformers/tt/model.py).
* Cross-rank transport: `WeightBridge` in
  [`grpo_remote_rollout/utils/weight_bridge.py`](../sources/examples/grpo_remote_rollout/utils/weight_bridge.py).
