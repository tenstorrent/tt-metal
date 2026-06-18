# Qwen3.6-27B vLLM + TP=8 port notes

This documents the vLLM (continuous-batching) + tensor-parallel (TP=8) port of
the **dense** Qwen3.6-27B model, adapted from the working coder_next (MoE) vLLM
template at `/home/yito/vllm_template/coder_next_src/`.

**Status: CODE ONLY — never run on HW here. Everything below needs HW
verification (see TODO).**

## Files created (all imports under `models.demos.qwen36_27b.tt.*`)

| File | What it is |
|------|------------|
| `generator_vllm.py` | vLLM adapter class `Qwen3_5ForConditionalGeneration` (+ alias `Qwen36ForCausalLM`). `initialize_vllm_model`, `allocate_kv_cache`, `prefill_forward`, `decode_forward`, `process_decode_output_host`, persistent DeltaNet row/slot management. Builds a **dense** TP=8 model (`dense_tp=True, tp_size=8`; NO EP). |
| `model_vllm.py` | `TtQwen36VllmModel` — dense decoder stack, TP, replicated embedding/norms, vocab-col-parallel lm_head + `all_gather`, `forward_vllm_prefill/decode`, `create_deltanet_state(batch=)`. |
| `decoder_vllm.py` | `TtHybridDecoderLayer` — token-mixer (DeltaNet or GQA) + **dense gated MLP**; replicated RMSNorms; passes vLLM paged/contiguous hooks to attention. |
| `attention_vllm.py` | `TtGatedAttention` — TP GQA (fused Wqkv col-parallel, row-parallel o_proj + all_reduce), KV-head ×2 replication (nkv=4 < TP=8), contiguous trace decode + CPU-fallback prefill that fills the contiguous cache. |
| `deltanet_vllm.py` | `TtDeltaNetState` (with `batch=` + slot management) + `TtGatedDeltaNet` (TP-sharded qkv/z col-parallel, out_proj row-parallel; fused kernels). |
| `mlp_vllm.py` | `TtMLP` — dense SwiGLU; fused gate|up col-parallel, down row-parallel + all_reduce. |
| `load_weights_vllm.py` | `StreamingStateDict` for the HF Qwen3.6-27B key layout. |
| `mesh_utils.py` | `to_torch(tensor, dim=...)` — read single/multi-device tensors to host (compose shards or de-dup replicas). The template imported this from coder_next; it did not exist as source here so it was created. |

The existing single-device files (`model.py`, `attention.py`, `deltanet.py`,
`decoder.py`, `mlp.py`, `generator.py`, `load_weights.py`, `model_config.py`)
were left intact; only `model_config.py` was edited **additively** (new TP/vLLM
fields + helper methods) so single-device imports keep working.

## ModelRegistry line

```python
ModelRegistry.register_model(
    "TTQwen3_5ForConditionalGeneration",
    "models.demos.qwen36_27b.tt.generator_vllm:Qwen3_5ForConditionalGeneration",
)
```

## HF config -> Qwen36ModelConfig mapping

HF arch `Qwen3_5ForConditionalGeneration`, `model_type="qwen3_5"`. The text
backbone dims live under `config.text_config` (the top-level config is the
multimodal wrapper). `_hf_config_to_cfg` reads from `text_config` and maps:

- hidden_size 5120, num_hidden_layers 64, vocab_size 248320
- num_attention_heads 24, num_key_value_heads 4, head_dim 256
- full_attention_interval 4 (layer_types repeat `[lin, lin, lin, full]`)
- intermediate_size 17408 (DENSE gated MLP)
- linear_num_key_heads 16, linear_num_value_heads 48 (ratio 3), key/value_head_dim 128, conv_kernel 4
- partial_rotary_factor 0.25 -> rotary_dim 64
- rope_theta 1e7 (read from `text_config.rope_parameters.rope_theta`)
- precision: dense weights bf8, math HiFi4 (fp32 accum), `ondevice_attn=True`

Checkpoint path: `hf_config._name_or_path` (falls back to
`/home/yito/work/qwen36_27b_hf`).

DeltaNet derived dims: key_dim = 16*128 = 2048, value_dim = 48*128 = 6144,
conv_dim = 2*2048 + 6144 = **10240**.

## HF weight key layout (verified from the real checkpoint index.json)

Text backbone nested under `model.language_model.`; `lm_head.weight` is top
level. `load_weights_vllm._normalize_key` strips `language_model.` to the
canonical `model.layers.*` / `model.embed_tokens` / `model.norm` names the
modules ask for. Vision-tower keys (`model.visual.*`) are excluded.

Per layer:
- DeltaNet (linear_attention): `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}.weight`, `linear_attn.{A_log,dt_bias}`, `linear_attn.conv1d.weight`, `linear_attn.norm.weight`.
  - **Important difference vs HF Qwen3-Next:** the projections are stored ALREADY
    SPLIT (`in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`), NOT fused
    (`in_proj_qkvz`/`in_proj_ba`). So NO `_fused_split` row-reordering is needed
    (unlike the coder_next loader). Confirmed by listing the index keys.
- Full attention (i where (i+1)%4==0): `self_attn.{q_proj,k_proj,v_proj,o_proj}.weight` + `self_attn.{q_norm,k_norm}.weight`. `attn_output_gate=True` => `q_proj` outputs `2*head_dim*nheads` (query | gate), handled by the attention split.
- Dense MLP: `mlp.{gate_proj,up_proj,down_proj}.weight`.
- `input_layernorm.weight`, `post_attention_layernorm.weight`.

## Dense vs MoE changes (vs the coder_next template)

- Decoder FFN is `TtMLP` (gate/up/down) instead of `TtMoE*` — all EP/MoE
  construction (`device_ep`, `ep_size`, `moe_ep*`, routing, shared expert) is
  removed.
- The model takes `dense_tp/tp_size` only (no `device_ep/ep_size`).
- The whole stack is TP-sharded (coder_next only TP-sharded the "dense stack"
  and kept experts as EP).

## TP=8 scheme (mirrors p300x2/tp_model.py, the authoritative dense-TP ref)

Mesh: `MeshShape(1, 8)`, `FabricConfig.FABRIC_1D`, `Topology.Linear`.

- **MLP**: gate|up fused -> column-parallel weight `[H, 2I]` sharded on dim 3;
  per chip computes its `[.., 2I/8]`, split to gate(silu)/up. down weight `[I,H]`
  row-parallel (sharded on input dim 2); local matmul + `all_reduce`.
- **Attention**: q|k|v fused -> column-parallel `Wqkv` sharded on dim 3. nh=24 ->
  3 query heads/chip. **nkv=4 < TP=8** so each KV head is replicated ×2
  (`kv_slots=8`, 1 KV head/chip) by tiling the k/v weight on the head axis before
  sharding. o_proj `[nh*hd, H]` row-parallel (input dim 2) + `all_reduce`.
  Contiguous per-chip KV cache `[B, nkvp=1, cache_len, hd]`.
- **DeltaNet**: in_proj_qkv / in_proj_z col-parallel (dim 3) -> `all_gather` the
  linear output back to full before the (replicated) fused kernel; out_proj
  row-parallel (dim 2) -> `mesh_partition` input + `all_reduce`. Per-head tensors
  (b/a/A_log/dt_bias/norm/conv) replicated.
- **lm_head**: vocab(column)-parallel, vocab padded to `tp_size*32`, `all_gather`
  on dim 3 to full vocab. Embedding + all RMSNorm weights replicated.

## TODO / likely bugs (un-runnable here — needs HW verification)

1. **GQA head<->KV-head mapping under TP with KV replication (HIGHEST RISK).**
   nkv=4, num_kv_groups=6, nh=24, TP=8. The KV-head ×2 replication makes
   `kv_slots=8` (1/chip), but with 3 query heads/chip the GQA grouping
   (query head g attends KV head g//6) does NOT line up cleanly with a naive
   contiguous q-head shard. tp_model.py uses TP=4 (nkv=4 -> exactly 1/chip, no
   replication) and does NOT exercise the nkv<TP case. **Verify the per-chip
   query-head -> KV-head assignment is correct**; may need to reorder query-head
   columns of `Wqkv` (or the KV replication pattern) so each chip's 3 query heads
   share the KV head that chip holds. If they don't all share one KV group, the
   fused single-KV-head-per-chip layout is invalid and KV must be laid out
   differently (e.g. 2 chips per KV group). This is the first thing to validate
   against `p300x2/test_tp_attention.py` extended to TP=8.

2. **`nlp_create_qkv_heads_decode` with nkvp=1, nhp=3** — confirm it accepts these
   per-chip head counts and the height-sharded layout SDPA-decode expects.

3. **DeltaNet fused-kernel arg dims under TP** — `_decode_step_full_fused` gathers
   qkv/z to full width then runs the replicated fused kernel with GLOBAL head
   counts (num_heads=48, etc.). Confirm that's what the fused
   `deltanet_decode_full` / `deltanet_prefill_full` kernels want when weights are
   sharded but the op input is gathered. (tp_model.py instead runs the kernel on
   PER-CHIP head counts `nvp/nkp` with sharded conv/A/dt — an alternative if the
   gather path is too slow or wrong.) conv_dim=10240.

4. **Attention prefill TP path** (`_prefill` / `_prefill_attn_out`) is a host
   correctness fallback: composes the sharded qkv back to host, does the attention
   math on CPU, then shards the full `[1,1,S,nh*hd]` attn output on the last dim
   (dim 3) to feed the row-parallel o_proj + all_reduce (same pattern as
   deltanet `_out_proj`). This assumes the contiguous head order of the o_proj
   input matches the Wqkv column-shard order (chip c -> query heads c*nhp:(c+1)*nhp)
   — re-verify alongside TODO #1. Also the contiguous KV-cache fill re-replicates
   KV heads to `kv_slots` then shards on dim 1; confirm each chip's cache ends up
   with the KV head matching its query heads.

5. **`all_reduce` / `all_gather` / `mesh_partition` on a 1x8 Linear mesh** — the
   template used `Topology.Linear` for an 8-chip line; tp_model.py used
   `Topology.Ring` for a 2x2 p300_x2 torus. Pick the topology that matches the
   ACTUAL target mesh (Linear for a true 1x8 line). All collectives currently use
   `cluster_axis=1` + `Topology.Linear`.

6. **`mesh_utils.to_torch` replica de-dup** — for a replicated tensor it composes
   on dim 0 and slices `[:per]`. Verify `ConcatMeshToTensor`/`get_num_devices`
   APIs exist on this tt-metal build; otherwise fall back to `ttnn.to_torch` with
   the appropriate `mesh_composer`.

7. **DeltaNet decode requires the fused kernel** — `deltanet_vllm.forward` raises
   if `deltanet_decode_full` is unavailable (no CPU fallback wired for TP). Ensure
   the image provides `ttnn.experimental.deltanet_decode_full` /
   `deltanet_prefill_full`.

8. **vocab_size 248320** is already a multiple of `8*32=256`, so lm_head padding
   is a no-op; double-check no off-by-tile in the `all_gather` slice.

9. **Memory** — bf8 dense weights across 8 chips; ~27B/8 plus the persistent
   DeltaNet state `[max_batch, 48, 128, 128]` per linear layer (48 layers) +
   conv `[max_batch,1,10240,32]`. Size `max_batch_size` (max_num_seqs)
   conservatively; tune like the coder_next adapter did.
