# Head-parallel attention for StagePrefillTP4 (pi0.5 VLM prefill)

**Date:** 2026-06-16
**Scope:** TP=4 (and TP=8) VLM **prefill** only. New `_GemmaAttentionTP4`; shared
`GemmaAttentionTTNN` (single-device + denoise) untouched.

## Goal
Today attention is **replicated** on every chip (~3.8 ms/chip fixed: SDPA 2.16 + RoPE 1.2
+ heads 0.4), which is the ceiling on TP scaling. Make attention **head-parallel** so the
Q-head work (QKV proj, SDPA, O-proj) shards across chips, at the cost of one added
`all_reduce`/layer. Target: shrink the per-chip attention so TP>4 keeps paying off.

## Architecture (MQA: 8 q-heads, 1 kv-head, head_dim 256, width 2048)
- **Q-heads sharded** across `tp` chips (`hpc = num_q_heads // tp`: 2 @TP=4, 1 @TP=8).
- **K/V replicated** (single kv-head — every chip needs it).
- Per chip: QKV→2(or 1) q-heads + 1 kv-head → SDPA over its heads → concat → row-parallel
  O-proj → **partial** → `all_reduce(num_links=2)` (in the block, like the MLP).
- Residual stream stays **replicated** (all_reduce output identical across chips) → MLP &
  norms unchanged.

## Approach A (chosen): reuse `GemmaAttentionTTNN` with a reduced-head config
The existing forward (fused-QKV → `nlp_create_qkv_heads` → RoPE → SDPA → `nlp_concat_heads`
→ O-proj → reshape) works **per-chip unchanged** when fed:
1. a per-chip wqkv `[hidden, hpc·256 + 512]` (this chip's Q-heads | K | V),
2. `config.num_heads = hpc` (via `dataclasses.replace(config, num_heads=hpc)`),
3. a row-parallel o_proj `[hpc·256, 2048]`.
It then returns a **per-chip partial** `[.,seq,2048]`; the block does the all_reduce.
No new forward logic — reuses RoPE/SDPA/head-split/KV-cache verbatim.

## Changes (all in `tt_metal/.../tt_bh_glx/stage_prefill_tp4.py`)
1. **`_load_block_weights_tp4`** — head-parallel attention weights:
   - **wqkv**: host-build interleaved `[Q_0heads|K|V | Q_1heads|K|V | …]` (K/V duplicated per
     chip), `ShardTensorToMesh(dim=-1)` → chip i gets `[hidden, hpc·256+512]`, bf8_b.
     `tp = mesh_device.get_num_devices()`, `hpc = num_q_heads // tp`.
   - **o_proj**: `.T` then `ShardTensorToMesh(dim=0)` (row-parallel) → chip `[hpc·256, 2048]`.
   - Assert `num_q_heads % tp == 0` and `tp in (4, 8)`.
2. **`_GemmaAttentionTP4`** — thin subclass/factory: build `GemmaAttentionTTNN` with
   `dataclasses.replace(config.vlm_config, num_heads=hpc)` + the sharded weights. Inherits
   `forward` (returns per-chip partial).
3. **`_GemmaBlockTP4`** — use `_GemmaAttentionTP4`; after `attention.forward(...)` returns
   the partial, `attn_out = ttnn.all_reduce(partial, num_links=2, memory_config=L1)`, then
   residual add. (Mirror the MLP all_reduce.)

## Correctness criteria
- `test_prefill_tp4_pcc` PCC ≥ 0.99 at **TP=4** and **TP=8** (`PI0_TP=8`, chips 8–15).
  Baseline replicated attention is 0.995127 — head-parallel should match within bf8 noise.
- Output shape unchanged `(1, 1024, 2048)`.

## Test / validation plan (main session, device, with go-ahead)
1. `test_prefill_tp4_pcc` TP=4 → PCC ≥ 0.99.
2. Same TP=8.
3. Tracy profile TP=4: confirm attention matmuls/SDPA shrink (per-chip) and read the net
   (sharded SDPA savings vs the added attention all_reduce). Compare to the 12.7 ms/chip
   production-env baseline.

## Risks / gotchas
- **wqkv interleave order** must match `nlp_create_qkv_heads`'s expected `[Q|K|V]` fused
  layout per chip; getting the head order wrong → wrong PCC (silent). Verify head slice math.
- **o_proj shard dim**: row-parallel shards the **K/in** dim (the concat-heads axis), `.T`
  dim=0. Wrong dim → shape mismatch or wrong reduction.
- Aliasing caveats in `GemmaAttentionTTNN.forward` (don't deallocate q/k/cos) still apply.
- `all_reduce` adds CCL to attention; net win is **not guaranteed** — it's the thing we measure.
- bf8 K/V replicated: same per-tile quantization as the replicated baseline (tile-aligned).

## Out of scope
Decode/denoise expert attention; the shared `GemmaAttentionTTNN`; arbitrary TP (only 4, 8).
