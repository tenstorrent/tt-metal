# Chunked Prefill Plan — gpt_oss_d_p

## Background

The current `GptOssPrefillPipeline` does a **single full-sequence forward pass** for the entire
prompt (see `tt_gpt_oss_prefill_pipeline.py:19` comment: "No chunking"). This means the SDPA
call in `attention/prefill.py:151` holds Q×K attention scores for the full sequence length in
SRAM — O(S²) memory. At 128K tokens this becomes unworkable.

**Chunked prefill** splits the prompt into fixed-size chunks (e.g. 4096 tokens) and runs one
forward pass per chunk. Each chunk writes its K/V into the paged KV cache, then computes
attention by reading back from the cache via `chunked_scaled_dot_product_attention`. SRAM
peak drops from O(S²) to O(chunk_size × S_cached).

**Constraint:** `chunked_scaled_dot_product_attention` requires a paged KV cache — it reads
K/V through a page table. Chunked prefill therefore implies paged mode. This is a good thing:
the paged-KV read path in gpt_oss is currently unexercised (fill is wired but the SDPA reads
fresh tensors, not the cache).

**Model stats:** 20B has 24 layers (12 sliding + 12 full); 120B has 36 layers (18 sliding +
18 full). Both are exactly half-and-half.

---

## What Needs to Change

Five concrete changes, roughly in dependency order.

### 1. `attention/prefill.py` — swap SDPA to chunked variant

**File:** `tt/attention/prefill.py:151`

Current:
```python
tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
    tt_q, tt_k, tt_v,
    is_causal=True,
    sliding_window_size=config.sliding_window,
    ...
)
```

After `paged_fill_cache` writes this chunk's K/V into the cache, switch to reading from the
cache instead of fresh tensors:

```python
if chunk_start_idx is not None and page_table is not None:
    # Chunked path: read K/V from paged cache; chunk_start_idx provides causal mask offset.
    tt_sdpa_out = ttnn.transformer.chunked_scaled_dot_product_attention(
        tt_q, k_cache, v_cache, page_table,
        chunk_start_idx=chunk_start_idx,
        program_config=...,
        compute_kernel_config=...,
    )
else:
    # Non-paged / single-forward fallback (accuracy tests, no page_table).
    tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q, tt_k, tt_v,
        is_causal=True,
        sliding_window_size=config.sliding_window,
        ...
    )
```

`chunk_start_idx` is the **absolute token position** of the start of this chunk (0 for the
first chunk, 4096 for the second, etc.). It shifts the causal mask so that tokens in this
chunk attend to all prior tokens in the cache.

**Sliding window caveat:** `chunked_scaled_dot_product_attention` may not accept
`sliding_window_size`. For sliding-window layers, verify whether the kernel supports it.
If not, sliding-window layers (layers where `config.sliding_window is not None`) may need
to keep the non-chunked path until the kernel is extended, or use a separate chunked-sliding
implementation. Mark this as a validation task (see §Test Plan step 3).

**Deallocate:** `tt_k` / `tt_v` are the freshly-computed tensors that were written into the
cache. In the chunked path they are no longer needed after `paged_fill_cache` and should be
deallocated before the SDPA call to free SRAM.

### 2. Thread `chunk_start_idx` through `layer.py` and `model.py`

`chunk_start_idx` needs to reach `attention/prefill.py`. The call chain is:

```
GptOssPrefillPipeline.prefill()
  → model.ttnn_prefill_forward(chunk_start_idx=...) [already accepts it, model.py:568]
    → model._forward_layers_and_head()              [does NOT accept it — ADD PARAM]
      → decoder_layer(...)                          [layer.py:92 — ADD PARAM]
        → self.self_attn(...)                       [layer.py:115 — ADD PARAM]
          → attention.prefill_forward(...)          [attention/__init__.py → attention/prefill.py]
```

**`model.py:_forward_layers_and_head` (line 326):** Add `chunk_start_idx=None` param, pass it
into each `decoder_layer(...)` call at line 380.

**`layer.py:DecoderLayer.__call__` (line 92):** Add `chunk_start_idx=None` param, pass it into
`self.self_attn(...)` at line 115.

**`attention/prefill.py:prefill_forward` (line 21):** Add `chunk_start_idx=None` param. Use it
in the branched SDPA call from step 1.

**`model.py:ttnn_prefill_forward` (line 560):** Already has `chunk_start_idx=None`. Wire it
into the `_forward_layers_and_head(...)` call at line 595.

### 3. Rope matrices — slice per chunk

**File:** `tt/model.py:588-592` (inside `ttnn_prefill_forward`)

Currently: `rope_setup.cos_matrix_prefill[:, :, :seq_len, :]`

For chunked prefill each forward pass covers only `chunk_size` tokens starting at
`chunk_start`. The rope matrices must reflect the absolute positions of this chunk:

```python
rope_mats = [
    self.rope_setup.cos_matrix_prefill[:, :, chunk_start:chunk_start + chunk_size, :],
    self.rope_setup.sin_matrix_prefill[:, :, chunk_start:chunk_start + chunk_size, :],
]
```

When `chunk_start_idx` is None (full-sequence path), keep the existing `[:seq_len]` slice.

### 4. `GptOssPrefillPipeline` — add chunk loop

**File:** `tt/tt_gpt_oss_prefill_pipeline.py`

Replace the single `ttnn_prefill_forward` call in `prefill()` with a loop:

```python
CHUNK_SIZE = 4096   # tunable; must be a multiple of block_size and tile size (32)

for chunk_start in range(0, actual_isl, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, actual_isl)
    is_last_chunk = chunk_end >= actual_isl

    # Slice the padded token IDs for this chunk.
    chunk_ids = padded[chunk_start:chunk_end]
    # Pad to CHUNK_SIZE if the last chunk is short.
    chunk_ids += [_PAD_TOKEN_ID] * (CHUNK_SIZE - len(chunk_ids))

    # Slice page table: block_size pages that cover [chunk_start, chunk_end).
    chunk_page_table = full_page_table[:, chunk_start // block_size : chunk_end // block_size]

    tt_input = self._prepare_input_tensor(chunk_ids)   # SP sharding happens here
    tt_logits = self.model.ttnn_prefill_forward(
        x=tt_input,
        kv_cache=self.kv_cache,
        page_table=tt_full_page_table,
        chunk_page_table=tt_chunk_page_table,
        chunk_start_idx=chunk_start,
        get_last_token=get_last_token if is_last_chunk else -1,
        on_layer_complete=callback,
    )
```

**`_prepare_input_tensor`:** Currently receives `max_seq_len` tokens and SP-shards them.
With chunking it receives `CHUNK_SIZE` tokens per call. Update `assert` and reshape to use
`CHUNK_SIZE` instead of `max_seq_len`. SP sharding logic is unchanged.

**`compile()`:** Warm-up with a single chunk of `CHUNK_SIZE` tokens (not `max_seq_len`).
After compilation, all chunks are fixed-shape and reuse the same trace.

### 5. Migration callback — per-layer-per-chunk vs per-layer

The `on_layer_complete` seam already exists (model.py:391). Two options:

**Option A — per-chunk-per-layer** (migrate after layer `k` of chunk `c`):
- `num_chunks × num_layers` migration events (e.g. 32 × 36 = 1152 for 128K with 4096-token chunks)
- Decode node gets layer 0's KV after the very first chunk finishes → earliest possible decode start
- More migration traffic; requires the migration endpoint to handle partial KV correctly

**Option B — per-layer-after-all-chunks** (migrate layer `k` only after all chunks complete it):
- `num_layers` events (36 for 120B)
- Decode cannot start until all chunks of layer 0 are done
- Simpler; matches the current NoOp seam exactly

Recommendation: start with **Option B** since it requires no change to `_build_migration_callback`.
Switch to Option A once the migration endpoint team has confirmed partial-KV semantics.

---

## Test Plan

1. **Non-paged path regression:** Run existing accuracy tests (`tests/accuracy/`) with no
   `page_table` — must exercise the `else` branch in step 1 and produce identical PCC to
   before. Nothing should change here.

2. **Paged single-chunk correctness:** Set `CHUNK_SIZE = actual_isl` (full sequence, single
   chunk). This exercises chunked SDPA without the loop. PCC vs HuggingFace reference should
   match the current non-chunked paged baseline.

3. **Sliding-window layer validation:** Check `chunked_scaled_dot_product_attention` kernel
   source for `sliding_window_size` support. If unsupported, keep the existing SDPA for
   sliding-window layers in the chunked path. Document which layers take which code path.

4. **Multi-chunk correctness:** Run with `CHUNK_SIZE = 512` on a short sequence (e.g. 2048
   tokens) → 4 chunks. Compare first-token argmax and PCC against HuggingFace reference.

5. **Long-context smoke test:** 32K tokens, `CHUNK_SIZE = 4096`, 8 chunks. Confirm SRAM
   does not OOM and first token matches reference.

6. **Compile / trace replay:** `compile()` then two back-to-back `prefill()` calls with
   different prompts — verify trace replay produces correct results for the second request.

---

## Open Questions

- Does `chunked_scaled_dot_product_attention` support `sliding_window_size`? (Check
  `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp` and any `sliding_window` param.)
- What `block_size` does the paged attention config use? (Needed to compute
  `chunk_page_table` slicing and validate that `CHUNK_SIZE` is a multiple of it.)
- Migration Option A or B? Confirm with migration team what granularity their endpoint expects.
- With SP=4, each row sees `CHUNK_SIZE / 4` tokens per chunk. `_prepare_input_tensor` must
  shard the chunk (not the full sequence) across rows. Verify this is consistent with the
  RoPE slice — each row should get the rope positions for its SP slice of the chunk.
