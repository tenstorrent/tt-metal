# AutoDebug: Llama 3.1 70B decode SDPA program overflow

## Headline finding

The decode call omits `program_config`, so on Blackhole SDPA decode is allowed to build its program over the device-default grid. The reported failure is at program finalization (`70976 > 70656`), before numerical execution, and is therefore best explained by an unnecessarily large decode program descriptor rather than by the 64-Q-head/8-KV-head GQA geometry.

This call-site omission is inconsistent with the repository's established decode configuration:

- `models/tt_transformers/tt/model_config.py::get_attn_sdpa_decode_program_config` returns an explicit 8x8 `SDPAProgramConfig`, with `q_chunk_size=0`, `k_chunk_size=0`, and `exp_approx_mode=False`.
- The nightly SDPA decode matrix exercises batch 32 with 8 KV heads on an explicit 8x8 grid, and separately exercises 64 Q heads with 8 KV heads on an explicit grid. Thus neither batch 32, 8 KV heads, nor 64 Q heads needs to be reduced to address this finalize-time failure.
- `models/demos/gemma4/tt/attention/decode.py` documents the same principle: always pass an `SDPAProgramConfig` on Blackhole instead of accepting the unbounded no-config fallback; it uses a smaller bounded grid where resource pressure requires it.

## Ranked hypotheses

1. **High confidence: missing explicit bounded SDPA decode program config.** The failing call at `functional_decoder.py:489-497` supplies no `program_config`. An 8x8 grid caps descriptor/runtime-argument coverage at 64 cores instead of the larger Blackhole default and matches the repository's canonical transformer decode setting. This directly predicts a small program-finalization overflow while prefill remains unaffected, because prefill uses a different operator and program.

2. **Low confidence / effectively refuted: 64 Q heads or the 8-KV-head GQA ratio is unsupported.** The nightly file includes explicit coverage for `nh=64,nkv=8` and for `b=32,nkv=8`. Its documented `q_chunk_size == head_size` regression is paged attention with `d=64`, unlike this unpaged `d=128` call. Changing head counts would also violate the model/IR contract and would not be a minimal fix.

## Exact minimal suggested change

Create one reusable config during decoder initialization and pass it to the failing call:

```python
self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=0,
    k_chunk_size=0,
    exp_approx_mode=False,
)
```

```python
attention = ttnn.transformer.scaled_dot_product_attention_decode(
    query,
    key_cache,
    value_cache,
    is_causal=True,
    cur_pos_tensor=update_indices,
    scale=self.scale,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    program_config=self.decode_sdpa_program_config,
)
```

Using zero chunk sizes is deliberate and follows `get_attn_sdpa_decode_program_config`: decode retains the operator's runtime-derived chunk sizing, so the existing `max_cache_len` behavior is not frozen to the synthetic test's 128-token cache. The 8x8 grid supplies 64 cores, which is at least the supported maximum batch of 32 and leaves the Q shape `[1, B, 64, 128]` and K/V shapes `[B, 8, S, 128]` unchanged.

## Verification status

Source inspection only, as requested. No hardware was run and no implementation file was edited. The decisive follow-up is the existing synthetic prefill/decode PCC test on Blackhole: the decode should finalize, retain 64 Q heads and 8 KV heads, and meet the existing PCC threshold.

## Follow-up after 8x8 experiment

The 8x8 result refines the diagnosis: bounding the default 11x10 grid reduced the serialized program from 70976 to 70848 bytes, but it remains 192 bytes above Blackhole's 70656-byte limit. The next minimal experiment is the established non-paged Llama-70B decode grid, 8x4.

Ranked hypotheses:

1. **High confidence: use 8x4 so the program has exactly 32 available and active cores.** For this call, `B=32` and `num_kv_heads=8`. The program factory requires `num_cores_available >= B`; 8x4 satisfies that exactly. Its allocation equations yield `num_cores_per_batch_uncapped=1`, `num_cores_per_head=1`, `num_heads_per_core=8`, and `num_active_cores=32`. Thus every batch item has one core which processes all eight KV heads (and their eight Q heads each), preserving the 64-Q/8-KV GQA geometry. This is also the grid used by `models/demos/llama3_70b_galaxy/tt/model_config.py::SDPA_DECODE_PROGCFG` for batch-32 non-paged decode.

2. **Low confidence: another config field is needed.** It is not required for this next experiment. `q_chunk_size` is not consumed as a chunk selector by the non-paged decode wrapper/program factory; `k_chunk_size=0` remains legal and makes the wrapper derive 128 for this 128-token cache. Changing it to the sibling model's 256 would be non-minimal and larger than this test's cache. `exp_approx_mode=False` is already explicit and should remain unchanged. The 8x8-to-8x4 change is the controlled variable.

Exact patch:

```diff
 self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
-    compute_with_storage_grid_size=(8, 8),
+    compute_with_storage_grid_size=(8, 4),
     q_chunk_size=0,
     k_chunk_size=0,
     exp_approx_mode=False,
 )
```

This is source-proven legal, but only the bounded Blackhole test can prove that the serialized descriptor falls below 70656 bytes. If 8x4 still overflows, do not change model geometry or chunk sizes blindly; capture the new exact size and inspect kernel-config serialization, because 8x4 is already the minimum rectangular grid that supplies one core per batch item.

## Final follow-up: emitted mask variant

The 8x4 hardware experiment was legal but did not fit: serialized size was 70864 bytes, still 208 bytes over. This refuted grid size as the decisive mechanism and showed kernel-variant resources dominated the failure.

Fresh source and IR inspection identified the graph-faithful boundary. The emitted decode path builds a BF16 additive mask on device, repeats it over the local Q-head axis, and calls SDPA with `is_causal=False`, `attn_mask=...`, and `cur_pos_tensor=None`. The attempted translation instead used causal SDPA with a cur-position tensor. In the program factory that variant allocates two extra circular buffers for the writer and compute consumers.

The final translation restores the emitted signature after dense TP collapse: `[1,1,64,128]`, zero through `current_pos` and `-inf` afterward. Cache-update indices remain device tensors used only by the two in-place append operations. The original watcher-enabled synthetic command then passed, including decode output PCC 0.995945 and K/V append PCC above 0.99986. The real layer-39 run also passed with prefill/decode PCC above 0.99999.

Independent stage review required one final isolation: the first passing masked run still carried the 8x4 experiment, so its necessity was unproven. Removing the `SDPAProgramConfig` entirely left the watcher-enabled synthetic PCC results unchanged and passing. The functional translation therefore uses the emitted default program configuration. This verifies the mask signature as the root cause and fix; the rejected 8x8/8x4 descriptor-fit hypotheses remain above as the evidence trail.
