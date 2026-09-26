# Static native decode and BFP8 cache integration

`tt/optimized_decoder.py` now implements optional native SDPA and BFP8 cache
support without source rewriting, global patches, or runtime function injection.
Factory defaults remain `native_sdpa=False` and `kv_cache_dtype=ttnn.bfloat16`.
The device owner supplied passing actual-text native-SDPA and cache-precision
probe results and authorized this integration; the new static implementation
itself still requires its normal device validation.

## Public cache contract

`OptimizedDecoder.kv_cache_dtype` is the caller's recommended allocation dtype;
the decoder does not allocate caches. Both public methods accept a matching
BF16 or BFP8 K/V pair with 32-token pages, independently of which supported
dtype was recommended at construction. This preserves callers that allocate
BF16 caches. Mixed K/V dtypes, BF4 caches, and other page sizes are rejected.
The precision policy records the declared dtype and native-SDPA configuration;
the harness should also report actual caller-owned cache dtypes.

The inherited methods contain inline BF16-only checks, so simply calling their
`super()` implementations cannot support BFP8. Static optimized overrides copy
the public orchestration and replace only those checks with a shared validator.
This retains all of the existing request handling:

- Prefill preserves every logical output, bounded chunk padding, short sliding
  tails, request-page slicing, and persistent-tail reset.
- Prefix continuation still calls per-token decode using preallocated device
  positions, so partially filled pages are not replaced by a bulk fill.
- Batch decode still records a fixed per-slot loop. Each slot selects its own
  page-table row, UINT32 RoPE position and INT32 cache position; batch 32 is not
  reinterpreted as a single native attention request.

An AST comparison confirms both public method bodies are identical to
`FunctionalDecoder` after removing the old/new cache validation statements.
The source copy is explicit and reviewable; no `inspect` or `exec` is present.

## Fill and update precision are different contracts

`OptimizedAttention.from_existing` creates a named subclass and copies the
already-constructed fused attention's fields. This preserves fused flags,
bound normalization, QKV/output projections, cache layouts, precise attention
fallback, and prefill state.

Prefill attention and its sliding tail retain BF16 Q/K/V. Only the value passed
to `paged_fill_cache` is cast to the destination cache dtype, when different.
The raw tile-copy fill path requires input and cache dtypes to match; this is
enforced in
`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:30`.
An AST comparison confirms the copied fused prefill body differs only in this
fill loop. Full-attention later chunks consume the supplied paged cache through
the existing `chunked_prefill_sdpa` helper.

Decode cache updates remain **BF16 rows**, even for BFP8 storage. The subclass's
`cache_cast` delegates to the fused conversion/layout implementation with BF16.
This handles both fused sliding updates and separate full-attention updates.
The native writers accept only BF16 or FP32 input and repack into cache format:
the single-update validator states this at
`device/update_cache/paged_update_cache_device_operation.cpp:293`, and the fused
validator does so at
`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:327`.
Casting decode inputs to BFP8 before these operations would violate their API.

## Native attention configuration

`NativePagedAttention` is installed as the attention component only when
`native_sdpa=True`; otherwise the existing precise attention object remains.
It matches the tested probe's graph:

1. Round Q to BF16 and place it in interleaved DRAM.
2. Call `paged_scaled_dot_product_attention_decode` with caller-owned K/V,
   device INT32 cache positions, page table, scale 1.0, and the configured
   sliding window only for sliding attention.
3. Request interleaved DRAM output, then cast the native BF16 result to FP32 for
   the existing projection path.

The explicit SDPA program uses grid 8x8, `q_chunk_size=0`, `k_chunk_size=0`, and
`exp_approx_mode=False`. Zero decode chunk sizes retain the operator's supported
dynamic causal policy. The compute configuration is HiFi4, approximate math
disabled, FP32 destination accumulation enabled, L1 packer accumulation disabled,
and **`dst_full_sync_en=True`**. Full synchronization is part of this integrated
candidate rather than an independently disabled default.

Native decode accepts BF16/BFP8 cache tensors and requires INT32 device cache
positions in
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp`.
The UINT32 tensor remains reserved for the separate RoPE embedding lookup.

## Validation and handoff

Compilation, Black, `git diff --check`, and the AST preservation checks pass.
No hardware command was run by this investigator. Runtime edits are complete;
the device owner can use the existing JSON factory override mechanism:

```json
{"native_sdpa": true, "kv_cache_dtype": "bfloat8_b"}
```

Caller allocation must use `decoder.kv_cache_dtype` or an explicitly supported
dtype. The old precision probe's source-string validator patch must be removed
or bypassed because validation is now a real static method.

The device-owner checks should cover actual-text 4096+128 and 512-step runs,
BF16 compatibility with native disabled/enabled, BFP8 fill and decode, prefix
continuation inside a page, short sliding tails, permuted page tables, and
batch-32 request isolation. Preserve the existing shape/PCC contracts, exact
trace replay checks, device-only runtime audit, and program-cache miss guard.
Earlier probe evidence is not a substitute for validating this integrated code.
