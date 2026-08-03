# Phi-3.5 layout AutoFix audit

Source-only audit; no hardware was run and no implementation was changed.
Starting evidence is `AUTODEBUG.md` plus the four bounded Tracy CSVs. Each CSV
contains four forward calls, so counts below are reported both as recorded and
per call.

## Observed families and source mapping

| Mode | Recorded layout/TM counts (four calls) | Source attribution | Source verdict |
|---|---|---|---|
| decode B1 | tilize-pad 24 (6/call), untilize 8 (2/call), untilize-unpad 16 (4/call), permute 24 (6/call) | Two tilize-pad calls per forward are proven to be the two decode embeddings: each `EmbeddingsDeviceOperation` is immediately followed by `TilizeWithValPadding` in `tracy/decode_b1.csv`, while `_decode_rope` asks both embeddings for TILE output (`functional_decoder.py:401-407`). The other four tilize-pad, two untilize, four untilize-unpad, and six permutes form two identical slice/neg/concat groups matching the two `_apply_rope` calls (`:413-416`, topology at `:228-236`). | **Retry/removable:** all listed layout ops belong to decode RoPE, not norm or head concat. Remove only by replacing the complete explicit RoPE topology. |
| decode B32 | tilize-pad 16 (4/call), untilize 8 (2/call), untilize-unpad 16 (4/call), permute 24 (6/call) | Same two embeddings and two `_apply_rope` groups. Unlike B1, TILE output height is already 32, so embedding's final `to_layout` need not pad; the remaining four tilize-pad per call are inside the two explicit rotate-half groups. | **Retry/removable:** same RoPE replacement; B1's extra 2/call is a batch-height padding effect, not a separate model boundary. |
| prefill B1/B32 | tilize-pad 16 (4/call), tilize 16 (4/call), untilize 8 (2/call), untilize-unpad 16 (4/call), permute 24 (6/call) | Exactly two `_apply_rope` calls are made by `_prefill_rope` (`functional_decoder.py:238-245`). In each ordered CSV group, `slice, untilize, slice, tilize-pad, unary, 2x untilize-unpad, 3x permute, concat, tilize-pad, tilize, binary, tilize, 3x binary` appears twice before either `PagedFillCache`; thus all listed families, including the four plain tilizes, are source-attributed to RoPE. Prefill profiles use sequence 128 and do not execute `_offset_causal_mask` (`tests/fused_decoder_perf.py:71-98`; branch at `functional_decoder.py:316-322`). Head split, cache fills, and head concat appear afterward as their own named rows. | **Retry/removable:** the complete explicit RoPE family. No independent cache-fill or mask layout retry is supported by these CSVs. |

The profiler summaries supporting the counts are
`tracy/decode_b1_summary.csv:8-14`,
`tracy/decode_b32_summary.csv:8-14`,
`tracy/prefill_b1_summary.csv:7-16`, and
`tracy/prefill_b32_summary.csv:7-16`.

## Hypothesis verdicts

### 1. ROW_MAJOR norm weights cause traced conversions — refuted / removable from retry list

The model uploads norm weights in ROW_MAJOR with last padded dimension 32
(`functional_decoder.py:166-173`). That is an explicitly supported RMSNorm
weight contract (`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_nanobind.cpp:98-104`);
only the activation must be TILE
(`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:32-37`). In every
bounded CSV, `LayerNormDeviceOperation` is not adjacent to a layout conversion.
Uploading these weights as TILE is therefore not a source-supported removal
candidate.

### 2. Decode embedding tables should be uploaded TILE — blocked by op contract

The tables are correctly ROW_MAJOR (`functional_decoder.py:187-190`).
Embedding validation requires ROW_MAJOR weights
(`ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:28-43`).
Because `_decode_rope` explicitly requests TILE output, the wrapper performs a
post-embedding `to_layout`
(`ttnn/cpp/ttnn/operations/embedding/embedding.cpp:65-77`). The two embedding
tilizes cannot be removed by changing table layout. They disappear only if a
replacement RoPE consumes ROW_MAJOR embedding output, or if embedding/roPE is
fused.

### 3. Use `ttnn.experimental.rotary_embedding` directly — blocked for width 96

The generic primitive accepts a padded width of 32 or a multiple of 64; rotate
half must meet a tile boundary
(`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp:32-42`).
Phi head width is 96 and its semantic midpoint is 48
(`functional_decoder.py:221-236`), so the primitive is source-proven
inapplicable. This is not a probe-worthy uncertainty.

**Retry target:** a Phi-compatible fused op that implements HF half-rotation at
48 for width 96 (or an existing op whose source contract explicitly supports
that midpoint). It must accept decode HEIGHT_SHARDED Q/K or return the identical
sharding required by paged cache update/SDPA. Reworking only `slice` or `concat`
will not remove the whole observed family.

### 4. Head split/concat are avoidable conversion sources — refuted for the recorded families

Prefill split requires TILE input and head width divisible by 32
(`ttnn/cpp/ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp:72-85,151-169`);
Phi width 96 satisfies this. Prefill concatenate similarly requires an
unpadded head width divisible by 32
(`ttnn/cpp/ttnn/operations/transformer/concatenate_heads/concatenate_heads.cpp:15-33`).

Decode create-QKV directly produces HEIGHT_SHARDED TILE Q/K/V
(`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp:30-57,119-156`).
Decode concat requires exactly HEIGHT_SHARDED TILE input with one user per core
and deliberately pads output batch to 32
(`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp:22-69,81-110`).
Thus `to_memory_config` before concat and the B1 logical slice
(`functional_decoder.py:490-493`) are current-contract requirements. They are
not the recorded tilize/untilize/permute groups; the CSV names the head ops and
sharded/interleaved transfers separately.

### 5. Explicit causal-mask `to_layout` causes these profiles — refuted

The profiles use sequence 128 (`tests/fused_decoder_perf.py:71-98`), selecting
the non-chunked causal SDPA branch (`functional_decoder.py:316-322`). The mask
builder and its explicit `to_layout` (`:247-266`) execute only for later chunks
above 32768, so they contribute zero rows here.

## Final status and focused probes

**Status: source-resolved, implementation retry remains.** The categorical
“no tilize/untilize” claim is false, but attribution is no longer wholly
unknown: the bounded conversion families map to explicit Phi RoPE, plus
the decode embedding output conversions. Norm, head split/concat, cache fill,
and the long-prefill mask
are removed from the retry list.

Only these device probes remain justified:

1. A focused width-96 fused-RoPE candidate probe at prefill `[B,32,128,96]` and
   decode `[1,B,32,96]`, B=1/32: compare PCC to `_apply_rope`, assert logical
   and padded shapes plus memory configs, then verify disappearance of exactly
   the RoPE-attributed counts above.

No stack-trace recapture is needed before that probe: ordered op
signatures and source contracts already decide the other hypotheses.
