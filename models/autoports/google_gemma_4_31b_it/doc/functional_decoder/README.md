# Functional decoder (stage 01)

Status: complete. Independent [stage review](stage_review.md): `clean-pass`.

Target: text-only `google/gemma-4-31B-it`, revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Reference: transformers 5.12.1
`Gemma4TextDecoderLayer`. Implementation is new and uses generic TTNN operations.
No removed Gemma implementation or Gemma-specific framework branch was copied.

## Interface

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device)`
accepts layer-relative HF keys or canonical checkpoint keys prefixed with
`model.language_model.layers.<layer_idx>.`. Conversion and scalar extraction occur
here, outside runtime. All weights, activations and caches are BF16; matmuls and
normalization request HiFi4 with FP32 destination accumulation.

`allocate_cache(*, num_blocks, page_size=32)` returns caller-owned K and V tensors
`[num_blocks, num_kv_heads, page_size, head_dim]`.

`prefill_forward(x, *, cos, sin, kv_cache, page_table, chunk_start=0)` accepts
logical `[B,1,S,5376]` TTNN inputs and matching `[B,1,S,head_dim]` RoPE tensors.
It splits into at most 1024-token physical chunks and exposes tile padding through metadata-only views. The
output preserves every logical token. `chunk_start` is the absolute prefix length
and must be page-aligned for explicit continuation. Fresh requests use zero.
Arbitrary positive logical prompt lengths through 262144 are intended; tested
limits are recorded separately in `../context_contract.json`.

`prefill_chunk_forward(x, *, cos, sin, kv_cache, page_table, chunk_page_table,
chunk_start, valid_tokens)` is the lower-level physical-chunk boundary. The
chunk page table maps the chunk's first page to the corresponding absolute page;
the complete table is used for attention. Padding is causal future state and is
never read by valid query rows. Tail pages belong exclusively to their request.

`decode_forward(x, *, cos, sin, kv_cache, page_table, current_pos)` accepts
`x=[1,1,B,5376]`, RoPE `[1,B,1,head_dim]`, and device int32 absolute positions
`[B]`. It writes current K/V then attends through the current token, inclusively.
The caller guarantees valid positions and sufficient, non-overlapping page mappings.
Page tables are row-major int32 `[B,ceil(capacity/page_size)]` tensors. Decode
reads positions from the device tensor and updates the selected K/V slots, with
no host readback. The caller advances the position tensor.

The caller prepares RoPE, inputs, page tables and all persistent state before
capture. Tests warm the exact decode path, capture once, then refresh the same
tensor allocations. Runtime forwards have no torch imports, host conversions or
fallback. RoPE construction and final comparison belong to the harness. These
are layer interfaces, not a generator or serving adapter.

## Layer kinds and prefix state

| Layer kind | Geometry | Attention semantics | Resume-state cost at prefix n |
|---|---|---|---|
| sliding_attention | 32 Q, 16 KV, head 256 | causal window 1024, scale 1 | retained K/V: 16384 × n bytes, rounded to pages |
| full_attention | 32 Q, 4 KV, head 512 | full causal, scale 1; shared K/V projection before separate normalization | retained K/V: 8192 × n bytes, rounded to pages |

Prefix-hit granularity is 32 tokens: continuation requires a page-aligned prefix;
the new suffix may have any positive logical length. A hit needs the same weights, token prefix, absolute positions, and the resident
physical K/V pages with the correct page-table mapping. The caller retains those
pages and resumes at a chunk boundary; there is no recurrent state or hidden layer
state to snapshot. A table adds 4 bytes per mapped page per request. Allocation
reserves full capacity, so retaining a prefix currently retains its allocated cache
unless the caller manages a shared page pool. Sliding attention could retain only
the last window in a later layout; this implementation conservatively retains all
prefix pages. No bounded-cache optimization is claimed here.

A 262144-token batch-1 cache occupies 4 GiB for sliding attention and 2 GiB for full
attention (K plus V, BF16). Prefix state remains live until the caller evicts or
releases the request; shared prefix pages must remain until their last owner leaves.
Release traces before freeing any captured cache or persistent tensor. Per-pass
intermediates are released through TTNN tensor lifetimes; returned outputs remain
owned by the caller. Do not retain newly allocated eager outputs across an older
trace replay without a proven allocation/lifetime plan.

## Validation and evidence

`run_decoder.py` loads only the target layer's tensors from indexed checkpoint
shards. `weight_stats.json` records every used tensor's name, shape, dtype, mean and
standard deviation. Normal pytest tests draw deterministic synthetic tensors from
these statistics with the real config shapes. Real-weight runs are separate.

The short reference is the ordinary HF decoder forward. The long-context harness
uses HF projection/norm/MLP modules and the same HF RoPE to build every necessary
prefix K/V, then computes selected output rows with grouped PyTorch SDPA. The short
control compares this reduced-memory reference against ordinary HF forward.
Long-context evidence reports its sampled output scope explicitly; the TTNN pass
still executes the complete prompt. Sliding references omit keys outside the HF
window, which cannot affect any selected query.

Commands and exact logs/results are in `work_log.md`. Required PCC is 0.995 for
both kinds and phases. Final real-weight validation, synthetic pytest and all eight profiles pass. See
[whole-layer performance](../functional_decoder.md), `profile_provenance.json` and
`performance_*.json`. Independent [stage review](stage_review.md) returned `clean-pass`.


| Capability | Evidence | Scope / remaining risk |
|---|---|---|
| Both actual kinds, real config/weights | `validation_summary.json`, `weight_stats.json` | Representative layers 0 and 5; no full-model quality claim |
| 262144-token context, prefill and traced decode | `sliding_context_post_view.json`, `full_context_post_view.json`, `../context_contract.json` | B1; long HF compares last32 prefill rows, exact-max last row and decode, TT executes all rows |
| Page mappings and changing positions | `*_batch{8,16,32}_final.json` and logs | Permuted nonzero physical pages, untouched sentinel pages, reversed batch rows, distinct positions |
| Arbitrary logical tails and increasing lengths | `*_boundaries_final.json` | 1,31,32,33,1023,1024,1025,2049,33 with changed data and page ownership |
| Prefix resume | `*_boundaries_final.json` | Split at1024; resumed vs continuous PCC1.0 for both kinds |
| Trace lifetime across eager prefills | `*_watcher_final.log` | One trace ID, six interleaved cases including unseen34 after warm33, no request-time capture/release |
| Determinism and runtime audit | `*_final.json`, `runtime_audit.json` | Bitwise repeated prefill/replay; conversion guards enabled |
| Watcher | `watcher/{sliding_attention,full_attention}_final/generated/watcher/` | Poll10 seconds; no error/fatal/overflow/invalid/corruption/sanitize messages |
| Synthetic CI | `pytest.log` | Four real-shape cases, B1/S33 and B8/S32, both kinds;4 passed |

All correctness runs use `TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`. No exemptions or corruptible-buffer
acknowledgments are used. The harness allocates persistent other-request state
before capture, consumes and drops eager outputs before replay, and refreshes only
existing device inputs. Warmup covers physical buckets and absolute chunk starts;
callers must prepare the signatures they will use before recording live traces.

## Investigated anomalies

- Full-context default decode SDPA failed PCC. Exact-HF-cache probes isolated the
  long BF16 recurrence. Sequence splitting with K128/core-cap16 and precise exp
  repairs full attention. `../../AUTODEBUG.md`, `../../AUTOFIX.md`,
  `long_attention_probe.json`, `long_reference_control.json`, and final context
  results record the diagnosis, controls and passing repair.
- An unseen logical length34 after warm33 allocated FillPad program state while a
  trace was live. Metadata-only logical sequence views remove that setup. Final
  watcher/boundary runs exercise the unseen length with all tracker categories on.
- FP32 PCC rounded high-correlation batch results to1. The final harness computes
  correlation in FP64; final queue results supersede earlier numerical summaries.
- The platform-identification warning names an unrecognized motherboard and uses
  bus_id for tray_id. Device open/close, correctness and watcher checks pass; this
  single-device stage does not use fabric topology inference.
