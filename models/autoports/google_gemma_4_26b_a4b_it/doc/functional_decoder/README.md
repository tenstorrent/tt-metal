# Gemma4 functional decoder

Functional validation and independent stage review are complete (`clean-pass`).
The target is
`google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, using one Blackhole ASIC
(`MeshShape(1,1)`, automatic trace-region sizing). Four ASICs are available;
only one participates. This directory contains decoder-layer work only.

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
mesh_device, chunk_size=1024)` accepts the complete, unprefixed HF layer state
dictionary. Setup validates every key and shape, converts weights, and prepares
constant device tensors. The default physical chunk is 1024; supported chunk
settings are multiples of 128 up to 16384 that divide the HF context and, for
sliding attention, cover its 1024-token window. These are internal physical
constraints, not logical sequence-length alignment requirements.

| Layer kind | Representative layer | Query heads | KV heads | Head width | Distinct behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| sliding_attention | 0 | 16 | 8 | 256 | Causal 1024-token window, full RoPE |
| full_attention | 5 | 16 | 2 | 512 | Tied K/V projection, partial RoPE |

Both kinds use hidden width 2816, seven norms, a shared 2112-wide MLP,
128 routed experts of width 704 with top-8 selection, and a learned layer scale.
The model has no shared-KV consumer layers or per-layer embedding input.

`prefill_forward(hidden_states, *, rope_mats, page_table, kv_cache, user_id=0,
start_pos=0)` accepts one request `[1,1,S,2816]` and returns all S logical rows.
Fresh requests chunk and pad on device. A nonzero start continues an already
populated prefix through token-wise paged updates, preserving partially filled
pages. `user_id` selects the request's page-table row.

`decode_forward(hidden_states, *, rope_mats, current_pos, cache_pos, page_table,
kv_cache)` accepts `[1,1,B,2816]`. A fixed slot loop becomes part of the device
trace. Positions and page IDs are tensor inputs and may change on replay.
The caller owns input/output lifetime, cache allocation, and trace lifecycle.

- K/V are BF16 `[physical_pages, kv_heads, 32, head_dim]`; only 32-token pages
  are supported. Page tables are INT32 `[slots, pages_per_slot]` and contain
  physical page IDs with disjoint ownership for independent active requests.
- Cache capacity must include planned decode tokens and round up to 128 tokens
  for SDPA read padding. RoPE capacity rounds to a tile for one prefill chunk,
  and to the physical chunk size for multi-chunk sliding prefill. Allocating
  both for the full advertised 262144-token context satisfies these bounds.
- Prefill RoPE tables are `[1,1,capacity,head_dim]`; decode tables are 2D
  `[capacity,head_dim]`. Harnesses currently supply BF16 table values.
- Decode RoPE positions are UINT32 `[1,padded_batch]`; cache positions are
  INT32 `[B]`. Both must identify valid, nonnegative positions within capacity.
  Inactive lanes are not part of this contract.
- Prepare buffers before capture, warm the exact signature, capture, then
  update input contents and replay. Tests read outputs only outside the pass.

Attention and routing use local FP32 SFPU normalization/rotary arithmetic,
FP32 projections, and explicit BF16 cache updates. Decode QKV projection
uses bounded groups of 256 SFPU row-dot products; prefill uses HiFi4 matmul. Decode gathers
paged K/V on device and computes attention with TTNN operations. The decode
router uses an SFPU dot product and selects top-k logits before normalizing
selected probabilities. Shared and routed MLPs reuse the existing single-chip
Gemma4 implementation. Final activations and caches remain BF16. There are no
runtime CPU oracle paths or global operator monkeypatches in `tt/`.

The unchanged acceptance threshold is PCC >= 0.995. All accepted checks below
use actual target shapes; real-weight tests use the pinned checkpoint. The
synthetic CI tests preserve each checkpoint tensor's mean, standard deviation
and source dtype using `weight_stats.json`.

| Integrated check | Sliding minimum PCC | Full minimum PCC | Exact evidence |
| --- | ---: | ---: | --- |
| Batch32 prefill / traced decode | .99765820 / .99968658 | .99785365 / .99982852 | `batch32_sliding_exact_qkv_integrated.json`, `batch32_full_identity_gather_integrated.json` |
| Nine-request reuse prefill / traced decode | .99569189 / .99978697 | .99922349 / .99982091 | `reuse_sliding_exact_qkv_integrated.json`, `reuse_full_exact_qkv_integrated.json` |
| 4096 prefill / 128 traced decode steps | .99852634 / .99904322 | .99944062 / .99977388 | `headline_sliding_exact_qkv_integrated.json`, `headline_full_identity_gather_integrated.json` |
| Maximum262144 prefill / traced decode | .99889163 / .99979693 | .99627766 / .99988453 | `long_sliding_262144_final.json`, `long_full_262144_fixed.json` |
| Non-aligned262143 prefill / traced decode | .99889113 / .99985036 | .99622397 / .99986080 | `long_sliding_262143_final.json`, `long_full_262143_fixed.json` |
| Partial-page continuation | .99914565 | .99967083 | `continuation_sliding.json`, `continuation_full.json` |

The maximum-context TT pass computes every output row and fills the entire
cache. The HF prefill oracle compares 291 sampled query rows against all K/V
positions, including chunk boundaries and the last33 rows; it avoids a full
quadratic host attention matrix. These prefill accuracy values are a subset,
not an all-output comparison. Decode checks the final two positions and repeats
the final position with full attention context.

Reuse lengths are31,32,33,1023,1024,1025,2049,33 and2047, with changed physical
page maps and current-position tensors on the same trace. Batch32 uses disjoint
random page mappings. Continuation splits a second request at31/33/65 and checks
prefix bytes plus the other request are unchanged. `cache_indices.json` checks
integer cache addressing through physical page ID262144, including indices
above2^24. Headline and batch tests record bitwise repeated-output equality.

| Capability | Evidence | Remaining risk |
| --- | --- | --- |
| HF context262144, both layer kinds | `../context_contract.json`, four maximum/near-maximum JSONs above | Long prefill HF parity is sampled291 rows; all K/V participates |
| Paged cache and current positions | reuse, batch32, cache-index and continuation JSONs | Caller supplies valid mappings, active positions and buffer ownership |
| Prefill-to-decode and request reuse | headline, continuation and nine-request tests | Later model-stack integration is a separate stage |
| Fully traced decode, no runtime host fallback | `RUNTIME_AUDIT.md`, runtime guards in all accepted runs | Audit covers these concrete contracts and reachable lowerings |
| Real-statistics synthetic CI | `synthetic_pytest_final.log`, `tests/test_functional_decoder.py` | Synthetic inputs supplement real-weight evidence |
| Watcher10 | `watcher_summary.json`, `watcher_sliding_final.json`, `watcher_full_final.json`: both clean | Separate from profiling; no disabled checks |
| Warmed4096/128 performance | `PERFORMANCE.md`, `tracy/<kind>/whole_layer.json` | Traffic is estimated; latency is measured over all layer operations and gaps |
| Independent stage-review / local commits | `STAGE_REVIEW.md`: clean-pass; checkpoint SHAs in `work_log.md` | Reviewed functional stage only |

This is a functional decoder baseline. `PERFORMANCE.md` explains whole-layer
device denominators, useful active-expert FLOPs, estimated DRAM transfers and
theoretical one-ASIC peaks. The headline uses layer inputs, not end-to-end text
generation. Report tables and CSV provenance are under `tracy/`.

Some diagnostic reuse logs contain a generic post-capture allocation warning.
`RUNTIME_AUDIT.md` classifies it against allocator source and passing buffer
lifetime controls; it does not grant safety for arbitrary caller allocations.
The large page-table gather failure, precision failures and profiler failures
are resolved with focused controls in `AUTODEBUG_long_context.md`,
`AUTOFIX_decode.md`, `AUTOFIX_profiler.md` and `HF_PRECISION_CONTROLS.md`.
Historical failing JSONs remain diagnostic provenance and are not acceptance
results. The optional combined maximum-context watcher run was cancelled after
steady progress; it supplies no passing gate and caused no capability reduction.

`COMMANDS.md` gives reproduction commands; `work_log.md` records exact artifacts. Scope is exclusively the
functional decoder under this autoport; later stages have not begun.
