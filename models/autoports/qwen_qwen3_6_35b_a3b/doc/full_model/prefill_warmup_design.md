# Universal prefill preparation design

CPU source investigation, corrected inheritance audit 2026-11-14. This is a proposed implementation, not hardware validation or stage acceptance. No device code was imported or executed.

## Finding

`QwenGenerator.prepare()` warms supplied exact `(length, start)` cases and then captures decode/sampling. `_plan()` later retains newly allocated device RoPE/position tensors for every unseen pair. The model also runs whole-logical-length embedding, terminal, slices and concatenations. Warming one long prompt does not prepare shorter prompts, different resume offsets, or their scalar slice signatures. The tracing skill explicitly requires global preparation of eager paths that can execute between live trace replays.

There are three independent causes to remove:

1. Request-keyed persistent position plans (`generator.py:_plan`, `fused_decoder.py:prepare_positions`).
2. Absolute-offset program keys: full attention slices `state['page_table'][:, start_pos // 32:]`, changing both slice constants and the table's physical width, then passes scalar `chunk_start_idx=start_pos` to SDPA. Hidden slices and the final `hidden[:, -1:, :]` also encode logical offsets.
3. Unbounded physical sequence shapes in full-sequence embedding/concat/terminal, despite bounded inner attention/MoE work. `prepare_moe()` also lazily allocates expert dispatch outputs.

## Preserve the actual selected schedule

The actual inheritance is `MultichipDecoder → OptimizedDecoder → _ProjectionDecoder → _ExpertDecoder → FusedDecoder` (plus `LinearTP4Mixin`). `FusedDecoder.prepare_prefill` explicitly returns linear chunks of 64 relative to the public call, with one exact tail of1..63. The default `chunk_size=4096` does **not** select 4096-token linear projection chunks. The earlier version of this note incorrectly followed `functional_decoder.py`; that conclusion is withdrawn.

Full-attention aligned chunks have lengths `{32,64,...,256}`, start on absolute multiples of 256, and otherwise use singleton decode-style attention. `MeshPolicy.moe_chunk` is 2048. MoE consumes request-relative chunks of 2048 followed by an exact logical tail. `_moe` already pads nonsingleton tails to the next multiple of 32, runs the grouped path, then slices back to the logical row count.
Preserve TP4/EP4, replicated BF16 inter-layer residuals, all selected CCL/dtype/fidelity/cache policies and the prefill-cast/decode-repack KV dtype split. Preserve64-token linear projection grouping as well as chunk64 math unless a separately validated optimization changes it.

A loop calling the existing public `prefill_forward()` once per 64/256/2048 tokens is **not** a sufficient refactor:

- Each linear call rounds recurrent state FP32→BF16→FP32 again. The accepted code does that once per layer per public prefill call.
- A singleton within a multi-token linear prefill uses prefill arithmetic, `projection_k=1024`, and `cache_boundary=False`; a public one-token call has different behavior. Tail tokens cannot be replaced indiscriminately with ordinary decode.
- Different public starts cause linear and full-attention chunk boundaries to disagree. An arbitrary outer cut can split an accepted attention chunk and shrink the selected 2048-token MoE batches.
- Padding recurrent/convolution updates with dummy tokens corrupts the resume state. No padded token may advance cache or position state.

## Concrete bounded implementation

Introduce a generator-owned physical prefill executor and remove retained `(length,start)` device plans. Keep logical plans as CPU integer tuples only. Separate three decoder operations: `begin_public_prefill` (one recurrent-state rounding), `attention_chunk` (explicit original-public-length mode, exact logical chunk length), and `moe_chunk` (up to 2048, original grouping). Internal chunk calls do not become new public cache boundaries.

The smallest robust executor is a **stream of request-relative 2048-row blocks between layers**, with a small full-attention lookahead and carry. It preserves the selected MoE throughput instead of turning every attention chunk into a smaller MoE call:

1. Embed prepared input blocks of at most 2048 valid rows. Keep each request's public start/length and each layer's next attention/MoE ordinal as CPU integers. Those integers drive the bounded scheduler, not absolute-offset TT slice keys.
2. A linear layer processes 32 chunks of 64 per complete block. Its last public block has one exact attention tail 1..63. Begin-public rounding happens once before the first chunk; carry FP32 recurrent and conv state normally across subsequent blocks. Set the public-singleton mode from the original request length, never the current block length.
3. A full-attention layer follows its original absolute 256 alignment, even if a chunk straddles a request-relative 2048 boundary. Buffer up to 255 lookahead rows from the next upstream block, evaluate that one complete attention chunk, and retain the excess attention output for the next MoE block. Prefix singleton attention also preserves the original schedule. Do not split the crossing 256 chunk merely to match2048.
4. Accumulate exactly 2048 attended residual rows and apply the existing MoE once; emit that 2048-row block to the next layer. At public end, apply the existing singleton or next-multiple-of 32 MoE tail behavior and emit only valid rows. A pull-based iterator per layer can request the next upstream block when lookahead is required, so each layer retains only bounded input/output/carry, not the full context. A 4096-row input window suffices when upstream emits 2048-row blocks; compact retained carry between refills.
5. Terminal work consumes emitted blocks. For token-out, run terminal only on the last valid hidden row. For all-logits compatibility, produce bounded logit chunks and consume/read them before reuse. Never allocate maximum-context full-vocabulary logits on device.

Use runtime row indices to gather from these fixed staging windows; preallocate index vectors and gather outputs before capture. `ttnn.gather(..., out=...)` provides that contract. Keep scalar slices only for a finite set of **local** window offsets/shapes, if their warmup inventory is explicitly enumerated; no slice uses an absolute context position. Combining attention outputs may initially use the existing bounded concat with at most a bounded window's chunks. However its input-count/shape tuples are program variants too: either enumerate them or use a stable append/copy helper. `indexed_fill` is not automatically that helper: it allocates/copies a whole output and may permute tiled row dimensions. A narrow in-place device row-copy helper is an alternative only if needed; no host activation roundtrip.

The streaming scheduler is more work than a direct public 2048-token loop because the two attention schedules do not share boundaries for arbitrary starts. If minimizing scheduler changes is more important initially, a fixed-capacity layer-major residual workspace plus runtime chunk gather/scatter preserves the same arithmetic. One `[262144,2048]` BF16 workspace is 1 GiB/chip; account for all simultaneous workspaces. Prefer streaming when that avoidable scratch affects capacity; do not lower advertised context to accommodate the staging design.

### MoE padding: retain 32 alignment, do not expand every tail to 2048

Current nonsingleton MoE already supplies zero rows up to `ceil(n/32)*32`; preserve exactly that rule. No new live-row mask is needed to reproduce the existing policy. Public singleton must retain its sparse path. Padding every tail to 2048 is an additional change: it increases the expert-capacity bucket and `input_m_tiles`, changes per-expert counts/region offsets, and sends many dummy routes through grouped kernels. Masking those rows out changes the current routing/count contract again; it is not a zero-work fix and must not be assumed equivalent or faster.

`dispatch_capacity` already buckets by powers of two, with minimum32. For lengths 2..2048 only seven expert-workspace buckets are needed: 32,64,128,256,512,1024,2048. The grouped compute has only 64 selected physical token lengths 32..2048 step 32, plus the distinct sparse singleton path. Exact logical tail cropping/padding still has finite helper signatures; eliminate or enumerate those separately. This is substantially smaller than the previous note's unnecessary 2048 distinct grouped-MoE variants.

### Stable positions and page tables

- Reuse the model's already allocated full RoPE tables. For each physical full-attention length, allocate a reusable index vector and position scalar before capture; refresh contents at an eager prefill chunk boundary, then obtain RoPE with fixed-shape device embedding/gather. Match the accepted cosine/sine shape and BF16 conversion. Do not allocate a new persistent tensor for each absolute position. A single buffer per concurrently executing shape is sufficient; prefill is serial today.
- Scalar and tensor SDPA overloads both call `ttnn::prim::sdpa` with causal mode and the same supplied scale, program config and compute config. The factory uses the same compute kernel, with runtime offset delivery and two small additional int32 circular buffers for the flexible path. This is a suitable preservation of the selected paged/chunk256 strategy, not a switch to another attention algorithm. Numerical equality and latency still require the reduced paired test; the flexible program sizes against the fixed page-table maximum and requires its byte width to be 32-aligned.
- Replace scalar SDPA offset with `chunk_start_idx_tensor=<persistent int32 [1]>`. This checkout already exposes that Python keyword. The flexible API explicitly compiles once for a fixed full page-table size. Both q_chunk32 and k_chunk256 alignment constraints remain: use it only for the existing 256-aligned interiors. Singleton boundaries retain dynamic-position paged decode attention.
- Allocate a stable **fill-page view `[1,8]`** for the maximum 256-token full-attention chunk (page size 32). Before the chunk, copy the relevant eight page IDs into it, padding unused entries with a valid owned page. Fill writes only actual source blocks. `paged_fill_cache` validates `input_padded_length <= page_table_width * block_size`; eight entries cover every aligned physical variant. Keep full-context `[1,8192]` page tables for SDPA at context 262144. Smaller configured contexts use their own fixed prepared width.
- `paged_fill_cache` presently has no runtime absolute page-offset argument. Its `batch_idx_tensor` selects a row, not an offset. `valid_seq_len_tensor` is not a general arbitrary-token mask. Therefore replace the shrinking device slice with this fixed view, refreshed from host-owned scheduler mappings at prefill chunk boundaries, or add a narrowly scoped runtime page-offset facility. Do not pretend the existing API already supplies one.
- Page mapping changes refresh existing full/fill table allocations. Request-time host copies for eager prefill are compatible with the trace contract; decode must continue unchanged-page-table zero-copy replay and device position/token advancement. A traced prefill variant would require its stable input refresh before replay.

## Finite program inventory and preparation order

Warm **actual signatures**, deduplicated across layers only when shapes/configurations match. A representative layer per exact implementation/config is sufficient for compilation; every layer's persistent state/CCL bindings must nevertheless exist. An initial deliberately conservative inventory is:

| Work | Physical variants independent of context/start |
| --- | --- |
| Linear attention | exact lengths 1..64; public-singleton versus multi-token-tail modes; selected projection/cache-boundary flags |
| Full attention | singleton dynamic-position path; lengths32..256 step 32 with tensor start and fixed full/fill page-table widths |
| MoE | sparse singleton and grouped 32..2048 step 32; seven persistent capacity buckets; finite valid-tail pad/crop helpers |
| Embedding / residual movement | the bounded lengths used above, fixed workspace dimensions, runtime indices, finite layout conversions |
| Terminal | fixed output chunk shapes plus single final row; distinguish full-logits and token-out consumers |
| Other | all advertised fixed-slot/batch modes, sampling modes, token/position/RNG updates, resets and page-table refresh helpers |

Warm the 64 grouped MoE shapes directly, not 2048 whole-model prefills. Separately warm or remove scalar logical-tail slice/pad signatures. Deduplicate only by observed physical operation signatures. The executor's input/carry/concat helpers are part of the inventory: linear tails 1..64, full start residues 0..255 and interior/tail sizes, fixed-window gather/copy modes, and their actual tensor layouts. Attention/MoE arithmetic variants are bounded independently of context; an absolute prefix position must never introduce a new program.
Preparation sequence: allocate all persistent buffers → warm each signature over the real configured KV-pool shape → reset recurrent/conv/KV/request/RNG state → verify no lazy helper allocation remains → capture model and selected common-sampler traces once. Buffer address changes are reconfiguration, not request handling. No recapture, trace retirement, tracker exemptions over an entire forward, or program-cache-skipping acceptance run solves this defect.

Acceptance probe: with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, alternate previously unseen logical lengths, nonaligned starts, partial final chunks, resumed prefills, changed and unchanged mappings while retaining trace IDs. End every case with decode replay and assert program-cache count does not increase after the preparation frontier. Compare relevant logits and exact cache/position advance, including last-context boundary. Keep the user-required max-context prefill/decode run to the single final acceptance run; reduced probes establish the runtime-offset and finite-shape contracts first.

## Sources read

- Installed `tt-enable-tracing/SKILL.md`: global two-phase warmup, scalar slice keys, lifetime, split sampling and tracker requirements.
- `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tt/fused_decoder.py` (actual inherited planner); `tt/functional_decoder.py` only as a non-inherited comparison in this autoport.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`: flexible tensor offset and alignment contract.
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp` and paged-cache bindings: fill-table width/dtype constraints and dynamic argument limits.
- `ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp`, `indexed_fill/indexed_fill.cpp`, and its device operation: runtime indices, gather output buffer, indexed-fill allocation/layout behavior.
