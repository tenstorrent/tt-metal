# Grouped decode: bounded fourth-hypothesis candidates

## Current disposition after the device discriminator

**Reject the 4×8 candidate.** The owner ran `tests/run_grouped_decode_grid_probe.py`; its first candidate invocation failed native validation before expert computation. `readiness_vllm/optimized_grouped_grid_probe/result.json` records `status=FAIL`, case `shared_early_late/active_0`, and successful mesh closure; `doc/optimized_vllm/grouped_grid_probe.log` preserves the execution log. The native validator at `moe_fused_swiglu_device_operation.cpp:218` requires `grid_x >= grid_y >= 2`, and reports `got 4x8`. The initial source audit missed this mandatory guard. Its geometry arithmetic was therefore not a valid runnable-configuration proposal. Do not remove the guard, claim an exactness pass, or infer a performance result. The original proposal below is retained as rejected investigation history.

The current bounded follow-up is **8×8 with startup-only gate/up ND weight shards `[32,64]`; down weights remain interleaved**. The optional, unapplied `grouped_nd_weight_candidate.patch` adds `ep_weight_nd_sharding=False` to model/decoder construction, passes the explicit opt-in through every layer, and constructs only gate/up in that layout during weight upload. Default behavior remains interleaved. It does not change the native grid, dtype, routing, arithmetic, compute configuration, barriers or runtime state. A source-only `git apply --check` passed after the owner's removal of the rejected grid switch. Runtime validation remains outstanding.

### Actual bank mapping and the limited hypothesis

The historical device census in `doc/optimized_full_model/context_memory_audit.json` records eight DRAM banks. The proposed constructor checks that the runtime DRAM grid contains eight banks; it does not silently apply this analysis to a different device.

The mapping comes from actual source, not an assumed linear-buffer rule:

- `moe_fused_swiglu_program_factory.cpp:660–674` uses worker index `y*HGROUPS+x`, K start `8*y`, hidden start `2*x`, and embedding-output start `8*y+x` at 8×8/2048×512.
- `kernels/moe_fused_swiglu_dataflow.hpp::read_weight_chunk` iterates local K rows `i=0…7`, then reads the one-tile N span `n=2*x+c`, where gate/up chunk `c` is zero or one. It supplies logical page `(8*y+i)*16+n` to `TensorAccessor`.
- For ND tensors, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:279–295` reconstructs page coordinates, derives a flattened shard index using shard-grid strides, and maps it through `round_robin_mapping` (`:416–417`). For a one-K-tile by two-N-tile shard, the shard index is `k*8+floor(n/2)`, not the interleaved page index. Default `NdShardSpec` distribution is explicitly `ROUND_ROBIN_1D` in `tt_metal/api/tt-metalium/tensor/tensor_types.hpp:93–97`.

For a corresponding K-step/chunk across the 64 workers:

| Placement | Bank index | Request multiplicity over banks |
| --- | --- | --- |
| Current interleaved GU | `((8*y+i)*16 + 2*x+c) % 8 = (2*x+c)%8` | Four banks × 16 requests; even banks at chunk0, odd banks at chunk1 |
| Proposed GU `[32,64]` | `((8*y+i)*8 + floor((2*x+c)/2)) % 8 = x` | Eight banks × 8 requests at either chunk |

This is a source-derived **corresponding-loop-position** model, not measured simultaneous issue. The reader publishes one chunk and waits before issuing the next (`reader.cpp:890–905`); cores and reader/writer need not remain in perfect lockstep. The hypothesis is reduced per-chunk bank concentration. It is **not** K-row bank rotation: each worker remains associated with one bank. It is also not request coalescing: the existing `WeightRuns` span remains one tile, so request count and useful bytes stay unchanged. Each GU tensor has 64×16 tiles, exactly divisible by the 1×2 shard shape, giving 512 shards and 64 shards per bank. There is no added tile payload or incomplete-shard padding. Tensor/accessor metadata and program-cache entries differ; record their actual allocation/capture effects before adoption.

### Why down remains interleaved

The factory's actual output index is `n=8*y+x`, with `ec=1`. `read_wd_rows` reads that same one-tile N coordinate for every hidden K row; `issue_wd_batch` iterates hidden blocks, without partitioning N into a separate per-column loop. Thus current interleaved down has `bank=n%8=x`, giving eight banks × eight workers at a corresponding K-row. A hypothetical `[32,256]` ND down layout instead maps `bank=floor(n/8)=y`, also eight banks × eight workers. It changes physical affinity but does not improve the aggregate multiplicity under this source model. There is no measured NoC-locality evidence to justify bundling that change into the first candidate.

### Earlier ND proposals narrowed by aggregate analysis

The initial 6-tile GU / 5-tile down proposal rotated individual streams across eight banks but worsened source-derived instantaneous balance. GU6 maps each chunk's workers to three banks with 24/24/16 requests; all K partitions start at multiples of eight and rotate in lockstep. Down5 maps a corresponding K-row to counts 10/10/10/10/9/5/5/5. It also adds an estimated 757,596,160 BFP8 payload bytes/rank across all40 layers. A subsequent GU3-only idea uses five or six banks per chunk with peak16 requests and adds 713,031,680 bytes/rank. These are not preferred to GU2's eight banks/peak8/no tile padding. No ND variant has been measured here.

### Legality and exactness gate for the optional ND patch

Native validation accepts tiled BFP8 weights in either interleaved DRAM or DRAM ND-sharded memory. All64 experts must share per-role memory configurations; gate/up shapes and ND widths must agree. The patch constructs identical gate/up configs for every expert and retains all logical matrix dimensions. `[32,64]` satisfies the tensor layout's tile-aligned-shard guard (`tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:43–65`). The existing factory derives the true ND N width from each buffer, verifies gate/up agreement, and the generic tensor accessor supplies bank addresses. No arithmetic geometry or row-mode selection input changes.

First compare decoded BFP8 weight payloads between the existing interleaved and proposed ND copies for all64 gate/up experts on allfour chips, preserving exact logical shape/dtype/values. Reject any conversion difference. Use only startup copies for the diagnostic; production constructs the selected representation once before preparation and does not reshard during forward execution. The same layer0 synthetic-input cases from the failed grid probe can then compare **8×8 against 8×8 with alternate EP weight lists**, including raw live expert rows and combined rows, with reference repeated after candidate execution. Change the probe deliberately; do not rerun its rejected4×8 branch or relabel its old failure.

Follow a passing microgate with independent full-state/bucket exactness and all40 full-vocabulary checks, nonaligned/chunked prefill (the same EP weights serve prefill), retained trace and stale-input/async checks, unchanged full-context memory feasibility, and matched real-serving benchmarks. All diagnostics remain separate from primary benchmark timing. Same legal geometry and stored values make exactness plausible; they do not substitute for those tests or establish any speedup. This investigation does not exhaust all optimization ideas.

## Original source-only proposal, retained as rejected history

Source-only inspection, 2026-11-17. HEAD `5e2dca2e09485be6c4f18c688a1b7b532e45a99b`, with the owner's ongoing Stage 10 working-tree changes. No source edits, TT imports, device access, benchmarks, server requests, profiling or subprocess termination were performed. This document is the only output. Integer route count/prefix fusion is a separate experiment and is deliberately excluded.

The parent supplied current combined route-suppression/solo-attention timings of native 38.306 ms and adapter-async 38.242 ms. These are context, not independently remeasured results. The inherited useful-work floor remains 21.63129875 ms. Source inspection cannot assign the remaining gap to the grouped kernel or promise that any candidate below closes it.

## Rejected candidate: decode-only 4×8 grouped geometry

Try `core_grid=ttnn.CoreCoord(4, 8)` only for deterministic grouped decode, preserving the 8×8 prefill path, weights, datatype policy, BF16 destination, stochastic-rounding-off setting, route plan, counts/offsets, expert output, combine order and mandatory reader ACK drain. This is an existing operation parameter, so it needs no plugin or native implementation change. It is a hypothesis requiring exactness before timing, not an approved replacement.

`tt/multichip_decoder.py::_grouped_moe` currently fixes 8×8 for both prefill and decode. Native `device/moe_fused_swiglu_geometry.cpp::Blocking` splits the 64 input K tiles across grid **height** and the 16 hidden tiles across grid **width**. Changing only width from eight to four retains the eight K contributors and their eight-tile contractions. The native geometry derives:

| Quantity | Current 8×8 | Candidate 4×8 |
| --- | ---: | ---: |
| K groups / K tiles per group | 8 / 8 | 8 / 8 |
| Hidden groups / hidden tiles per group | 8 / 2 | 4 / 4 |
| Gate/up N chunks / tiles per chunk | 2 / 1 | 2 / 2 |
| Output embedding tiles per worker | 1 | 2 |
| Active decode expert count | 1…32 | 1…32 |
| Effective M tiles | 1 | 1 |
| Gate/up reduce contributors per output | 8 | 8 |
| Reduction slice tiles per worker | 1 | 1 |
| Row-mode down K tiles / K blocks, if retained | 16 / 1 | 16 / 1 |

The chunk choice is derived from `choose_hn_pad()`: its preferred count is three, but the nearest legal divisors of width four are two and four, with the tie choosing two. This changes grouping of independent N outputs rather than deliberately changing their K contraction.

**Why this may help:** half as many worker participants and hidden groups reduce multicast/gather participation and metadata work; two adjacent N tiles can become one read when paired with suitable ND weight placement. **Why it may hurt:** half as many compute workers each produce twice as many output columns, and larger local buffers can change the native resource fallback. Neither cost is measured here. At current interleaved placement, adjacent pages still issue as separate requests. Do the geometry contrast alone first; do not attribute a combined layout/geometry result to one cause.

### A correction to the generic down-path explanation

Do not assume the small decode block executes eight down blocks with intermediate BF16 spills. `kernels/moe_fused_swiglu_common.hpp::h_row_plan` explicitly admits short blocks when the reduction slice fits within one token tile-row. For 8×8, `m_eff=1`, `HN_PAD=2` gives two workers with one tile each, so it qualifies. For 4×8, `HN_PAD=4` gives four such workers and also qualifies.

When the host retains `wd_mrow_rounds && wd_resident`, `kernels/moe_fused_swiglu_compute.cpp:824–856` consumes one complete `HID_T=16` row and invokes a **single** down K block; the fallback HGROUPS-block path is a different branch. `geometry.cpp` can turn row mode off to satisfy L1, so source constants alone are not evidence of the instantiated path. At both candidate geometries the preliminary divisibility guard passes: `DEPTH_H*hid_t=48` is divisible by `(M_BLOCK/2)*hn_pad`, respectively eight and sixteen. The final L1/resource branch remains an explicit discriminator. Do not describe avoiding down spills as a benefit if the current instantiated path already has none.

### Exactness discriminator

1. Instantiate the two geometries with unchanged frozen activation, IDs, scores, route activity, count and offset tensors. Keep every active output mapped through the same logical `(slot, expert ID, top-k position)`. Compare complete BF16 native expert outputs for all active rows, followed by combined outputs, bit-for-bit. Do not use PCC, greedy agreement or raw packed-buffer comparison if offsets differ for an unrelated reason.
2. Cover local experts in early and late positions, all-empty ranks, one/multiple local experts, shared experts with counts 1, 2, 4, 16 and 32, holes, active row 31 and changed token positions. Use the same prepared six-trace lifetime and no recapture. B2/B4/B32 have at most 32 physical rows; top-k is unique per row, so each expert has at most 32 decode rows and `m_eff=1`. This bound must remain enforced for the optimization; larger/chunked **prefill** stays 8×8.
3. Verify native host blocking metadata before capture, through an existing host-side construction/inspection mechanism if available, to confirm both variants retain row mode and resident W-down. The factory passes this conjunction into all three kernels (`moe_fused_swiglu_program_factory.cpp:166,245,295`). No profiler is needed. A changed branch is not automatically wrong, but defeats the proposed equal-contraction proof and requires investigating any mismatch before proceeding.
4. Run the existing independent-reference reduced probe and full-40 full-vocabulary/state/96-token comparisons. Equal row-mode contraction and contributor order are necessary source arguments, not a bitwise proof: matmul N blocking, pack scheduling and layout can still reveal a difference. Any active logit difference rejects this candidate under the existing equality requirement.
5. Only after exactness, compare whole traced decode and real vLLM primary/mixed benchmarks under identical selected configuration. Retain all mandatory barriers and make no native timing subtraction. The floor remains a separate field even if runtime improves.

The native fast SiLU helper, BFP8 partial/H packing and BF16 destination stay unchanged. This candidate deliberately does **not** change grid height, compute fidelity, norm/router family, output precision, per-expert order or EP ownership.

## Second bounded candidate: startup-only DRAM ND weight placement

Current `tt/multichip_decoder.py` uploads every EP gate/up/down weight with `DRAM_MEMORY_CONFIG`. The native operation already accepts DRAM ND-sharded weights, uses one descriptor per role for all 64 experts, and validates consistent per-role layouts. Its existing `WeightRuns<SHARD_W>` reader/writer uses the tensor accessor and coalesces a run only inside an N shard. There is no new runtime reshard required: construct the chosen layout at startup, before preparation/capture, and keep those weight buffers persistent.

At **8×8**, gate/up chunk width and down output width are each one tile. Therefore ND placement cannot reduce transaction count by merging those reads. Claims that it automatically halves the current kernel's reads would be false. It can still change bank distribution. At **4×8**, both read spans are two tiles, so aligned shard boundaries can permit a two-tile transaction. This is the narrower setting where the coalescing hypothesis has a source-backed mechanism.

Use a one-tile K shard height. Choose N width using both actual chunk starts and the device's DRAM-bank count: every two-tile read should remain within its shard, and successive K-row shard indices should rotate through banks rather than remain pinned. The factory's `warn_if_gate_up_shard_splits_runs` checks the actual walk; its comments explicitly distinguish a correct split read from a coalesced one. Gate and up must share the same N shard width. Down may have a different width. Wider sharding can add physical padding, so preserve logical shapes and account for weight-memory growth without changing useful-work bytes.

Do not copy an arbitrary “production default”: `models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py::routed_expert_weight_memory_config` is tied to a separate unified operation's grid. Its source documents one-tile shard height and bank rotation, but its selected width is not evidence of optimal placement for this 2048×512 model. Consult the actual bank count and evaluate a bounded pair of layouts; no bank-count or speed assumption was made in this inspection.

A layout-only conversion must preserve every stored BFP8 tile bit. Validate it against existing weights once at preparation, then apply the same expert-output/full-model exactness tests and whole-serving comparison. If the same weights serve prefill, include a non-aligned prompt and chunk continuation comparison; runtime geometry can remain unchanged for prefill. No `from_torch`, conversion or reshard belongs inside decode.

## Singleton indexed replacement is not a minimal exact repair

The current B1 path in `_moe` uses indexed TP sparse matmuls, local hidden size128 per rank, FP32 DEST, ordinary TTNN SiLU/multiply, an eight-expert reduction and cross-rank sum. Grouped EP uses whole hidden512 experts on their owner rank, eight within-chip K partitions, BFP8 staged input/partial/H carriers, BF16 DEST and the native fused activation helper. `precision.py::baseline_policy` and validation intentionally distinguish these accumulator policies; simply changing a B1 config to BF16 does not make its contraction and reduction tree match grouped EP. Merely choosing the same top eight IDs does not establish equal outputs.

A genuinely equal singleton kernel would need to reproduce all of the grouped path's boundaries: BF16-row-major to BFP8-tile conversion, per-K-group LoFi contraction and partial packing, ordered eight-contributor reduction, the exact native fast SiLU/rounding sequence, BFP8 H packing, the selected down contraction, BF16 output, top-k weighting and EP cross-rank sum. It must handle an expert's live row moving among packed lanes without changing any of these results. A per-expert device-count band is exposed by the native API, but band selection alone does not supply this alternate equal kernel.

Such a kernel could retain the arithmetic while changing transport/scheduling, but it is a new substantial implementation, not an existing configuration or safe singleton delegation. It would also need uniform device-side count predicates, trace-safe ownership, independent NoC completion and shared-expert/mixed-batch proofs. It is lower priority than the bounded 4×8 parameter contrast and startup layout contrast. No evidence here supports relaxing exactness, replacing native sampling, or removing the required ACK drain.

## Explicitly rejected shortcuts

- Grid-height changes alter K partitions and rounded partials. Grid width changes are only plausible here because the short-row down branch preserves the whole16-tile contraction; arbitrary grids are not equivalent.
- A 2×8 grid is not the next presumed-equivalent choice: `HN_PAD=8` makes the preliminary row-mode divisibility check fail (`48 % 32 != 0`), selecting a different down arithmetic branch.
- Native `GU_CHUNKS`, `WD_SPLIT`, `M_BLOCK`, pipeline depths, `MOE_SILU_FAST` and posted-H settings are compile-time implementation controls, not Python read-tuning parameters. Changing them is a separate native kernel change. Posted-H is explicitly documented as lacking a confirmed ordering guarantee and must remain disabled.
- On 8×8, the W-down writer split receives `floor(2*3/8)=0` hidden rows per ordinary H block. This observation does not establish a bottleneck or authorize changing traffic ownership/barriers; row-mode handling and the actual instantiated branch must be considered.
- Counts/offsets, unchanged/changed scheduler tensors, sampler trace ownership, async return behavior and cleanup remain independent serving gates. A passing standalone geometry result cannot replace them.

No candidate in this report is marked kept, measured, passing or sufficient for the 21.63129875 ms floor.
