# Remaining solo candidates after route-plan fusion

Source-only audit, 2026-11-17. Production source was not changed and no hardware, profiler, or server was used. Parent reports route-fusion native36.061 ms and async36.001 ms versus prior async38.242 ms, with exact full40 logits/serial96. The required 21.63129875 ms floor is unchanged. None of the candidates below has a measured saving. Line anchors refer to the inspected pre-ND source snapshot; the named functions remain the authoritative anchors if the concurrent ND patch shifts lines.

The historical Stage7 B1 path and grouped B2 solo path still differ in expert arithmetic and execution geometry. Replacing grouped EP experts with the indexed B1 path is not a numerical-preserving shortcut. Retain the selected grouped math, deterministic rounding, precision, reduction order, and all required model/sampling traces. The independently prepared legal GU ND-layout test should run first.

## Ranked experiments

### 1. Remove the dead inactive singleton normalization in solo0

Anchors: `tt/multichip_decoder.py`, `decode_batch`, the linear-attention loop around1930–1936; `tt/mesh_compensated_norm.py:213–246`, `normalized_singleton`.

Move the existing `if solo_slot0_attention and i == 1: ... continue` above `_normalized_input(row, "input_layernorm.weight")`. That row1 norm output is currently discarded. The retained solo0 selection already guards authoritative activity `(0,)`; active row0 normalization and every B2 postnorm/router/shared/full-projection carrier remain identical. No new kernel, host refresh, state predicate, adapter path, or arithmetic substitution is required.

This deletes 30 singleton compensated norms. Each contains square-multiply, fast reduction, terminal scan, norm epilogue, normalization multiply and weight multiply: at least180 dead device operations across the linear stack, before shape/layout movement. This makes it the strongest next **small** experiment by removed-work-to-change-size ratio. It is not an estimate of180 launches' wall time and cannot be claimed sufficient to close the14.43 ms native floor deficit. Run exact full40 logits/state first, then matched native timing; retain all mixed-traffic gates because the captured solo signature changes.

### 2. Fuse conditional recurrent-state commit into the existing state-slab writer

Anchors: `tt/multichip_decoder.py:389–395,518–535`; `tt/mesh_state_slab_recurrence.py:54–73,288–325`; `tt/slot_state.py:159–199`.

Masked decode calls the same ordered recurrence as B1, but supplies `output_state=None`. The state-slab helper then allocates its `(1,8,128,128)` FP32 state output in L1. The caller moves that entire state to DRAM, then `copy_if` reads it and writes the persistent DRAM state if activity is true. B1 already supports direct in-place state output. This is a source-proven extra staging boundary repeated across 30 linear layers, even in the retained solo0 graph.

Minimal experiment: add an optional persistent activity scalar to the model-owned state-slab writer. Bind its state destination directly to the existing persistent state. Read activity on device each invocation and predicate only the four per-worker state NoC writes. Always consume/pop all four state-output CB entries and still write the attention result, so inactive rows preserve state without changing compute or deadlocking the writer. Leave the fallback recurrence and its old conditional commit unchanged when native state-slab guards do not match.

The state is 524288 bytes per rank per layer. The current active path performs an extra DRAM read plus write of this state relative to a direct commit: 30 MiB of avoidable DRAM traffic across 30 layers per rank, plus the intermediate L1 state transport and two operations. This is a byte count, not a latency estimate. `copy_if` already partitions pages across workers; it is **not** a single-core 128-page serial copy.

Arithmetic remains the exact existing SFPU state slab, including BF16 consumption rounding, FP32 persistent state, and ordered contractions. Do not substitute native FDR/TF32 arithmetic, reassociate reductions, or change the Q/K normalization/SiLU lookup contracts. Test exact recurrent/output bits with active→inactive→active replay, dormant low/high slots, then full logits and trajectories. In-place writes are already supported, but the new predicate must be read before the write and must not bypass CB drains.

### 3. Fuse per-layer tiled row extraction and activity masking

Anchors: `tt/multichip_decoder.py:1970–1973`; `tt/model.py:438–441,456–465`; existing face-layout example `tt/serving_state.py:575–598`.

Every layer currently converts the B2 result to ROW_MAJOR, slices each row, converts each slice back to TILE, and then `QwenModel._mask_hidden_rows` allocates/copies both rows again. These are exact data movement operations surrounding unchanged residual arithmetic. The solo graph pays this at all 40 boundaries.

Minimal experiment: a model-owned BF16 TILE row-extract-and-mask helper takes the already-DRAM result and the persistent activity vector, and returns the same logical singleton TILE rows. Copy active logical row bits into each output's row zero and explicitly zero every padded row; inactive outputs are entirely zero. Distribute vocabulary/hidden tiles across workers. This removes untilize/slice/tilize plus the subsequent separate conditional copy, without changing the B2 input to the next normalization/projection or any arithmetic kernel.

Keep the initial embedding mask until separately tested. Validate output dtype, padded shape, logical values and physical padding against the complete old split-plus-mask sequence, including changed activity and retained outputs under trace tracking. A view into the shared B2 tile is insufficient: row writes must not alias another request. The exact gain is unknown; this candidate targets repeated dispatch and copies rather than weight reads.

### 4. Change grouped weight-read issue order without changing arithmetic

Anchors: native `moe_fused_swiglu/device/kernels/moe_fused_swiglu_dataflow.hpp:94–127`; reader `:729–741,899–945`; native geometry header `:29–40`.

`read_weight_chunk` issues each K row into a fixed `l1_base + k*chunk_w*tile_bytes` destination. A bounded read-wave experiment can rotate the **issue traversal** over these K rows per worker while preserving every DRAM source, L1 destination, CB slot, chunk publication, and completion barrier. Compute then receives byte-identical tiles in the original K order. This is legally different from reordering FMA terms, changing K/H groups, changing the gate/up fold, or enabling posted multicast.

Use an explicit opt-in factory/kernel signature or model-owned derivative. Test only after the ND-layout candidate, with its chosen layout held fixed. Do not change global shared constants silently or read an environment variable after a program was cached. The source establishes equivalence of the completed payload, not that the current bank wave is bad. Read-wave staggering could regress bandwidth. A reduced exact expert-output comparison must precede full40 timing.

Existing overlap is substantial: the reader issues GU chunk0 before activation staging, issues later chunks after the staging barrier, and batches W_down reads before the reduction rendezvous. Moving every chunk into the prologue would drain all weights at its blanket barrier, explicitly defeating this overlap. Cross-expert prefetch needs additional protected CB capacity and synchronized weight-base selection; it is not a trivial issue-order edit.

### 5. Compact zero-count expert mailbox traffic, only as a coordinated native-kernel experiment

Anchors: native reader `:501,622–627,671–673,1366–1369`; compute `:524–551`; writer `:267–278`; native device validation `:53–57,197–199`.

All 64 local experts still publish/consume count mailboxes, even with zero count. Reader, writer, and all three compute TRISCs walk the same sequence. A solo token activates at most eight experts globally, so most mailbox turns carry no arithmetic.

The source does **not** support claiming 64 network ACK rounds: zero-count experts already have `m_blocks=0`, skip all per-block collective traffic, and do not advance the active-block counters. The final reader `noc.async_write_barrier()` is a required launch-boundary drain, including all-empty launches; preserve it.

A legal experiment would derive an ascending list of active local expert indices from device counts, then have reader/compute/writer consume the same compact schedule and explicit completion marker. Weight-base addressing must still use the original local expert index, while mailbox sequence numbers and active-block counters use a consistent schedule. Preserve per-expert H-buffer padding, deterministic math, and every active block's ACK protocol. The public API cannot simply pass eight `expert_ids`: validation ties expert count to all weight lists, and changing an ID-table entry alone does not select a different physical weight base.

This is a larger protocol change with hang and stale-mailbox risk. An all-empty-rank case, one active expert at index63, changing active lists between replays, and repeated producer/consumer launches are mandatory discriminators. Do not implement a reader-only `continue` before publishing the empty expert: compute and writer would still wait for it. Prefer the preceding model-owned candidates before this change.

## LM head and sampler disposition

`tt/model.py:299–324` keeps the same FP32 terminal norm/BF16 boundary. `tt/lm_head.py:16–73` retains the Stage7 DRAM-sharded vocabulary projection: three readers per bank, block4, 16 input cores, per-core N32, selected weight dtype and FP32 accumulation. Both B1 and B2 occupy one physical tile of rows, so switching B2 to singleton logical rows does not remove a tile of LM-head matrix work.

The generator still pads local logits and copies into its persistent sampler buffer (`tt/generator.py:492–501`). An output-bound head/row-staging contract could remove an intermediate full-logit copy while preserving the canonical sampler, but this runs once per token rather than 40 times. Existing `PrefillLogitRows` provides an exact row-staging precedent. Preserve zero dormant logits and stable sampler addresses; do not replace split device sampling with host argmax, force-argmax, generic fallback or full-logit readback.

`tt/generator.py:503–511,633–650` retains native32/native64 sampling traces, active token merge, and persistent seed advance. Dormant parameters can require native64, so checking only the active user's top-k is unsafe. `models/common/sampling/tt_sampling.py:1260–1360` retains the split candidate machinery. The historical clean Stage7 sampler-only artifact (`doc/optimized_full_model/final/sampler_timing.json`) measured split-greedy around 0.488 ms for 32 physical rows and vocab248320, versus roughly 2.7 ms for force-argmax. That is historical evidence with recorded old hashes, not a current component measurement. Nothing inspected proves the current sampler dominates the remaining floor deficit.

## Other tempting changes to defer

- Shrinking grouped dispatch capacity from2240 to a solo bound of256 rows reduces reserved space, but existing native readers already read only active expert regions; a smaller allocation is not itself evidence of less execution work.
- Combine still zeros a `(1,1,32,8,2048)` BF16 result (1 MiB/rank/layer). A reduced logical combine/post-combine output could remove inactive traffic, but it changes output geometry and requires matching weights/indices and padded carrier restoration. Native post-combine tilization and sparse indexing need a dedicated legality probe; source validation alone is not enough. Keep initialization until every unread route is proven skipped.
- Substituting indexed B1 experts, changing grouped core geometry or reduction groups, dropping precision, or skipping required ACK completion would not establish the canonical grouped numerical-family contract.

Each candidate must be default-off, prepared before capture, retain the existing B2/B4/B32/solo0 and sampler signatures, and pass exact stale-input/state/logit checks before matched native and real-vLLM measurements. Source reasoning cannot promise that any combination reaches the floor.

## Other dead solo0 row work, after the isolated norm experiment

The archived `solo_dead_norm_candidate.patch` changes only the order of the existing row1 skip and independent norm call. Device position construction, activity reads, state handling, all B2 carriers, and terminal/sampling are unchanged.

1. **Initial row1 embedding and hidden all-gather.** `QwenModel.decode` currently calls `embed` for slot1, and `embed` runs a local embedding plus hidden-shard all-gather before `_mask_hidden_rows` zeros the inactive row. A separate solo0-only candidate can replace that row with an immutable constructor-allocated BF16 `(1,1,2048)` zero tensor with `(1,32,2048)` padding, keeping slot0's activity mask. This removes one embedding/all-gather sequence per token, not per layer. Do not reuse the existing FP32 linear partial or1024-wide full-attention zero. The collective semaphore sequence will change and must be prepared/captured afresh with all variants; active outputs must remain exact.
2. **Discarded row1 boundary extraction/masking.** After each layer, the next solo0 carrier needs row1 to be zero. It can use the same immutable residual-zero tensor instead of extracting and then masking a computed row1, while keeping the full B2 residual/MoE/terminal arithmetic. This saves a row slice/tilize/copy at40 boundaries. The generic split-and-mask fusion above is preferable when preserving the device activity predicate for every row is part of the helper contract; a solo0 constant path is legal only under the existing authoritative `(0,)` signature guard. Never select it from a capture-time tensor value or use it in ordinary B2 traffic.
3. **Slot1 position/RoPE and advance.** The specialized attention paths no longer consume slot1's RoPE/current-position tensors; the final inactive position advance is a device-guarded no-op. These are once-per-token opportunities, but keep them unchanged in the norm patch and first follow-up. Dropping them broadens the persistent-input contract and is lower priority than the repeated layer boundaries.

There is no independent embedding normalization to remove. Full-attention input norm/projection, post-attention norm, both batched residual adds, grouped router/shared expert, and terminal norm/head operate on required B2 or internally32-row carriers. Do not switch these operations to B1 or remove one lane by changing their shape; that could select a different numerical implementation. The inactive linear singleton norm is exceptional because it is a separate already-discarded result.
