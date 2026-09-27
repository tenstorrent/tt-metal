# AutoDebug: indexed expert replay nondeterminism

## Verdict

The latest whole-layer probe localizes the first changing **observed logical**
tensor to the gate slice, before GELU, down projection and mixing. With N1,
only columns 0:32 change across all eight active expert slots. Changing only
gate/up N ownership to N2 expands the affected region to exactly columns 0:64
on a different rank. Both regions belong to sparse worker (0,0), which is also
the A multicast sender and the prior generalized-router compute core. This
strongly localizes a worker-specific interaction, but does not yet distinguish
the sender role from predecessor compute state on that core.

Retaining raw gate/up storage suppresses the failure for 128 steps. The
isolated original expert chain also passes 128 repeats after uploading all
saved physical BF16 input bits, including unused-row NaNs. Its stable result
differs from the saved whole-layer reference only on rank 1. Whole-layer
program/allocator state is therefore necessary in the reproducer currently
available. K88 without spill and a 40 ms delay between replays both still
fail. Neither is a demonstrated repair.

Moving A to DRAM preserves all physical tile bytes and still fails with the
first-32-column pattern. Moving the native generalized router from core (0,0)
to (1,0) passes the first 128-step output-only control; the hardware owner is
checking the original-core reproduction and slice boundaries next. This is a
diagnostic result, not yet an accepted repair. No inspected source defect has
been proven to cause this current-mode failure, so no runtime fix is justified
by this audit alone.

A separate source-proven FP32 intermediate-buffer sizing defect exists in the
sparse factory; it is outside the failing `fp32_dest_acc_en=False` mode. Do not
use a naive FP32-accumulation toggle as a clean diagnostic control. Details are
under Other issues.

## Evidence and provenance

- HEAD: `9a529836fc91b1117a48f6d63ef30455a69cd42d`.
- `tt/multichip_decoder.py` SHA256:
  `8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`.
- `tt/optimized_decoder.py` SHA256:
  `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.
- `tests/diagnose_attention_ccl_boundaries.py` recorded SHA256:
  `3a783db8067d5a39cd5646abab6d0aa1d8cccdbdcc6848206492426e06e8b53f`.
- `bfp8_boundary_v3_all.json`: real layer 0 weights, input 4096, requested 128
  steps, failure at zero-based step 6 / absolute position 4102. `routed_local`
  differs only on rank 1: 2697 changed logical elements, max absolute difference
  0.01953125, no nonfinite values. All preceding recorded boundaries are exactly
  stable, including the actual sharded normalization. `shared_local` is stable.
  `routed_reduced` differs identically on all four ranks (2525 elements,
  max 0.03125); final output differs identically on all four ranks (1891
  elements, max 0.1875). This is propagation of a local difference through a
  collective, not evidence that the collective generated it.
- `bfp8_boundary_v3_all.pt` contains the retained local tensors. No raw fixture
  tensor values or weights are copied into this report or telemetry.
- `AUTODEBUG_bfp8_ccl.md` documents the prior localization and the invalid v1/v2
  diagnostic equivalence assumptions. The v3 result uses the actual sharded
  normalizer and supersedes their passing controls as localization evidence.
- The BF16 attention policy's passing layer/stack/batch results and the
  single-layer full-attention BF8 pass are useful contrasts. They use different
  expert inputs and, for full attention, different expert precision. They do
  not establish determinism of this frozen layer-0 expert input.

This was a fresh delegated source-only investigation under AutoFix/AutoDebug.
The required `.agents/scripts/autodebug.sh` CLI was attempted with the model
focus path. Its fresh `gpt-5.5`/xhigh session could not perform shell reads
because bubblewrap was unavailable. After repeated blocked reads, only that
auxiliary CLI process was terminated. No sandbox workaround or dependency
installation was attempted. This already-isolated delegated audit continued
with working read tools. No TTNN import, device operation, reset, runtime edit,
or C++ build was performed. The shared tree contains existing stage work.

## Lowered sparse path and concrete parameters

`tt/multichip_decoder.py:717-764` constructs the TP decode experts, then replaces
both sparse program configs. `_HybridExperts` selects those experts for decode.
`OptimizedExperts._chunk` at `tt/optimized_decoder.py:193-241` uses eight indexed
slots, the same indices for both projections, BF16 outputs in interleaved L1,
LoFi math, `math_approx_mode=False`, `fp32_dest_acc_en=False`, and
`packer_l1_acc=False`. Layer 0 input is BF16; gate/up weights are BFP8 and down
weights are BFP4. Every operand tile is 32x32.

| Quantity | Gate/up | Down |
| --- | --- | --- |
| Logical A | `[1,1,1,2816]` | `[1,8,1,192]` |
| Logical B | `[1,128,2816,384]` | `[1,128,192,2816]` |
| Physical M/K/N tiles per expert | 1 / 88 / 12 | 1 / 6 / 88 |
| `is_input_a_sparse` | false | true |
| `num_active`, outer batchA | 8, 1 | 8, 1 |
| Grid / workers | 6x2 / 12 | 11x8 / 88 |
| K block / K iterations | 44 / 2 | 6 / 1 |
| Per-core N / block N / subblock N | 1 / 1 / 1 | 1 / 1 / 1 |
| A CB capacity / bytes | 88 tiles / 180224 | 12 tiles / 24576 |
| B CB capacity / bytes | 88 tiles / 95744 | 12 tiles / 6912 |
| Output/partial CB capacity | one shared BF16 tile | one shared BF16 tile |
| Total compact output tiles | 8x12 = 96 | 8x88 = 704 |

The sparse factory is
`ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`.
It selects the shared in0 sender/receiver, in1 reader/writer, and
`bmm_large_block_zm_fused_bias_activation.cpp` compute kernel at lines466-574.
The selected JIT kernel source copies under `build_Release/libexec/tt-metalium`
were byte-equal to the checkout copies during this inspection. This does not
prove which binary/kernel-cache artifacts a future process will load.

## Reader, ownership and CB ledger

### Indexed count and address mapping are consistent

The factory forces `get_batch_from_reader=False` when indices are present and
sets compute/receiver batch count to eight (lines75-94,292). The in0 sender's
indexed branch never reads the route mask; it broadcasts the same A eight
times for gate/up and walks consecutive compact A slots for down
(`reader_bmm_tile_layout_in0_sender_padding.cpp:101-109,186-243,413-421`). Thus
zero/underflow in route weights cannot create the known `nnz` versus
count-nonzero handshake mismatch on this selected path.

The in1 reader loads the single row-major UINT16 ID stick once, barriers the
read, then addresses weight expert `indices[slot]` and output `slot` separately
(`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:284-344`). Weight tile
index is `worker_N + expert_id * Kt * Nt`; output tile index is
`worker_N + slot * Mt * Nt`. Down's A advances by `Mt*Kt=6` tiles per slot,
independent of expert ID. The op validates that indices occupy one row-major
stick and that B has one expert batch axis. The frozen shape satisfies both.

Runtime cache overrides patch the indices address in the in1 reader's argument
6, A and B addresses, and output address (factory lines815-853). The current
program attributes include `use_indices`. There is no source evidence here of
an old-index-buffer cache hit or use of expert IDs as compact A addresses.

### No core holes or tail transactions for these configs

Factory lines164-171 require enough workers, and lines243-267 reject a
nonrectangular multicast receiver grid. Gate/up uses exactly all 12 workers in
6x2; down exactly all 88 in 11x8. Each worker owns exactly one N tile, every
output block/subblock is one tile, M is one full physical tile, and K divides
44 or 6 exactly. The writer's last-column counts remain one and its padded
skip counts are zero. A host-only arithmetic check enumerated compact output
pages and confirmed exactly one owner for all 96 gate/up and 704 down pages.

### The transactions balance, including across expert slots

Gate/up has 16 input transactions per core: eight experts times two K blocks.
Each transaction reserves/pushes 44 A tiles and 44 B tiles, and compute
waits/pops those same counts. Down has eight transactions per core, each with
six A and six B tiles. Each input CB holds two whole transactions. Neither
case requests a partial group or wraps in the middle of a transaction.

For the A multicast, receivers reserve space, reset their valid semaphore,
increment the sender counter, wait for valid, then push the same count
(`reader_bmm_tile_layout_in0_receiver.cpp:79-96`). The sender waits for all
11 or 87 receivers, resets its counter, multicasts data, flushes writes on
Blackhole, then multicasts valid (`in0_sender_padding.cpp:360-402`). The
sender ends with a write barrier; receivers drain outgoing ready atomics at
exit. This inspection found no missing current-mode reserve/wait/push/pop.

Gate/up spills its first K44 partial to CB5, reloads it through
`copy_init`/`copy_block`, pops CB5, computes the second K44 block, and pushes
the final tile to CB4. These CBs alias SRAM because both formats are BF16.
An apparent cross-expert alias race is **not established**: the same in1 RISC
reads that expert's weights and writes its output, and performs the output
write barrier and CB4 pop before loading the next expert's weights
(`in1_sender_writer_padding.cpp:657-717`). Compute cannot form the next
expert's partial without those next weights. Down never uses a spill partial.

The gate/up reload phase reconfigures SrcA from BFP8 weights to BF16 partials,
initializes copy, waits/copies/pops one partial, restores SrcA to BFP8 and
reinitializes matmul (`bmm_large_block_zm_fused_bias_activation.cpp:90-126`).
SrcB remains BF16 input. Startup uses reversed source order as the matmul API
requires. No missing caller-level format restoration was identified. A
lower-level LLK timing defect is still possible and needs a failing sub-op
before speculative LLK changes.

Existing `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`
covers nonmonotonic IDs, compact A, BFP4/BFP8 weights and new-index-buffer cache
hits. Its representative geometry is M32/K128/N256 with K-block one, default
compute config and DRAM output. Those are useful address-contract tests, but
they do not reproduce this M1/K2816/N384, K44, LoFi, L1, traced replay case.
No result from those tests was assumed or generated during this audit.

## Follow-up evidence: first gate tile, whole-layer context required

The following runs were performed by the hardware owner and read as artifacts
by this source-only investigation. Runtime and expert source hashes remain as
listed above; diagnostic versions are recorded in each JSON.

| Artifact | Observed result | Consequence |
| --- | --- | --- |
| `frozen_experts_output_only.json`, `frozen_experts_boundaries.json` | Both pass 128 repeats using logical input upload. | Insufficient padding control: that upload erases 37,552 nonfinite unused physical-row values. |
| `bfp8_boundary_retain_gu.json` | Retaining raw GU passes 128 whole-layer steps. | Lifetime/address changes suppress the observed failure; does not clear the unretained producer. |
| `bfp8_boundary_retain_hidden.json` | Fails step 0 / position 4096: rank-1 hidden changes 118 logical elements, max 0.296875, only columns 0:32. All physical expert-input bits, routes and IDs are stable. | Changes in input padding are not necessary for every observed failure. |
| `bfp8_boundary_retain_slices_v2.json` | Fails step 2 / position 4098: first logical difference is rank-1 gate, 173 finite values, max 0.1875; up is exact. Hidden changes 162 values. | First observed difference precedes activation and down/mix. All changed gate/hidden columns are within 0:32. |
| `bfp8_output_k88.json` | Original output-only chain still fails at step 47 with N1/K88 on the same 6x2 grid. | Removing partial spill/reload alone is not a fix. |
| `bfp8_output_delay40.json` | Original output-only chain still fails at step 13 with a 40 ms delay between replays. | This delay is not a fix and does not isolate an intra-trace ordering issue. |
| `frozen_experts_physical_output.json` | Original expert method passes 128 repeats with all saved physical BF16 input bits restored. Stable output differs from saved whole-layer reference on rank 1 only: 2450 values, max 0.0078125. | Physical input contents alone do not reproduce the whole-layer failure. That changed count/max matches the hidden-only failure's local routed output. |

The first slices attempt, `bfp8_boundary_retain_slices.json`, stopped on a
physical-padding-only difference while all logical boundaries were stable.
The v2 diagnostic records those differences but continues until a logical
boundary changes. Its diagnostic SHA256 is
`864d2110a16bb3a2591f3fc19aaa896161ed152e48c4193883aa0ad5d46bd74c`.
For the hidden-only failure, the diagnostic SHA256 is
`d7bed7526147a254c3eecb5ff902d5752973d7f5aafbb96114dd1accf18ada4c`.

CPU-only mask analysis found gate changes in all eight slots (21 or 22 values
per slot) and none outside the first tile. The 22 affected columns form a
repeated pattern between the two 16-column tile faces. Changed values are
finite and are not zero replacements. Difference amplitudes/signs are not
identical across experts. These are statistics, not raw fixture values.

### Producer ownership versus slice and activation ownership

For gate/up, the sparse factory assigns `worker_N=i` at lines699 onward.
Each worker owns one output tile per expert. Thus logical core (0,0) produces
columns 0:32 for all eight expert slots; it is also the in0 multicast sender.
The first up tile, columns 192:224 of GU, belongs to worker 6, core (0,1).
The failure's shared producer is therefore concrete. However, GU output is
interleaved L1: producing core, destination bank/core and destination address
must be distinguished before blaming this worker or its arithmetic.

`slice/slice.cpp:333-372` pads the single logical row to 32 and invokes the
tile slice primitive, then restores the logical metadata. Both starts and
widths are tile-aligned. `slice_program_factory_tile.cpp` uses 48 workers for
the 48 physical output tiles, one tile each. Gate output tile `t` reads GU
tile `12 * (t // 6) + (t % 6)`; up adds six. There is no first-column branch.
The reader reserves, reads, barriers, then pushes; the writer waits, writes,
flushes before pop, and performs a final write barrier. No missing drain was
identified in this concrete one-tile-per-worker path.

`ttnn.mul` resolves to the binary-NG multiply path. With equal shapes and
default `fast_and_approximate_mode=False`, it selects the SFPU no-broadcast
kernel, not the FPU broadcast kernel. The factory again distributes 48 tiles
to 48 workers. Accurate GELU preprocesses the left input into a BF16 CB, then
the kernel multiplies it with the BF16 right input. The changed first tile of
each expert maps to flattened tiles 0,6,12,...,42, hence eight distinct
activation workers. Source inspection found the expected CB waits, register
acquire/commit/wait/release protocol and output write barriers. This and the
gate-slice failure demote activation as the first producer of the difference.

### Retention changes the allocator prefix too

`BoundaryExperts._chunk` clears its own `boundaries` at entry, while
`BoundaryDecoder._forward` clears the decoder dictionary earlier, before
attention. A GU handle saved in the expert dictionary from the prior warm
call therefore survives into the next call's attention/norm prefix until
the expert entry. Retaining GU can change allocation addresses both before
and after its current production. A pass with GU retention is not proof of
a lifetime issue exclusively between GU production and its consumers.

## N2 and sender/predecessor follow-up

`bfp8_output_n2k44.json` fails at step 7 in the original output-only method.
`bfp8_boundary_n2_slices_addresses.json` fails at step 0 in the gate slice on
rank 2: 254 changed values, max 0.328125. Counts per 32-column tile are
`[126,128,0,0,0,0]`; up remains exact. Thus the affected width doubles with
worker (0,0)'s N ownership. The diagnostic SHA256 is
`dc22ebba887ee93fa2279d7e5c3481de4e1d5ea5a60d319101c7e2309548ccd8`.
The repeated changed-column mask modulo 16 is `{0,1,4,7,11,12,13,15}`, across
the four faces of those 64 columns and all slots, apart from two BF16 rounding
exceptions in the last slot. This motivates a register/compute-state check in
addition to the A-sender protocol.

The independent lifetime audit found no recorded prefix-allocation base
overlap for raw GU or gate/up slices in the 221 scalar address records.
The captured physical input does vary in unused rows: 1596 zero-to-Inf and
226 Inf-to-zero values per rank, while logical input remains exact. This is
different from the earlier hidden-only failure, whose entire physical input
was identical. Neither result alone attributes the gate change to padding.

`bfp8_output_expert_input_dram.json` still fails at step 26.
`bfp8_boundary_input_dram_slices.json` still fails at step 6: rank-0 gate,
126 values, max 0.1015625, first 32 columns; up is exact. The copy control is
source-supported: interleaved L1-to-DRAM `to_memory_config` falls through to
`prim::copy` and selects `DefaultTilized`. It copies all 88 complete BF16
tiles; `convert_df=false` directly connects reader and writer CBs without a
compute kernel. The 2048-byte page copies preserve unused-row and NaN bits.
The added operation still changes timing and allocation context, so this
control clears neither of those factors.

### Sender versus receiver ledger

Both variants use BRISC on dedicated NoC 1 for A, NCRISC on dedicated NoC 0
for B/output, and the identical shared compute kernel on every worker.
`reuse_in0_in_CB`, sharded A, fused-op signaling and indexed validity-mailbox
traffic are all disabled. For each K block:

- Sender reserves the complete A block, reads all tiles, waits for the read
  barrier, waits for receiver reservations, multicasts data, flushes outgoing
  writes on Blackhole, queues the valid multicast, then publishes local CB0.
- Receivers reserve the same block, clear their valid flag, increment ready,
  await the valid multicast, then publish CB0 to their local unpacker.
- Compute waits for the full block and pops it only after its matmul work.
  Both local CB0 rings contain two whole blocks. Indexed mode runs precisely
  eight slots; there is no fake reuse or partial-ring transaction here.

The sender can expose its own tiles before the receivers expose theirs,
because it does not wait for the valid multicast to arrive remotely. Its
NoC source bytes have already departed before local push. Source inspection
found no missing read completion or source-lifetime flush. Replacing that
flush with a full write barrier would therefore be a scheduling control;
a pass would not by itself demonstrate a missing required barrier.

### Generalized router and sparse startup state

The router is a native one-core operation on (0,0), including all persistent
sharded inputs/outputs, at `optimized_decoder.py:895-915`. It uses the
single-block ungrouped top-8 path, full DEST sync, SFPU sorting/softmax and
FPU SrcB scratch/transposes. Its multi-block source warns about same-acquire
SFPU/SrcB interactions, but that branch is not selected for this 256-padded
router. `bfp8_output_router_core1.json` moves the native core and persistent
buffers to (1,0) and passes 128 output-only steps. The original-core repeat
and moved-core slice check are still required to interpret this contrast.

The current source has several explicit boundaries against generic stale
state; do not assume they are missing:

- Blackhole `common/chlkc_list.h:29-36` zeroes all DEST on MATH and all SrcA/B
  banks on UNPACK before `kernel_main`. `firmware/src/tt-1xx/trisck.cc` calls
  `do_crt1` for each kernel, resetting software format/zero-flag caches.
- `firmware/src/tt-1xx/trisc.cc:222` calls `tensix_sync()` before signaling
  kernel completion. A router instruction merely remaining queued past its
  declared completion is not supported by this source.
- Generalized gate's final transpose step2 issues `CLR_AB`; its final
  `tile_regs_release` waits for pack and clears full DEST. The code uses
  `deepseek_compute_kernel_init<false>`, so the opt-in DeepSeek math-remap
  toggle is not active in this path.
- Sparse `compute_kernel_hw_startup<SrcOrder::Reverse>` programs formats,
  complete tile/face geometry, FP32/INT8 modes and source-format overrides.
  `configure_unpack_AB` resets stochastic-rounding enable bits. Matmul init
  replaces math/unpack MOPs, address modifiers and counters and explicitly
  restores operand-driven source zero substitution after copy/transpose.
  Pack startup restores destination section selection and output format.
- Inspected shared ALU config writes use masked RMWCIB instructions with
  disjoint masks. No concrete lost-update pair was identified.

The checked installed copies of startup, math common/matmul, pack/unpack
common, zero-flag tracking and generalized-gate SFPU/transpose headers match
the checkout byte-for-byte. This does not establish the identity of any
already-cached executable kernel.

## Updated focused verify/refute sequence

The N2 ownership, DRAM-input and address controls above have now run. The next
discriminator is the moved-router-core repeat with unchanged native expert
program. If the relocation remains passing, prepend the actual generalized
gate on core (0,0) to the passing frozen physical-input expert probe while
keeping the saved routes and indices consumed by experts unchanged. Use
existing persistent router outputs, then compare the same gate on core (1,0).
This can isolate predecessor state using existing native operations before
trying a kernel barrier or cleanup modification. A blanket LLK cleanup pass
would only implicate some state/timing interaction; narrow it before retaining
any repair.

1. Compare original router core (0,0), moved router core (1,0), then the
   original core again with the same native N1/K44 expert program. Read slices
   in the moved-core control to distinguish disappearance from migration of
   the changed tile to the new router core. Track addresses as scalars so a
   changed allocation layout is visible without new retained handles.
2. If the minimal gate-prefix probe reproduces, retain that reproducer and
   inspect the transition before changing shared LLK code. Compare the
   relevant config fields on sender/receiver cores at sparse startup, with
   the hardware owner using existing assertion/debug tooling. A cleanup
   intervention must be narrowed to a field/dependency before it is a fix.
3. If the result remains tied to the sender regardless of router placement,
   the smallest timing control is a full write barrier before local CB0
   publication. Moving the sender role while preserving output ownership
   would distinguish sender behavior from an intrinsically affected core,
   but requires a carefully reviewed factory experiment, not a runtime model
   workaround. Neither experiment is authorized or performed by this audit.
4. Keep the address/lifetime evidence available. If a storage-write candidate
   emerges, resolve interleaved pages to physical bank/core and byte ranges;
   base-address equality alone is insufficient. Require an actual writer
   and missing completion dependency or an observed write after storage
   reuse. The independent buffer-lifetime audit owns this cross-program check.

No speculative geometry, precision or barrier change should be retained until
its isolated hypothesis and the original layer-0 4096/128 duplicate-replay
check pass. The layer-5 control and stack/batch/cache checks remain acceptance
gates. Passing BF16 remains the selected attention policy in the meantime.

## Other issues: FP32 partial-buffer sizing (not this failure)

The sparse factory computes `interm0_single_tile_size` from
`output_data_format` at line120 **before** it chooses `interm0_data_format` at
lines192-194. If output is BF16 and FP32 accumulation is enabled, the latter
becomes Float32, but CB5 size and page size remain the BF16 2048 bytes at
lines223 and644-645. A 32x32 Float32 partial needs 4096 bytes. The corresponding
dense 1D factory computes partial tile size from `interm0_data_format`
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:139-148`). The sparse
factory also lacks the dense path's FP32 partial reload configuration.

This is a concrete separate format/allocation mismatch in source, not proof
that an FP32 experiment ran or corrupted anything here. The failing and
selected decode configs explicitly disable FP32 accumulation, so this defect
cannot explain their nondeterminism. No fix for it was made in this source-only
investigation.

## Final status

The first observed logical change follows sparse worker (0,0)'s gate-column
ownership across N1 and N2 geometry and multiple ranks. Sender behavior versus
the router's prior use of that core remains unresolved. Source inspection refutes several
specific sparse count, compact-address, core-hole, partial-tail and obvious
CB-alias explanations for the concrete current parameters. Full physical
input upload, K88, replay delay and DRAM-input results narrow the necessary
reproducer to whole-layer state without proving a root cause. The first moved
router-core control passes and is under follow-up. This investigation made
only a documentation change and ran CPU/source checks; device results above
are explicitly attributed to the hardware owner's artifacts. No performance
or build result is claimed.
