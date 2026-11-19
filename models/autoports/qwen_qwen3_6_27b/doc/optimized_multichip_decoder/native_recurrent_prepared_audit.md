# Prepared native recurrence: coherent TP4 families

Source audit, 2026-11-13, checkout HEAD `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. This document proposes bounded experiments; it does not implement, benchmark, accept, or reject a candidate. No device calls, TTNN imports, builds, production edits, or probe changes were made for this audit. The existing [family audit](native_recurrent_family_audit.md) supplies the public API/model mapping; this document concentrates on removing its wrapper and wave costs.

**The next meaningful candidate is one preparation launch plus one multihead native recurrence launch, with compact prepared inputs, direct compact output, and persistent state. Test both FP32 and BF16 persistent-state variants.** The factory's one-head-per-core assertion and public FP32-state validation are implementation boundaries with concrete adaptations, not reasons to reject this family.

## Evidence and boundary

`perf/integrated14_profile_l0_b32_s33/decode_perf_report.txt` attributes 125 us (20.9% of the instrumented decode sum) to the local recurrence. This is a profiler observation, not the final default path's uninstrumented latency. The associated `metrics.json`/`accounting.json` use a modeled 512 GB/s per chip on P300c.

`native_recurrent_public_b1_smoke_v2.json` records `status=pass`, TP4 `[1,4]`, watcher enabled, allocation tracking enabled, program-cache skipping disabled, and two eager steps. Timing was disabled. Parent reports the allocation-tracker positive control passed. This proves a bounded public-operation owner, not the prepared kernel below. At inspection, `native_recurrent_wide18.json` was still `status=running` with 17 recorded cases; do not treat this snapshot as a final wide-suite result. Parent reports the existing custom `nativepack512` state-rounding path has passed its reference checks; that supports reuse of its persistence semantics, not acceptance of native FPU arithmetic.

Measure from the current post-convolution packed BF16 ROW_MAJOR tensor `[B,1,2560]`, plus actual log decay `g` and sigmoid `beta`, through preparation, recurrence, compact FP32 output `[1,12B,128]`, and committed persistent state. The actual default g is FP32; beta is BF16. Support BF16 g only as a separately recorded input policy. Include every device conversion/preparation/packing operation in this boundary and subsequently measure the whole real-weight decoder. State reset, host reads, and reference preparation stay outside latency timing.

The local mapping is Q/K heads 4, V heads 12, K=V=128; value head `hv` uses Q/K head `hv//3`, global local-head index `h=user*12+hv`. Q is normalized with `rsqrt(sum(q*q)+1e-6)` and scaled by `128**-0.5` exactly once. K is normalized without that scale. Compute decay `exp(g)` exactly once and do not apply another sigmoid to beta. Preserve decay-before-prediction and updated-FP32-state readout:

```
sd   = S * exp(g)
u    = beta * (v - k @ sd)
live = sd + k.T @ u
out  = q_scaled @ live
S    = live                         # persistent FP32 family
S    = BF16_RNE(live)                # persistent BF16 family
```

There is no inter-rank dependency or new collective in this local operation. Preserve the decoder's existing inter-layer residual contract and TP4 shard ownership.

## Concrete changes that remove dominant overhead

| Existing cost or restriction | Prepared candidate | What must be measured/proved |
|---|---|---|
| Public normalization, dtype conversion, head splitting/repetition, scale/exp, and per-wave slicing | One device preparation launch, normalizing each grouped Q/K vector once | Complete-boundary latency; normalization/exp rounding against the current input preparation |
| Factory maps one head to one core and rejects `12B > cores` | Reader/compute/writer share a bounded outer head loop | B10/17/32 correctness and workload balance; no stale CBs or early state reuse |
| Head-major padded FP32 Q/K/V and scalar tensors | Compact prepared ROW_MAJOR storage; reader constructs native tiles in local L1 | Actual DRAM traffic and reader overhead, correct zero padding |
| Per-wave state owners and index reads | One persistent global local-head state tensor, T1 direct slot `h` | Exact ownership/rebinding and changed replay; no state mirror or full-state conversion |
| Padded head output plus untilize/permute/retile/concatenate/copy | Native writer writes each head's row directly into persistent compact output | Disjoint aligned writes, tail ownership, full-output replay freshness |
| FP32 persistence doubles state traffic | Additional native BF16 input/output-state kernel specialization | Matched BF16-control accuracy and long drift, including FP32 live readout before rounding |

A private `ttnn.generic_op` descriptor with adapted reader/compute/writer sources is feasible without claiming that the public API supports these contracts. Alternatively extend the native factory with an explicit, tightly validated T1 mode. Reusing unmodified native FILE_PATH kernels in four global-head waves remains a useful arithmetic control, but does not test the two-launch endpoint. Do not silently change the shared public operation's multi-token/ring semantics.

## Multihead scheduling and native CB lifecycle

The native factory (`device/fused_recurrent_gated_delta_rule_program_factory.cpp`) checks `BH <= grid.x*grid.y`, creates one core per head, and passes one `h` to reader/writer and `T` to compute. The kernels have no cross-head communication. The measured grid is 11x10. For `H=12B`, use `C=min(H,110)`, `base=H//C`, `rem=H%C`; core i owns `start=i*base+min(i,rem)`, `count=base+(i<rem)`. Preserve the factory's supported core-coordinate mapping; do not invent an assumption about physical NoC coordinates.

| B | Heads | Active cores | Heads/core |
|---|---:|---:|---|
| 1 | 12 | 12 | 1 |
| 3 | 36 | 36 | 1 |
| 9 | 108 | 108 | 1 |
| 10 | 120 | 110 | 10 cores with 2, 100 with 1 |
| 17 | 204 | 110 | 94 cores with 2, 16 with 1 |
| 32 | 384 | 110 | 54 cores with 4, 56 with 3 |

Do not force whole Q/K triplets onto cores if that increases the maximum per-core head count. Grouped preparation already avoids repeated normalization; inexpensive grouped reads can remain duplicated. The initial adapter can bound B to 1..32 and T exactly 1.

Required changes and barriers:

1. Reader iterates its contiguous head range. For each head, reserve and load all 16 state tiles, complete the NoC read barrier, then push state. Read/prep Q/K/V/scalars into their own reserved CBs; complete each required read/fill before push. Only one core owns a state head. Reading the entire old head before publishing it makes in-place T1 writeback safe.
2. Compute adds the same outer head loop around the existing T1 body. Preserve decay, prediction, delta, rank-one update, and readout order. At the end, **pop the 16 tiles of `cb_s2` after readout**. The current kernel deliberately leaves its last `nxt_S` resident for a subsequent token; carrying that behavior across independent heads creates stale data or a full-CB stall. All Q/K/V/gate/intermediate CBs must have balanced consumption per head. T1 never needs `cb_s3`; remove it only as a separately verified resource reduction.
3. Writer loops the identical head range, drains compact output then state, barriers each set of writes before popping source CBs. Existing output/state CB double buffering can remain initially. State output may be produced before output, but the state CB must hold a complete head so compute can reach output production while writer waits for output. Do not reduce it below that requirement.
4. The command queue orders complete recurrence launches. Before the next launch can read mutable state, previous writeback must be complete. No host sync or cross-rank fence is introduced. Distinct owners can use the cached program only when all tensor base addresses and head bounds are rebound by runtime arguments.

At Kt=Vt=4,T1, the existing factory allocates `4+4+4+1+1+16+8+32+16+16+16+4+4+4=130` FP32 tiles: 532,480 bytes/core, excluding ring index scratch. Multihead loops reuse this space; allocation does not grow with heads/core. L1 feasibility must include descriptor alignment, kernels, trace-resident allocations and any prepared storage. A paper aggregate total is not physical allocator evidence.

## Compact preparation and output, without hidden padded DRAM copies

One concrete aligned prepared ROW_MAJOR record is `[B,1,2592]` FP32, 10,368 bytes/user:

| Float offsets | Contents |
|---|---|
| 0..511 | Four normalized/scaled Q vectors |
| 512..1023 | Four normalized K vectors |
| 1024..2559 | Twelve V vectors |
| 2560..2571 | Twelve decay scalars |
| 2572..2583 | Twelve beta scalars |
| 2584..2591 | Owned zero padding |

The reader fetches three 512-byte vectors/head and an aligned 128-byte gate segment beginning at byte 10,240 of that user's record. Each scalar's actual offset is selected locally. Populate native FP32 Q/K/V tile row zero and scalar `[0,0]`, zeroing all other rows/columns. Native rank-one matmul relies on the other 31 lanes being zero; allocator contents cannot satisfy that contract. A simple first implementation explicitly zeros each reused tile's padding. Hoisting zero initialization requires proving every CB slot address is fixed and padding never changes. Preparation writes complete records, owns alignment, and accepts arbitrary supported logical B; no public alignment requirement is added.

The reader needs a ROW_MAJOR accessor with the actual aligned page size; it cannot reuse the native reader's assumption that every input accessor has a 4096-byte tile page. Use whole-record or aligned sub-page accesses through the tensor's actual descriptor. Packing prepared rows into L1-native tiles is local movement and remains timed.

For compact FP32 TILE output `[1,H,128]`, head h supplies row `r=h%32` in tile-row `h//32`. For value tile column c=0..3 and horizontal face f=0..1:

```
destination tile page = (h//32)*4 + c
destination byte offset = (r//16)*2048 + (r%16)*64 + f*1024
source byte offset within native output tile c = f*1024
copy bytes = 64
```

These eight aligned writes/head transfer only the 128 useful FP32 output values. Different heads may share a destination tile page but own disjoint 64-byte segments; no read-modify-write or whole-page writer is needed. Give the final head's writer sole responsibility for zeroing rows H..round_up(H,32)-1 from an initialized 64-byte zero source. Never zero the whole final output page while other heads write their rows. Each write must honor the current NoC API's address/alignment contract, and writer barriers must precede source CB release. This removes the public output retiling plus the isolated probe's stable-output copy while preserving a preallocated output address.

CPU integer checks covered B1..32: every head has exactly one owner; every compact output row/face segment, including tail rows, is covered exactly once; all offsets are 64-byte aligned and within a 4096-byte tile. This is a layout proof, not a hardware NoC/write-order proof.

At B32 the compact prepared buffer is 324 KiB and compact output is 192 KiB. In contrast, fully padded native FP32 Q/K/V occupy 18 MiB, the two scalars 3 MiB, and native output 6 MiB. Writing and reading the padded inputs alone costs 42 MiB of DRAM traffic. A tiled-prepared candidate in DRAM therefore does not fairly represent the compact family. A prepared-L1 alternative is also coherent: roughly 21 MiB of expanded input spread over the device may be possible, but its coexistence with native CBs/other live allocations needs physical evidence and measured NoC costs.

## Direct BF16 persistent-state native adaptation

The public/device operation currently requires FP32 state. The kernel can instead have a T1 specialization with BF16 persistent input/output, while all decayed state, live update, matmul accumulation, and readout stay FP32. This family retains BF16 state continuously; it must not perform a full DRAM BF16-to-FP32-state conversion followed by a full reverse conversion each step.

The minimal datatype changes are:

- Host/descriptor: state input/output tensors stay `[H,128,128]` BF16; `cb_S` and `cb_state` use `Float16_b`, tile/page size 2048. Other math/output CBs remain FP32 with 4096-byte pages. Restrict shapes/dtypes/ownership explicitly. For the initial descriptor keep `cb_S=16` tiles and `cb_state=32` tiles, now saving 98,304 bytes/core against the FP32 factory.
- Reader: obtain state page bytes from its state CB/descriptor, separately from Q/K/V FP32 tile size or the compact record size. State accessor, NoC transfer byte count, and each destination tile stride must all use 2048. The current reader's shared `tb=get_tile_size(cb_q)` and its lambda's shared `t*tb` are both wrong after this change. Keep state read barrier before push and retain all 16 tiles before any writeback.
- Compute input: permit BF16 state to unpack into FP32 DST. For the existing scalar-broadcast decay, explicitly establish the BF16 state source format and FP32 decay source format before `mul_bcast_scalar_init`; restore FP32 operand formats before subsequent `mm(k,sd,...)`. `bcast.h` invokes `state_configure`/`llk_unpack_AB_init` for the selected CBs, but mixed-format support and the required explicit reconfiguration must be verified in a bounded hardware probe. The source fact that BF16 is exactly representable in FP32 does not by itself prove this configuration.
- Conservative input adaptation if the mixed-format broadcast fails: locally copy/unpack the 16 BF16 input tiles into an FP32 CB, then execute the unchanged native FP32 decay/math. This is on-core expansion of one head, not a full persistent-state DRAM conversion. It adds at most 64 KiB/core and a local traversal; compare it before rejecting the BF16 family. BF16 unpack uses the default BF16 route with FP32 DST, as the existing custom kernel does; do not blindly assign `UnpackToDestFp32` to a BF16 CB.
- Compute output: preserve a complete FP32 `nxt_S` for readout. A straightforward first correct specialization calls native rank-one update without its existing state emission, reads out from FP32 `nxt_S`, then rounds/copies its tiles into BF16 `cb_state`. In bounded DST groups: wait on FP32 `nxt_S`, reserve BF16 state output; acquire registers; direct-unpack FP32 source tiles to DST; invoke Blackhole `typecast_tile_init<Float32,Float16_b>` and `typecast_tile` for each tile; commit/wait; reconfigure pack to BF16; pack; restore FP32 pack; release. Push the complete state only after all tiles are packed, then pop `nxt_S`. The writer may already have emitted output while waiting for this state, which is safe. Use `UnpackToDestFp32` for this persistence-copy source so an extra FP32-to-TF32 truncation is not accidentally added before BF16 rounding.
- Writer: keep output FP32, but instantiate state accessor and state source strides with 2048-byte tiles. State-write barrier precedes popping BF16 state output; output's separate barrier/FP32 page stride remains unchanged.

The two-pass local state packing is a correctness-first BF16 branch with extra L1 traffic. A subsequent fused emission may pack FP32 `nxt_S` and BF16 persistence from the rank-one DST batch, but **must synchronize the FP32 pack before rounding/reusing those registers**. Calling SFPU rounding after `tile_regs_commit` without reacquiring/register synchronization is not a valid optimization. Likewise rounding shared DST before packing FP32 `nxt_S` changes this step's readout. Reuse the proven custom round-from-copy method only with its exact register ownership and integer RNE initialization, or add a separate persistence register batch. Native's four FP32 DST tiles/half-sync convention must be reconciled explicitly; copying the custom eight-register/full-sync helper verbatim is unsafe.

`optimized_decoder.py::_fused_delta_row_compact_fp32_compute_config` and the current custom recurrence show the applicable conventions: BF16 state CBs use default unpack; FP32 copies use direct unpack; state load temporarily switches source format; the native typecast establishes RNE constants; BF16 packing switches and restores pack format. That implementation performs FP32 live update/readout before persistent rounding. The shared typecast header has a Quasar-specific no-op branch, but this target is Blackhole: do not infer its rounding behavior from the Quasar comment.

BF16-specific proof risks are signed zero/subnormal behavior, RNE half-way cases, exact-versus-truncated copy semantics, stale format state after a BF16 operation, and output/state ordering. Test adversarial BF16 half-way values and cancellation in addition to PCC; source-format/packer mistakes can yield deceptively high aggregate PCC. Native FPU arithmetic itself remains TF32-class and may differ from the custom SFPU FP32 control even with identical state persistence. Do not weaken the existing accuracy gates.

## Traffic, memory, context, and expected opportunity

For H=12B, state bytes are `H*128*128*element_bytes`. A T1 update reads and writes each state element once:

| Family at B32, per rank | Persistent state | Minimum state DRAM read+write | Ideal state-only time at modeled 512 GB/s |
|---|---:|---:|---:|
| FP32 | 24 MiB | 48 MiB = 50,331,648 B | 98.304 us |
| BF16 | 12 MiB | 24 MiB = 25,165,824 B | 49.152 us |

These are modeled bandwidth floors, not achieved latency. The all-padded FP32 DRAM preparation family already requires at least 90 MiB for state plus writing/reading prepared inputs, or 184.32 us at that model, excluding output and raw-input reads. That comparison explains why removing public waves alone may fail. The compact FP32 family has a roughly 100-us traffic floor at B32 before math/launch overhead; its improvement over the observed 125-us custom recurrence may be modest. The compact BF16 native branch has more theoretical headroom, but must pay for native local CB traffic and any separate persistence pass. Neither estimate establishes a rejection or a speedup.

At B1 the state-only FP32 floor is 3.072 us; preparation/dispatch/output work can dominate. Report B1 and B32 separately, and B9/B10/B17 to expose the head-loop transition. A two-launch candidate trades an extra preparation launch against removal of repeated public wrapper/wave operations. Further fusing preparation into recurrence can be tried later, retaining grouped normalization and bounded per-core work; do not assume launch savings exceed duplicated normalization.

Retaining FP32 rather than BF16 state adds 384 KiB/user/rank/layer: 18 MiB/rank across 48 linear layers at B1 or 576 MiB at B32. These all-layer figures are capacity hand-off estimates, not permission to begin full-model work. Direct BF16 native state has no persistent-state capacity increase. Count prep/output owners, trace-pinned scratch, retained constants and allocator padding in physical peak memory. The current multichip context contract is 262144 with a nonaligned 262143 case. This operation changes no KV-cache layout, sequence padding, or inter-layer CCL. If selected FP32 state or scratch changes actual feasible context, revalidate and update `doc/context_contract.json`; this source audit does not change the contract or authorize a reduction.

## Fair experiment and acceptance sequence

1. Establish prepared FP32 recurrence at B1 with one head/core, comparing to the public native arithmetic/ring control and custom FP32 state. Validate actual g/beta and grouped mapping, independent FP64 reference, each head/user/rank output and persisted-state PCC >=0.995, and stricter existing stage gates where applicable. Include zero QKV, zero state, rank-distinct values and poisoned padding.
2. Add direct compact output and multihead counts at B3/9/10/17/31/32. Compare complete-boundary output/state, not only kernel output before packing. Tail rows must be deterministic. Make the compact preparation path and persistent ownership work before drawing performance conclusions from padded DRAM preparation.
3. Add direct BF16 state as a matched family: compare BF16 native to BF16 custom and an independent reference rounding state after every update while keeping current-step readout FP32. Keep FP32 native/custom controls. If mixed-format decay or fused output rounding fails, adapt with the local expansion/separate packing described above and retry; the first API/compiler error is not a rejection.
4. Keep input/prepared/output/state/zero scratch and any bank tensors allocated before the first capture. Trace commands bind all mutable addresses. Test fresh-owner program-cache rebinding while old owners remain unchanged, changed inputs and changed state on replay, 50 queued changed updates without host drains, and 512-update drift to the reference. Run watcher separately from profiler. Allocation gate remains `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, including a deliberate postcapture-allocation positive control that checks rejection without replaying a corrupt trace. No acknowledged-corruptible scope is acceptable. The previous public probe's postcapture-output allocation failure is directly relevant.
5. After correctness, compare warmed complete-boundary traced timings in alternating order against custom BF16, custom FP32, and public native FP32. Record all preparation/native/output operation tables, CSV, kernel/source fingerprints, allocation peaks, and state dtype. Then integrate only the winning coherent branch and rerun real-weight TP4 decoder PCC, prefill-to-decode state transitions, nonaligned context support, whole-step traced decode, runtime-fallback audit and final default-path metrics. No isolated microbenchmark is a stage acceptance.

No option is rejected here. The public-wave measurements, when final, remain valid measurements of that control. The compact multihead FP32 and direct BF16 persistent families remain concrete candidates until tested or until a qualifying stage stop condition is reached with explicit hand-off notes.

## Source provenance

Inspected native files are under `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/`: public `.cpp`; device operation and program factory `.cpp`; compute kernel; dataflow reader/writer. Also inspected `tt_metal/hw/inc/api/compute/bcast.h`, `tile_move_copy.h`, `eltwise_unary/typecast.h`, and the current model's `optimized_decoder.py`, `multichip_decoder.py`, frozen `native_recurrent_candidate.py`, probe outputs and context contract. CPU checks used only Python standard-library integer arithmetic/JSON parsing. The existing smoke/wide JSON artifacts carry exact helper/probe/native source fingerprints; this audit makes no new hardware provenance claim.

Current source SHA-256 values (audit-time files, including working-tree changes):

```text
d404654b8160297504b6f769a74c05ed596c555111b354d312f26f5b3fb2589d  ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/fused_recurrent_gated_delta_rule_program_factory.cpp
08fd7439a9196a2d6b76ae18f505c4dfa1e77035f33f5569834f43d30454647b  ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp
8aab756a78118605090fb76a96725575d5226be666a70034aa15cb2aba2e7179  ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/dataflow/reader_fused_recurrent_gated_delta_rule.cpp
4c31dd8bcc449e77720e0f0a49ba1d49b8730adcf4597afd4ee6080bcbe30404  ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/dataflow/writer_fused_recurrent_gated_delta_rule.cpp
56d6c5ab025a9da54d91333187148d965fb042dcb31b43494ff9c9c083b4cde6  tt_metal/hw/inc/api/compute/bcast.h
11d605f85b0cb205620dc55ee6a24cd067635f9fe6ef7c4ef68478025526614f  tt_metal/hw/inc/api/compute/tile_move_copy.h
ccac16ae4ad9803b5921c265e9b338c0141375bc8ecd0b89531dec1f6c970763  tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h
315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22  models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py
```


## Isolated implementation hand-off (2026-11-13)

Implemented only `tt/prepared_native_recurrent_candidate.py` and `tests/probe_prepared_native_recurrent.py`; no production or native source changed. These files are frozen pending the parent's bounded hardware run. No kernel build, TTNN/Torch import, or device execution was performed by the implementing subagent.

The helper launches exactly two generic programs per step. Preparation uses one core/user, reads packed BF16 QKV plus tiled g/beta, applies FP32 SFPU grouped normalization/Q scale/exp, and writes a compact 2592-float record. Recurrence uses the native helper functions verbatim, a balanced outer head loop, and direct compact output. Both FP32 and BF16 state owners retain their state addresses; output/prepared buffers exist before capture. BF16 uses an explicit 16-tile local expansion (CB14 BF16 -> CB5 FP32) and a separate local direct-unpack/typecast/pack pass after FP32 readout. The native final CB8 is popped between independent heads. No persistent state conversion or public-wave/padded-input restoration occurs in the timed prepared path.

Per-core recurrence CB totals are 540672 bytes for FP32 and 507904 bytes for BF16, including reader scratch and writer zero scratch. Preparation uses 65536 bytes/core. Prepared storage at B32 is 331776 bytes/rank; compact output is 196608 bytes/rank. These are source-accounted sizes, not measured physical allocator peaks. One-core/user preparation and the separate BF16 persistence traversal are deliberate correctness-first choices; measure their profiler contribution before treating a slow first version as a family rejection.

The probe keeps five paths: custom BF16, custom FP32, public-native FP32 ring, prepared-native FP32 and prepared-native BF16. `--custom-variant` defaults to `cumulative_exp_scalar` and is passed explicitly/recorded, so ranking uses the current custom implementation. `baseline` remains an explicit diagnostic option. Output and state PCC are checked independently per user and per head on all four ranks against FP64 reference arithmetic with explicit persistence rounding, then against both custom controls. Prepared records are checked separately for Q/K normalization, exact V/beta, decay, and zero padding. The probe covers eager repeated updates, new-owner cache rebinding with old owners unchanged, changed-state-and-input replay exact to own eager, 50 queued changed updates, allocation-tracker rejection positive control, and alternating-order complete-boundary traced timings. It records source hashes. No acceptance gate was weakened.

CPU validation passed: Python AST parsing, B1..32 head partition coverage, compact record offsets, unique/aligned complete output face ownership including tails, independent tile-row output roundtrip, CB byte arithmetic, and exact reuse of the native helper-source prefix. The validation process asserted that neither `torch` nor `ttnn` entered `sys.modules`. C++ compilation, NoC behavior, BF16 format transitions, numerical gates, tracker/watcher gates and latency remain unverified until hardware execution.

Initial bounded command, to be run only by the parent's serialized device queue (capture stdout/stderr and command provenance alongside the JSON):

```bash
env -u TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_prepared_native_recurrent --batches 1 --steps 2 --bank-size 2 --warmup 1 --queued 0 --skip-timing --custom-variant cumulative_exp_scalar --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prepared_native_b1_smoke_v1.json
```

After a clean bounded run, the quality command expands to `--batches 1 3 9 10 17 32 --modes normal zero_qkv zero_state --steps 3 --bank-size 3 --queued 50 --skip-timing` under watcher/tracker, with a fresh artifact name. Add B31 for tail stress and `--steps 512` for long drift. Separate uninstrumented timing unsets watcher/profiler and removes `--skip-timing`; use a fresh artifact name and preserve all five complete-boundary comparisons. Parent owns further experiments and stage integration.

Implementation source SHA-256 at hand-off:

```text
0e5991945bc595b784ece95a7c04db1395aa3abf81cab9f83b5b859b576d96f4  models/autoports/qwen_qwen3_6_27b/tt/prepared_native_recurrent_candidate.py
37ad5114114719750faedf746074ec2905bdf0215dfce97c51d6e6857c69e633  models/autoports/qwen_qwen3_6_27b/tests/probe_prepared_native_recurrent.py
```
