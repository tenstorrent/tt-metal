# TP4 recurrent readout epilogue fusion audit

2026-11-13. CPU/source inspection only. No TTNN/torch imports, device execution, candidate timings, or source edits. This report is the only file written by this audit. Scope: the local 12-head GDN decode epilogue; no recurrence scalar/column sweep is repeated.

## Conclusion

A bounded fusion is feasible. Every recurrent head owner computes all 128 channels of its readout, and RMS normalization is independent per head. Neither a standalone fused epilogue nor later integration into recurrence needs a cross-core reduction or a mesh collective. Keep the current BF16 persistent-state update completely unchanged.

The recommended first experiment is **one standalone generic operation replacing FP32→BF16 typecast, unweighted RMSNorm, and local affine/gate/repack**. It isolates the numerical and compact-output contracts from recurrence scheduling. Only after that passes should the same epilogue consume recurrence-local CB data. A first compile/API failure is not evidence against this family: adapt CB bindings/formats and retain the unfused device oracle.

The existing parallel Q/K norm helper provides useful plumbing and a candidate mathematical implementation, but is **not an exact native RMS replacement proved by its current tests**. Its tests require PCC ≥0.995; exact checks apply to unchanged inputs and trace reuse, not native versus candidate output. A head128 specialization is mechanically straightforward but needs new numerical evidence and additional BF16 rounding boundaries.

## Inspected source pins

Paths are relative to `models/autoports/qwen_qwen3_6_27b`:

| Source | SHA-256 |
|---|---|
| `tt/multichip_decoder.py` | `4c56cbadb7768054100f921ced79d725a57bfa477beca9648f8f2d932e9bd765` |
| `tt/optimized_decoder.py` | `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22` |
| `tt/local_affine_gate_candidate.py` | `807dd82ff67dbce95dafe6d327be2f77a02554d3157e636dbe33f46f23512048` |
| `tt/parallel_qk_norm_candidate.py` | `a963453458436b777c924c280168126ed91f6d8846eff6a8a1d3a0a1867dde24` |

Key sites: `_gdn` at multichip line 853; `_local12_recurrent` at 1120; frozen recurrence writer at optimized line 4935, compute main at 5424, affine reader/compute at 5705/5795. Native RMS implementation is `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp`, factory `device/layernorm_op_multi_core.cpp`, and `normalization/kernel_util/compute/numeric.h`. Epsilon construction is `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp`. Line numbers identify this inspected state; hashes own the model-source provenance.

## Exact current numerical contract

For user `u`, local value head `h`, channel `c`, let `r` denote the FP32 recurrence readout. Current decode performs:

1. Compute `r` using the live, **unrounded** FP32 state update. Independently BF16-round and persist state. Do not read rounded persisted state back for the current output, and do not change recurrence products/additions or grouped Q/K reuse.
2. Explicit FP32→BF16 typecast of `r`, producing `x`. This is a separate rounding boundary, not just an output dtype annotation.
3. Native unweighted RMSNorm over precisely 128 channels of `x`, epsilon supplied by `config.rms_norm_eps`, HiFi4, accurate math, FP32 destination, packer L1 accumulation enabled. Output is BF16: `n`.
4. FP32 multiply of widened `n` and BF16 gamma, then explicit BF16 rounding. The accepted affine helper uses `sfpu_mul_bcast_row<true>` followed by `typecast_tile<Float32,Float16_b>`.
5. Read BF16 gate `z` from packed projection columns `[2560,4096)`. Widen to FP32; apply `silu_tile<true>` with the same init; multiply by widened rounded affine value in FP32; explicitly BF16-round the result.
6. Emit compact BF16 TILE `[1,B,1536]`, physical `[1,32,1536]`, with head order `h*128+c`. The output projection consumes this exact local-head order.

The norm boundary before gamma and the affine boundary before SiLU multiplication must both remain. In particular, combining gamma into an otherwise FP32 norm intermediate changes the current GDN path even if the mathematical expression looks identical.

### Native RMS details that must not be assumed equivalent

The current interleaved RMS factory selects block size 4 for FP32 destination. Width128 is exactly four tiles, with no width tail. `ckl::square` uses FPU multiplication of BF16 `x`, packs FP32 squares, then `numeric::row_wise_mean` calls `reduce_init/reduce_tile` in increasing tile order 0,1,2,3 before scaling by exact FP32 `1/128` (`0x3c000000`). This is not the recurrence's SFPU reduction helper. Native `FLOAT32_REDUCTION` is true by factory policy, but `accumulate_compute_loop` still calls the FPU reduce API; its name is not proof of an SFPU tree.

The mean is packed FP32. Native epsilon CB is BF16 and `generate_bcast_col_scalar` stores `epsilon_bits >> 16`: **truncation of the input FP32 epsilon's high 16 bits**, not full FP32 epsilon and not host nearest-even BF16 conversion. Native then performs FPU add of mean plus epsilon, exact SFPU rsqrt, FP32 inverse pack, and FPU column-broadcast multiplication with original BF16 `x`, packing a BF16 output.

For exactness, transplant these operations/bindings or a demonstrably identical lowered sequence. Do not replace them silently with a sequential sum, SFPU fold, full-precision epsilon, rsqrt polynomial change, or a fused multiply/add. Direct FP32 unpack settings must match each consumed FP32 scratch CB. An explicit integer RNE cast is already available in the affine helper, but replacing native output packing with that cast still needs tie/subnormal/NaN controls.

### Comparison with parallel Q/K helper

`parallel_qk_norm_candidate.COMPUTE` folds eight squared tiles elementwise with SFPU mul/add, then SFPU row-reduces once, scales by `1/256`, adds full FP32 epsilon, rsqrts, normalizes and multiplies gamma in FP32, and rounds only its final output. Its docstring explicitly says native reduction order may differ.

A width128 candidate changes eight tiles to four and scale to `0x3c000000`, reads four gamma tiles, and adds (a) initial FP32 readout→BF16 round, (b) norm-output BF16 round before gamma, and (c) affine-output BF16 round before gate multiplication. Epsilon should match the native truncated value for the closest comparison. It must also replace the Q/K helper's sharded L1 input ownership with the recurrence DRAM readout mapping below. Consequently this is a small candidate design, **not a literal safe width substitution**. It can be a useful alternate after the native-arithmetic control, subject to unchanged stage gates and reported bit differences.

## Standalone fused epilogue ownership

Input `r`: FP32 TILE DRAM `[1,B*12,128]`. Packed projection: BF16 TILE DRAM `[1,B,P]`, `P>=4096`, padded width multiple32. Gamma: BF16 TILE DRAM `[1,1,128]`. Output: BF16 TILE DRAM `[1,B,1536]`. Limit initial candidate to actual compact decode `1<=B<=32`; preserve the existing noncompact/one-token prefill path unchanged until explicitly adapted and tested.

For user `u`, head `h`, channel tile `j` in 0..3:

- Readout source row `rh=12*u+h`, tile `(rh//32)*4+j`, FP32 face-row byte offset `(rh//16%2)*2048 + face*1024 + (rh%16)*64`.
- Assemble a worker-local tile with user in row `u`; this preserves per-row arithmetic while changing only rows' physical ownership.
- Packed gate source tile `80+4*h+j`; physical user rows already match the desired output.
- Gamma source tile `j`, row0, broadcast over users.
- Output tile `4*h+j`; writer can issue the existing affine writer's full 2048-byte tile write. All users share this single tile owner. Zero logical padding rows `B..31` explicitly before publication.

Two coherent launch families deserve comparison:

1. **48 existing affine tile owners:** owner `(h,j)` reads all four readout channel tiles, computes all-channel statistics, emits only channel tile `j`. This keeps the existing affine output ownership and full-tile writer, but repeats readout reads/squares/reduction four times per head. It is a clear low-risk correctness control, not an assumed winner.
2. **12 head owners:** owner `h` reads four channel tiles, computes statistics once, emits four complete compact tiles. This eliminates the repeated statistics and reads and preserves full-tile exclusive output ownership. More heads per worker can cover smaller grids, but default12 workers suffice on the target; B32 fills all rows of each tile. This alternative should not be rejected using only the redundant 48-owner control.

Use one private output allocation or caller-supplied validated output. Runtime args must rebind every input/output pointer on program-cache hits. No module-global tensor cache, no alias with readout/gamma/packed, no reuse before the last writer barrier. CBs may have fixed capacities for four FP32 readout tiles, four BF16 rounded inputs, four FP32 squares, FP32 statistics/inverse, gamma/gate, and BF16 output; compute exact footprint before choosing buffering. The retained four rounded input tiles are needed after statistics, so do not pop them during the square pass. Reconfigure copy/FPU/packer formats explicitly when moving between FP32 and BF16 CBs.

## Later recurrence integration

The recurrence main holds eight FP32 DST tiles per value-column and releases them after each of four columns. There is **no room to retain all readout columns in DST during recurrence**. Store readout to a private four-tile CB (current CB12 already serves that role), then run the proven epilogue only after column3. This changes no recurrence arithmetic. Ensure the writer no longer consumes/pops CB12 while compute is using it; allocate a distinct final BF16 epilogue-output CB and publish it after normalization/gating.

Keep state CB11 publication and writer progress independent of final epilogue where feasible. A minimal safe version retains current state/output synchronization and only replaces the final output wait; later state-first writer overlap must be measured separately. The current writer comment claiming compute rereads updated state is stale; readout is already complete in live FP32 DST.

Head owners now emit **row slices** of compact tiles: output tile `4*(head%12)+j`, row `head//12`. Distinct heads/users own disjoint byte ranges. Unlike current FP32 readout's 64-byte face rows, BF16 output face rows are 32 bytes; validate NoC source/destination alignment and byte-enable constraints. Stage rows in matching-parity scratch if required. Never write a whole tile from an individual head owner, because adjacent users may belong to another core. Either retain head-to-user compact scatter with explicit zero ownership for each output tile's padding, or use a model-local handoff/collector; do not introduce a global mesh gather. Full-tile output ownership makes the standalone version simpler and may remain faster.

Reader must fetch gamma and gate for the correct `(user,local_head)` while preserving existing Q/K grouping predicates and raw input handshake counts. Existing Q/K normalization and persistent state dtype/layout remain unchanged. Do not feed noncompact `[B,1,P]` packed input into the compact indexing contract without an explicit alternate reader path.

## Cost and limitation evidence

`perf/interim_profile4_l0_b32_s33/decode_perf_report.txt` reports the earlier recurrence at about125 us and the post-recurrence chain as typecast~2 us, RMS~5 us, gamma multiply~3 us, reshape~11 us, slice~2 us, two typecasts~2 us each, gated multiply~4 us. These rounded rows are historical **existing measurements**, not a new forecast. The current `local_affine_gate` candidate has already replaced the gamma/reshape/slice/typecast/gate family. Do not count their removal again or claim the old chain as the current baseline.

A standalone fused epilogue can remove two launches relative to current typecast + RMS + local-affine (three to one), intermediate DRAM writes/reads for rounded readout and normalized readout, and their allocation lifetimes. At B32 the logical rounded readout and normalized readout are each 98,304 bytes per rank; eliminating their complete write/read round trips is 393,216 bytes per rank, excluding scratch/gamma/gate traffic and row-read overfetch. The 48-owner control can reintroduce more readout traffic than it removes, hence the paired12-owner test. Integration additionally removes the FP32 readout global write/read, but changes writer occupancy and may serialize an epilogue after recurrence on the same cores. Neither the launch count nor saved bytes proves a speedup. No reduction in the dominant 24 MiB/rank/token state read/write is claimed.

## Existing dedicated operations

`transformer.fused_recurrent_gated_delta_rule` is not a direct fused epilogue implementation. Its public wrapper rejects `use_qk_l2norm`, expands grouped Q/K via repeat_interleave, normalizes/scales/reformats externally, computes exp externally, and the device op requires all FP32 TILE inputs including state. It returns FP32 output, without RMS/gamma/SiLU gate. Adapting it would change state storage and add conversions rather than provide the bounded epilogue fusion; this source fact is not a timing rejection of a future specialized version.

`transformer.gated_delta_attn` consumes prepared triangular/inverse/intra-attention chunk tensors, requires chunk size128 and FP32 tensors, and produces FP32 chunk output/state. It likewise provides no readout norm/gamma/gate epilogue. It is a different prefill-oriented topology, not a drop-in one-token local12 replacement. The current chunk_gated_delta_rule prefill path stays outside this audit.

## Minimal implementation and discriminating validation

1. Add an opt-in model-local standalone helper with the exact contracts above and output reuse validation. Keep default unchanged. Start with native arithmetic and48-owner control; implement12-owner family in the same helper or separate named variant.
2. Frozen device oracle: `typecast(r,BF16) → rms_norm(no weight,current ckc/epsilon) → local_affine_gate`. Also compare against the original unfused affine/gate sequence to isolate any prior-helper discrepancy. Record intermediate rounded-x, norm, affine, and final output bit differences, maximum error, per-user/head/rank PCC and finiteness. For an exact candidate require bitwise output equality before performance claims.
3. Use all B1..32 for addressing; short diagnostic B1/2/3/17/31/32 initially. Include actual captured FP32 recurrence readouts and real gamma/packed gates, rank/user/head/channel-coded synthetic inputs, zero and near-zero rows, mixed magnitudes, signed zeros, BF16 halfway ties, and epsilon values whose low16 bits are nonzero. Poison unused padding with finite/NaN/Inf values and assert useful rows cannot observe it. Nonfinite stress is a liveness/bit-contract probe, not model PCC evidence.
4. Rebind every input/output with same shape and different addresses under program cache; retain and poison retired buffers; verify no input mutation, exact changed-input eager/trace output, output padding, and sentinel neighbors. Repeat≥50 queued calls and traced reuse. Use watcher+allocation tracking separately from timings.
5. Measure matched warmed/traced whole GDN decoder for48-owner,12-owner,native-chain controls at B1 and B32, plus both short/long prefill public contracts. Profile the selected complete path, including output-projection reshard cost. A faster isolated helper is not sufficient if allocation/sharding or following matmul loses.
6. If exact native arithmetic fails, debug intermediates in order: input RNE, four-tile square, ordered reduction, epsilon truncation, rsqrt, norm BF16 pack, gamma RNE, gate arithmetic, output routing. A failed SFPU-tree variant does not reject native arithmetic. If CB/Metal2 wrapper incompatibility blocks compile, use legacy-equivalent FPU APIs with the same reduction tree and explicit buffer descriptors rather than abandoning the material fusion.
7. Only after standalone validation, integrate proven epilogue after recurrence column loop. Require **bitwise persisted state unchanged** relative to the selected recurrence across repeated tokens, state BF16/FP32 supported interfaces, low-core wraps, grouped-QK boundary splits, changed-input trace, reversed users, and zero states. Run current long-step full-layer PCC/cache-state/stress/watchers unchanged. Update graph audit and runtime-helper audit for every reachable new helper.

This report establishes a source-level implementation path and falsifiable tests; hardware feasibility, exact equivalence and latency remain unverified.

## Follow-up: coherent head-major family measured and rejected (2026-11-13)

This section supersedes the earlier hardware-unverified status for the implemented
standalone and head-major candidates, not for the further proposals below.
The parent ran the probes; this follow-up only inspected source and JSON on CPU.

The compact-layout standalone fusion was exact but slower (`norm_gate_perf16`):
B1 native/head/tile = 14.055/34.052/32.796 us; B32 =
24.959/41.179/58.051 us. It was therefore adapted coherently instead of rejected
solely for its old head-strided input reads. The recurrence writer now emitted
FP32 `[12,B,128]` directly, and the fused norm/gate read four whole tiles per head.
There was **no restore to the old compact FP32 readout** between producer and
consumer. Both 12-head and 48-output-tile ownership were measured. State and
recurrence arithmetic stayed unchanged; final BF16 output retained the existing
compact projection-consumer contract.

[head_major_wide19.json](head_major_wide19.json) records 258 passing exact cases
on mesh `[1,4]`: B1..32, BF16/FP32 state, normal/zero-QKV/zero-state/scale modes,
plus one-core wrapping, input/address rebinding, 50 queued calls and changed-input
traces. Watcher and allocation tracking were enabled; timings were skipped.
Output, readout and persisted state had zero bit mismatches in the recorded
comparisons. This establishes tested equivalence, not a universal numerical proof.

[head_major_perf19.json](head_major_perf19.json) separately records warmed traced
**complete recurrence + epilogue chains**, BF16 state, cumulative_exp_scalar,
seven samples of 100 replays, with watcher/profiler/allocation tracking disabled.
All paths perform matched state updates; no reset or restoration is inside the
timed trace. Native means current recurrence + typecast + native RMS + accepted
local affine/gate, not an unoptimized historical chain.

| Logical batch | Native chain us | Head-owner chain us | Tile-owner chain us | Head delta us | Tile delta us |
|---|---:|---:|---:|---:|---:|
| 1 | 63.26191 | 72.17776 | 70.41669 | +8.91585 | +7.15478 |
| 3 | 67.80740 | 78.01131 | 76.15867 | +10.20391 | +8.35127 |
| 17 | 103.66233 | 113.33483 | 111.75641 | +9.67250 | +8.09408 |
| 32 | 149.64061 | 154.90267 | 153.02419 | +5.26206 | +3.38358 |

Reject these two coherent head-major implementations for selection: both lose
at every measured batch despite eliminating two launches and intermediate
BF16 tensors. B1 expands FP32 readout allocation from 16 KiB to 192 KiB; B32 is
192 KiB in both paths. That explains an additional physical cost at B1, but
is not proof of the measured bottleneck. This is a complete-chain rejection,
not a measured whole-decoder regression or rejection of every possible fusion.
The candidate helper SHA256 is
`1632adca8b0faa477eaa5bccc3718b4a0c8d5ec02ce30078c9404b488c259643`;
probe SHA256 is
`0e7a517d159802ea8a0febba8e7427ebf7272fa586cd5cb6531bcc235e2f3bcd`.
The JSON records all other input source hashes and trace-contract details.
Helpers and probes remain frozen.

## Two remaining custom-recurrence candidates, not measured

The current integrated14 profile identifies recurrence at approximately125 us.
No per-core phase timing proves how much of that time is preparation, SFPU work,
DRAM service or waiting. BF16 state read/write remains 24 MiB/device/token at B32
(the nominal512 GB/s state-only floor is49.152 us). Neither proposal below
reduces those bytes; neither is a predicted speedup. Both preserve the existing
local12 owner, state layout and inter-layer residual contract. Prepared native
recurrence is being investigated independently and is deliberately excluded.

The completed cumulative reader/scalar changes, value-column/pair sweeps,
state-streaming smoke, and native-BF16-pack precision candidate are not new
recommendations. In particular, `recurrent_column_candidate.md` already records
an exact but non-improving streaming smoke, and `work_log.md:157` records native
packing's persisted-state bit failure plus a separate full-layer PCC pass.

### 1. Restrict grouped Q/K preparation arithmetic to its live row

Source anchor: frozen `tt/optimized_decoder.py:5379-5420`, `prepare_vector`,
called at local-core entry or global triple boundaries from lines5454-5462.
The current code squares four full FP32 tiles, folds them in order0,1,2,3,
performs native SFPU REDUCE_ROW, adds1e-6 and rsqrts a full tile, then performs
four full-tile normalize/optional-Q-scale passes and packs them before
transposition. Only original row0 ultimately becomes each prepared column.
The earlier AutoDebug liveness proof establishes that row-reduction lanes are
independent and the transpose/broadcast bit mask excludes dead original rows;
this does not imply that an arbitrary new reduction tree is equivalent.

Bounded first implementation: preserve every CB, pack, transpose, prepared-column
expansion, grouping predicate and the existing REDUCE_ROW helper. Replace only
pointwise square/fold and normalize/scale with explicit live-row lane operations,
and restrict scalar add/rsqrt to the consumed norm lane using the same accurate
primitive. Preserve four separate FP32 squares, three ordered additions,
full FP32 epsilon0x358637bd, exact existing rsqrt, normalization multiply, then
separate Q scale0x3db504f3. Do not substitute FPU/TF32 contractions or combine
normalization and Q scale. Leave original reduction and transpose in place first;
removing the intermediate four-tile pack is a later change requiring a different
register/transpose proof.

A logical32x32 input tile contains one useful row: pointwise work covers
32 useful values instead of1024, but lane granularity, loads/packs, reduction,
initialization and synchronization remain. B32 currently prepares164 Q/K pairs
per rank, including core-boundary fragments; this work is repeated enough to
merit a bounded control. No claim that preparation or total recurrence improves
32x follows. For B1 there are12 pair preparations across12 active cores.

Risks/tests: live row0 occupies both upper faces and several SFPU subrow slots;
`VectorMode::R` alone must not be presumed equivalent to precisely row0. Preserve
native lane mapping and native reduction input bits; compare dumped prepared Q/K
bits before recurrence, including zero vectors, magnitude/epsilon boundaries,
poisoned unused padding, all32 batches and split triples/one-core wrapping.
Then exact output/state at repeated steps, 50 queued calls, rebinding and changed
traces, followed by matched recurrence and full-layer timings. This only changes
custom recurrence preparation, not the independently investigated native-op
adapter. Its likely benefit is bounded preparation overhead, not a demonstrated
solution to the entire roofline gap.

### 2. Fold all four prediction segments while retaining accumulators in LREGs

Source anchor: `_gdn_decay_predict_` at optimized lines5123-5174 and its
four-call m0..3 loop at5490-5500. For every value-column, the four state tiles
are already simultaneously resident in DST0..3. Yet each segment traverses the
whole tile separately, writes all prediction accumulators to DST6, and the next
segment reloads them. This is distinct from changing value-column/core ownership.

Bounded first implementation: only change the prediction helper and its four
calls to one helper. Traverse `(half,band)` outside, and ordered m0..3 inside.
Retain the four lane-vector accumulators in four LREGs while two work LREGs,
one prepared-K register and the existing decay register cover each state's
separate decay/store/product/add. A concrete budget is ACC=LREG1..4,
K=LREG0, work=LREG5/6, decay=LREG7. Load the native scalar before starting,
initialize accumulators directly from m0 products (never zero-plus-product),
then preserve the existing operand order of each SFPMUL and SFPADD at m1..3.
Write each decayed state back to its original DST tile before overwriting work
registers; store the four completed accumulators once at the end of the band.
Leave the existing REDUCE_COL, delta, update/readout/RNE helper, state packing,
CB publication and writer byte-identical. No extra inter-core communication,
CB space, global tensor or NoC transaction is required.

Static opportunity: the existing four segments issue16 accumulator stores and
12 accumulator loads per band; the new schedule needs4 stores. Thus it removes
24 accumulator DST accesses per band, eight bands/value-column and four columns:
**768 explicit SFPU DST loads/stores per head**. This is an instruction-count
bound, not removed DRAM bytes or predicted cycles. First-segment register moves,
load/use latency slots, loop/control overhead and reduced instruction overlap
can offset it. It removes only prediction accumulator traffic; claiming the
same saving for update/readout would require an additional register-pressure
proof because live unrounded updates and two integer-RNE scratch registers
must survive there.

Arithmetic/ownership risks: verify retaining an SFPU result across segments is
bit-equivalent to current FP32 DST store/load, including signed zero/subnormal
behavior. No FMA, reassociation, vector reduction or rounded-state readout is
allowed. Preserve scalar LREG7 and do not call initialization/replay helpers
that overwrite the allocated accumulators. Verify instruction dependency gaps
for loads and SFPMUL/SFPADD using Blackhole primitive rules; do not assume the
old interleaving remains valid after loop interchange. Current state ownership
and pack indices are unchanged, so this introduces no new global write race.
The register constants and eight-band count come from
`ckernel_sfpu_binary_bcast.h:110-152`.

Discriminating validation: export prediction DST6 immediately before the
unchanged REDUCE_COL as a diagnostic and compare bits to the current helper,
then compare reduced prediction, final output and persisted BF16/FP32 state.
Use ordinary, zero, cancellation-heavy, mixed-magnitude and tie-adjacent inputs;
all batches, grouped-boundary/one-core controls, address rebinding, queued reuse
and changed-state traces. Only after watcher/tracking-clean exact checks,
measure warmed isolated recurrence and the default coherent whole GDN chain.
If the first bounded prediction transform loses, retain that evidence before
considering the harder update/readout counterpart.

No candidate code, hardware run, default-path change or hidden-norm helper edit
was made for this follow-up. These are two source-supported experiments with
explicit correctness risks; neither has measured feasibility or performance.
