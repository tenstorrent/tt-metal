# AutoDebug: TP4 local12 compact recurrence

Inspection only, 2026-11-13. No device use, source changes, launcher invocation, or candidate timing. This report and `static_audit.json` are the only investigation artifacts written here.

## Findings

**There is concrete redundant reader work, but no demonstrated correctness bug or proven single cause of the 133.811 us latency.** The best small independent experiments are:

1. Stop zeroing dead raw-input padding: a conservative V/g/beta-only control removes **48 KiB and 96 local NoC requests per core**; the independently checked Q/K extension removes **all 112 KiB and 224 requests per core**, without changing useful arithmetic.
2. Stop fetching/widening raw Q/K that compute immediately discards: removes **440 DRAM reads, 112,640 bytes and 56,320 widening iterations per rank at B32**.
3. Cache gate scratch by user: removes **548 DRAM reads and 26,304 bytes per rank**.
4. Evaluate accurate exp only over the vector containing the consumed scalar, retaining the same exp implementation and initialization.

Separately, the active compute no longer consumes zero/one CB contents; deleting their initialization and matching waits/pops removes **2,048 scalar stores / 8 KiB per core**. Preserve their descriptors/startup format selection initially. State/output publication can be streamed by value-column, but that is a larger protocol change with uncertain benefit.

The compulsory BF16 state read/write is **24 MiB per chip per token**. At the parent's nominal 512 GB/s ceiling, that alone takes at least **49.152 us**. Saved reader bytes are not equivalent to saved latency: movement and compute overlap, and measured RISC spans include waits.

## Frozen source and observed execution

All source line numbers below refer to commit **8502f42bca20dcad1507fbc8cebf7032220b7d8e**, inspected through `git show`, not the concurrently changing working file. `082c82fa` is the prefix of the multichip file's SHA-256, not a Git revision:

- `tt/multichip_decoder.py`: `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`.
- `tt/optimized_decoder.py`: `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.

Both match `doc/multichip_decoder/perf/delivery_profile_retry_l0_b32/metrics.json`. Short names M and O below mean those two Python files.

**Direct observations:** `decode_perf_report.csv:37` identifies GenericOpDeviceOperation, 110 cores, 133.811 us, global call count 2003971. The raw `ops.csv` maps it to recurrent program hash 4846907665546838252, trace 0, replay session 10, device 3. Other ranks take 133.643/133.287/133.536 us. It is the slowest rank of the separately signposted replay, following the five timed replays (`tests/run_multichip_decoder.py:402-411`), not an average of their recurrent timings. The compute hash 17669065949326256812 matches the final optimized single-chip recurrent record; the local reader hash differs. The raw report exposes no separate writer hash; unchanged writer source follows directly from M imports/call configuration.

The selected invocation is M:675-676, with no `cores`, `normalized`, or `grouped_qk` override:

| Operand | Concrete B32 per-rank representation |
|---|---|
| Q/K/V | Same aliased BF16 ROW_MAJOR buffer `[32,1,2560]`, packed Q512/K512/V1536 |
| g | FP32 TILE `[32,1,12]`, padded `[32,32,32]` |
| beta | BF16 TILE, same logical/padded shape as g |
| persistent state | BF16 TILE `[384,128,128]` |
| output | FP32 TILE `[1,384,128]` |
| placement | DRAM interleaved for every operand |

These values are visible in raw profiling metadata and agree with M:632-676,869-940. B32 here is 384 local value heads, not 32 recurrent work items or 1,536 heads per chip. The grid is 11x10. `GROUP_QK=true`; all active recurrence-fusion flags are true (O:4807-4825,5432-5442). FP32 direct unpack, full DST synchronization, accurate math and separate persistence rounding remain part of the contract.

The accepted instrumented result has output PCC 0.999071, minimum per-user decode PCC 0.996754, state PCC 0.999434 and changed-input trace equality to eager. This establishes an existing numerical baseline; it does not validate any proposal here. Watcher and allocation tracking were disabled in this particular profile.

**Interpretation limits:** BRISC/NCRISC/TRISC0/1/2 spans are 133.802/93.520/127.950/130.184/130.439 us. They contain synchronization and CB waits, and per-core spans/wait metrics are absent. They do not prove a compute bottleneck, a reader bottleneck, or a 3.37 us ceiling for writer-overlap improvements.

## Ownership and buffer ledger

M:801 maps `h` to `user=h//12`, `hv=h%12`, `hk=hv//3`. On mesh rank r, global Q/K head is `4*r+hk` and global value/state head is `12*r+hv`. Packed row reads use byte offsets `256*hk`, `1024+256*hk`, `2048+256*hv` within the user's 5,120-byte page (M:761-763,811-813). State tile `(m,n)` is page `16*h+4*m+n`. Output page is `(h//32)*4+n`, row `h%32` (O:4952-4959). All three sibling value heads use identical Q/K, but distinct V/g/beta/state.

The partition assigns cores 0-53 four heads each and cores 54-109 three heads each (M:850-866). It covers every head exactly once. It produces **164 Q/K fragments**: 128 natural triples plus 36 core-boundary splits. Every four-head core prepares two Q/K pairs; every three-head core prepares one. Example: start=4,count=4 processes heads 4,5,6,7, with preparation at 4 and 6, not at 4 and 7.

| CB | Capacity / steady-state producer-consumer behavior |
|---|---|
| 0/1 raw Q/K | 8 FP32 tiles each; reader pushes 4/head; compute prepares or discards 4/head |
| 2 raw V | 8 FP32 tiles; push/pop 4/head |
| 3/4 g/beta | 2 FP32 tiles each; push/pop 1/head |
| 5 input state | 32 BF16 tiles; read/push/wait/pop 16/head |
| 11 persisted update | 32 BF16 tiles; reserve/pack/push/wait/write/pop 16/head |
| 12 readout | 8 FP32 tiles; reserve/pack/push/wait/write/pop 4/head |
| 13/14 prepared Q/K | 1 FP32 tile each, retained through the local part of a global triple |
| 6/7 zero/one | 1 FP32 tile each, initialized once, waited at startup, popped at exit |
| 15 scratch | 1 FP32 tile, reserved once and addressed directly |

Total CB storage is 328 KiB/core with BF16 state, 456 KiB with FP32 state. Contrary to the reader/compute comments, raw/state/output rings have **two-head capacity** (M:971-1004). Halving them casually would remove overlap.

Reader reserves inputs, issues state/vector/gate reads, waits for read completion, widens values, then publishes (M:800-844). Compute waits, prepares/reuses Q/K, computes all four value-columns, and publishes updated state plus output after all four finish (O:5448-5539). Writer waits for both, performs state and output writes, waits for completion, then retires both CBs (O:4949-4971). Distinct heads have disjoint state pages and output row slices. B32 has no output tail; other batches must retain the last-head padding writes.

Per rank, per B32 invocation:

| Traffic | API calls | Bytes |
|---|---:|---:|
| State DRAM reads | 6,144 | 12,582,912 |
| Q/K/V DRAM reads | 1,152 | 294,912 |
| g/beta DRAM reads | 768 | 36,864 |
| State DRAM writes | 6,144 | 12,582,912 |
| Output DRAM writes | 3,072 | 196,608 |

There are 384 read barriers and 384 write barriers, plus startup synchronization. Total requested DRAM bytes are 25,694,208; state alone is 97.94%. API calls are not necessarily individual wire packets. Initialization loopback requests below are separate from these DRAM requests.

## Minimal discriminating candidates

### A. Skip V/g/beta padding initialization; leave Q/K initialization intact

M:773-777 clears both slots of CB0-4: `2*(4+4+4+1+1)*4096 = 112 KiB/core`, or 12.03125 MiB/rank. Blackhole's implementation performs **512-byte local NoC loopback reads**, not DRAM writes: `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl:13-46`, `blackhole/dev_mem_map.h:88`. This is 224 requests/core and 24,640/rank. This startup work is fixed per active core while each B32 core processes only three or four heads; it is not amortized over the thirteen/fourteen-head work per core of the single-chip B32 path.

Leave Q/K initialization unchanged to bound this experiment; its necessity is not assumed. V row0 alone reaches `_gdn_delta_row_` (O:5055-5082): SFPTRANSP selects V and prediction row0, the chosen delta overwrites its temporary, and only that value is replicated. Other V rows do not participate in any reduction. g's accurate exp is elementwise; `_broadcast_scalar_from_dest_` extracts only `[0,0]` with predicated assignment before broadcast, so unused infinities/NaNs are discarded rather than multiplied by zero (`ckernel_sfpu_binary_bcast.h:651-680`). Beta uses the same extraction. Output padding originates from the reduction pack mask (O:5530), independently of these input pads.

**Intervention:** retain all reservations and live-element stores, but issue startup zeroing only for CB0/1. No compute/writer change. **Bound:** removes 48 KiB/core, 5.15625 MiB/rank and 10,560 local loopback requests/rank, or 42.86% of this zero traffic. It removes no state DRAM bytes and does not establish any minimum latency gain. Explicitly poison unused V/g/beta slots in an isolated exactness check to test the source-level liveness proof.

### A2. Remove remaining Q/K padding initialization as a separate control

Review of the lowered normalization supports removing Q/K padding zeroing too. O:5389-5399 squares/folds elementwise, then REDUCE_ROW produces independent row sums; only norm `[0,0]` is consumed. The lower row-reduction merge uses fixed-lane predicated **swaps**, not multiplication by a zero mask (`ckernel_sfpu_reduce.h:470-485,576-583,619-703`). Padded rows, including NaNs, cannot enter row0's sum. Scalar normalization then selects only norm `[0,0]`, multiplies pointwise, and transposes original row0 into column0. `gdn_expand_column` uses a replay beginning with bitwise SFPAND before rotate/add (`ckernel_sfpu_binary_bcast.h:177-182,214-218`), so unwanted columns become exact zero even when their bits encode NaNs. The normalized-input interface skips normalization and has the same transpose/mask protection.

**Intervention:** remove all CB0-4 startup zero calls and their now-unused zero-completion barrier, while preserving every initial reservation, live-value store and ring capacity. Regular reads initialize their own NoC read state; they do not depend on the preceding zero operation. No normalization arithmetic changes are needed. **Bound:** removes the entire 112 KiB/core and 224 loopback requests/core, or **12.03125 MiB and 24,640 requests/rank**. This includes A's savings; the additional Q/K saving is 64 KiB/core and 14,080 requests/rank. Run A and A2 separately to distinguish the liveness proofs and cost. Poison all raw padding with varying finite/NaN/Inf patterns and compare useful output/state bits against the frozen baseline. Static confidence is high; hardware confirmation remains outstanding.

### B. Remove dead zero/one initialization and matching handshakes

CB6/7 data are never tile-loaded by the active imported compute. References are startup format setup, CB construction, startup waits and final pops (O:5443-5447,5541). Old `contract()` and other uncalled helpers do not change that liveness result. M:778-782 nevertheless performs 2,048 volatile stores per core.

**Intervention:** remove those reader fills/pushes and compute waits/pops together; initially retain CB6/7 descriptors and the CB6 startup format argument. `compute_kernel_hw_startup` uses that descriptor to configure source formats (`api/compute/compute_kernel_hw_startup.h:60-81`), so deleting the descriptor alone is not equivalent. **Bound:** saves 8 KiB/core, 0.859375 MiB/rank, 225,280 scalar stores/rank. This is a fixed invocation cost; no recurrence arithmetic changes.

### C. Match raw Q/K production to existing grouped consumption

O:5455-5462 prepares raw Q/K only at local entry or global triple boundaries. Otherwise it waits/pops raw Q/K without using them. M:811-812,822-833 still fetches and widens every duplicate.

**Intervention:** give the reader the same GROUP_QK choice and predicate:

`prepare_qk = !GROUP_QK || h == start || h % 3 == 0`.

Only when true, reserve/read/widen/push CB0/1. Delete compute's corresponding duplicate wait/pop block. Keep every state/V/g/beta operation, existing prepared-CB retention/retirement, all arithmetic and output writes. Adding a reader compile-time flag must shift the TensorAccessorArgs base consistently, or encode the choice without shifting existing accessor offsets.

**Bound:** 384 raw pairs become 164; remove 220 duplicate pairs = 440 reads, 112,640 DRAM bytes (0.44% of total requested DRAM traffic), and 56,320 BF16-load/shift/FP32-store iterations per rank. Q/K normalization count stays 164. Existing raw-ring zeroing can initially remain unchanged. At default B1-B9, one head/core means no duplicate pair can be removed; B10/B17/B31/B32 and reduced-core cases distinguish the change.

The historical `doc/optimized_decoder/AUTOFIX_gdn_grouped_reader.md` is a useful structural reference, but its `(h-start)%3==0` assumes aligned triples. That predicate is wrong for start=4,count=4 above. Do not transplant it or its older partition unchanged.

### D. Reuse gate scratch within one user

For local hv=0..11, `gate()` always fetches the same user page and offset zero (M:795-798). The subsequent per-head store selects `scratch[hv]` (M:835-841). Q/K/V scratch occupies 0..1535; g/beta slots at 1536/1600 are disjoint and survive later vector reads.

**Intervention:** refresh both gate scratch slots at local entry and whenever user changes; continue publishing distinct g/beta scalars every head. For this local12 specialization all heads use the first face. A generalized implementation must include the selected gate face in its cache key.

**Bound:** current B32 ranges each touch one user, so 768 reads become 220: save 548 calls and 26,304 bytes/rank (0.10% of total DRAM bytes). This tests command issue overhead more than bandwidth. It makes no change to per-head exp or beta multiplication. Do not replace per-head scalar extraction with a shared gate scalar.

### E. Restrict accurate exp to its consumed vector

O:5450-5452 executes `exp_tile<false>(0)` over the entire 1,024-element g tile. Only its `[0,0]` output is used. The API defaults to VectorMode::RC and eight iterations; Blackhole invokes the function once per face, hence 32 SFPU vector evaluations (`api/compute/eltwise_unary/exp.h:79-96`; `tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h:45-85`). The selected accurate FP32 implementation honors ITERATIONS, including 1 (`ckernel_sfpu_exp.h:584-656,713-739`).

**Intervention:** first test `exp_tile<false>(0,VectorMode::R)` as a one-argument control, retaining accurate exp init and all packing. A tighter specialization is `exp_tile<false,false,InputClamping::ClampToNegative,1>(0,VectorMode::None)`. It evaluates the vector containing `[0,0]` using the same polynomial/instruction sequence. Other lanes are dead under the scalar extraction proved above. Check SFPU address/replay state and bitwise output/state on device before accepting it.

**Bound:** the control halves exp face work; the tight variant reduces 32 vector evaluations to 1, not total kernel time by 32x. Full-tile copy/pack, exp initialization, state math, state I/O and synchronization remain. Do not switch approximate mode or use host exp. Broader row-only Q/K normalization is possible but changes more helpers and should follow this isolated control, preserving the existing reduction tree.

### F. Stream updated-state columns to the writer

O:5528-5531 packs each n-column's four state tiles and one output tile, but O:5535 publishes only after all four columns finish. Fused compute uses live DST, not a reread of CB11; O:4951's comment claiming that output wait protects compute reads of updated CB is stale.

**Intervention:** a state-only streaming experiment keeps output publication unchanged, packs each n's state tiles contiguously into CB11, publishes four tiles, and lets the writer map stream `(n,m)` back to page `h*16+m*4+n`. Writer must not wait for output before starting state transfer, and must complete a transfer before popping its source. Output can still be written once/head. A full stream could publish one output tile per n too.

CB11 would change from one reserve/push/pop of 16 to four matched groups of 4; pack indices must become relative to the current write pointer, not retain old `m*4+n` indexing. Preserve all readout math, persistence rounding, masks, source formats and destinations. The reader has already copied the old head's complete state before compute starts, so earlier persistence does not overwrite unread old state.

**Bound:** zero saved bytes or arithmetic; only overlap/drain can improve. Extra handshakes/barriers can lose. Removing only `out.wait_front(4)` is ineffective because both queues currently publish together. This is lower priority than A-E.

## Partition tradeoffs and expected bounds

Four heads/core is already the minimum possible maximum, `ceil(384/110)=4`. With the present 3/4-head assignment, each four-head core necessarily intersects at least two Q/K groups and each three-head core at least one; the existing 164 fragment preparations attain that lower bound. Forcing complete triples gives 18 cores six heads and 92 cores three, increasing the longest recurrence workload by 50% despite reducing aggregate preparation to 128. It is not an established optimization.

An isolated 96-core control has four heads/two Q/K fragments per core, the same maximum work as current critical cores, but raises aggregate fragments to 192. It can test contention versus load balance; no speedup follows from counting cores. Splitting independent value-column n work across cores preserves each cell's operation order and can help small batches, where B1 currently uses 12 cores. At B32 it adds preparation/communication and is a larger scheduling change, not a cheap reduction in arithmetic.

With unchanged state geometry, the nominal state-only floor is 49.152 us; adding all requested baseline DRAM bytes gives 50.184 us at 512 GB/s. These ideal aggregate bandwidth floors do not predict achieved throughput or include loopback initialization/SFPU work. Even attaining the state-only floor would save at most 84.659 us from this 133.811 us row, before accounting for unavoidable other costs. With all other profile costs fixed, that idealized saving would cap kernels+gaps speedup at about 1.126x. The 512 GB/s number is the parent's accounting assumption (`tests/account_multichip_perf.py:104-129`).

Matched profile totals are 715.86775 us kernels, 40.15 us gaps and 756.01775 us kernels+gaps. Recurrence is 18.69% / 17.70%, respectively. Even deleting it entirely gives at most **1.230x kernel-sum** or **1.215x kernels+gaps** speedup if other costs stay fixed. A 10% recurrence reduction saves 13.381 us, only 1.77% of kernels+gaps. Uninstrumented delivery decode is 733.114 us; the 133.811 us instrumented row must not be subtracted from it as a measured forecast.

The initial TP4 B32 recurrent profile was 142.741 us with the same compute hash but tiled separate vectors; accepted packed recurrence is 133.811 us, a 6.3% reduction. Whole-layer improvements were much larger. This contrast supports investigating reader cost but does not show that remaining reader cost dominates. S513 accepted recurrence is 133.673 us; state size is independent of prefix length.

## Exactness boundaries and other potential issues

The active kernel already retains all four state tiles of a value-column in DST, fuses decay/prediction, and fuses update/readout (O:5443-5533). Each state element needs separate FP32 decay, K-product, delta*K update product/add and Q-product; the four K segments fold in the existing order before each SFPU column reduction. Do not factor decay outside prediction, use FMA, switch to TF32 FPU contraction, or reassociate the reductions. The current output uses the **unrounded FP32 update**, while only persistence is BF16-rounded (O:5194-5255). Reading rounded persisted state for current output changes semantics.

Manual BF16 RNE appears expensive: relative to a direct FP32 state store, its six additional instructions per 32-lane chunk total 3,072 instructions/head. However, removing it is **not a proven exact optimization**. The LLK rounding enum documents default nearest-even (`tt_llk_blackhole/llk_lib/llk_defs.h:150-156`), while the explicit typecast implements an integer masked-FP32 result (`ckernel_sfpu_typecast.h:273-286`). Packer/gasket configuration alone does not prove identical handling of ties, subnormals, overflow and NaN bit patterns. Treat relying solely on BF16 pack conversion as a separate numerical-contract experiment, requiring exact persisted-state evidence; keep it out of the safe headline recommendations.

Consecutive logical state TILE pages are DRAM-interleaved. Replacing sixteen page reads/writes with one contiguous 32 KiB transaction is not valid without a new physical-layout proof. Keeping state sharded in L1 may eliminate DRAM traffic, but changes prefill/decode storage, ownership and trace contracts; it is beyond the minimal candidates here.

The ring-depth and writer-protection comments are stale documentation, not evidence of a current queue mismatch. Static ownership/queue inspection finds balanced production and retirement; no wrong-user/head/state mapping was found.

## Checks performed and parent validation

`static_audit.json` records frozen hashes, 7,040 AST-only partition/provenance cases across B1-B32, core limits 1-110 and grouped enable/disable, plus 5,888 whole-head abstract queue cases for baseline and Q/K-elided protocols. It verifies head coverage, local Q/K cache provenance/lifetime and the byte/request calculations. It imports no TTNN and opens no device. The abstract schedule proves counts/progress in that model, not silicon ordering, math, compilation or latency.

Existing `tests/check_multichip_compaction.py:163-215,226` supplies strict packed-versus-independently-uploaded-RM, reversed-user output/state and four-step controls at B2/3/9/10/17/31/32. It does not compare a modified kernel against the frozen recurrence or force one-core wraps. `tests/run_recurrence_exact_comparison.py` has the desired bitwise comparison style but hardcodes 48-head single-chip dimensions; it cannot directly validate local12 unchanged. Host validation's string/partition assertions do not establish device execution.

For each isolated candidate, the discriminating parent check is exact frozen-versus-candidate output **and** persisted-state comparison over multiple steps, before whole-layer timing. Include B1/2/3/10/17/31/32, a low-core case crossing users/ring wraps, nonaligned starts such as 4, both supported state dtypes, and the raw TILE/normalized interfaces if those remain supported. Use distinct rank/user/head values and zero Q/K/V/state cases; include poisoned dead padding for A/A2/E and partial output tails for non-tile-aligned head counts. Re-run changed-input trace reuse and the existing per-user/rank checks, with explicit completion/readback. Keep profiling and watcher jobs separate as in the parent's workflow.

Measure A/A2 and B-E independently with unchanged policies and frozen arithmetic, then combine only supported wins. F should follow only if the movement controls or finer timing still point to a writer pipeline opportunity. No hardware correctness or performance result is claimed by this investigation.
