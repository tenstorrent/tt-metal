# Megakernel attention block (track `-attn`): design v0

Conforms to DESIGN_megakernel.md §6 "Interface spec v0": role bit ATTN, CB 9-12 (Q, K, V, O), semaphores 16-19,
the shared 300 KB working region (used only during ATTN phases), and D=2 weight prefetch on STREAM DM RISCs.
Interface changes go through the megakernel doc first.

## 1. What one attention layer does per chip (TP8, KV-split "user" mode, as prod at 32K u8)

Rows R = 32 (8 users × T=4 verify candidates; row 4u+j = user u, candidate j). The chip owns users u % 2 == r
("own users": 4 at u8). Its KV head h = chip/2; it runs all 6 q heads of the pair over the full context for its own users.

| Step | Shape per chip | Bytes | Today µs (u8, 32K) |
|---|---|---|---|
| norm (folded) + qkvg projection, wqkvg_pair | x[32×5120] · W[5120×3584] BFP8 | 19.5 MB | 49.5 (~394 GB/s) |
| q/k RMSNorm (head_dim 256) + partial RoPE (first 64 dims = 2 tiles) | q 6 heads, k 1 head | — | in glue |
| KV append (k, v rows of all 8 users) to the paged BFP4 cache | 2 × 32 rows × 256 | ~10 KB | in glue |
| verify SDPA, own users, 6 q heads / 1 KV head, BFP4 KV | 4 users × 32K × 256 × 2 | ~37.7 MB | ~190 (KV-read bound) |
| gate: o ⊙ sigmoid(gate pair) | 32 × 1536 | — | in glue |
| o-proj, wo_pair | A[32×1536] · Wo[1536×5120] BFP8 | 8.4 MB | 25.2 |
| total incl. norms/glue/gaps, excl. the 2 all-reduces | | ~65.6 MB | ~314 |

qkvg output columns per chip (112 tiles): q pair 0-47 (head h = tiles 8h..8h+7), k 48-55, v 56-63, gate pair 64-111.

## 2. Phase flow inside the megakernel (phase types from the spec)

1. **PROJ qkv** (type 1, tensor 4) on STREAM (80): core c owns 1-2 output tiles (cores 0-31 two, 32-79 one; bank-local
   BFP8 slice, K=160). Writer scatters each output tile to its consumers: q and k/v tiles → the 4 ATTN group roots'
   raw buffers; gate tiles → the concat core.
2. **NORM q/k + RoPE** (type 8, block-private) on the 4 group roots: v0 = stall stand-in (TODO MKA2: real RMSNorm
   row-reduce over 8 tiles + RoPE on tiles 0-1 of each head, tile-aligned: out0 = x0·cos − x1·sin, out1 = x1·cos + x0·sin).
3. **SDPA** (type 5) on ATTN cores, time-multiplexed with STREAM (v0: 110 cores = 4 rectangular groups, one per own
   user, sizes 25-30):
   - group root extracts Q_u (rows (h, j) = 4h + j, 8 tiles) and the user's 4 new K/V rows (a masked extra block)
     from the raw q/k/v tiles by 32-byte row copies (tile faces), multicasts Q_u to the group;
   - each core streams its contiguous slice of the user's KV (BFP4, 32-position blocks of 8 K + 8 V tiles) through
     the 300 KB working region (reader, NOC1) and accumulates O_aug = Σ_b exp(Q·K_bᵀ) · [V_b | 1] in an fp32 CB with
     packer L1 accumulation (the ones-tile column gives the softmax denominator, so no separate row reduce);
   - partials reduce up a fanout-3 tree to the group root (36 KB per partial, 2 receive slots per core);
   - root: o = O / den, then 32-byte row copies of o's (h, j) rows into the concat core's A rows (4u + j, cols 8h..).
4. **Gate + broadcast** on the concat core: A ⊙ sigmoid(gate) (48 tiles; non-own rows stay zero), multicast to STREAM.
5. **PROJ wo** (type 1, tensor 5) on STREAM: A_g[32×1536] · Wo slice (K=48, 2 tiles per core) → CB_OUT, signal sem.

Weight prefetch (writer/NOC0 on STREAM): this layer's wo slice and the next layer's MLP slices are issued right after
the qkv reads, so they stream *during* SDPA, contending with the KV reads (reader/NOC1). That contention is the main
thing MK1 did not model.

## 3. v0 numerics and stated simplifications (sim MKA1)
- Softmax without max subtraction (exact for bounded test scores; production adds the running max / rescale, which is
  one reduce + one bcast-mul per block). Q is pre-scaled by 1/sqrt(256) in the test data.
- No causal mask inside the new T=4 block (every candidate sees all 4 new positions); golden matches.
- q/k norm + RoPE and the KV append are stall stand-ins (cycles stated in the run), not computed.
- KV is a contiguous interleaved BFP4 tensor per user (prod: paged, page table); the per-tile reads are the same.
- Context scaled down (sim runtime); the SDPA term is extrapolated linearly in context from two context points.

## 4. Budget target (u8, 32K)
DRAM bytes per attention layer ≈ 65.6 MB (+ the next layer's prefetched MLP weights streaming during it). At ~400 GB/s
that is ~164 µs; plus the non-overlappable chain (qkv compute after its weights land ~5 µs, root extract/multicast,
tree reduce, concat/gate/broadcast, wo compute) ≈ 20-30 µs → **~190-200 µs per attention layer vs ~314 today**
(−~1.8-2.0 ms per pass over 16 layers), if the KV stream reaches ~400 GB/s while weights prefetch alongside.

## 5. MKA1 results (2026-11-10; silicon = P150 chip 0 of d30u20, 120 cores; sim = ttsim-perf BH, 110 cores)
SDPA phase only (4 own users, 6 q heads / 1 KV head, T=4), PCC 0.99998 in every configuration.
| Config (32K unless noted) | Silicon span | Bytes | Aggregate |
|---|---|---|---|
| BFP4 KV, K+V on NOC1 only, no prefetch (v0) | 186 µs | 37.7 MB | 203 GB/s |
| BFP4 KV, K NOC1 + V NOC0, NBB 4, no prefetch | 124 µs | 37.7 MB | ~330 GB/s (KV-bound) |
| BFP4 KV, + 27 MB STREAM weight prefetch (role-weighted, NBB 8) | 195-205 µs | 64.9 MB | ~325-335 GB/s |
| BFP4 KV 64K, + prefetch | 320 µs | 102 MB | ~320 GB/s |
| **BFP8 KV (§8 default), NBB 2, no prefetch** | **166-175 µs** | 71.2 MB | **~410-430 GB/s** |
| **BFP8 KV, NBB 2, + 27 MB prefetch** | **234-249 µs** | 98.4 MB | **~395-420 GB/s** |
- Larger KV reads (BFP8 1088 B vs BFP4 576 B tiles) reach higher aggregate bandwidth: per-read overhead, not bytes, limits
  BFP4. Multi-tile contiguous KV reads (page = whole 32-position block) would likely lift BFP4 too.
- Bank-local KV placement was worse (220 µs: NoC distance dominates); a single NoC for all KV is the main limiter.
- Compute is not the bound (LoFi = HiFi2); per-core compute median ~80-145 µs vs read-bound spans.
- Sim vs silicon (final BFP4 config, +prefetch): 4K 34 vs 100 µs, 8K 48 vs 114 µs; no prefetch 8K 48 vs 59 µs. The sim
  is 1.2x optimistic without contention and 2-3x with heavy concurrent DRAM traffic: use silicon for contention numbers.
- L1: BFP8 NBB 2 = 288 KB; NBB 4 needs 1 reduce slot (328 KB, over the 300 KB region) and is slower anyway.
- Today's prod verify SDPA at 32K u8 ≈ 190 µs (BFP4 KV, KV only). MKA1 with BFP8 KV does the KV + 27 MB of next-op
  weights in ~240 µs, i.e. the next layer's wo + MLP weights stream for ~50-70 µs extra instead of ~68 µs at 400 GB/s
  after it.

## 6. MKA2: attention-type layers inside the megakernel engine (silicon, d27u08 chip 0, 2026-11-10)
One generic_op on the full 12x10 grid (80 STREAM + 40 ATTN-only cores), NL attention-type layers per pass. Per layer:
mixer (qkv stand-in, MIXN=1 BFP4 + the rest of qkv's 19.5 MB as scratch reads) -> SDPA phase (type 5, CBs 9-12, BFP8 KV,
global semaphores, cumulative per layer) -> wo stand-in (8.4 MB real reads + 10 µs) + attention AR stall (18 µs) ->
fm11 MLP -> MLP AR stall (18 µs). Next-layer mixer + MLP weights are issued during SDPA under the word-13 budget.
Correct in every layer: SDPA PCC 0.99998, mixer 0.99999, MLP 0.99998 (2 and 3 layers, 4K and 32K, repeated iterations).

| 32K, u8 (4 own users), BFP8 KV | layer 0 (cold) | layer 1 | layer 2 (steady) |
|---|---|---|---|
| v0: all weight prefetch on the writer (NOC0) | 705-722 µs | 645-655 | 581-589 |
| + wo / extra-qkv reads moved to the reader (NOC1), budget 0 | 491 | 427 | 371 |
| + budget 32 KB per KV group (word 13) | 484 | 407 | **368** |
| KV shares weighted (role or measured-rate maps) | worse: 387-850 | | |
§8 projects ~322 µs per attention layer at u8: **steady MKA2 is ~368 µs (+14%)**.
Steady-layer budget: SDPA phase ~270 µs (bound by NoC1 column distance: left columns finish ~200 µs before the right
columns and ATTN-only cores), post-SDPA tail ~93 µs (reduce/norm ~15, wo 10 + 2 AR stalls 36, the even gate|up groups
streamed on the reader after SDPA, down), mixer ~24 µs.
Bugs found and fixed on the way:
- Double-buffer deadlock: the next layer's slots are planned after this layer's push, so reserve ONE slot at the write
  pointer (MK1 reserved two, pre-push).
- CB ring straddle: a short last KV group made later groups straddle the CB_K/CB_V end; every group now reserves,
  pushes and pops the full NBB (only nb used).
- NoC0 overload: V + all ~47 MB of weight prefetch on NOC0 vs K alone on NOC1 → 580 µs; balancing ~19 MB onto NOC1 → 370 µs.
Next levers: overlap the post-SDPA gate|up stream (prefetch even groups during SDPA, needs ~150 KB L1), multicast-aware
KV scheduling for the column skew, and the real qkv/wo phases replacing the stand-ins.
