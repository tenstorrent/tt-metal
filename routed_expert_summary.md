# Routed-expert FFN — DRAM bandwidth investigation (TEMP working notes)

Scratch notes for the `UnifiedRoutedExpertFfnDeviceOperation` DRAM-bandwidth work.
Delete before the real PR. Board: Blackhole **p150b**, 8 DRAM banks, NoC 64 B/cycle
@ 1.35 GHz ⇒ **86.4 GB/s per core per NoC port**, ~512 GB/s DRAM peak.

Op under test: `tests/.../deepseek_prefill/test_single_routed_expert.py::test_single_routed_expert_isl_sweep`
(`x_rm` layout = the production Blackhole fused-tilize path).

## 1. Traffic model

Per call, for one expert on one chip:

| stream | bytes | notes |
|---|---|---|
| x read | `tiles·32 · emb · 2` | bf16 ROW_MAJOR, read once, count-guarded |
| weights read | `3 · emb · hidden · 0.5625` **× num_chunks** | bfp4_b (576 B/tile); re-read once per M-chunk |
| output write | `tiles·32 · emb · 1.0625` | **bfloat8_b** tiles, not bf16 |

`num_chunks = ceil(count_tiles / chunk_M_tiles)`, `chunk_M_tiles = 64` (per_core_M 8 × GRID_Y 8).

Kimi-K2.6 (emb 7168, hidden 2048) ⇒ weights = 24.77 MB. Verified against measured
per-core request counters: gate/up/down each issue exactly 14,336 tile reads = 8.26 MB.

## 2. Baseline vs. current (kimi_k26, device kernel time)

| ISL | baseline µs | after sender remap µs | speedup | total MB | GB/s now |
|---|---|---|---|---|---|
| 128 | 210.4 | **179.6** | 1.17× | 28.4 | 158 |
| 256 | 222.4 | **198.0** | 1.12× | 30.4 | 154 |
| 512 | 280.1 | **256.5** | 1.09× | 39.5 | 154 |
| 1024 | 403.4 | **328.9** | 1.23× | 54.1 | 165 |
| 2048 | 657.2 | **591.4** | 1.11× | 83.5 | 141 |
| 4096 | 1294.0 | **1175.0** | 1.10× | 167.0 | 142 |
| 5120 | 1673.9 | **1489.1** | 1.12× | 221.1 | 149 |

glm_51 sum 4136 → 3679 µs (1.124×). Baseline was ~127–145 GB/s; now ~141–165 GB/s.

Perf-test baselines recalibrated (3 samples/case, midpoint of min/max). `_MARGIN`
widened 3% → 5%: most cases are stable to <1% but kimi isl-256 spans **7.5%**
(186.8 / 188.2 / 201.3 µs) — the short-ISL cases are dominated by fixed per-K-block
sync latency, not streaming work, so they pick up jitter.

## 3. Root cause: the reads are ISSUE-bound on the reader RISC

Instrumented the reader/writer kernels with wall-clock accumulators splitting every
DRAM read into **ISSUE** (cycles the RISC spends pushing requests into the NoC command
buffer) and **BARRIER** (cycles waiting for responses after the last issue).

x = [256, 7168], core (0,0) — pre-remap, when it was both x-sender and weight-sender:

| stream | req size | issue cy | barrier cy | reqs | **cy/req** | bytes | B/cy | GB/s | % of 86.4 |
|---|---|---|---|---|---|---|---|---|---|
| x read | 512 B | 34,503 | 26,381 | 896 | **38.5** | 458 KB | 7.5 | 10.2 | 12% |
| gate read | 576 B | 57,879 | 11,573 | 1344 | **43.1** | 774 KB | 11.2 | 15.1 | 17% |
| down read | 576 B | 55,759 | 4,169 | 1344 | **41.5** | 774 KB | 12.9 | 17.4 | 20% |
| up read (NoC1, writer) | 576 B | 57,764 | 13,709 | 1344 | **43.0** | 774 KB | 10.8 | 14.6 | 17% |

Findings:

1. **ISSUE ≫ BARRIER** everywhere (down: 55.8K vs 4.2K). Not DRAM-bound, not link-bound.
2. ~**38–43 cycles of RISC issue cost per request**, independent of size (512 vs 576 B)
   and of NoC (0 vs 1). A 576 B payload only needs **9 cycles** of a 64 B/cycle port ⇒
   **~79% of each port is idle waiting for the RISC.**
3. A single reading core therefore delivers only ~11–13 B/cycle ≈ **15–17 GB/s**.
4. Multicast, by contrast, achieves **45.4 B/cycle = 70% of port** — the write side is fine.

Reader-RISC occupancy at isl-256 (276K-cycle op), pre-remap:

| core | role | busy | idle |
|---|---|---|---|
| (0,0) | x **and** weight sender | 254.5K = **92%** | 8% |
| (5,0) | weight sender | 177.2K = 64% | 36% |
| (0,3) | x sender | 76.0K = 27% | 73% |
| (5,3) | receiver only | ~3K = **1%** | 99% |

**69% of the whole op was core (0,0)'s reader issuing/awaiting DRAM reads.** Only 19 of
88 reader RISCs touch DRAM at all (8 x-senders + 11 weight-senders); the other 69 readers
and ~77 writers are idle.

Why one tile per request is forced: DRAM is interleaved with page = tile, so
`bank = page_id % 8`. Consecutive N tiles land in *different* banks (can't coalesce);
same-bank tiles (stride 8) are 4608 B apart, not contiguous. So 576 B/request is
structural for the current weight layout.

Important qualifier added later (§7d): the ~37-43 cy/request is **not** slack in our kernel.
The hardware floor for a request of this size class is 32.2 cy, so we are within 14% of it —
that residual is `TensorAccessor` page->address generation plus the D2.0 `Noc::async_read`
wrapper. "Issue-bound" here means *the request count is too high*, not *our issue path is
slow*. The fix is fewer/larger requests (§8 item 3), not a cheaper issue path.

## 4. Change landed: staggered DRAM-read sender placement

Weight senders were all on row 0 and x senders all on column 0, so **(0,0) owned both
streams** and serially issued x + gate + down (~190K cycles). Staggered them:

- weight sender for column `gx` → row `gy = gx % GRID_Y`
- x sender for row `gy` → column `gx = (gy + 1) % GRID_X`

```
gx:    0    1    2    3    4    5    6    7    8    9   10
gy=0   W    X                                  W
gy=1        W    X                                  W
gy=2             W    X                                  W
gy=3                  W    X
gy=4                       W    X
gy=5                            W    X
gy=6                                 W    X
gy=7                                      W    X
```

No core is in both sets (`gx == (gx % 8 + 1) % 11` has no solution on 11×8).
Multicast rectangles now span the **full** column/row and rely on non-loopback multicast
to skip the sender, since the sender is no longer on an edge row/column.

Program-factory-only change (runtime args + rectangles); kernels untouched.
Result: core (0,0) 92% → **75%** busy, isl-256 276K → 246K cycles, **1.12× across all ISLs**.

## 5. Ideas tried and REJECTED (with measurements)

| idea | result | why |
|---|---|---|
| Weight all-gather across the N-column (readers/column = 1/2/4/8) | sum 4836 / 4754 / 5081 / **6083** µs | Cut per-core weight read 3.3× at 8 readers but total regressed 26%: the extra readers were the gx=0 column, which *already* carried the whole x read, and the per-block gather barrier widened. Worth retrying now that the remap made those sets disjoint. |
| Distribute the x row-major tilize (each core tilizes only its own tile-rows, mcast bfp8) | **2× slower** | The tilize was already overlapped with the reader's x DRAM read; turning it into a cross-core dependency added two 11-way barriers per K-block. |
| Full 11×10 grid (110 cores, 2 weight passes instead of 3 at isl-5120) | **hangs** | Top grid rows are needed by dispatch — wedged the board, needed `tt-smi -r`. |
| Option B: strided-by-8 column ownership to get multi-tile reads with NO host relayout (GRID_X 8, 64 cores) | **rejected on paper, 0.87×** | Three independent problems. (a) `bank = (64k + gx) % 8 = gx` for every k, so each core is pinned to ONE DRAM bank and stuck on the `READ 1 DRAM` curve: 8 × 29.4 = 235 GB/s, *below* today's 310 (§7e). (b) 64 cores raise token work by 88/64 = 1.375×. (c) `per_core_N_d` 21→28 grows L1 by 390 KB against ~184 KB of headroom, forcing `per_core_M` 8→6 so isl-2048 goes 1→2 chunks and isl-5120 3→4. Also cannot be a runtime A/B switch, because CBs would have to be sized for the union of both modes. |

| N weight senders per column, count-derived at RUNTIME (W=4 at isl<=512, 2 at <=1024, 1 above) | **rejected, 128: 167->280 us** | Implemented and measured on the staggered placement, i.e. with the weight-sender and x-sender sets disjoint — the reason the earlier attempt was dismissed. Still regressed: isl-128 167->280, isl-256 187->237, isl-512 252->282, isl-1024 309->323; isl>=2048 unchanged, which confirms the runtime switch itself worked (W=1 there). The per-block barrier PAIR — a ready counter over all GRID_Y members plus a gather over the W senders — costs ~3.6 us/block, swamping the ~28 us/chunk of multicast it saves. Two independent attempts, so this is settled. |

## 6. Secondary findings

- **Compute is near HW peak**: 14.8 cycles per 32³ tile-MAC. matmul busy ≈ 1.35M cycles
  ≈ 1020 µs at isl-5120 with 88 cores ⇒ a hard ~217 GB/s ceiling at long ISL. Long-ISL
  work is compute-bound; the DRAM headroom is at **short/mid ISL**.
- **x row-major → bfp8 tilize costs 27% of runtime at isl-5120**, and it is executed 11x on
  identical input. The tilize itself is NOT avoidable — in the real model x arrives
  ROW_MAJOR, which is why the `x_rm` path exists. What is duplicated is the work: on the
  RM path the sender multicasts `cb_x_rm` (the row-major bf16 bytes), then every core runs
  `tilize(n_strips)` with `n_strips = per_core_M`, i.e. the whole block, producing byte-identical
  bfp8 tiles. Core (5,4) — a pure receiver that issues no DRAM read and sends no multicast —
  still spends 27% of the op in the tilize.
  **RETRACTED:** an earlier draft suggested having dispatch emit bfp8 TILE x. Our own data
  contradicts it: in the `x_tile` variant the conversion appears as standalone Tilize 786 us
  + Typecast 491 us = 1277 us at isl-5120, against ~450 us for the in-op tilize. Moving it
  upstream is ~3x worse; the fused in-op tilize is the right design.
  Of the three ways to tilize once per M-row, two are ruled out: (a) sender tilizes and
  multicasts bfp8 but keeps its matmul -> ZERO gain, the sender's tilize+matmul equals what
  every core pays today and everyone waits on it; (b) distributing the tilize by tile-row ->
  measured 2x SLOWER (see section 5). The workable form is a **dedicated producer column**:
  gx=0 does x read + tilize + bfp8 multicast and no matmul, the other 10 columns only
  matmul, so the producer is never the straggler; it also halves x multicast bytes
  (2048 -> 1088 B/tile). Costs: 10% more N work per matmul core, per_core_N_gu 6->7 and
  per_core_N_d 21->23 (~10% more L1 against an already tight budget), phase-4 activated
  rotation over 10 cores not 11, and a changed writer output column mapping.
- The down matmul runs the full compile-time M ring and skips only the MACs, not the packs
  — 137K vs 214K cycles at per_core_M 1 vs 8. Wasteful at short ISL.
- Overlapping mcast with reads / relaxing `async_read_barrier` is worth only ~11% on the
  critical core (barrier 20.9K of 185.4K), because read-issue and mcast-issue are both on
  the same RISC and therefore additive.

## 7. Post-remap: DRAM read bandwidth reference (the number that reframed everything)

Everything in §1–6 was measured *before* the sender remap. After it landed, the open
question was how much headroom is actually left, so we measured the machine directly
rather than inferring it. A timed reference pass was injected at the top of the reader
kernel: each participating core issues N independent reads of the real gate tensor into a
small rotating L1 window, then barriers. Participation is gated on `(my_mt, my_nt_gu)`, so
the core count sweeps with a kernel-only edit (JIT — no host rebuild per point).
`TT_METAL_BUILD_TESTS=OFF` in this build, so metal's `6_dram_offchip` microbenchmark was
not available without a full reconfigure; this was cheaper and runs on the exact buffers
and page sizes the op uses.

### 7a. Scaling with the number of ISSUING cores (576 B = one bfp4 tile page)

| issuing cores | cy/req | per-core GB/s | **aggregate GB/s** |
|---|---|---|---|
| 1 | 36.6 | 21.2 | 21 |
| 8 | 37.4 | 20.8 | 166 |
| 16 | 42.7 | 18.2 | **292** |
| 32 | 79.5 | 9.8 | **313** |
| 64 | 163.8 | 4.7 | 304 |
| 88 | 218.8 | 3.6 | 313 |

**Ceiling for 576 B reads is ~310 GB/s and it saturates at 16–32 cores.** Beyond that the
aggregate is flat while per-core issue cost inflates 6× (36.6 → 218.8 cy/req) — the RISC
is absorbing NoC backpressure, not doing useful work. `bar` stays ~430 cycles at every
point, confirming the responses are never the constraint.

This retro-explains the rejected experiment in §5: the 88-core weight all-gather pushed
~3× past saturation, so it bought no aggregate bandwidth and paid full price in per-core
stall and gather-barrier width. **"More issuing cores" is exhausted as a lever.**

### 7b. Scaling with REQUEST SIZE (64 cores, 2 MiB per core)

Read from the x buffer, whose pages are whole emb-wide bf16 sticks (14,336 B), so any size
up to a full stick is a single-page read.

| request size | elapsed cy | **aggregate GB/s** |
|---|---|---|
| 512 B | 673,643 | 269 |
| 1024 B | 486,259 | 373 |
| 2048 B | 477,727 | **380** |
| 4096 B | 477,129 | **380** |
| 14336 B | 487,894 | 371 |

**Request size does move the ceiling: ~310 GB/s at 576 B → ~380–420 GB/s at ≥1024 B.**
Caveat: the 8-core/2048 B point measures 461 GB/s, but that is optimistic — 8 cores reading
2 MiB each out of a 3.67 MB buffer get DRAM row-buffer hits from page reuse. Treat the
64/88-core numbers as the honest ceiling.

### 7c. What the op is actually achieving

Critical core (weight sender) at isl-256, post-remap: 185.4K cycles of work in a 198 µs op.

| | cycles | µs | rate |
|---|---|---|---|
| DRAM read, gate + down (up runs concurrently on NoC 1) | 134.6K | 99.7 | 24.8 MB ⇒ **249 GB/s** |
| multicast | 50.8K | 37.6 | 45.4 B/cy = **70% of port** |
| sync waits | ~61K | | |

**The weight read is already at 249 of a 310 GB/s ceiling — only 1.25× remains at this
request size.** That is the single most important correction to §3's framing: the reads are
issue-bound per core, but in aggregate the op is already close to what 576 B pages can
deliver. Scheduling is no longer the problem; the page size is.

### 7d. Cross-check against the P150 NoC/DRAM characterisation table

An independent BH-P150 NoC/DRAM sweep (provided by the op owner; per-transfer-size bandwidth
for one core, plus latencies) agrees with everything in §7a–c and adds one constraint we had
missed. Reading its 11 data columns as skipping the two with no rows (`WRITE 1 DRAM`,
`All workers READ 1 DRAM`):

| quantity | table | our measurement | |
|---|---|---|---|
| 1-core DRAM read @512–576 B | 21.47 GB/s | 21.2 GB/s | match, 1% |
| flat cost per request, ≤1024 B | 32.2 cy | 36.6 cy | match, 14% |
| 1-core NoC port ceiling | ~83 GB/s | 64 B/cy × 1.35 GHz = 86.4 | match, 96% |
| aggregate ceiling ≥2048 B | 8 banks × 47.4 = 379 GB/s | 380 GB/s (§7b) | match |
| our column multicast | MCAST FAR 49.7 / MANY-LINKED 57.7 / NEAR 78 | 61.3 GB/s | in band |

The knee agrees too: flat per-request cost up to 1024 B (23.85 ns at every size from 16 B to
1024 B), then size/64 link-bound from 2048 B. And LINKED multicast is 1.67x unlinked
(57.7 vs 34.5 GB/s), which validates the `linked=true` the kernels already use.

Two corrections it forces on earlier sections:

1. **§3's "stateful NoC reads" idea is worth at most 14%, not 2x.** The floor is 32.2 cy per
   request, we spend 36.6; the residual is `TensorAccessor` page->address generation plus the
   D2.0 `Noc::async_read` wrapper. Not worth chasing.
2. **The 379 GB/s ceiling is a BANK limit shared by gate + up + down.** All three hit the same
   8 banks regardless of which NoC issues them, so the weight-read floor is
   24.8 MB / 379 GB/s = **65 us per chunk**, not the 41 us an earlier draft assumed. Neither
   cheaper issue nor more reader cores can go below it.

### 7e. New constraint: reads must ROTATE banks

`READ 1 DRAM` (one core pinned to a single bank) saturates at **29.4 GB/s @4096 B** (31.5 at
16 KB, 47.4 only at 64 KB), against `READ ALL DRAMS` at **81.4 GB/s @4096 B**. So a read
pattern must land a core's *consecutive* requests on *different* banks. Since
`bank = page_id % 8`, that is a property of the page-index arithmetic:

- **Option A** — `page = k * num_N_blocks + n_block`, `num_N_blocks = 11` ⇒
  `bank = (11k + n) % 8`. For k = 0..7 that is banks `[0,3,6,1,4,7,2,5]` — all eight, rotating.
  A lands on the `READ ALL DRAMS` curve. GOOD.
- **Option B** — `page = k*64 + gx + 8j` ⇒ `bank = (64k + gx) % 8 = gx` **for every k**. Each
  core is pinned to one bank for the entire read, so it is stuck on the `READ 1 DRAM` curve:
  8 cores x 29.4 = **235 GB/s aggregate, BELOW today's 310**. B is a regression on the read
  itself, before counting its 64-core compute penalty or its extra chunks.

**Option B is therefore dead** (see §5), and with it the question of switching A/B at runtime
from the device-side count — there is nothing worth switching to.

## 8. Revised path, with the bank-limited floor

The weight read has a **hard floor of 65 us per chunk** (24.8 MB / 379 GB/s, §7d). Multicast
is 37.6 us and divides cleanly by sender count. So the per-chunk FIXED cost decomposes as
`read (>=65, currently 99.7) + mcast (37.6 / senders_per_column)`.

Ranked by measured value:

1. **Producer column (token side) — the biggest single lever, 1.44x on a total run.**
   `gx=0` does x read + tilize + bfp8 multicast and **no matmul**; the other 10 columns only
   matmul. See §6 for why the two cheaper variants of "tilize once per M-row" are ruled out
   (one gives zero gain, the other measured 2x slower). Costs: 10% more N work per matmul
   core, `per_core_N_gu` 6->7 and `per_core_N_d` 21->23 (~10% more L1 against a tight
   budget), phase-4 activated rotation over 10 cores not 11, changed writer column mapping.
2. **N weight senders per column — now the primary WEIGHT-side lever**, not the secondary one
   it was ranked as. Since the read is floored at 65 us, multicast is 37.6 of the remaining
   FIXED budget and is the only part that still divides: 37.6 -> 18.8 (2 senders) -> 9.4 (4).
   Multicast volume itself is irreducible (it is already a true multicast). Free of L1 cost —
   it only splits an already-sized block among more issuers — so it can be a **kernel-level
   runtime decision from the device-side count**, exactly like `adaptive_chunk` picks
   `per_core_M`: more senders at small counts where FIXED dominates, one at large counts
   where extra issuers only add contention (§7a: past 32 issuing cores the aggregate is flat
   while per-core issue cost inflates 6x). The compile-time `kWeightReadersPerColumn`
   prototype should become count-derived.
3. **Multi-tile weight pages (option A)** — takes the read 99.7 -> 65 us, i.e. down to the
   bank floor. Make each core's (K-row x `per_core_N`) slice a *single* DRAM page (6 tiles =
   3456 B). Needs a host-side weight relayout (DRAM-sharded or custom page size) in
   `TtRoutedExpert` plus the matching `TensorAccessor` in the reader — it changes the layout
   the model hands the op. Verified bank-rotating (§7e). Open question for the op owner:
   relayout upstream, or accept both layouts behind a flag?
4. **Residual sync (~61K cy)** — deeper weight CBs (3-4 slots) plus per-subset
   (transaction-ID) barriers so the mcast of K-row *n* starts while *n+1* is still in flight,
   instead of one `async_read_barrier` over everything. Bounded at ~11% of the critical core.

### 8a. Estimated device time on a total run

Model `T = sum_chunks (FIXED + TOKEN * per_core_M)`, validated against measured times to 2%
on the total (worst point 11% at isl-1024). Currently FIXED = 99.7 (read) + 37.6 (mcast) =
137 us per chunk and TOKEN = 56.8 us per `per_core_M` unit, decomposing as tilize 18.8 +
rest 38.0.

| scenario | 128 | 256 | 512 | 1024 | 2048 | 4096 | 5120 | TOTAL | vs now | isl-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| measured now | 180 | 198 | 256 | 329 | 591 | 1175 | 1489 | 4218 | 1.00x | 1.00x |
| 2 senders/column only (no relayout) | 147 | 175 | 232 | 346 | 572 | 1145 | 1490 | 4108 | 1.05x | 1.13x |
| A: multi-tile pages only | 131 | 159 | 216 | 330 | 557 | 1113 | 1443 | 3949 | 1.09x | 1.24x |
| A + 2 senders/column | 112 | 141 | 197 | 311 | 538 | 1076 | 1386 | 3761 | 1.14x | 1.41x |
| A + 4 senders/column | 103 | 131 | 188 | 301 | 528 | 1057 | 1358 | 3667 | 1.17x | 1.51x |
| **A + 2 senders + producer column** | 105 | 126 | 167 | 251 | 418 | 836 | 1087 | **2991** | **1.44x** | 1.58x |

(An earlier draft of this table quoted 1.20x for A + 2 senders and 1.53x for the producer
column. Both were optimistic: they used a 41 us weight-read floor instead of the correct
bank-limited 65 us. Option B's row is gone -- see §7e. And per §10b the "+ 2 senders"
rows are now known to be unachievable: the multi-sender split was implemented and
regressed, so the multicast stays at 37.6 us/chunk and is ADDITIVE with the read.
Treat the "A: multi-tile pages only" row as the realistic weight-side outcome.)

Three things this settles:

1. **Any weight-side work is capped at 1.48x on a total run.** The sweep contains 10 chunks,
   so the entire weight cost is 10 x 137 = 1370 us of 4218 (32%); even a *free* weight read
   and multicast leaves 2848 us. Item 3 + item 2 together capture 457 us = 33% of that.
2. **3x on a total run is not reachable** by anything in this list. 3x at **isl-256 alone** is
   also out: the FIXED floor is 65 (read, hard) + ~9 (mcast at 4 senders) = 74 us, plus 57 us
   of token work = 131 us, i.e. **1.51x** at isl-256. Reaching 3x there would need the token
   side too, and the producer column only takes TOKEN 56.8 -> 41.8.
3. **The token side is where the remaining time is**: 2848 of the 4218 us. Only the producer
   column attacks it, and it is the one lever whose value grows with ISL.

Open input needed: which ISLs dominate production DeepSeek prefill? The table above is the
artificial sweep. If experts typically see <=512 tokens, items 2+3 are a 1.4-1.5x win; if
they see thousands, the producer column is the only thing that matters.

## 9. How to reproduce

```bash
# correctness (74 cases, all models × both x layouts)
pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_single_routed_expert.py -q

# device perf vs baselines
pytest tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_single_routed_expert_perf.py -q

# all ISLs in one Tracy process + bandwidth table (scratch harness)
python3 -m tracy -p -r -o <outdir> -a device_kernel_duration -t 5000 \
  -m "pytest <worker> -k 'kimi_k26 and x_rm' -q"
```

Kernel-side instrumentation used for §3 and §7 (reverted, not in the tree): wall-clock
accumulators via `get_timestamp_32b()` around each region, dumped with the new-style
`DPRINT(fmt, ...)` at kernel exit, read back with

```bash
TT_METAL_DPRINT_CORES="(0,0),(1,0),(0,3),(5,3)" TT_METAL_DPRINT_RISCVS="NC,BR" \
TT_METAL_DPRINT_FILE=dump.txt pytest <worker> -k "kimi_k26-isl-256 and x_rm" -q
```

Kernel `.cpp` edits are JIT-compiled, so instrumentation and the §7 core/size sweeps need
no host rebuild — one data point per ~15 s test run.

Gotchas hit along the way:
- After any C++ change, `cmake --build build --target install` is required — plain
  `--target ttnn` leaves the Python-visible `ttnn/ttnn/_ttnn.so` stale.
- A killed pytest can wedge the chip (`Timeout waiting for physical cores`); recover with
  `tt-smi -r 0`.

## 10. Implementation log (branch mbezulj/2607-routed-expert-dram)

### 10a. LANDED: down-matmul ring padding (1.02x total, 1.08x at isl-128)

The down matmul must cycle the FULL compile-time-MAX per_core_M ring through its
partials CB — PACKER_L1_ACC needs push == drain == out_block_num_tiles so the write
pointer wraps onto block 0's slots. But an out-of-bounds row only needs the pointer
to ADVANCE; it needs no pack. So the K-loop now does a bare reserve_back/push_back
for `sb_m >= m_subblocks` (ring wrap bit-identical, discipline preserved), and the
final partials -> cb_out copy drains padding subblocks without copying or packing
them, with the writer bounding its cb_out drain by the same runtime per_core_M.

| ISL | before | after | |
|---|---|---|---|
| 128 | 179.6 | **166.9** | 1.08x |
| 256 | 198.0 | **186.6** | 1.06x |
| 512 | 256.5 | **251.9** | 1.02x |
| 1024 | 328.9 | **308.8** | 1.07x |
| 2048-5120 | | unchanged | 1.00-1.01x |
| SUM | 4222.5 | **4149.2** | 1.02x |

Zero change at per_core_M = 8 (no padding rows), exactly as predicted. This was a
prerequisite: compute at per_core_M = 1 was ~85 us against a 73-80 us DRAM floor.

### 10b. REJECTED: count-derived weight senders per column

See section 5. Measured, reverted. **Consequence for every estimate in this doc:**
the multicast is irreducible AND additive with the read, because one RISC issues
both. So the per-chunk FIXED floor is **65 (read) + 37.6 (mcast) = 102.6 us**, not
the 75 us section 8a assumed. That lowers the isl-256 ceiling from ~2.5x to about
**1.25x from today / 1.5x from the original baseline**.

### 10c. CONFIRMED BUT NOT LANDED: multi-tile weight pages (option A)

Premise measured directly, by emulating option A's access pattern with no relayout
at all: pages p, p+8, p+16, ... are contiguous inside one bank (aligned_page_size ==
576 exactly), so a raw 3456 B read at page p fetches 6 tiles; stepping p by 11 per
request rotates the bank the same way option A's page index would.

| cores | req size | cy/req | per-core GB/s | aggregate GB/s |
|---|---|---|---|---|
| 11 | 576 B | 37.9 | 20.5 | 226 |
| 11 | **3456 B** | 139.6 | **33.4** | **368** |
| 88 | 576 B | 213.3 | 3.6 | 321 |
| 88 | 3456 B | 986.2 | 4.7 | 416 |

**1.63x on the weight read with the same 11 cores** — 24.77 MB at 368 GB/s = **67 us
against today's 99.7**, matching the predicted bank floor. The request becomes
data-bound (3456/64 = 54 cy of link time) instead of issue-bound. Worth ~1.09x on a
total run and ~1.21x at isl-256.

### 10d. The right layout is DRAM ND/BLOCK sharding -- but NOT width sharding

Measured the two candidate sharded access shapes (2 MiB per core, emulated on the
existing tile-paged buffer; a read of S bytes at page p covers pages p, p+8, ... so
it is confined to one bank, and the page STEP between requests decides whether the
core rotates banks):

| config | per-core GB/s | aggregate | weight read |
|---|---|---|---|
| today: 576 B, 11 cores | 20.5 | 226 | 100 us |
| **plain WIDTH_SHARDED** (1 shard/bank), 27 KB req | 22.4 | **246** | **101 us** |
| same, 13 KB req | 21.8 | 240 | 103 us |
| same, 55 KB req | 22.3 | 245 | 101 us |
| 8 cores, one bank each, 27 KB req | 31.6 | 253 | 98 us |
| **bank-ROTATING**, 27 KB req | 34.0 | **374** | **66 us** |
| bank-ROTATING, 3456 B req (section 10c) | 33.4 | 368 | 67 us |

**The canonical dram-sharded-matmul layout -- WIDTH_SHARDED with one shard per bank
-- buys nothing here (246 vs 226 GB/s).** A single bank saturates at ~22-31 GB/s
REGARDLESS of request size (13/27/55 KB all land at ~245), so one-shard-per-bank caps
the aggregate at ~8 x 30 = 246 GB/s, which is what the op already achieves. Request
size is not the lever; BANK ROTATION is. Note this also re-explains section 7e:
option B failed for exactly this reason.

So the weights must be sharded along **K as well as N** (DRAM ND_SHARDED / block
sharded, which ttnn supports -- the matmul validator accepts "DRAM ND_SHARDED", and
`TensorAccessor::get_shard_noc_addr(shard_coord)` takes a coordinate array, i.e.
(k, gx)) so that consecutive K-blocks of a core's slice land in different banks.

Use **shard height = 1 tile-row**, not in0_block_w_gu tile-rows: shard id = k*GRID_X
+ gx then rotates the bank with k, a K-block becomes in0_block_w_gu consecutive-shard
requests of 3456 B, and that measured 368 GB/s -- statistically the same as the 27 KB
single-request 374. This matters because in0_block_w_gu can be LOWERED by the op's L1
guard on large models, so binding the shard shape to it would be fragile;
per_core_N_gu = ceil(hidden_tiles / GRID_X) is stable.

| | shard shape | shard grid | padding | requests per K-block |
|---|---|---|---|---|
| gate, up | `[32, per_core_N_gu*32]` = [32, 192] | 224 x 11 | hidden 2048 -> 2112 (66 tiles) | 8 (was 48) |
| down | `[32, per_core_N_d*32]` = [32, 672] | 64 x 11 | emb 7168 -> 7392 (231 tiles) | 6 (was 126) |

The reader becomes one `noc_async_read(gate_acc.get_shard_noc_addr({k, my_nt_gu}),
..., per_core_N_gu * 576)` per k, and the compute kernel is untouched -- tiles land in
the k-major / n-minor order cb_in1_gate already expects. Expected: weight read
99.7 -> 66 us, ~1.09x total and ~1.21x at isl-256.

This supersedes the hand-rolled permutation below: sharding expresses the same bank
rotation declaratively, the allocator does the shard -> bank assignment at runtime, so
nothing arch-specific is baked into the DATA and weights need no per-arch preparation.
The hand-rolled formula is kept only as the explanation of WHY the rotation works.

**Why it is not landed.** It touches three tensors' memory configs plus padding in both
the test and `TtRoutedExpert`, the op's weight validation, and the reader/writer read
loops, and the non-sharded path has to stay for the other callers
(`test_swigluoai_routed_expert`, `test_routed_expert_bias`, the MoE tests) -- so it
wants a `weights_dram_sharded` op attribute with the current layout as default rather
than a half-migrated tree. The superseded permutation below needs no
change to the buffer's page size — only to tile ORDER, and it fits inside the
existing 2D tile grid (padded 64 -> 66 tile-cols, +3% weight bytes), so the mesh
mapper and 4D shape survive. For core gx, K-row k, slice element n (col = 6*gx + n):

```
g = k * GRID_X + gx          # GRID_X = 11
b = g % 8                    # NUM_DRAM_BANKS
o = g / 8
P = b + 48 * o + 8 * n       # target DRAM page index
place logical tile (k, col) at tile-grid position (P / 66, P % 66)
```

This is a bijection onto [0, 224*66): page P has bank b and bank-offset 6o + n, so
each (k, gx) group owns 6 CONSECUTIVE offsets within one bank — hence one 3456 B
request — and consecutive g rotate the bank. The reader then issues one request per
k at `gate_acc.get_noc_addr(b + 48*o)`, landing 6 tiles in the k-major/n-minor order
cb_in1_gate already expects, so the compute kernel is untouched.

The blocker is contract, not code. The weights are mesh-distributed and written to a
**disk cache** in `_convert_and_cache_expert_weights`, so this layout would be baked
into cached weight files and coupled to three op-internal facts: GRID_X = 11,
per_core_N_gu = 6, and NUM_DRAM_BANKS = 8. Weights would need re-preparing per arch
(Wormhole has a different bank count), and every other caller of the op
(`test_swigluoai_routed_expert`, `test_routed_expert_bias`, the MoE tests) would need
the same layout or a flag. That is a deployment decision, so it needs the op owner's
call: relayout upstream and version the cache name, or add a
`weights_bank_grouped` op attribute so the old layout stays the default.

## 11. ND-shard settings actually in use (measured, landed)

### 11a. The settings

Selected by `TtRoutedExpert(weights_dram_sharded=True)` (default `False`). The op detects
the layout from the tensor itself — `memory_config().created_with_nd_shard_spec()` — so
there is no extra op attribute to keep in sync.

| | value |
|---|---|
| buffer type | `ttnn.BufferType.DRAM` |
| memory config | `ttnn.MemoryConfig(buffer_type=DRAM, nd_shard_spec=...)` |
| shard shape | `[TILE_SIZE, per_core_N * TILE_SIZE]` |
| shard grid | full DRAM grid, `device.dram_grid_size()` → `CoreRange((0,0)-(7,0))` on P150 |
| orientation | `ShardOrientation.ROW_MAJOR` |
| distribution | `ShardDistributionStrategy.ROUND_ROBIN_1D` (the default) |
| `per_core_N` | `ceil(n_tiles / FFN_GRID_X)`, `FFN_GRID_X = 11` |
| applied by | `ttnn.to_memory_config()` after the interleaved build and the squeeze to 2D |

Concrete shapes for the two swept models:

| tensor | logical shape | n_tiles | per_core_N | **shard shape** | shard grid |
|---|---|---|---|---|---|
| kimi gate/up | [7168, 2048] | 64 | 6 | **[32, 192]** | 224 x 11 |
| kimi down | [2048, 7168] | 224 | 21 | **[32, 672]** | 64 x 11 |
| glm gate/up | [6144, 2048] | 64 | 6 | **[32, 192]** | 192 x 11 |
| glm down | [2048, 6144] | 192 | 18 | **[32, 576]** | 64 x 11 |

Why each choice, since none of them is free:

* **Shard width = `per_core_N`** is what makes one FFN core's K-row slice exactly one
  shard, hence ONE NoC request instead of `per_core_N`. Requests per K-block drop
  gate 48 → 8, down 126 → 6, up 48 → 8. The op validates this width and fails host-side
  if the spec and its own N split disagree.
* **Shard height = exactly one tile-row.** With ROUND_ROBIN_1D, shard id =
  `k * shard_grid_n + gx`, so consecutive K-rows land in DIFFERENT DRAM banks. That
  rotation — not the request size — is what buys bandwidth (§7e, §10d). A K-block-tall
  shard would be a single 27 KB request but would pin each core to one bank, measured at
  246 GB/s, i.e. no better than interleaved. It would also couple the spec to the op's
  `in0_block_w`, which the L1 guard can lower on large models.
* **`n_tiles` need not divide `per_core_N`** (64 vs 6 → 11 shards covering 66). The last
  shard is partially valid and those columns are dropped by the op's existing N-bounds
  guards, so no tensor padding is needed.
* **Device-side reshard, not an ND memory config on `as_tensor`**: the mesh-mapper path
  rank-squeezes the 4D weight and ND-sharded tensors reject that view. This also keeps the
  on-disk weight cache interleaved, so toggling the layout needs no cache rebuild.
* **Kernels select the path with `-D WEIGHTS_ND_SHARDED`**, not `if constexpr`:
  `TensorAccessor` exposes `get_shard_noc_addr` only on its sharded specialisation, and a
  discarded `if constexpr` branch outside a template is still type-checked.

### 11b. Measured result (P150 card 0, same build, one sample per case)

| model | ISL | interleaved | ND-sharded | speedup |
|---|---|---|---|---|
| kimi | 128 | 174.0 us | **139.5** | **1.25x** |
| kimi | 256 | 194.1 | **162.7** | **1.19x** |
| kimi | 512 | 244.3 | **180.4** | **1.35x** |
| kimi | 1024 | 309.7 | 308.2 | 1.00x |
| kimi | 2048 | 592.2 | 590.9 | 1.00x |
| kimi | 4096 | 1186.9 | 1167.6 | 1.02x |
| kimi | 5120 | 1471.0 | 1478.0 | 1.00x |
| glm | 128 | 146.5 | **122.2** | **1.20x** |
| glm | 256 | 156.9 | **132.2** | **1.19x** |
| glm | 512 | 232.3 | **157.1** | **1.48x** |
| glm | 1024-5120 | | ~same | 1.00x |

Sum over the sweep: kimi 4176 → 4031 us, glm 3644 → 3519 us, both **1.036x**.

**1.19-1.48x at isl <= 512, exactly neutral from isl 1024 up.** That split is the
prediction from §7c/§8a confirmed: below ~1024 tokens the fixed weight read is the
critical path, above it the x read and the matmuls are, and this change touches neither.
The isl-256 prediction was 1.21x against 1.19x measured.

Cumulative from the session baseline at isl <= 512: **1.45-1.56x**, with the achieved DRAM
traffic rate going 137 → ~200-220 GB/s.

### 11c. Perf test now covers both layouts

`test_single_routed_expert_perf.py` parametrizes `_WEIGHTS_IDS = ("w_interleaved",
"w_ndshard")` with a separate baseline per (layout, model, ISL) — 32 cases — so a
regression in either layout is caught and the gap between them stays visible. `_MARGIN` is
**8%**: repeated runs of the same build show long ISL stable to <1% but short ISL spanning
up to 11% (w_interleaved glm isl-256 measured 156.9 and 175.4 us), because those cases are
dominated by fixed per-K-block sync latency rather than streaming work. Cases with more
than one observation are centred on their min/max midpoint. Proper fix is multi-iteration
averaging in the harness, not a narrower band.

## 12. Full measurement matrix (2026-08-04, P150 card 0)

The perf test now sweeps **x layout x weight layout x model x ISL = 72 cases**, and the ISL
sweep gained a **64-token** point — 2 tile-rows, so only 2 of the 8 M-rows carry real
tokens and the op is almost entirely the fixed weight read. Device kernel time in us:

| model | ISL | x_rm + IL | x_rm + ND | x_tile + IL | x_tile + ND | best GB/s |
|---|---|---|---|---|---|---|
| kimi | 64 | 162.2 | 132.6 | 143.0 | **120.8** | 213 |
| kimi | 128 | 172.8 | 140.7 | 145.2 | **133.7** | 200 |
| kimi | 256 | 183.1 | 162.6 | 150.0 | **135.6** | 211 |
| kimi | 512 | 251.1 | 179.7 | 178.5 | **167.3** | 195 |
| kimi | 1024 | 308.2 | 307.9 | **279.0** | 280.5 | 145 |
| kimi | 2048 | 591.8 | 590.8 | 534.1 | **533.4** | 105 |
| kimi | 4096 | 1167.8 | 1174.0 | 1044.6 | **1043.4** | 107 |
| kimi | 5120 | 1479.7 | 1488.1 | 1313.2 | **1311.6** | 116 |
| glm | 64 | 142.3 | 118.5 | 127.9 | **113.1** | 195 |
| glm | 128 | 146.8 | 123.1 | 132.4 | **117.6** | 195 |
| glm | 256 | 158.4 | 138.0 | 130.6 | **117.4** | 209 |
| glm | 512 | 218.5 | 157.6 | 150.4 | **147.9** | 189 |
| glm | 1024 | 270.3 | 269.9 | 262.5 | **242.9** | 142 |
| glm | 2048 | 518.1 | 517.8 | 469.2 | **467.1** | 103 |
| glm | 4096 | 1026.8 | 1052.3 | 918.0 | **913.0** | 105 |
| glm | 5120 | 1283.7 | 1291.8 | 1158.2 | **1141.3** | 114 |

(IL = DRAM-interleaved weights, ND = DRAM ND-sharded. GB/s counts x read + weights x chunks
+ output write for the best cell of that row; the x_tile rows read x as bfp8, so their byte
count is lower and the rate is not directly comparable with an x_rm row.)

### 12a. What the two axes are each worth

**ND-sharded weights**, holding x layout fixed:

| | isl 64 | 128 | 256 | 512 | 1024+ |
|---|---|---|---|---|---|
| on x_rm | 1.20-1.22x | 1.19-1.23x | 1.13-1.15x | **1.39-1.40x** | 0.98-1.00x |
| on x_tile | 1.13-1.18x | 1.09-1.13x | 1.11x | 1.02-1.07x | 0.99-1.08x |

**x_tile vs x_rm** (i.e. what the in-op row-major tilize costs), holding weights fixed:

| | isl 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 5120 |
|---|---|---|---|---|---|---|---|---|
| on interleaved | 1.11-1.13x | 1.11-1.19x | 1.21-1.22x | **1.41-1.45x** | 1.03-1.10x | 1.10-1.11x | 1.12x | 1.11-1.13x |
| on ND-sharded | 1.05-1.10x | 1.05x | 1.18-1.20x | 1.07x | 1.10-1.11x | 1.11x | 1.13-1.15x | 1.13x |

Three things fall out of this:

1. **The two optimisations partly overlap.** Each is worth ~1.1-1.4x alone, but stacking
   them gives less than the product — at kimi isl-512, ND alone is 1.40x and x_tile alone
   1.41x, yet both together are only 1.50x (251.1 -> 167.3). They are competing for the same
   critical path: whichever one is removed first exposes the other.
2. **x_tile is worth 1.10-1.13x even at long ISL**, where ND-sharding is worth nothing. That
   is the tilize cost (section 6) showing up directly, and it is the only lever measured so
   far that scales with ISL. Note this is the OP's time only — it excludes whatever the
   caller must pay to produce bfp8 TILE x, which section 6 measured as far more than the
   in-op tilize when done as standalone Tilize + Typecast ops. So x_tile is only a real win
   if the producer emits that layout natively.
3. **Best achieved rate is ~195-213 GB/s at isl <= 512**, against the ~310 GB/s ceiling for
   576 B reads and ~379 GB/s for bank-rotating ones (section 7). Still short, and section 10b
   explains why: the multicast is irreducible and additive with the read on the same RISC.

### 12b. Baseline stability, and a self-inflicted trap

Verification of all 72 cases: **71 pass, 1 residual jitter case**, zero infra errors. Five
short-ISL cases needed re-centring on their min/max midpoint after a second observation
(kimi x_rm+IL isl-256 measured 183.1 and 197.8 us; glm x_tile+ND isl-256 117.4 and 129.6).
`_MARGIN` = 8% covers the observed spread; long-ISL cases sit inside 1%.

**Update after the M=4/w=16 rebaseline (section 13): the noise is BIMODAL and the band had
to become ISL-dependent.** Two full 72-case runs of the same build on the same card had a
median run-to-run ratio of 1.004 — no systematic drift — but the spread split sharply:
within 4.4% for isl >= 1024, out to 15.5% for isl <= 512. Resampling the five outliers four
times each showed two clusters rather than a spread, e.g. w_ndshard glm isl-128 at
115,641 / 115,724 / 125,984 / 126,724 ns (~11 us apart), and four of the five outliers were
high by a near-identical **+17-18 us** regardless of model or ISL — a fixed additive step,
not a proportional one, which is the signature of one occasional stalled read round rather
than clock or thermal drift.

The cause is chunk count: at isl <= 512 the op runs a SINGLE chunk, so one stall lands whole
in the measurement with nothing to average against; from isl 1024 up there are several
chunks. Making the op faster made this worse in relative terms — the same ~17 us absolute
stall is a larger fraction of 150 us than of 190 us. Hence `_MARGIN_SHORT_ISL` = 15% at
isl <= 512 and `_MARGIN_LONG_ISL` = 8% above, with multi-sample cases centred on their
MEDIAN (not the min/max midpoint, which a bimodal sample set misrepresents). Tightening
either band needs multi-iteration averaging in the harness.

Trap worth recording: an earlier verification showed **7** failures, of which only ONE was a
perf miss. The other six were Tracy infrastructure errors
(`profile_log_device.csv is also missing`, `CalledProcessError`) caused by running TWO
device-perf pytest sessions concurrently — `run_model_device_perf_test_per_op` writes to a
fixed `generated/profiler/<subdir>`, so concurrent runs clobber each other's artifacts. The
failure looks exactly like a perf regression. Never overlap perf runs.

### 12c. Perf-test structure

`_EXPECTED_NS` is keyed `(x_layout, weights_layout, model, active)`; `_LAYOUT_IDS` and
`_WEIGHTS_IDS` are both swept. The `-k` filter pins all four id components, so each of the
72 invocations profiles exactly one worker case — without that the ops CSV would hold
several FFN rows and the harness would sum them. Runtime is ~15 min for the full sweep.

## 13. NEXT: in0_block_w and where the K-blocking overhead actually goes

### 13a. The proposal, and what is right about it

The mechanism description is exactly right: the K dimension is broken into blocks of
`[per_core_M, in0_block_w] @ [in0_block_w, per_core_N]`, each accumulated into a separate
partials CB (PACKER_L1_ACC), so per K-block there is a fetch whose latency has to be
covered by something. And the parameters ARE maxed for L1: the guard picks the largest
`per_core_M` that fits (8), then the widest `in0_block_w` that still fits — which for kimi
snaps 16 down to **8**. Measured L1 use is ~1.31 MB of a ~1.49 MB budget. So at small ISL,
where there is very little compute per block to hide anything behind, that per-block fetch
latency is exposed. All of that matches what section 3 and section 7c measured.

### 13b. But the proposed direction looks inverted

Decomposing the measured per-K-block cost on the weight-sender core at isl-128 (kimi,
x_rm, interleaved) into a part that is FIXED per block and a part that SCALES with
`in0_block_w`:

| component | cy/block at w=8 | scales with w? |
|---|---|---|
| ready-barrier + peer-valid wait | 2503 | no — one round per block |
| read completion tail (`async_read_barrier`) | 429 | mostly no |
| read issue | 2070 | yes (w x per_core_N requests) |
| multicast | 1099 | yes (w x per_core_N tiles) |

⇒ fixed ≈ **2932 cy/block**, scaling ≈ **66 cy per weight tile**.

Total gate/up overhead is then `(K/w) * 2932 + K * per_core_N * 66`. The second term does
not depend on w at all — the same `K * per_core_N` tiles are moved either way — so only the
first term moves, and it goes as **K/w**:

| in0_block_w | K-blocks | fixed total | scaling total | overhead |
|---|---|---|---|---|
| 2 | 112 | 328 K cy | 89 K cy | **309 us** |
| 4 | 56 | 164 K | 89 K | **187 us** |
| **8 (today)** | 28 | 82 K | 89 K | **127 us** |
| 14 | 16 | 47 K | 89 K | **101 us** |
| 16 | 14 | 41 K | 89 K | **96 us** |
| 28 | 8 | 23 K | 89 K | **83 us** |

So **shrinking `in0_block_w` should make short ISL worse, not better** — it multiplies the
number of times the per-block barrier round is paid. The lever with the predicted sign is
the opposite one: make `in0_block_w` BIGGER. It is currently clamped to 8 only because L1
is full.

Caveat on this model: it is extrapolated from a single measured point (w=8), and it assumes
the barrier cost really is w-invariant. If part of what is booked as "fixed" actually
scales, the curve flattens. That is precisely why the next step is a sweep, not an argument.

### 13c. The L1 that a short sequence is wasting

The thing that makes a bigger `in0_block_w` affordable: **CB sizing is compile-time
(`per_core_M_max = chunk_M_tiles / GRID_Y = 8`) while the RUNTIME `per_core_M` at short ISL
is 1.** At isl <= 256 the op reserves eight times the M-dimension CB space it actually uses
— `cb_x_rm`, `cb_in0_x`, both gate/up partials, `cb_activated`, `cb_gate_intermed`,
`partials_d` and `cb_in0_down_full` all scale with `per_core_M`. That is the L1 that could
be paying for a wider K-block or a deeper weight pipeline instead.

And it needs no new plumbing: **`chunk_M_tiles` is already an op attribute** (default 64,
in the program-cache key). A caller that knows the token count is small can pass a smaller
one; the guard then has room to keep `in0_block_w` at 16, and at short ISL the smaller chunk
costs nothing because there is only one chunk either way. The runtime picker already adapts
`per_core_M` downward — what it cannot adapt is the CB sizing, so the caller has to choose it.

### 13d. Measured: the sweep (kimi_k26, x_rm, w_interleaved, card 0)

Both knobs were exposed as temporary env overrides in the program factory (`DS_CHUNK_M`,
`DS_W_GU`, plus a `log_info` reporting the config the guard settled on), so the
`(per_core_M_max, in0_block_w)` plane could be swept from the shell with one build. Hooks
removed afterwards; the numbers below are device time in ns, one sample each.

**Knob 1 — w alone at per_core_M_max = 8.** 13b's sign is confirmed:

| w | K-blocks | L1 | isl-64 | isl-256 | isl-2048 |
|---|---|---|---|---|---|
| 2 | 112 | 996 KB | 282,734 | 290,267 | 1,200,176 |
| 4 | 56 | 1104 KB | 200,636 | 251,276 | 736,816 |
| **8 (was default)** | 28 | 1320 KB | 161,792 | 192,305 | 591,630 |
| 16 (guard dropped M to 5) | 14 | 1235 KB | 143,213 | 156,136 | 715,515 |

Predicted +60 us for w=4; measured **+59**. Predicted -31 us for w=16; measured **-36**. The
model's sign and magnitude both hold. It over-predicts only at w=2 (+182 predicted vs +98
measured), i.e. some of the "fixed" barrier cost does overlap once blocks get very short.

**Knob 2 — w at per_core_M_max = 1**, which isolates w from M entirely (chunk_M=8):

| w | K-blocks | L1 | isl-64 | isl-128 | isl-256 |
|---|---|---|---|---|---|
| 8 | 28 | 396 KB | 160,984 | 177,224 | 189,007 |
| 16 | 14 | 545 KB | 143,167 | 146,296 | 153,581 |
| 28 | 8 | 767 KB | 142,900 | 145,512 | 151,513 |
| 56 | 4 | 1287 KB | 141,631 | 152,989 | 151,218 |

The curve **plateaus at w=16**. Beyond that the model over-attributes: it predicts another
13 us going 16 -> 28, measured is ~2 us. So w=16 captures essentially the whole win, and the
extra L1 that w=28/56 costs buys nothing.

### 13e. The prime-divisor trap — a real latent perf bug, found by accident

With w forced to 16 the guard dropped `per_core_M` to 5, and isl >= 512 appeared to regress
badly (+103 us at isl-512). That is **not** a w effect. `per_core_M_for_chunk()` picks the
smallest **DIVISOR** of `per_core_M_max` that covers the tail chunk, and the L1 guard walked
`per_core_M` down by 1 at a time — landing on **5, a prime**, whose only divisors are {1, 5}:

| isl | m_tiles | rows/core the tail needs | max=8 picks | max=5 picks | cost |
|---|---|---|---|---|---|
| 512 | 16 | 2 | 2 | **5** | 2.5x the M-work, +103 us |
| 1024 | 32 | 4 | 4 | **5** | 1.25x, +64 us |
| 2048 | 64 | 3, plus one full chunk | 8, one chunk | **5, two chunks** | 1.25x + an extra chunk |

Confirmed by holding w at 8 and setting chunk_M=40: isl-512 measured 356,874 vs 247,924 at
chunk_M=64 — the regression follows the prime `per_core_M_max`, not the K-block width.

This bites any model whose dims make the guard fire, silently, with no wrong answers — the
tail chunk just does up to 2.5x the M-work it needs to. **Fixed** by offering the guard only
per_core_M values that are DIVISORS of the requested max (8 -> 8, 4, 2, 1), which keeps the
tail ladder at least as fine as the request's own.

### 13f. Chosen config: per_core_M_max = 4, in0_block_w = 16

`kMaxChunkMTiles` 64 -> 32. With per_core_M_max=4 the **existing** default `in0_block_w=16`
already fits (1062 KB of 1379 KB), so the guard no longer fires at all for kimi and no w
logic had to change. 4 is a power of two, so the tail ladder stays fine (13e).

| isl | M=8, w=8 (was) | M=4, w=16 | speedup |
|---|---|---|---|
| 0 | 3,983 | 3,979 | 1.00x |
| 64 | 161,792 | 143,172 | **1.13x** |
| 128 | 172,144 | 146,963 | **1.17x** |
| 256 | 192,305 | 152,941 | **1.26x** |
| 512 | 247,924 | 197,268 | **1.26x** |
| 1024 | 309,273 | 305,239 | 1.01x |
| 2048 | 591,630 | 607,656 | 0.97x |
| 4096 | 1,174,488 | 1,192,784 | 0.98x |
| 5120 | 1,468,690 | 1,490,742 | 0.99x |

The cost above isl-1024 is extra chunk passes (chunk_M 32 vs 64), bounded at 3%. Net win
across the prefill range, concentrated exactly where the sweep spends its time.

**Rejected: per_core_M_max = 2, w = 28** (chunk_M=16, 1001 KB). ~2 us better at isl <= 256
(142,510 / 144,927 / 150,265) but far worse above: 217,529 @512, 390,259 @1024, 764,730
@2048, 1,526,454 @4096, 1,887,283 @5120 — 1.28x SLOWER at 4096. Too many chunk passes to
buy 2 us.

### 13g. Why per_core_M_max is the price of K-block width

The footprint model, validated to the KB against the guard's own arithmetic:

```
footprint(M, w) = base + w * (13.5 + M * 5.06) KB          [x_rm]
    cb_in0_x = M * w * 1088 B        cb_x_rm = M * w * 2 * 2048 B     <- the M*w PRODUCT
    cb_in1_gate / cb_in1_up = w * per_core_N_gu * 2 * 576 B each      <- bfp4, cheap
```

The weights are the *cheap* part of a wider K-block (13.5 KB/w, bfp4). What makes width
expensive is **x staging**, which is sized `M * in0_block_w` tiles in bf16 — at M=8, w=16
that is 136 KB + 512 KB. So `per_core_M_max` sets the PRICE of width: 54 KB/w at M=8,
38.9 at M=5, 18.6 at M=1 (measured slopes matched all three). Halving the max halves the
price, which is why 4/16 fits where 8/16 does not.

### 13h. Full 72-case validation of M=4 / w=16 (2026-08-04, card 0)

Speedup vs the 2026-07-29 baselines, all four layout combinations, both models. 72/72
captured, zero Tracy infra errors, single perf session.

| isl | kimi x_rm/int | kimi x_rm/nds | kimi x_tile/int | kimi x_tile/nds | glm x_rm/int | glm x_rm/nds | glm x_tile/int | glm x_tile/nds |
|---|---|---|---|---|---|---|---|---|
| 64 | **1.13x** | 1.07x | 0.99x | 1.04x | 1.08x | 1.07x | 1.02x | 1.06x |
| 128 | **1.18x** | 1.08x | 0.99x | 1.00x | **1.15x** | 1.06x | 1.03x | 0.99x |
| 256 | **1.21x** | 1.10x | 1.00x | 0.96x | **1.18x** | 1.08x | 1.01x | 1.06x |
| 512 | **1.31x** | **0.91x** | 1.02x | 1.06x | **1.20x** | **0.93x** | 1.00x | 1.04x |
| 1024 | 1.02x | 1.02x | 1.02x | 1.03x | 1.01x | 1.03x | 1.09x | 1.02x |
| 2048 | 0.98x | 1.01x | 0.99x | 1.00x | 1.00x | 1.00x | 0.99x | 1.00x |
| 4096 | 0.99x | 1.01x | 0.99x | 0.98x | 1.00x | 1.03x | 0.99x | 0.99x |
| 5120 | 1.00x | 1.02x | 0.99x | 1.00x | 0.98x | 1.02x | 0.98x | 1.00x |

Two structural readings:

1. **`x_tile` is flat by construction (0.96-1.09x).** It has no `cb_x_rm`, so its footprint
   already fit `in0_block_w=16` at per_core_M_max=8 — the guard never fired on that path.
   It only gave up per_core_M_max, and lost nothing for it. All of the win is on `x_rm`,
   which is the production path (the dispatch emits ROW_MAJOR).
2. **The gains do not stack with ND-sharding.** Both changes attack the same fixed
   per-K-block weight read, so the ND-shard advantage at isl <= 512 shrank from 1.19-1.48x
   to 1.05-1.23x. The second optimization to land collects what the first left.

**One real regression: `x_rm`/`w_ndshard` at isl-512, 0.91x / 0.93x** — the only cells below
0.96 in the matrix, and the only place ND-sharding is now slower than interleaved (196,139
vs 191,997 for kimi). Confirmed real, not jitter, over three samples each (kimi 203,310 /
196,139 / 188,815; glm 172,155 / 166,087 / 170,537; the baselines record the medians).
Chunking is provably identical at isl-512 across old and new (m_tiles=16 => tail needs 2
rows/core; 2 is a divisor of both 8 and 4, so both pick per_core_M=2, one chunk), so the
cause is the width alone. Working hypothesis: with per-tile interleaved requests there are
already `w * per_core_N` = 96+ reads per block, far past any outstanding-transaction limit,
so widening changes nothing; with ND-shard whole-K-row requests there are exactly `w`, so
8 -> 16 doubles the reads in flight per block and may cross that limit. Not verified —
it also fails to explain why only isl-512 regresses while 64/128/256 improve.

### 13i. Why M=4 / w=16 is the knee — bounds on both knobs

Asked directly: what stops us going further, smaller M and wider w? Both ends are measured,
and they bind for different reasons.

**w saturates at 16.** Measured at per_core_M_max=1, where L1 permits anything up to w=56
(13d knob 2): w=16 -> 28 buys ~2 us, w=28 -> 56 buys nothing. The overhead is
`(K/w)*2932 + K*per_core_N*66`; by w=16 the first term is down to ~30 us while the second is
**w-invariant** — it is the bytes actually moved. Nothing can be optimised below the traffic,
so once the per-block fixed cost stops dominating, w has nothing left to collect. The
fixed-cost model over-predicts beyond 16 precisely because it books as "fixed" some barrier
cost that in reality overlaps.

**Lowering M is bounded by chunk count, not by L1.** `chunk_M = M * GRID_Y`, and every chunk
re-reads the ENTIRE weight set (~137 us fixed: 99.7 read + 37.6 mcast, section 7). So halving
M halves the ISL at which an extra full weight pass starts being paid:

| per_core_M_max | chunk_M | one chunk up to | isl-256 | isl-1024 | isl-4096 |
|---|---|---|---|---|---|
| 8 (was) | 64 | m_tiles 64 = isl-2048 | 192,305 | 309,273 | 1,174,488 |
| **4 (chosen)** | 32 | m_tiles 32 = **isl-1024** | 152,941 | 305,239 | 1,192,784 |
| 2 | 16 | m_tiles 16 = isl-512 | 150,265 | 390,259 | 1,526,454 |

M=4 is exactly the knee: chunk_M=32 = 32 m_tiles means **one chunk all the way to isl-1024**,
so the full low-ISL plateau is bought at the largest M that still avoids a second weight pass
in the sweep's dense region. M=2 halves that reach to isl-512 and pays 1.28x at isl-4096 to
buy 2 us at isl-256.

**L1 is no longer the binding constraint at all.** M=4/w=16 uses 1062 KB of 1379 KB — 317 KB
spare that w cannot productively spend. The one axis that can still use it is weight CB
depth, at 13.5 KB per extra K-row (13g: the bfp4 weight CBs are the cheap term). That is
13j.

### 13j. Are we DRAM bound? The read is; the op is not

Achieved rate for the new config, counting ACTUAL DRAM traffic served
(`chunks * 3*K_e*N_h*576` weight bytes + x + output, x and output at `m_tiles * K_e` tiles).
The byte model reproduces section 12's published GB/s to within 0.7%, so it is the same
accounting:

| model | isl | best cell | us | chunks | GB/s actual | GB/s useful | % of 310 | % of 379 |
|---|---|---|---|---|---|---|---|---|
| kimi | 64 | x_tile/nd | 116.7 | 1 | **221** | 221 | 71% | 58% |
| kimi | 256 | x_tile/nd | 141.3 | 1 | 203 | 203 | 65% | 54% |
| kimi | 512 | x_tile/nd | 157.8 | 1 | 206 | 206 | 67% | 54% |
| kimi | 1024 | x_tile/nd | 271.9 | 1 | 148 | 148 | 48% | 39% |
| kimi | 2048 | x_tile/nd | 534.9 | 2 | 151 | 105 | 49% | 40% |
| kimi | 5120 | x_tile/nd | 1315.7 | 5 | 153 | 78 | 49% | 40% |
| glm | 64 | x_tile/nd | 106.5 | 1 | 207 | 207 | 67% | 55% |
| glm | 512 | x_tile/nd | 141.6 | 1 | 197 | 197 | 64% | 52% |
| glm | 5120 | x_tile/nd | 1146.7 | 5 | 151 | 77 | 49% | 40% |

("useful" counts the weights ONCE — the gap between the two columns is redundant re-reads.)

Op-level, 221 GB/s is 58% of the 379 GB/s bank ceiling and ~43% of the 512 GB/s spec, so it
looks far off. But decomposing kimi isl-64's 116.7 us:

```
weight read      67 us    24.77 MB at 368 GB/s = 97% of the 379 GB/s bank ceiling
multicast        37.6 us  DRAM completely IDLE
everything else  ~12 us
                116.6 us   (matches the measured 116.7)
```

**The read phase is already at 97% of the bank limit.** The op only shows 221 GB/s because
~32% of the wall clock is multicast with DRAM idle, and read and multicast are serialised
because THE SAME RISC does both (section 10b: the mcast is irreducible at 37.6 us/chunk and
additive with the read). So more DRAM bandwidth would buy essentially nothing at low ISL.

The lever is **overlap, not bandwidth**: if the multicast hid fully behind the read, isl-64
would go 116.7 -> ~79 us (**1.48x**) and the rate to ~325 GB/s = 86% of the ceiling. That
needs the read and the mcast on separate RISCs, or per-transaction-ID barriers so K-row n's
mcast can start while n+1 is still in flight — 13k.

And note the tension the "useful" column exposes: at isl >= 2048 half the DRAM traffic is
redundant weight re-reads (151 GB/s actual vs 78 useful, 5 chunks). Low ISL wants a wider
`in0_block_w`, high ISL wants a bigger `per_core_M_max` to cut chunk passes, and the two
compete for the same L1 — which is precisely why M=4 sits at a knee rather than an optimum.

### 13k. Can the multicast be overlapped, or turned into unicast?

**What is actually multicast, per K-block** (kimi, `per_core_N_gu = ceil(64/11) = 6`, bfp4
576 B/tile; `gate_block_bytes = in0_block_w_gu * per_core_N_gu * 576`):

| unit | tiles | bytes | count/chunk | total |
|---|---|---|---|---|
| gate block, w=16 | 96 | **55,296 = 54 KB** | 14 | 756 KB |
| up block, w=16 | 96 | **54 KB** | 14 | 756 KB |
| down block | 6 x 21 = 126 | **72,576 = 71 KB** | 11 | 780 KB |
| **per sender per chunk** | | | | **2.35 MB** |

Two independent cross-checks: 2.35 MB at the measured 61.3 GB/s column-multicast rate is
**38.3 us**, against the 37.6 us/chunk measured directly (section 10b); and `cb_in1_gate` =
16*6*2*576 = 108 KB, exactly double-buffering one 54 KB block. At the old w=8 the block was
27 KB. The natural finer unit is one K-row = 6 tiles = **3.4 KB**.

**Dual NoC: already done.** `unified_routed_expert_ffn_reader.cpp:233` sets up `Noc noc_read(0)`
for DRAM weight reads and leaves the kernel's default NoC for mcasts/semaphores, and phase 4
already runs in1_down reads concurrently with the activated mcast. So NoC ports are NOT the
constraint and "multicast on the other NoC" is not an available further win.

The real serialisation is at `reader.cpp:689`: all `w` reads are issued, then
`noc_read.async_read_barrier()` waits for **the whole block**, and only then is the whole
54 KB block multicast. Nothing can go out until the last tile of the block has landed. The
fix is barrier GRANULARITY, not a different NoC: barrier per transaction ID on a subset and
multicast that subset while the rest is still in flight. Splitting the read/mcast across the
two RISCs is the other form, and needs weight CB depth >= 2 blocks so producer and consumer
are not lockstep.

**Unicast instead of multicast: no — 5.8x worse.** Multicast sends ONE copy down a reserved
path to all 8 column receivers; unicast sends 8, i.e. 8 * 2.35 MB = 18.8 MB out of a single
core's port. That port ceils at 86.4 GB/s and the multicast already achieves 61.3 GB/s of it
(71%), so unicast costs **>= 218 us/chunk against 37.6**. It buys an earlier start by paying
8x the traffic. The intent is achievable without the 8x: split the multicast finer (per
K-row, 3.4 KB) so receivers start ~w x earlier on the same single copy.

**And the multicast is not the biggest target anyway.** Per K-block at w=8 (section 13b):

| component | cy | share |
|---|---|---|
| ready-barrier + peer-valid wait | 2503 | **41%** |
| read issue | 2070 | 34% |
| multicast | 1099 | 18% |
| read completion tail | 429 | 7% |

The largest item is the HANDSHAKE — the sender waiting for receivers to free the previous
slot — which is attacked by CB depth, not by NoC placement. That is the one thing the chosen
config leaves on the table: **317 KB of L1 is spare and a K-row of weight buffering costs
13.5 KB** (13g). Deeper weight CBs and per-trid barriers are complementary, and between them
they are what would take isl-64 from 116.7 us toward the ~79 us full-overlap figure in 13j.

### 13l. TRIED AND REVERTED: hoisting the DRAM read above the ready handshake

The per-block decomposition (13b) books 2503 cy — 41% — as "ready-barrier + peer-valid wait",
and the sender reads DRAM into ITS OWN cb_in1 slot, already reserved at the top of the
iteration. So the read has no dependency on the receivers, yet `reader.cpp` waited for the
handshake BEFORE issuing it. Hoisting the read above the wait should have overlapped ~2503 cy
of peer wait with ~2499 cy of read issue + completion, predicting per-block
6101 -> ~3602 cy (~1.7x on the overhead, ~1.28x at isl-64). There was even an in-repo
precedent: in UP_SPLIT mode the writer's `up` read is already triggered from the top of the
loop, above the wait.

Implemented for both the in1 (weights) and in0 (x) senders. 72/72 functional tests pass, so
the handshake was not broken. But the effect is absent:

| case | baseline | hoisted | delta |
|---|---|---|---|
| x_rm/w_int kimi isl-64 | 143,402 | 146,686 | +2.3% |
| x_rm/w_int kimi isl-128 | 146,822 | 142,319 | -3.1% |
| x_rm/w_int kimi isl-256 | 157,270 | 151,001 | -4.0% |
| x_rm/w_int kimi isl-512 | 202,118 | 197,799 | -2.1% |
| x_tile/w_nds kimi isl-64 | 119,066 | 122,973 | +3.3% |
| x_tile/w_nds kimi isl-256 | 141,341 | 142,561 | +0.9% |

Every delta sits inside the +-8-15% short-ISL noise band (12b), and a 1.28x effect would have
been unmistakable. **Reverted** rather than keep churn on the mcast handshake for an
unmeasurable change.

**What this corrects in the model.** The 2503 cy is NOT sender idle time waiting for peers.
Receivers ack BOTH senders at step 1, which precedes the senders' step 2, so by the time a
sender polls its ready sem the count is usually already there — there was no stall for the
read to fill. That cost is the semaphore round trip itself: posted sem writes, L1 polling, and
the valid-sem `set_multicast`. Local work cannot hide it.

**And it explains why w saturates (13d).** If the handshake were a hard 2503 cy per block,
w=16 -> 28 (14 -> 8 blocks) would have saved 6 * 2503 cy = 11 us; measured was ~2 us. So the
handshake is largely in the SHADOW OF COMPUTE — the sender waits while receivers matmul — and
a wider block gives each handshake more compute to hide behind. That is a self-consistent
story for every measurement in section 13: the win from w=8 -> 16 came from halving the
number of rounds, and it flattens once the rounds are cheap relative to the compute they hide
behind.

Corollary: **the two remaining levers named in 13k are both weaker than they looked.** Deeper
weight CBs cannot pipeline the mcast at all (the mcast targets the sender's own CB write
pointer, and a receiver's pointer only advances on push_back, so a receiver cannot hold two
distinct slots — one block in flight is intrinsic to multicasting into a CB slot). And the
handshake is not idle time to be filled. What is left is genuinely the DRAM floor: at isl-64
we are at 116.7 us against ~67 us of weight read at 97% of the bank ceiling plus 37.6 us of
multicast, i.e. close to the structural floor of the current one-RISC-does-both design.

### 13m. Still on the table

- **Weight CB depth at fixed w** (the part of the original latency argument that survives):
  with only double buffering the reader runs one block ahead, so the fetch is exposed. Deeper
  weight CBs cost only 13.5 KB per extra K-row — the cheap axis — and hide latency without
  multiplying the barrier count. Pairs with per-subset (transaction-ID) barriers so K-row n's
  multicast can start while n+1 is still in flight (section 8 item 4).
- **Runtime-selected CB sizing** would remove the tradeoff in 13f entirely, but CB extents
  are compile-time; the op is compiled once for the 5K buffer and must serve any runtime
  token count. `chunk_M_tiles` being an op attribute means a caller *could* compile a
  short-sequence variant, at the cost of a second program-cache entry.

## 14. Where the time ACTUALLY goes: differential ablation (2026-08-05)

Method: temporary env-gated kernel defines that drop ONE component while leaving every
barrier, CB and semaphore intact, so the kernel still completes and timing stays comparable
(`DS_NO_IN1_MCAST`, `DS_NO_IN0_MCAST`, `DS_NO_ACT_MCAST`, `DS_NO_W_READ`, `DS_NO_X_READ`,
`DS_NO_OUT_WRITE`), plus `DS_SKIP_PCC` in the worker so a deliberately wrong result still
reports device time (the perf harness runs the worker with `--check-exit-code`). All hooks
reverted afterwards; 72/72 functional tests pass on the restored tree.

**This is a better instrument than the per-block cycle sums used in sections 3/7/13b.** Those
sums attribute CB waits and handshake stalls to whatever code they sit in, so they
systematically over-credit the read. The differential cost of a component is what actually
disappears when you remove it.

### 14a. Full decomposition, kimi isl-128, x_rm + interleaved

| variant | us | marginal cost | share |
|---|---|---|---|
| base | 146.7 | — | |
| no weight mcast | 127.0 | **19.7** | 13% |
| no weight DRAM read | 115.5 | **31.2** | 21% |
| no x DRAM read | 144.0 | 2.9 | 2% |
| no output DRAM write | 143.9 | 3.0 | 2% |
| no activated mcast | 145.5 | 1.1 | 1% |
| **all DRAM + all mcast removed** | **84.7** | — | **58% is neither** |

The weight read is **31 us, not the 67-100 us** section 7/13 inferred from instrumented
per-block timing. Most of what those attributed to "read" was waiting.

### 14b. The floor is compute, and it is CONSTANT below isl-256

| isl | per_core_M | base us | floor us (no DRAM, no mcast) |
|---|---|---|---|
| 64 | 1 | 142.7 | **84.745** |
| 128 | 1 | 146.7 | **84.687** |
| 256 | 1 | 156.8 | **84.725** |
| 512 | 2 | 204.5 | 146.2 |
| 1024 | 4 | 309.8 | 272.9 |
| 2048 | 4 (x2 chunks) | 591.1 | 541.4 |

Identical to **0.05%** across isl 64/128/256 — because `per_core_M = ceil(m_tiles/GRID_Y)` is 1
for all of them. Fits `floor ~= 23 us fixed + 61 us per per_core_M unit`.

**No amount of DRAM or multicast work can take isl <= 256 below 84.7 us.** At isl-64 the op
performs exactly the same compute as at isl-256 while carrying 4x fewer real tokens: 2 of the
8 M-rows hold tokens, the other 6 compute padding. That waste is in FLOPs, NOT in latency —
every core runs per_core_M=1 either way, so skipping idle cores' math would not shorten the
critical path. (The down phase already skips MAC and pack for OOB rows; section 13's ring fix.)

**RETRACTED (see 14e): the floor is NOT attributable to the matmul MACs.** An earlier draft
divided 84.7 us by the 4074 per-core tile-MACs to get 28 cy/tile-matmul and concluded "near the
FPU limit". That is an inference from a total, not a measurement of the term, and direct
ablation contradicts it.

### 14c. Answers to the six proposed experiments

**(a)/(c) Comment out the multicast — refuted.** Predicted isl-128 ~80 us with mcast gone;
measured **127.0 us**. Multicast is 13% of device time (19.7 us weights + 1.1 us activated +
~0 for x). We are NOT primarily waiting for the multicast to complete. The ~80 us figure turns
out to be almost exactly the floor with the multicast AND the weight read AND every other DRAM
access removed (84.7 us) — the right number attached to the wrong cause.

**(d) Other RISC/NoC for an overlapped weight mcast — already built and RETIRED.**
`program_factory.cpp:483` documents it as `UP_WRITER_MCAST` (mode 1): the writer NoC-1
multicasts `up` down its column. "Bandwidth-optimal, but the NoC-1 worker multicast + posted
atomics collide with fabric CCL ops on NoC 1 and hang the run ... so this scheme is retired."
The perf test runs without fabric, so re-enabling it would look green here and hang in a real
fabric run. Its ceiling is 19.7 us regardless.

**(e) Replace mcast with unicast — refuted twice over.** The premise ("mcast NoC is slow over
10 cores") does not hold: the ACTIVATED mcast spans **11** cores along the M-row and costs
**1.1 us**, while the weight mcast spans 8 and costs 16-17 us — the difference tracks payload
(2.35 MB vs ~135 KB per chunk), not span. And unicast to a column of 8 means 8x the bytes out
of one core's port (8 x 2.35 MB = 18.8 MB) to attack a 19.7 us component.

**(f) Gather >= 8 KB before multicasting — already far above that.** `gate_block_bytes =
in0_block_w_gu * per_core_N_gu * 576`; at w=16 that is 96 tiles = **55,296 B = 54 KB** per
gate block, 54 KB for up, and 71 KB per down block (126 tiles) — 2.35 MB per sender per chunk.
6.75x the proposed threshold already. Section 13d also measured that making blocks *wider*
still (w=28 -> 96 KB, w=56 -> 192 KB) buys ~2 us and then nothing.

### 14d. What this implies for the next lever

The binding constraint at isl <= 256 is per-core tile-MACs, and the grid is badly balanced for
it: `per_core_N_gu = 6` of 64 hidden tiles splits N across GRID_X=11, K is reduced **entirely
within one core** (224 tiles), and M is split across GRID_Y=8 where only 2-8 rows hold real
tokens. So at isl-64, 6 of 8 core-rows do padding work while the 2 real rows each serialise a
224-deep K reduction.

**Split-K across the idle M-rows** is therefore the lever with real headroom: give each of the
idle rows a slice of K, then reduce partials across the column. That attacks the 61 us/M term
directly, up to ~4x at isl-64 where 6 of 8 rows are idle. It needs cross-core partial
reduction, which is a genuine restructure (new reduction phase + semaphores), not a tuning
change — and it is the only remaining idea measured to have multiple-x headroom rather than
the 13-21% that traffic-side changes are bounded by.

Secondary, smaller: the floor's 23 us fixed term is ~31k cy/chunk of pure orchestration across
25 block iterations (~1240 cy each), and the 28 vs 16-19 cy/tile-matmul gap suggests ~1.5x in
subblock sizing / reconfig / SFPU overlap. Both are LLK-level work.

### 14e. Corrections after review

**(d) The weight read and the weight multicast share NoC 0. My earlier "already dual-NoC"
claim was WRONG.** `kernel_types.hpp:134`: `preferred_noc_for_dram_read` returns NOC_0 for
Blackhole (the `default` arm) and `ReaderDataMovementConfig` uses RISCV_1 with that NoC. So in
the reader, `Noc noc_read(0)` and the default-constructed `Noc noc` used for the mcast are the
SAME NoC. The comment at `reader.cpp:233` claiming "dual-NoC parallelism" for the mcast is
misleading; the only genuine dual-NoC work is the writer's NoC-1 `up` read (UP_SPLIT) and its
output writes.

So a **reader-issued** NoC-1 multicast is untried, and is distinct from the retired
UP_WRITER_MCAST (which was the WRITER multicasting on NoC 1). Its ceiling is still 19.7 us
(14a), since removing the mcast entirely already captures any NoC-0 contention relief.
Attempting it — data mcast on NoC 1 with the ordering semaphore left on NoC 0, which breaks the
`linked=true` path ordering by construction — **wedged the device**: the first multi-core op of
the next run (a Tilize, per `dump_running_operations`) hung with no lightweight assert. Any real
attempt must move the valid-sem multicast onto the same NoC as the data.

**(a)/(c) The 84.7 us floor is measured, but NOT yet attributed.** Two attempts failed:

1. *Ablating the MACs* (`#ifndef DS_NO_MATH` around all three `matmul_block` calls) is
   unreliable: with the MAC gone, `dst` holds uninitialised garbage, and NaN/denormal inputs
   change SFPU (silu/sigmoid) timing. It produced physically impossible results — at isl-128 the
   floor got 11 us SLOWER without the MACs (84.8 -> 96.1), while isl-512 got 35 us faster.
2. *Per-RISC kernel durations* do not separate work from waiting. At isl-128 every RISC spans
   essentially the whole op in both configurations:

   | | TOTAL | BRISC (reader) | NCRISC (writer) | TRISC0 unpack | TRISC1 math | TRISC2 pack |
   |---|---|---|---|---|---|---|
   | base | 147.0 | 146.9 | 142.2 | 144.4 | 143.9 | 146.1 |
   | no traffic | 90.0 | 89.8 | 87.6 | 89.8 | 89.3 | 89.7 |

What still stands: the floor is 84.7 us, invariant to 0.05% across isl 64/128/256, and fits
`23 us + 61 us x per_core_M`. **Decomposing it is OPEN.** The right instrument is
`DeviceZoneScopedSumN1` / `DeviceZoneScopedSumN2` (`kernel_profiler.hpp:1043`), which ACCUMULATE
a zone across loop iterations — two slots per RISC per run, so several runs are needed. Plan:
in the reader, one zone around the ready-sem wait and one around read-issue+barrier; in the
compute kernel, one around `cb_wait_front` for weights (starved time) and one around the
matmul+pack subblock loop (busy time). That distinguishes "compute starved" from "compute busy"
directly, which neither attempt above could.

**(e) minimal_matmul (unicast) benched on our shape — it is 1.89x SLOWER, not faster.**
`ttnn.experimental.minimal_matmul` contains no multicast at all. On the isl-64 gate shape
[64, 7168] @ [7168, 2048] with bfp4 weights, grid 11x8, best of 8 configs:

| cfg (Mb,Kb,Nb,sh,sw) | us | weight traffic | GB/s |
|---|---|---|---|
| (1,16,4,1,4) | **90.0** | 16.51 MB (2x dup) | 183 |
| (1,32,1,1,1) | 119.3 | 16.51 MB | 138 |
| (2,16,2,2,2) | 106.9 | 8.26 MB (no dup) | 77 |
| (2,8,1,2,1) | 169.4 | 8.26 MB | 49 |

90 us for ONE matmul; our op does gate+up+down+SwiGLU in 143 us, so 3x = 270 us. Note the
fastest config only reaches 183 GB/s by reading the weights TWICE (M split into 2 blocks); its
useful rate is 92 GB/s, and every duplication-free config is slower. minimal_matmul reaches high
utilisation at M=4096 (the shape its own test uses), where weights amortise over many rows; at
M=64 there is nothing to amortise and unicast strictly loses to a shared multicast.

**(f) Confirmed for WEIGHTS specifically: 54 KB per multicast, already 6.75x the 8 KB
threshold.** `gate_block_bytes = in0_block_w_gu * per_core_N_gu * 576` = 16 * 6 * 576 = 55,296 B
for gate, the same for up, and 126 * 576 = 72,576 B per down block; 2.35 MB per sender per
chunk. Independent of ISL (weights do not scale with per_core_M). 13d already measured that
wider blocks (w=28 -> 96 KB, w=56 -> 192 KB) buy ~2 us and then nothing.

### 14f. RESOLVED: decomposing the floor with scoped device zones

The instrument that worked, after the two failed attempts in 14e. `DeviceZoneScopedN` inside the
gate/up K-block loop, one zone around the `cb_wait_front` for x/gate/up (time BLOCKED on weights)
and one around the matmul+pack subblock loop (time in the compute region). Summed over the
14 gate/up K-blocks, per thread, at kimi isl-128 x_rm+interleaved (same run reported op device
time **146,392 ns**, so these correlate directly):

| thread | matmul+pack region | blocked on weight CBs |
|---|---|---|
| TRISC_0 unpack | 32.9 us | **29.6 us** (per-core range 2.7-32.8) |
| TRISC_1 **math** | **62.3 us** (max 65.5) | 0.2 us |
| TRISC_2 **pack** | **62.9 us** (max 66.1) | 0.2 us |

Readings:

1. **The gate/up matmul+pack region occupies ~62 us of MATH and PACK time — 43% of the op's
   146 us**, for the gate/up phase ALONE (the down phase's 11 K-blocks are not instrumented).
   So the compute pipeline is genuinely occupied, not idle: the 14b "floor is compute pipeline"
   reading holds.
2. **But it is not the MACs.** Ablating `matmul_block` moved almost nothing (14e), so within
   that 62 us the packs, the MATH<->PACK `tile_regs` handshakes and the SFPU dominate over the
   multiply-accumulate itself. "Compute-bound" here means pipeline-bound, not FPU-bound — which
   is why maximising `out_subblock_w` (already done: gu_out_subblock_w = 6 = per_core_N_gu, so
   in1_num_subblocks = 1) matters more than anything MAC-side.
3. **`cb_wait_front` blocks UNPACK for 29.6 us**, and only UNPACK — MATH/PACK sit at 0.2 us
   because the wait is an unpack-side operation. That 29.6 us is the traffic-bound share and it
   lines up with the independently measured 31.2 us weight read + 19.7 us mcast differentials
   (14a) once overlap is accounted for. The per-core spread (2.7 to 32.8 us) is the sender vs
   receiver asymmetry: the weight sender waits far less than its column receivers.

**Recipe for next time** (both mistakes cost a rebuild each): compute kernels need
`#include "tools/profiler/kernel_profiler.hpp"` explicitly — dataflow kernels get it
transitively, and without it the macro fails to compile inside a template with
"there are no arguments to DeviceZoneScopedN that depend on a template parameter". And
`TT_METAL_PROFILER_ACCUMULATE=1` is NOT needed: plain `DeviceZoneScopedN` emits begin/end pairs
per iteration that can be paired and summed offline, which also keeps the per-op perf report
that accumulate mode disables (it warns it is "for INTERNAL RUNTIME-TEAM use only").
