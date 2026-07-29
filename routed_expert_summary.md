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

**Why it is not landed.** The required permutation is fully derived and needs no
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
