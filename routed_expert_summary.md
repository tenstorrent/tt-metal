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

## 6. Secondary findings

- **Compute is near HW peak**: 14.8 cycles per 32³ tile-MAC. matmul busy ≈ 1.35M cycles
  ≈ 1020 µs at isl-5120 with 88 cores ⇒ a hard ~217 GB/s ceiling at long ISL. Long-ISL
  work is compute-bound; the DRAM headroom is at **short/mid ISL**.
- **x row-major → bfp8 tilize costs 27% of runtime at isl-5120** and is done redundantly
  on all 11 cores of each M-row. Cheapest fix is upstream: have dispatch emit bfp8 TILE x
  (removes the tilize *and* halves x bytes, 73.4 → 39 MB).
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

## 8. Revised path to 3× on isl-256

Ranked by measured value. Target: 198 → ~70 µs (≈350–410 GB/s of total traffic).

1. **Multi-tile weight pages — the only change that can reach 3×.** Make each core's
   (K-row × `per_core_N`) slice a *single* DRAM page (6 tiles = 3456 B) instead of 6
   separate 576 B pages. Two compounding effects: request count ÷6 (issue 113.7K → ~18K cy)
   **and** the ceiling moves 310 → ~400 GB/s per §7b. Needs host-side weight relayout
   (DRAM-sharded or custom page size) in `TtRoutedExpert` plus the matching
   `TensorAccessor` in the reader — i.e. it changes the layout the model hands the op.
   Open question for the op owner: relayout upstream, or accept both layouts behind a flag?
2. **2 weight senders per column.** Now safe: the remap made the weight-sender and
   x-sender sets disjoint, and 22 issuing cores sits inside the 16–32 sweet spot. Its real
   value is halving the **multicast** on the critical RISC (50.8K → 25.4K) — multicast
   volume itself is irreducible (it is already a true multicast), it can only be split
   across senders. Expect ~1.7× on its own; read time improves only 249 → ~310 GB/s.
3. **Residual sync (~61K cy)** — deeper weight CBs (3–4 slots) plus per-subset
   (transaction-ID) barriers so the mcast of K-row *n* starts while *n+1* is still in
   flight, instead of one `async_read_barrier` over everything.

Hard arithmetic: 24.8 MB of weights at the 576 B ceiling (310 GB/s) is **80 µs**, so
without item 1 the best possible isl-256 is ~2.5×. At the ≥1024 B ceiling (400 GB/s) it is
62 µs, so **3× is right at the hardware limit and requires item 1.**

Superseded from the pre-reference plan: "stateful NoC reads (`set_state`/`read_with_state`)
to cut per-request issue cost" — still real for the per-core rate, but §7a shows aggregate
bandwidth is capped by the 576 B page size well before per-core issue cost matters, so it
is a second-order fix behind item 1.

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
