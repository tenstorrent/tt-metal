# WIP handoff (temporary — delete once the plan below is done or moved)

Picked up 2026-09-29 on bh-lb-120-a07u24: every P150b opens at 12×10 (checked chips 0 / 3 / 7), the Release build is
current, bs1 baseline matches the old box (cold 15.6 / sustained 15.8 ms).

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. ~~bs1 fused SwiGLU with K_block 40.~~ Done, negative (NEGATIVE_RESULTS §61): at M=512 the SwiGLU is only 3% exposed
   and the fused kernel is data-movement-bound. bs1 stays unfused.
2. ~~Cheaper SwiGLU in FF1+FF3.~~ Landed (§62): Schraudolph exp + bare SFPARECIP, 71% less SFPU work, STS-B within
   noise; sustained −1.8 / −0.3 / −0.5% at bs8 / 16 / 32 against a no-SFPU ceiling of −3.3 / −3.8% (bs16 / 32).
   The tail exposure was minimal_matmul's output writer (§63, landed: cold bs16 / 32 −1.8 / −2.1%, sustained
   −0.7 / −0.6%, bs8 sustained +0.4%). Left: the partial-sum add (66-69 µs, 4% at bs16; only K_block 80 avoids it and
   that fits only small blocks), and the power cap eating most cold gains (sustained gets ~⅓).
3. SDPA DRAM traffic. Landed at bs8 / 16 (§64): only the K / V reads matter; K / V in L1 (`QWEN_HEADS_KV_L1`), SDPA
   −15 / −13%, cold −1.7 / −0.8%, sustained 0.0 / −0.6%, bit-identical. bs1 already has Q/K/V in L1. bs32 landed with
   4 quarter-batch QKV chunks (§66: replay −1.1 ms, cold −0.7 / −1.5 ms). The head-major QKV write (#57722) is untried.
4. SDPA compute (§65). Compute-only / DM-only floors measured (bs32: 519 / 483 µs against 597); the pack thread paced
   the Q·Kᵀ / exp loop. Landed: row sums on the math thread (SDPA −6 / −7 / −6%). Done with SDPA compute: the exp is
   already the SFPLOADMACRO Schraudolph path at its rated speed. SDPA's remaining gap to its floors is each core's
   cold start on its first head's K / V (a next-head prefetch was negative).

5. ~~Trid pipelining in the custom ops (§67).~~ Done (2026-09-30, commits 7d40ada / 9844602 / b9c2258). Heads op:
   compute-bound at bs8 / 16 / 32 (full within 3-5 µs of compute-only; Q's DRAM write fully hidden), cos / sin
   double-buffered (−2.7% per call, bit-identical, default `QWEN_HEADS_ROT_DB=1`). Add+norm: data-movement-bound
   everywhere (90-92% of its DM floor: DRAM traffic at bs32, per-wave read / write latency at bs8 / 16); exchange on its
   own trid landed but neutral (`QWEN_ADD_NORM_PART_TRID=1`); two-wave CBs negative (bs8 +1.8%, bs16 / 32 do not fit).
   E2e after all of it: bs8 76.7 / 99, bs16 143.3 / 194, bs32 291 / 378 ms cold / sustained (chips 1 / 2 / 0).

## Where we stopped (2026-10-01, HEAD bc9b069 on `amorrison/high-batch-optim`, not pushed)

**Numbers now** (e2e from `sustained_run.sh`, cold / sustained ms; device = one profiled replay at 1.35 GHz):

| batch | cold | sustained | × H200 sustained | device replay | practical roofline | ideal, whole model (120 cores, 512 GB/s) |
|---|---|---|---|---|---|---|
| 1 (3164ba1) | 14.1 | 14.3 | 2.63× | 13.96 | 7.65 | 5.84 |
| 8 (cdb9143) | 76.7 | 99.1 | 3.00× | 76.07 | 59.7 | 46.7 |
| 16 (cdb9143) | 143.3 | 193.1 | 2.87× | 142.1 | 114.3 | 93.4 |
| 32 (cdb9143) | 291.7 | 377.6 | 2.71× | 279.1 | 230.4 | 186.9 |

bs8 / 16 / 32 code is unchanged since cdb9143 (today's changes are bs1-only), so their profiles are current.

**Build.** The installed `_ttnncpp.so` / `_ttnn.so` include 61fb987 (host C++ in the matmul 1D factory and the device-op
factory selector), built with `ninja -C build ttnn/_ttnncpp.so ttnn/_ttnn.so` and copied over `build/lib/` and
`ttnn/ttnn/_ttnn.so`. A fresh checkout or another box needs that rebuild, or bs1's FF1 / FF3 fail (the Metal 2.0 1D
factory rejects a DRAM width-sharded in1). All 8 chips were reset at 15:30 and open at 12×10.

**Profile artifact** (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn, version 26). Per-op tables with practical
rooflines (attainable 450 GB/s, measured device-time floors for SDPA / heads / add+RMSNorm at bs8-32 and for the bs1
heads op and SwiGLU), a bs16 / bs32 section (measured vs practical vs ideal, per op group, where the gap goes) and the
whole-model ideal line on the e2e chart. The page template (sections, JS) lives only in the published page: regenerate
by reading the artifact, then `python3 perf_tools/profile_page.py <saved page.html> runs.json <out.html>` (PERF_GUIDE
§5) and republish to the same URL. runs.json for the current state:

```json
{"commit": "3164ba1 (bs1) · cdb9143 (bs8 / bs16 / bs32)", "date": "2026-10-01", "batches": [
 {"bs": 1, "report": "2026_10_01_19_06_29", "e2e": {"cold": 14.1, "sus": 14.3, "aiclk": "1350", "power": "n/a"}},
 {"bs": 8, "report": "2026_10_01_15_38_45", "e2e": {"cold": 76.7, "sus": 99.1, "aiclk": "1056 (1025-1168)", "power": 86}},
 {"bs": 16, "report": "2026_10_01_15_43_17", "e2e": {"cold": 143.3, "sus": 193.1, "aiclk": "1000 (993-1012)", "power": 104}},
 {"bs": 32, "report": "2026_10_01_15_48_29", "e2e": {"cold": 291.7, "sus": 377.6, "aiclk": "993 (981-1037)", "power": 139}}]}
```

**Done today (bs1, 15.6 -> 14.1 ms cold):**
- SwiGLU product through `silu_mul` mode 3 (minimal_matmul's single-pass bfp8-sized SwiGLU): 52.1 -> 30.0 µs per call,
  more accurate than stock (§68). Now at 95% of its compute floor. The 3-segment LUT sigmoid hits the read floor
  (18.6 µs) at 7x the error; a 6-segment `lut2` sigmoid is the untried middle (<= 10 µs / layer).
- FF1 / FF3 on the 1D matmul over 120 cores with DRAM-streamed weights (tt-metal 61fb987 + model 3164ba1): 69.7 ->
  62.2 µs per call (§69). The bs1 matmuls are compute-kernel-bound (~21 cycles per tile matmul vs 16, data movement
  hidden; config sweeps exhausted); QKV / WO / FF2 cannot fill 120 cores (192 / 80 N tiles).
- bs1 SDPA closed (§70): compute-bound at 91% of its compute floor; more cores need ragged row groups in the streaming
  compute (q160 picks 1-row groups) and K / V delivery for heads spanning grid rows; bounded at ~ -0.17 ms.

**Where the batched gap is** (bs16, from the artifact's new section; bs32 proportionally the same): sustained 193.1 ->
cold 143.3 is the power cap (49.8 ms, AICLK ~1000 MHz; the ideal at that clock is ~126 ms); cold -> device 1.2 ms;
device -> practical 27.8 ms of kernels above their bounds (FF1+FF3 14.7, QKV 4.8, FF2 3.7, WO 2.2); practical -> ideal
per op 13.5 ms, all of it the SDPA / heads / add+RMSNorm floors (softmax, norms, RoPE); ideal per op -> whole model
7.4 ms of activations crossing DRAM.

**Next, in the order I would take them:**
1. Batched matmuls above their bounds (~25 ms at bs16, ~41 ms at bs32; `minimal_matmul` FF1+FF3 at 1.27-1.33x its
   bound). Start with compute-only / DM-only floors at today's configs (`bench_mm_ablate.py`, `bench_ff13_fused_ablate.py`)
   to see whether it is still the partial-sum add (§60/§62) or per-block overhead as at bs1.
2. The power cap (the largest single step). Faster kernels have lowered the settled clock before (§59): measure
   J / pass, not only ms, when A/B-ing; tt-smi sampling is too sparse for J / inference today.
3. Add+norm at bs8 / 16 (DM-bound on per-wave latency; leads below) and the heads op read path.
4. Housekeeping: upstream 61fb987 as its own tt-metal PR (432 matmul unit tests passed); `lut2` sigmoid for bs1.

- **Add+norm leads, if revisited.** What is exposed is each wave's read / write latency (no reads: −15 / −15 / −42 µs;
  no writes: −4 / −4 / −58 µs). Options not tried: software-pipeline the compute (next wave's add / square / partial
  before this wave's normalise) with CB 8 turned into a ring of 2-3 waves to free L1 (it holds every wave now: 147 KB
  per core at bs32); larger read batches per barrier / trid ping-pong in the reader.
- **Heads op, only if its compute gets faster:** the read path is next (read-only ≈ compute at bs8 / 32; ~2 GB/s per
  core from L1, one barrier per 26 KB unit) — that is where trid-pipelined reads would pay.
- **Build hygiene.** After any host-side C++ change: rebuild (PERF_GUIDE §4). A stale `.so` against newer kernels
  deadlocks in the first warmup prefill instead of erroring (NEGATIVE_RESULTS §67).
- **Tools added today** (`perf_tools/`): `profile_page.py` (artifact data + rooflines), `device_kernel_us.py` (device
  µs per call of a traced bench under the profiler; use it for floors, wall-clock includes dispatch gaps),
  `mm_legacy_variants.py` + `bench_bs1_mm_ablate.py`, `bench_bs1_ff13_1d.py`, `bench_bs1_swiglu.py`,
  `bench_silu_mul_floors.py`, `bench_sdpa_bs1_floors.py`; `bench_heads_placement_ablate.py` `HB1=1` (bs1 call),
  `bench_add_norm_ablate.py` `AN_A_L1=1`. Kernel-variant trees must run from a directory with no `ttnn/` tree.

`sustained_run.sh` reports AICLK / power over the sustained window and J/inference, but tt-smi samples swing 30-155 W
within a window (host gaps), so J/inference is too noisy to rank variants yet.
