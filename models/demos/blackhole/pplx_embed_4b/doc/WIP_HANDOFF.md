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

## Where we stopped (2026-09-30)

- **Profile artifact: re-profiled and updated (2026-10-01, version 21).** All four batch sizes profiled at cdb9143 on
  chip 0 after a full board reset (reports `2026_10_01_15_34_46` / `15_38_45` / `15_43_17` / `15_48_29`; tt-smi held
  chip 0 at 1350 MHz through every replay). E2e (3 rounds, chips 4 / 1 / 2 / 0): bs1 15.6 / 15.7, bs8 76.7 / 99.1,
  bs16 143.3 / 193.1, bs32 291.7 / 377.6 ms cold / sustained; AICLK 1056 / 1000 / 993. Roofline changes: DRAM at the
  attainable 450 GB/s (best stock streaming op, bf16 add; 512 is the datasheet); the heads op / add+norm get measured
  device-time floors like SDPA (`device_kernel_us.py`; wall-clock bench floors include the trace's dispatch gap and
  read 3-8% high). Replay roofline vs device: bs1 38%, bs8 78%, bs16 80%, bs32 83%. The generator is now
  `perf_tools/profile_page.py` (PERF_GUIDE §5). Not modelled yet: the all-L1 vector ops (bs1 ~3.3 ms of 15.3 have a
  zero bound: SwiGLU mul, residual adds, LayerNorm, heads op).
- **bs1 focus (2026-10-01).** Gap table at cdb9143 (device µs per call vs floor): matmuls 10.2 ms (67%; FF1 / FF3 /
  FF2 70 µs vs 48 on their 96 cores, 38 on 120), SwiGLU product 1.88 ms, heads op 1.02, SDPA 0.98 (floor 9 µs
  compute), residual adds + LayerNorm 1.1 (142 calls of 7-8 µs). Landed: SwiGLU product via `silu_mul` mode 3,
  15.6 -> 14.8 ms cold (NEGATIVE_RESULTS §68). Next candidates: a 6-segment `lut2` sigmoid (<= 10 µs / layer if it
  holds accuracy), then the matmuls (M = 16 tile rows does not split over 10 grid rows: a different decomposition to
  use all 120 cores), heads op, SDPA.
- **Add+norm leads, if revisited.** What is exposed is each wave's read / write latency (no reads: −15 / −15 / −42 µs;
  no writes: −4 / −4 / −58 µs). Options not tried: software-pipeline the compute (next wave's add / square / partial
  before this wave's normalise) with CB 8 turned into a ring of 2-3 waves to free L1 (it holds every wave now: 147 KB
  per core at bs32); larger read batches per barrier / trid ping-pong in the reader.
- **Heads op, only if its compute gets faster:** the read path is next (read-only ≈ compute at bs8 / 32; ~2 GB/s per
  core from L1, one barrier per 26 KB unit) — that is where trid-pipelined reads would pay.
- **Build hygiene.** The Release build was stale on 2026-09-30 (older than 0458a99's SDPA host change) and every run
  hung in the first warmup prefill; rebuilt at 22:31. After any host-side C++ change: `./build_metal.sh` (PERF_GUIDE §4).
  All 8 chips were reset and open at 12×10 as of 22:13.

`sustained_run.sh` reports AICLK / power over the sustained window and J/inference, but tt-smi samples swing 30-155 W
within a window (host gaps), so J/inference is too noisy to rank variants yet.
