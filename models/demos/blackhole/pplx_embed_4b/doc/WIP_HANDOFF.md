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

- **Profile artifact, not yet edited.** Plan: give the two GenericOp rows measured floors the way the SDPA section
  does (tag "compute + DM", hover shows both floors): heads op compute-only 115.6 / 210.7 / 117.0 µs (bs8 / 16 / bs32
  chunk), DM-only 109.2 / 223.9 / 110.9; add+norm compute-only 62.5 / 88.7 / 150.0, DM-only 79.6 / 118.3 / 253.2
  (§67 tables; the DM-only floors include the copy compute, so they are upper bounds). Only roof / bound fields
  change, no new profile needed, but the profiled times predate the cos / sin change (~2% stale on the heads rows).
  Waiting on the user: edit now, or after a fresh profile. The artifact's current tags: heads "L1 / compute" (fine),
  add+norm "L1 / compute" at bs8 / 16 (wrong: DM-bound) and "DRAM" at bs32 (right). Generator scripts are still not
  in the repo (see memory: profile-artifact).
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
