# DSpark: a path to 100 tok/s/user

Status: plan. Builds on `DSPARK.md` (what was built), `DSPARK_PARALLEL_VERIFY.md` (wide verify) and
`DSPARK_ROUND_OVERHEAD.md` (round bookkeeping). This document sets the target against measured numbers,
says which of those plans still matter, and orders the work.

100 tok/s/user is **10 ms per emitted token**. Today plain greedy is 27.9 ms per token and DSpark is
31.9 ms per token. Every number below is either *measured* (with its run) or *estimated* (with how).

## 1. Where we are

Measured on Oct 29: the 2000-token movie conversation from `test_dspark_spec.py`, k = 4, bf8 drafter
experts, temperature 1, device sampling and device accept, pipelined verify (barrier round, not yet
streamed):

| run | ms/round | tokens/round | tok/s/user |
|---|---|---|---|
| plain greedy | 27.9 per token | 1 | 35.8 |
| oracle drafts (1/3 corrupted) | 54.5 (fixup 5.0, verify 48.8, accept 0.7) | 2.57 | 47.2 |
| DSpark | 76.9 (fixup 2.2, verify 49.9, accept+draft 24.7) | 2.41 | 31.3 |

DSpark accepted-draft histogram for 0..4 drafts: `[316, 202, 110, 62, 140]` over 830 rounds. So
\(P(m = 0) = 0.38\) and \(E[m] = 1.41\).

The quantities the rest of the plan uses:

| symbol | what | value | source |
|---|---|---|---|
| \(L\) | one row through all 7 stages | 27.9 ms | greedy |
| \(s\) | the cost of one more pipelined row: the slowest stage | 5.1 ms | (48.4 - 27.9) / 4 |
| \(D\) | draft trace (3 stages, head, 5 Markov steps, T = 5) | 16.8 ms | `test_dspark_draft` replay |
| \(O\) | per-round bookkeeping (fixup, uploads) | 2.2 ms | phase split |
| \(\tau\) | tokens per round, \(E[m] + 1\) | 2.41 | histogram |

Already implemented but **not yet measured**: the streamed round (`DSparkSpeculator.streaming`). The host
accepts row by row and starts the next draft at the first reject, while the rows after the reject drain
behind it. Expected ~57 ms/round, ~42 tok/s.

### 1.1 How far the base step is from the hardware

A layer reads about 230 MB of weights per token: attention ~130 M parameters at bfp4 (~73 MB), 6 routed
experts at bfp4 (~120 MB) and the shared expert at bf8 (~38 MB). At TP4 that is ~58 MB per chip, or
~115 µs at Blackhole's ~512 GB/s. The whole token, 40 layers plus the head, is a **~5 ms** floor.

The Oct 27 profile (20 layers, 8 stages, 32k context) measured **~530 µs of kernel time per layer**:
fused_experts ~170 µs, ~208 µs of work every TP chip repeats (mHC 84, MixStreams + norms ~40, step
context share ~25, router ~16, replicated `q_a` / `kv` ~25), and the rest split attention. 40 x 530 µs
is ~21 ms of kernels, so the remaining ~6 ms of \(L\) is hops, trace boundaries and host.

So the base step runs at about **4.6x its bandwidth floor**, and it is latency-bound, not
bandwidth-bound. That has two consequences:

* \(L\) can come down a lot, but by cutting op count and replicated work, not by adding chips.
  Re-partitioning to TP8 / TP32 would not shrink a latency-bound layer, and it adds collectives.
* The pipeline is the right layout for speculation. A pipelined row costs only \(s \approx L/7\).

### 1.2 Why parallel verify is parked

A traced parallel verify measured **102.7 ms for 5 rows** (Oct 28), about 3.5x a single-row stage.
Better kernels will not save it, because the MoE structure is against it:

* Pipelined, 5 rows cost \((7 + 4)/7 = 1.57\times\) a single step. Parallel only wins below 1.57x.
* Each row routes to its own 6 of 384 experts. 5 rows touch up to \(384(1-(1-6/384)^5) \approx 29\)
  distinct experts (independent routing, an upper bound). Routed experts are about half a layer's
  bytes.
* So once a stage is bandwidth-bound, a 5-row stage reads \((73 + 38 + \tfrac{29}{6} \cdot 120)/230
  \approx 3.0\times\) the bytes of one row.

The better the single-row kernels get, the more parallel verify loses. It stays as a test and a
reference. It is not on the path. The same reasoning retires adaptive k: in a streamed pipeline, rows
past a reject are not on the critical path at all (section 2).

## 2. The cost model of a streamed round

The round should follow one rule: **a token enters stage 0 as soon as it is known.** Concretely:

1. Rows `[a, d1..dk]` stream into stage 0, one every \(s\).
2. The host (later, the device) accepts row by row. At the first reject, row \(m\), it has \(a_m\).
3. **\(a_m\) is dispatched into stage 0 at once.** It is the next round's row 0, and it needs no draft.
   The fixup for the rejected rows goes ahead of it on each stage's CQ.
4. The drafter starts from \(a_m\) at the same moment. Its drafts enter stage 0 as soon as they exist,
   behind \(a_m\).
5. The rejected rows still in flight drain ahead of \(a_m\). They cost nothing as long as \(k \cdot s < L\)
   (20 < 28 today), because \(a_m\) is never waiting on them.

The streamed round already coded does 1, 2 and 4. Step 3 is new. A round that accepts \(m\) drafts
then costs

\[
R(m) = L + m\,s + [m \ge 1]\,\max(0,\, D - s) + O
\]

and throughput is \(E[m+1] / E[R]\). Two properties matter:

* **A round that rejects \(d_1\) costs exactly one plain step**, so DSpark can never be slower than
  greedy.
* The draft is exposed only by \(D - s\). Once \(D \le s\) it is free.

For 100 tok/s this gives

\[
L + O + P(m \ge 1)\max(0, D - s) \;\le\; 10\,\tau - (\tau - 1)\, s \quad [\text{ms}]
\]

| step | \(L\) | \(s\) | \(D\) | \(O\) | \(\tau\) | ms/round | tok/s/user |
|---|---|---|---|---|---|---|---|
| today, barrier round (measured) | 27.9 | 5.1 | 24.7 | 2.2 | 2.41 | 76.9 | 31 |
| streamed, draft at first reject (coded) | 27.9 | 5.1 | ~19 | 2.2 | 2.41 | ~56 | ~43 |
| + dispatch \(a_m\) at the reject | 27.9 | 5.1 | ~19 | 2.2 | 2.41 | ~46 | ~52 |
| + drafter at \(\le s\), bookkeeping folded | 27.9 | 5.1 | 5 | 0.5 | 2.41 | ~36 | ~68 |
| + base step 20 ms, stages balanced | 20 | 3.0 | 5 | 0.5 | 2.41 | ~26 | **~93** |
| + acceptance 3.0 | 20 | 3.0 | 5 | 0.5 | 3.0 | ~28 | **~107** |
| or: base step 18 ms, today's acceptance | 18 | 2.6 | 3 | 0.5 | 2.41 | ~22 | **~107** |

So there are two ways to reach 100: **a base step of ~18 ms at today's acceptance, or ~20 ms with
\(\tau \approx 3.0\).** Both need the round restructure and a fast drafter first; neither works without
cutting \(L\). Plain greedy at \(L = 20\) is itself 50 tok/s, so DSpark is then a 2.1x speedup, which is
in line with what speculative decoding usually gives.

## 3. Workstreams

### WS1. Round structure (host, small; 31 -> ~52 tok/s)

1. **Measure the streamed round as it is** (Phase 0). Confirm ~57 ms and exactness at temperature 0.
2. **Dispatch \(a_m\) at the reject**, before the draft is read (step 3 above). `spec_verify_async`
   already holds and releases the drafter's per-step writes. The new part is writing \(a_m\)'s packet
   at the reject and appending the drafts when `spec_draft` returns. The drafter receives \(a_m\)'s tap
   row while it is drafting, so the release order needs the same care as the deadlock fixed on Oct 29
   (an unread D2H page blocks the output stage, which blocks the drafter's receive).
3. **Fold the fixup into the packet.** After step 2 the fixup sits between the reject and \(a_m\)'s
   step on every stage, so its 2.2 ms is on the critical path. Either use the deferred ring commit
   (`DSPARK_ROUND_OVERHEAD.md` 3.2) or carry the rejected range in the step packet and restore at the
   top of the trace (`DSPARK.md` Phase 5 item 2).
4. **k = 5.** The block already computes the 5th proposal and throws it away. In a streamed round, the
   extra row costs \(s\) only when the 4 before it were accepted (17% of rounds today), and those
   rounds then gain up to one more token.
5. **Keep per-row host work under \(s\).** Today each row brings 258 KB of logits to the host. At
   \(s \approx 3\) ms the host accept could become what paces the pipeline. Put argmax and sampling in
   the last stage's trace, and send only the token id plus the two probabilities the accept needs,
   \(p(d_{j+1})\) and the residual draw.

Exit: ≥ 50 tok/s at today's \(L\); greedy stream equals plain greedy; the phase split shows the fixup
off the critical path.

### WS2. Drafter latency (device, medium; D 16.8 -> ≤ s)

The draft's bandwidth floor is ~2-3 ms: per stage up to ~15 distinct bf8 experts across the 5 rows plus
the shared expert (~150 MB/chip, ~0.3 ms), attention ~0.07 ms, the 5-row head ~0.2 ms, plus 5 sequential
Markov steps. 16.8 ms is about 6x that.

1. **Profile one `draft` replay with Tracy first.** Nobody knows its split yet. The mHC change
   (`DSPARK_ROUND_OVERHEAD.md` 3.5) left it at 16.8 ms, so the time is somewhere else.
2. Expected items: one 5-row `fused_experts` call per stage instead of 5 (the serial-path split read
   landed Oct 28); 5-row all-reduces; the 5 sequential Markov argmaxes over the full vocabulary.
   For that last one, a top-k of the head logits, then the Markov bias on the candidates only, may be
   enough.
3. Keep bf8 experts if \(D \le s\) holds. Trade down to bfp4 only with an acceptance A/B (WS4).

Exit: draft trace ≤ 5 ms at T = 5, and ≤ \(s\) once WS3 lands.

### WS3. Base step latency (device, large; L 27.9 -> 18-20 ms, s 5.1 -> ~3 ms)

This also makes plain decode ~1.4-1.5x faster (to ~50 tok/s), so it pays off with or without DSpark.

1. **Profile the full 40-layer, 7-stage single-row decode at ~2k context** (Phase 0). The numbers in
   1.1 come from a 20-layer, 32k-context profile that predates the vocabulary-sharded `lm_head`. Need:
   kernel time per stage, the gaps between traces, and hop latency.
2. **Hops and gaps, ~6 ms (estimated) -> ≤ 2 ms.** Per-stage payload (4 streams x 5120, `pre_mix`,
   cache rows), socket latency, trace boundaries, and the per-row host packet (including the Engram
   hash).
3. **Replicated per-layer work, ~208 µs -> ~80 µs.**
   * mHC: 84 µs. The `fn` weight is copied DRAM -> L1 on every call, on a 10-core program; keep it
     resident or prefetch it.
   * Fuse MixStreams and the norms into the mHC pre/post (~40 µs).
   * Router gate `InterleavedToSharded` (7 µs).
   * Compressor layout conversions on kv-source layers (~60 µs).
   * Single-core output inverse RoPE (23 µs).
   * Build the step context once and forward it, instead of rebuilding it on every stage (75 µs per
     stage).
   * `tt-blaze`'s fused per-layer DeepSeek-V4.1 kernels (mHC + CSA + MoE in one program) are prior art
     for how far this can go.
   * Worth ~5 ms over 40 layers.
4. **`fused_experts` ~170 µs -> ~110 µs per layer** against a ~75 µs floor. Worth ~2.4 ms.
5. **Balance the stages.** \(s\) is the slowest stage (5.1 ms) while the mean is 4.0 ms. Stage 0 carries
   the embedding and Engram; the last stage carries the head, the taps, and the logits send to the
   drafter. Give `plan_layer_placement` per-layer weights so that \(s \approx L/7\).

Exit: greedy ≥ 50 tok/s, \(s \le 3.2\) ms, every stage within 10% of the mean.

### WS4. Acceptance (measure first; 2.41 -> ~3.0, the margin)

The flat per-position acceptance (~0.6 at every position, `DSPARK_ROUND_OVERHEAD.md` 3.6) means a
gain multiplies through every row. But we don't yet know how much of the gap is ours to close.

1. **Find the ceiling with host-only analysis.** During one device run, dump the stash taps and the
   target's tokens. Then run `reference_dspark` on the CPU over that dump, teacher-forced. Its
   \(\tau\) on the same stream is what a perfect device drafter would reach. The gap between it and
   2.41 is device numerics.
2. **Measure greedy \(\tau\)** over the same 2000 tokens with bf8 experts. The only greedy figure so far
   (2.50) is from 26 rounds with bfp4.
3. If there is a numerics gap:
   * Give the drafter its own bf8 copy of the LM head. Today it shares the decode head, which is bfp4;
     a copy is ~175 MB per chip on the drafter row, which has room.
   * Use fp32 router scores (the routing near-ties).
   * Check the precision of the Markov head.
4. k = 5 (WS1.4).
5. **Fix the workload the target is quoted on.** A light-hearted movie chat at temperature 1, top_p 1
   is close to the worst case for acceptance. Report \(\tau\) on a fixed mix (chat, code, math) at a
   fixed sampling config, and state 100 tok/s against that mix.

Exit: the reference ceiling is known; \(\tau\) is reported per prompt type, greedy and sampled.

## 4. Order

| phase | work | expected tok/s/user | needs |
|---|---|---|---|
| 0 | measure: streamed round (sampled and greedy), Tracy of decode, Tracy of `draft`, greedy \(\tau\), tap dump for WS4.1 | ~43 (streamed, as coded) | device runs only |
| 1 | WS1.2-1.5 | ~52 | host + small traces |
| 2 | WS2 | ~68 | drafter kernels |
| 3 | WS3 (can start in parallel with 1-2) | ~93 | decode kernels |
| 4 | WS4 fixes, if 4.1 shows a gap | 100-110 | drafter numerics |

Phase 0 can redirect the plan. If the reference ceiling in WS4.1 is well above 3.0, acceptance moves
ahead of WS3. If the profile shows that hops are most of \(L\), WS3.2 goes first.

Phase 0 device commands (to be run by hand; each one opens the device):

```
DEEPSEEK_V4_SPEC_TEMPERATURE=0 DEEPSEEK_V4_SPEC_VERIFY=pipelined pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_dspark_spec.py::test_dspark_spec_decode' -x --tb=short 2>&1 | tee /tmp/dspark_greedy.log
DEEPSEEK_V4_SPEC_TEMPERATURE=1 DEEPSEEK_V4_SPEC_VERIFY=pipelined pytest 'models/experimental/deepseek_v41_flash/tests/decode/test_dspark_spec.py::test_dspark_spec_decode' -x --tb=short 2>&1 | tee /tmp/dspark_sampled.log
```

## 5. What not to do

* **Parallel verify** (section 1.2). Revisit only if routing turns out to be strongly shared across
  adjacent tokens, which would show as a small expert union in the 5-row `fused_experts` call.
* **Adaptive k.** In a streamed round the rows past a reject are off the critical path. A longer k
  costs nothing until it is accepted.
* **Wider TP or a re-partition.** The layer is latency-bound (section 1.1).
* **A device-resident round loop** (last stage -> stage 0 token socket, drafter -> stage 0 draft socket).
  It removes the host from the per-row path, but it needs data-dependent flushes inside traces. Only
  build it if WS1.5 leaves the host pacing \(s\).

## 6. Risks

* **Deadlocks in the streamed round.** Holding and releasing the drafter's per-step receives against a
  single-page D2H FIFO has already deadlocked once. Dispatching \(a_m\) early adds one more ordering.
  Every phase should run the 2000-token conversation, not 64 tokens.
* **The host pacing the pipeline.** At \(s \approx 3\) ms, the per-row host loop (packet, Engram hash,
  output read, accept) must stay well under 3 ms. Measure it in Phase 1.
* **L1 on the last stage.** In-trace sampling and the logits send to the drafter sit beside the
  prefetch rings, the same budget that forced the drafter onto single-row ops.
* **Acceptance depends on the workload.** At today's 2.41, 100 tok/s needs \(L \approx 18\) ms, which is
  WS3's optimistic end. The target should be stated with its prompt mix and sampling config (WS4.5).
* **Estimates.** The \(L\) and \(s\) targets in WS3 rest on a profile of a different configuration.
  Phase 0's profile replaces them, and the table in section 2 should be recomputed from it.
