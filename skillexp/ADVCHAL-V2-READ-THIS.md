# advchal-v2 — read this one file

What stage 02b (`$advisor-challenger`) actually did in each of 15 test cells, what it left on the
table, and what separates a model the shard advisor can help from one it cannot.

Everything here is reconstructed from the cells' own session transcripts
(`~/skillexp-logs/p-advchal-v2-*/02-02b-advisor-challenger.jsonl`) — 149 harness measurements and
26 oracle runs — not from the cells' self-reported summaries. Where a cell's summary and its log
disagree, the log wins and the disagreement is noted.

**Where the detail lives.** This file is the account. The supporting tiers are
[`ADVCHAL-V2-MEASUREMENTS.md`](ADVCHAL-V2-MEASUREMENTS.md) (every measurement, in order, per cell),
[`ADVCHAL-V2-ORACLES.md`](ADVCHAL-V2-ORACLES.md) (the correctness bar each cell held itself to),
[`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md) (every op the advisor wanted to move),
[`ADVCHAL-V2-PER-CELL.md`](ADVCHAL-V2-PER-CELL.md) (attribution accounting per cell),
[`ADVCHAL-V2-RESULTS.md`](ADVCHAL-V2-RESULTS.md) (the headline table), and
`advchal-v2-narrative.json` / `advchal-v2-data.json` (machine-readable).

---

## 1. What the stage was asked to do

Measure how much of a decoder's speed the **shard advisor** (`$shard-advise` / `ttnn-advise`, pinned at
tt-mlir `618cd4e75d`) can be credited with — not how fast the decoder can be made.

The method is deliberately narrow:

1. **Freeze** the incoming decoder as the control. It is never re-tuned. Whatever it does is the baseline.
2. **Capture** the shipped graph, run the advisor on it, and reconcile the advisor's plan against a
   real op-level profile: which conversions does the shipped decoder pay that the advisor's plan does
   not place? That difference is the **ceiling** — the most the advisor could possibly be worth.
3. **Screen** the ranked candidates on hardware, one at a time, under a fixed harness: ≥10 untimed
   warm-ups, then 5 timed blocks, each the mean of ≥50 traced decode replays.
4. **Ship** a candidate only if *every* one of its 5 blocks beats *every* one of the control's 5 blocks
   (the non-overlap rule), it passes a correctness oracle, and it re-confirms in a fresh process.
5. The **delta is the result.** A tie goes to the incumbent. A zero is a publishable finding.

Two numbers matter for reading any cell:

- **noise floor** — the max−min spread of the control's 5 blocks. Nothing smaller than this is
  measurable. It ranged from **0.146 µs** (llama-1B) to **14.5 µs** (north-mini onA), a 99× spread.
- **band** — the noise floor scaled by the layer count, i.e. the uncertainty on the extrapolated
  full-model number. A per-model gain smaller than its band is real per layer but **not established at
  model level**. Two cells shipped inside their band; one shipped 4× outside it.

---

## 2. The 15 cells at a glance

| model | cell | control ms/layer | what the advisor pointed at | what shipped | model-level |
|---|---|---|---|---|---|
| llama-3.2-1B | `exp` | 0.3731 | 2.8 µs/layer ceiling | **nothing** | 0.0 % — honest zero |
| llama-3.1-8B | `exp17` | 0.6650 | 4.4 µs/layer ceiling | **nothing** | 0.0 % — honest zero |
| phi-3.5-mini | **A** | 0.6570 | RoPE → L1, 32-core rect | `rope_l1_rect32` | **−8.75 %** (−1,594 µs) |
| phi-3.5-mini | **B** | 0.7888 | RoPE → L1 chain | `rope_l1_chain` | **−5.74 %** (−1,285 µs) |
| phi-3.5-mini | **FN** | 0.8072 | RoPE → L1 **+ 1-core norm → 11** | RoPE only | −4.91 % — **−13.24 % was measured and discarded** |
| phi-3.5-mini | exp17 | 1.1009 | 83.6 µs/layer ceiling (largest) | **nothing** | 0.0 % — every direction overlapped or hard-failed |
| qwen3.6-27B | `fuse` | 1.2083 full<br>19.1402 linear | packed-QKV boundary | `packed_qkv_l1_chain` | −445.7 µs — **inside its ±618.5 µs band** |
| qwen3.6-27B | **B** | 1.4494 full<br>15.8498 linear | 33.7 µs/full-layer ceiling | **nothing** | 0.0 % — geometry hard-failed |
| gemma-4-12B | `exp11` | 1.2541 sliding<br>1.3774 full | K/V/Q residency + MLP handoff | `Q+K+V+MLP` both kinds<br>+ output chain full only | **−1.14 %** (−666.8 µs) |
| gemma-4-26B | `exp` | 1.2597 | sliding-O DRAM sharding | `sliding_attention_o_chain` | **−147.9 µs**, band ±36.5 |
| gemma-4-26B | **onA** | 1.8252 sliding | **1-core norm → 88** | `advisor_norm88` | **−12.98 %/layer** |
| gemma-4-26B | **FN** | 1.3412 sliding<br>1.5394 full | concat-heads → output projection | `advisor_concat_projection` | **−2.04 %** (−791.7 µs)<br>*88-core norm regressed here* |
| north-mini | **FN** | 0.5537 MoE | **1-core MoE norm → 22** | norm at **32** cores | **−10.23 %** (−2,551 µs) |
| north-mini | **B** | 0.6138 / 0.2033 dense | dense MLP DRAM-sharded residency | **nothing** | 0.0 % — all geometries slower or stalled |
| north-mini | **onA** | 0.2918 dense<br>0.8465 sparse | — | **nothing** | 0.0 % — sparse MoE untraceable |

Eight cells shipped something. Seven shipped nothing. Of the seven zeros, **two are honest zeros on
well-placed decoders, three are structural (the advisor could not see the layer at all), one is a
geometry wall, and one is the phi exp17 case where every direction was legal but not separable.**

---

## 3. What happened, one test at a time

### 3.1 llama-3.2-1B — the cleanest zero in the corpus

Control **0.373080 ms/layer**, noise floor **0.146 µs** — the tightest floor anywhere, 0.039 % of the
layer. This cell could resolve differences smaller than any other, and still found nothing.

The advisor's whole ceiling was **2.822 µs/layer**. Three directions were screened:

| candidate | result |
|---|---|
| `dense_2_residual_chain_64` — legal exactly-dividing 64-core extended chain | 0.4203 ms, **+12.67 %** |
| `concat_output_dram` — DRAM concat output | 0.3732 ms, +0.04 %, repeats **overlapped** the control → rejected |
| SDPA → concat chain | infeasible: paged GQA SDPA rejects sharded output |
| Rotary-K DRAM | incompatible with paged cache update |

The advisor **agreed with 72.84 %** of the profiled window (258.260 µs/layer). Its RMSNorm was already
on **32 cores** and the advisor wanted to *reduce* it to 22 — the fewer-cores direction, worth a
notional 6.0 µs and slower in practice.

**Verdict: nothing to win. 5,672.576 µs/model before and after, ±2.336 µs.** The stage caught its own
bookkeeping defect here: the hard-error SDPA candidate was carrying the incumbent's median in
`measured_ms` even though it failed before timing; corrected to `null`.

### 3.2 llama-3.1-8B — the same story, bigger model

Control **0.665046 ms/layer**, floor 0.697 µs. Ceiling **4.394 µs/layer** = 140.6 µs/model.

| candidate | result |
|---|---|
| `dense_2_skip_attn_output_reshard` | 0.6676 ms, **+0.38 %** |
| `dense_3_skip_mlp_output_reshard` | 0.6677 ms, **+0.40 %** |
| `dense_geometry_64` — exactly-dividing 64 cores | 0.6931 ms, **+4.22 %** |
| top-ranked boundary candidate | hard error: concat-heads requires sharded input |

The advisor independently re-derived **76.32 %** of the measured window. RMSNorm again already on 32
cores, advisor wanted 22 (7.1 µs, the fewer-cores direction).

**Verdict: 20,747.296 µs/model before and after, ±22.304 µs. Zero.**

**Why both llamas are zero is the most useful negative result in the corpus.** These decoders arrived
with their reductions already spread across 32 cores and their chains already L1-resident. The advisor
had nothing structural to fix, and the only thing it wanted — fewer cores on the norm — is its known
bias and measured slower. This is a *demonstrated* zero, not a failure to look.

### 3.3 phi-3.5-mini — four versions, four different answers

Four arms started from four different `optimized_decoder.py` snapshots. Their controls span **1.68×**
(0.6570 → 1.1009 ms/layer). All four met the same advisor family: phi's split-half RoPE leaves
conversions on the boundary between the rotary halves and the QKV projection, and the advisor wants
that whole run L1-resident.

| | control ms | best candidate | shipped | model-level |
|---|---|---|---|---|
| **A** (`fuse-advise onA`) | 0.6570 | `rope_l1_rect32` 0.6072 (−7.58 %) | yes | **−8.75 %** |
| **B** (`fuse-noadvise`) | 0.7888 | `rope_l1_chain` 0.7487 (−5.09 %) | yes | **−5.74 %** |
| **FN** (`nofuse-noadvise`) | 0.8072 | `rope+norm11` 0.7003 (**−13.24 %**) | **RoPE only** (−4.84 %) | −4.91 % |
| **exp17** | 1.1009 | `rope_l1_tail` 1.1007 (−0.02 %) | no | 0.0 % |

**A** found the win on the first legal try: a 32-core rectangular L1 RoPE chain, every repeat
(0.606833–0.608693 ms) below every control repeat (0.656754–0.657184). Differential real-weight oracle
PCC 0.9999987790, fresh confirmation 0.607902 ms. Shipped: **18,210.88 → 16,616.75 µs/model**. One
false start worth recording — a helper named `profile.py` shadowed Python's stdlib `profile` module
during a Transformers import and crashed the process before it opened the device. The cell correctly
ruled that a crash containing no measurement is **not** a rejection, renamed the helper, and re-ran.

**B** hit a tracer wall first: the pinned tracer has no handler for `paged_fused_update_cache` and
rejects traced tensors at that terminal op. Per the skill's version-skew rule it did **not** rebuild
tt-mlir or change the decoder — it made the terminal explicit in the capture adapter, preserved the
graph either side of it, and reported the fused-cache share as *unreachable* rather than attributing it
to the advisor. The RoPE L1 chain then won at 0.748709 ms. Its gate flagged a possible **settling
signature** in the first control block; rather than dismiss it, the cell re-froze the control with 20
warm-ups and re-captured afterwards to preserve the temporal ordering, then recomputed the band from
the cleaner control. Shipped **−1,284.9 µs/model (5.74 %)**. It also discovered another agent already
running this exact stage in the shared workspace and continued from that state rather than starting a
competing hardware run.

**FN is the most instructive cell in the corpus.** Its full sequence:

| # | candidate | ms | vs control |
|---|---|---|---|
| 1 | control | 0.8072 | — |
| 2 | `rope_l1_query` | 0.7934 | −1.71 % |
| 3 | `rope_l1_key` | 0.7926 | −1.81 % |
| 4 | `rope_l1_query_key` | 0.7681 | **−4.84 %** (superadditive) |
| 5 | `rope_l1_query_key_confirm` | 0.7675 | −4.92 % |
| 6 | `norm_11c` — the advised count | 0.7459 | **−7.60 %** |
| 7 | `norm_12c` | 0.7490 | −7.21 % |
| 8 | `norm_24c` | 0.7485 | −7.27 % |
| 9 | **`rope_l1_query_key_norm_11c`** | **0.7003** | **−13.24 %** |

It did everything right: swept the advised grid *and two grids above it*, found the norm win, and
combined it with the RoPE win for a **13.24 % layer-level improvement** — the largest measured
improvement anywhere in this corpus. Then it threw it away.

The reason was not speed. It was the correctness oracle. phi FN built a **differential** oracle
(candidate vs the frozen incumbent's own output) and set the bar at **0.999999**. The combined
candidate came in at PCC 0.9999910667 and was rejected; the RoPE-only candidate was bitwise identical
(PCC 1.0) and shipped. In its own words:

> The combined set is rejected despite being faster … Per the placement-stage oracle rule, I'm shipping
> RoPE-only and keeping the 11/12/24-core norm knob default-off.

**phi FN is the only cell in the corpus that used a 0.999999 bar.** Everything else used 0.995 or a
recorded model-specific value. phi **A** — the same model — passed its differential oracle at
0.9999987790, which is itself below 0.999999 and would have failed FN's bar. phi FN's own shipped
real-weight test passes at PCC **0.998902**, i.e. it rejected a candidate ~120× closer to the reference
than the thing it shipped. And **north-mini FN shipped exactly this class of change at PCC 0.999526**.
Splitting a sum-of-squares reduction across 11 cores changes floating-point summation order by
construction; a near-bitwise bar can never be met by any norm re-grid. See
[`ADVCHAL-V2-ORACLES.md`](ADVCHAL-V2-ORACLES.md).

**exp17** is the honest hard case. Its checked-in single-chip decoder *rejects any decode batch other
than 1* and builds its Q/K/V sharding for `max_batch_size=1`, so batch-32 enablement had to happen
first — correctly treated as experiment setup, not advisor contribution, with the control frozen only
after that path ran. It then had the **largest ceiling of any phi arm, 83.551 µs/layer (2,673.6
µs/model)**, dominated by the two split-half RoPE chains — and realised none of it. The full RoPE L1
family was slower (1.101395 ms). The legal L1-tail isolate had a *lower median* (1.100683) but its
repeats **overlapped** the control (1.099390–1.101032 vs 1.100128–1.101220), so the non-overlap rule
refused it. Sharded SDPA output hit `TT_FATAL: Sharded output not supported for GQA`. A publishable
zero with every knob left default-off.

**Reading the four together.** The slowest phi arm got nothing and the fastest got the most. "More
headroom in a slower decoder" does not hold here, because these are not the same code at different
speeds — they are different structures. What actually decided each arm was whether the specific
RoPE boundary existed in a *legally removable* form: A had it as a clean 32-core rectangle, B as a
chain, FN as a two-sided query+key pair, and exp17 only as a tail isolate too small to separate from
its own noise floor.

### 3.4 qwen3.6-27B — where 75 % of the model is invisible

Both arms are 64 layers: **48 linear-attention + 16 full-attention**. The linear layers cost ~13× a
full layer (15.85–19.14 ms vs 1.21–1.45 ms), so they dominate the model. **The pinned tracer cannot
cross the linear kind's mutable-state `ttnn.copy` boundary.** For 48 of 64 layers the advisor never saw
the graph. Both arms therefore report a structural zero over three-quarters of the model, and every
number below concerns only the 16 full-attention layers.

**`fuse` arm.** Control 1.208257 ms (full). The top-ranked repeat placement failed exactly as the skill
predicts for partial application — its sharded output met an interleaved passthrough at concat. The
cell did the prescribed retry, extending the candidate through the advisor's adjacent `add → concat`
reconfiguration rather than abandoning the direction; that combined form also lost (1.233225 ms). What
won was the **packed-QKV boundary consolidation at 1.180402 ms (−2.31 %)**, ~27.9 µs/layer — far larger
than the isolated conversion estimate, which the cell correctly read as evidence that the *chain
extension*, not the single conversion, carries the value. Differential real-checkpoint oracle **PCC
1.0**; fresh confirmation 1.181348 ms. Two further probes lost: `rope_query_c32` (1.2225) and
`rope_both_c32` (1.2332).

Shipped **938,063.85 → 937,618.16 µs/model = −445.69 µs, band ±618.50 µs.** The gain is real per layer
and **not established at model level** — the cell headlined this rather than hiding it. It also caught
a genuine limitation of the reconciliation tool: IR inspection showed several soft name/position
pairings claiming removed boundaries that the authoritative IR explicitly *retains*. Those false
candidates were recorded and excluded from attribution.

**`B` arm.** Control 1.449416 ms full, 15.849799 ms linear. Ceiling 33.698 µs/full-layer. Everything
legal was screened and everything lost:

| candidate | result |
|---|---|
| `advisor_rope_q_l1` | 1.4544, **+0.35 %** |
| `advisor_rope_k_l1` | 1.4536, **+0.29 %** |
| `advisor_qkv_direct` | 1.4501, +0.05 % — non-overlap **failed** |
| `advisor_rope_dram` | 1.4516, **+0.15 %** |
| 10-core geometry | hard fail: neighbouring K=192 projection gives `per_core_K=20`, not evenly shardable |
| 11-core geometry | hard fail: same shared-projection contract |
| 16-core above-advice probe | hard fail: exceeds the **one-row worker-grid contract** |
| L1 per-head norm, direct SDPA→head handoff | exact hard contract errors |

This arm is the clearest demonstration of the skill's own warning that a **whole-graph plan cannot be
dropped piecemeal onto a shipped graph**. It also did the best recovery work on the untraceable kind:
rather than write off 48 layers, it established that the linear layer's *gated-delta token mixer* is
terminal but its **residual/norm/MLP envelope is traceable**, captured that envelope explicitly, and
declared only the stateful token-mixer ops uncapturable — preserving the layer count without
pretending the advisor had seen the recurrent core.

### 3.5 gemma-4-12B — the most thorough screening campaign

**28 measurements, the most of any cell.** Control 1.254146 ms sliding / 1.377420 ms full.

It worked outward in stages rather than testing one candidate:

| stage | candidates | best |
|---|---|---|
| RoPE row-major | `b17` 1.2488, `b19` 1.2474, both 1.2413 | −1.02 % |
| K residency | `sliding_keep_k_l1_chain` 1.2308 / full 1.3359 | −1.86 % |
| corrected form | `shipped_l1_interleaved` 1.2288 / 1.3359 | −2.02 % |
| V, MLP | `keep_v_l1` 1.2322, `mlp_direct_down` 1.2352 | — |
| products | `k_v_mlp` 1.2238, **`q_k_v_mlp` 1.2188** | **−2.82 %** |
| output chain | `grouped_o_l1` 1.2413 sliding, `full_q_k_v_mlp_o` 1.3060 | full-only win |

Three things here are worth carrying to other cells:

1. **Its narrow PCC oracle missed a real bug.** A broader optimized-decoder regression run found that
   leaving K height-sharded **violates the per-head norm contract**. The earlier measurement was
   *disqualified rather than published*, and the shipped form corrected to K in **L1 interleaved** — the
   advised boundary removal — and re-measured.
2. **It re-froze the control mid-run** (1.2415 / 1.3559) after shipping the first change, so subsequent
   candidates were measured against the decoder they would actually extend.
3. **It reversed its own suppressions.** Two sliding candidates had been set aside; measured properly
   against the frozen control they were real wins (V residency 1.232241, direct MLP handoff 1.235160,
   each with complete 5-repeat separation), and their product with K materially changed the headline.

For the oracle it refused a shortcut: the local 12B cache held only the config, and rather than use the
skill's otherwise-acceptable synthetic oracle for a shipped change it selectively fetched the layer-0
shard. When the final product came to include a full-attention change, it extended the oracle to an
actual full-attention checkpoint layer instead of treating a sliding-only oracle as sufficient.

Shipped **58,520.1 → 57,853.3 µs/model, −666.8 µs (1.14 %), ±55.5 µs** — outside its band.

### 3.6 gemma-4-26B — three versions, and the corpus's decisive experiment

Three cells, all 25 sliding + 5 full attention layers, all at **decode batch 1**.

| | control ms/layer | 88-core norm | concat→projection | sliding-O DRAM | shipped |
|---|---|---|---|---|---|
| **exp** | 1.2597 sliding | observed, not shipped | — | **1.2540 (−0.46 %)** | sliding-O → −147.9 µs/model |
| **onA** | **1.8252** sliding | **1.5873 (−13.03 %)** | — | — | 88-core norm → −12.98 %/layer |
| **FN** | **1.3412** sliding | **1.3469 (+0.43 %)** ✗ | **1.3184 (−1.69 %)** | — | concat→proj → −791.7 µs (2.04 %) |

**This is the experiment that answers "were the winners just worse to start with?"** The *same
candidate* — re-gridding a 1-core RMSNorm to 88 cores — was measured on two arms of the same model with
the same tool at the same pin:

- on **onA**, whose control is 1.8252 ms/layer, it won **−13.03 %**;
- on **FN**, whose control is 1.3412 ms/layer — **26 % faster** — the identical candidate **regressed
  both layer kinds** (+0.43 % sliding, +15.57 % full) and was left default-off.

The fusing arm had already fixed what the advisor was recovering on onA. The advisor's 13 % was not 13 %
of value it created; it was 13 % of a deficiency that existed only in that arm's starting point.

**onA is also the cell where the accounting said zero and the measurement said 13 %.** Its reconciliation,
after an IR-aware re-run, put the advisor-attributable ceiling at **exactly 0.000 µs for both layer
kinds** — every comparable shipped conversion also appears in the advisor's plan. By the contribution
method that closes the cell at zero, and it recorded that zero. It then measured the "doubtful" norm
candidate anyway, and it was worth 236.8 µs/layer. The ceiling is a *boundary-conversion* ceiling; it
prices a re-grid of an op that stays inside its chain at zero. See §4.2.

Its oracle ran all three real-weight cases (sliding decode PCC 0.999629, full decode 0.999787, bar
0.995), and — importantly — it ran them **against the actual shipped default with every candidate
override unset**, closing the gap between what was timed and what ships. **64.70 % of the sliding
window and 58.51 % of the full window is sparse-expert work the tracer cannot reach**, so all of this
concerns roughly a third of each layer.

**FN** hit `Sharded output not supported for GQA` on the connected SDPA-output extension, then found the
compatible **concat-heads → output-projection** chain won cleanly (1.318449 sliding / 1.494548 full,
every repeat below every control repeat). It also hit a Tracy limitation worth recording: trace-replay
device rows carry no host op markers between the signposts, so `tt-perf-report` correctly returned an
*empty* bounded window. Rather than loosen the timing protocol it added a profile-only
one-eager-replay wrapper purely to generate the required op CSV, and kept latency decisions on the
untouched harness. Shipped **38,887.6 → 38,095.8 µs/model (2.04 %), ±80.9**.

**exp** is the batch-1 cell that shipped sliding-O DRAM sharding on a decisive split: sliding O won
every repeat (1.253223–1.254487 vs control 1.258866–1.260157) while full QKV clearly lost
(1.290230–1.291618 vs 1.261299–1.262151). One detail shows the right instinct about shipped-vs-timed
code: the winning measurement used O-projection `in0_block_w=1`, and the cell **pinned that
role-specific value in the auto-selected default** so the shipped constructor could not depend on
helper inference and diverge from what was timed.

### 3.7 north-mini — one big win, and two cells that could not see their own model

**FN** is the corpus's second-largest win and its best example of a review catching a real defect.

The advisor wanted the MoE RMSNorm off 1 core onto **22**, width-sharded. FN swept **22 / 32 / 64**:

| grid | ms | vs control 0.5537 |
|---|---|---|
| 22 (advised) | 0.5433 | −1.88 % |
| **32** | **0.5183** | **−6.40 %** |
| 64 | 0.5733 | **+3.54 %** — worse than doing nothing |

Going *above* the advice by one exactly-dividing step was worth 4.5 percentage points more than the
advice itself; going two steps up was worse than the 1-core original. Then an independent reviewer
rejected the whole measurement set for a **control-design defect**: the candidate policies had
inherited constructor defaults instead of cloning the frozen incumbent, so they changed several dormant
policy fields alongside the norm core count — not "frozen-incumbent-plus-one-knob". All six candidates
and both confirmations were **remeasured** with `dataclasses.replace` cloning the shipping policy and
changing only `advisor_moe_norm_cores`. The same reviewer also found the full-attention MoE oracle had
**programmatically copied layer 1's PCC into the layer-4 artifact**, and that the shipped fast path
skipped `current_positions` shape and required-RoPE-input validation. All three were fixed.

Refreshed result: **24,949.218 → 22,397.946 ± 74.289 µs/model (−10.23 %)**, oracle PCC 0.999526 via
official layer-1 tensors transparently remapped onto the layer-4 path.

**B** screened the dense MLP residency family exhaustively and lost every time. Against a 0.203313
ms/layer dense control: `dense_advised_down_ds` 0.2074, the legal exactly-dividing 48→64-core chain
0.234085, and a 110-core above-advisor chain 0.292203. The hard walls it documented are the useful
part: the 32-core residual shard exposes only **two K tiles per shard**, incompatible with
`in0_block_w=16`; and the aggregate sliding attempt was rejected by `rotary_embedding_hf` with **"Cos
must be sharded in decode mode."** Both DRAM-sharded full-chain geometries *stalled* after compilation
rather than failing; the cell kept the device serialized, waited inside its bounded timeout, confirmed
device health afterwards, and closed the layout family as **not measurable** rather than slower. It
also disproved several reconciliation "boundaries" against the IR — they were rotary-input layout
conversions, not QKV→rotary or SDPA→concat edges — and kept them as rejected, default-off evidence.

**onA** never got to screen anything. `ttnn.sparse_matmul` cannot consume tracer tensors, so the sparse
MoE tail is unreachable; the cell captured each sparse kind's reachable attention prefix, marked the
tail uncapturable, and quantified the untraced share. Every verdict came back `not_measurable`, so
nothing was legally screenable and the contribution is a measured zero. Its other finding is a harness
trap: the first dense profiler companion **overflowed the device profiler buffers** during the
template's 250 timing replays, leaving the late signposted replay with no device rows. It discarded
that profile as invalid and re-ran with Tracy's mid-run device-data dumping — preserving the fixed
timing harness rather than shortening it to fit the profiler.

---

## 4. What makes a model advisor-compatible

Ranked by how much it actually decided outcomes in this corpus.

### 4.1 Can the tracer see the layer at all? (decides more than everything else combined)

| model | unreachable | cause |
|---|---|---|
| qwen3.6-27B | **48 of 64 layers** | linear attention's mutable-state `ttnn.copy` boundary |
| north-mini onA | the entire sparse MoE tail | `ttnn.sparse_matmul` rejects tracer tensors |
| gemma-4-26B onA | **64.70 % sliding / 58.51 % full** of the window | sparse experts |
| phi-3.5 B | the fused-cache share | no tracer handler for `paged_fused_update_cache` |

If the advisor cannot see an op it cannot advise on it, and no amount of screening recovers it. Every
structural zero in this corpus is a tracer-coverage zero. **This is a `$shard-advise` / tt-mlir
coverage problem, not a placement problem**, and it is where the leverage is: qwen's dominant cost is
in the 48 layers the advisor never read.

### 4.2 Does the model have a *reduction* on too few cores?

Every large win in the corpus is the same shape: an RMSNorm sitting on **1 core** moved onto many.

| cell | norm | measured |
|---|---|---|
| gemma-4-26B onA | 1 → 88 | **−13.03 %/layer** |
| phi-3.5 FN | 1 → 11 | **−7.60 %/layer** (discarded on the oracle) |
| north-mini FN | 1 → 32 | **−6.40 %/layer** |

And every model *without* one produced a zero from this family: both llamas arrived with the norm on 32
cores, and the advisor's only suggestion was to *reduce* to 22 — worth a notional 6–7 µs and slower
measured.

This class has a systematic accounting problem. The reconciliation ceiling prices **boundary
conversions** — conversions the shipped decoder pays that the advisor's plan does not place. Re-gridding
an op that stays inside its L1 chain removes no boundary, so it prices at **0.000 µs**. gemma-4-26B onA
recorded a 0.000 µs ceiling and then measured 236.8 µs/layer from exactly such a re-grid. Any cell that
trusted the ceiling and stopped would have shipped a zero. This is the **second attribution channel**,
and it must be read with its direction: *up* (advised more cores than shipped, especially from ≤2) is
where the value is; *down* is the advisor's fewer-cores bias and lost every time it was measured.

### 4.3 Will the op's neighbours accept sharded I/O?

The most common reason a legal-looking candidate never gets a number. Verbatim walls from this corpus:

- `TT_FATAL: Sharded output not supported for GQA` — phi exp17, gemma-4-26B FN
- `nlp_concat_heads_decode` **requires sharded input** — llama-3.1-8B's top candidate
- `rotary_embedding_hf`: **"Cos must be sharded in decode mode"** — north-mini B
- paged GQA SDPA rejects sharded output; rotary-K DRAM incompatible with paged cache update — llama-3.2-1B
- `paged_fused_update_cache` unfixable in the advisor's own report — phi FN

The advisor plans the whole graph; the shipped graph accepts changes only where its neighbours' contracts
allow. The prescribed response — **extend the candidate across the adjacent conversion rather than reject
the direction** — is what turned qwen's failing repeat placement into a measurable (if losing) chain and
north-mini B's 32-core attempt into a measurable one.

### 4.4 Does the grid divide the tensor?

Wins land on exactly-dividing grids; non-dividing grids hard-fail before timing.

- north-mini FN: 22 advised → **32 wins** → 64 regresses. One dividing step above the advice was the win.
- phi FN: swept 11 / 12 / 24 — all three faster than the control, essentially a plateau (−7.60 / −7.21 / −7.27 %).
- qwen B: 10 cores → `per_core_K=20` on the neighbouring K=192 projection, not shardable. 11 the same.
  16 exceeded the one-row worker-grid contract.
- llama-3.1-8B: the exactly-dividing 64-core candidate was **+4.22 %** — dividing is necessary, not sufficient.

**The advisor's suggested core count is a starting point, not an answer.** In the one cell that swept a
dividing grid above the advice and shipped it, that step was worth more than the advice.

### 4.5 Is there headroom above the noise floor?

The floor ranged 0.146 µs → 14.5 µs across cells. Two cells were decided by it rather than by placement:
phi **exp17**'s best legal candidate had a *better median* than the control and was refused because the
repeats overlapped; qwen `fuse` shipped a real per-layer win whose model-level extrapolation
(−445.69 µs) sits **inside its ±618.50 µs band**. Small models have proportionally less to win *and* a
harder time proving it.

---

## 5. Were the winners just worse to start with?

**Partly, and for one model provably yes.**

**Provably yes — gemma-4-26B.** Same model, same candidate, same tool: −13.03 % on the arm with the
1.8252 ms control, **+0.43 %** on the arm with the 1.3412 ms control. The advisor's large win existed
only because that arm's starting point had a defect the other arm did not. §3.6.

**Provably no — phi-3.5.** Its four arms span 1.68× in control speed and the ordering is *inverted*: the
**fastest** arm (A, 0.6570 ms) took the **largest** win (−8.75 %/model), and the **slowest** arm
(exp17, 1.1009 ms) with the **largest** ceiling (83.6 µs/layer) shipped **zero**. Headroom did not
predict realisation. What predicted it was whether the RoPE boundary existed in a legally removable
form.

**Neither — the llamas.** Their zeros are not "worse to start with" or "better to start with"; they are
*already correctly placed*. Norm on 32 cores, chains L1-resident, 73–76 % of the window independently
re-derived by the advisor. There was nothing to find and the cells demonstrated that rather than
assuming it.

**The honest general statement:** the advisor's measured contribution in this corpus is mostly the size
of the *placement defect it happened to find*, and defects were distributed by how each arm's
`optimized_decoder` was built, not by model architecture. It is a **defect finder** more than an
optimiser. That makes its value real but non-additive: fix the 1-core norm upstream in `$optimize` and
the advisor's contribution on that model drops to near zero — which is exactly what the gemma-4-26B FN
arm shows.

**Could the others have been helped?** Yes, in four concrete cases — §6.

---

## 6. What could still be won, ranked

| # | opportunity | evidence | est. value |
|---|---|---|---|
| 1 | **Re-run phi FN's combined candidate under the corpus-standard oracle bar.** The −13.24 % was measured, confirmed at both 11-core and combined form, and rejected only by a 0.999999 differential bar that no other cell used and that the model's own shipped test (PCC 0.998902) does not require. | §3.3, [`ORACLES`](ADVCHAL-V2-ORACLES.md) | **+8.3 pp** layer-level over what shipped |
| 2 | **Give the tracer a handler for qwen's linear-attention `ttnn.copy` boundary.** 48 of 64 layers, and the linear layer costs ~13× a full layer, so ~91 % of qwen's model time was never advised on. | §3.4 | unknown, but the largest unexamined surface in the corpus |
| 3 | **Sweep a dividing grid above the advice in every cell that shipped a norm re-grid.** north-mini FN gained 4.5 pp doing this (22→32). gemma-4-26B onA shipped the advised 88 and never tried 44 or 110; phi FN's 11/12/24 plateau was never extended to 32. | §4.4 | 1–5 pp per affected cell |
| 4 | **`ttnn.sparse_matmul` tracer support.** Blocks north-mini onA entirely and hides 58–65 % of every gemma-4-26B window. | §3.7, §4.1 | unblocks 2 models |
| 5 | **Re-screen qwen B's geometry family off the one-row worker grid.** Its 16-core above-advice probe failed the one-row contract, not the tensor shape; a 2-row geometry was never attempted. | §3.4 | unknown, currently a hard zero |
| 6 | **Price the second attribution channel in the reconciliation ceiling.** A 0.000 µs ceiling next to a 236.8 µs/layer measured win (gemma-4-26B onA) means any cell trusting the ceiling ships a false zero. | §4.2 | correctness of the method itself |

---

## 7. Corrections to previously published numbers

Reading the transcripts overturned two claims in the earlier documents, both now fixed at source.

**1. phi FN's norm sweep.** [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md) said phi FN "screened **11
only**, measured slower". Both halves are wrong. The log shows `for cores in 11 12 24`, and all three
were **faster** than the control (−7.60 %, −7.21 %, −7.27 %). The candidate was rejected by the
correctness oracle, not by timing. The corrected reading is in §3.3.

**2. "phi FN abandoned the 1-core RMSNorm."** It did not abandon it — it measured it, combined it with
the RoPE win for −13.24 %, and then discarded the combination on a non-standard oracle bar. This
changes the finding from "a bounded sweep missed the win" to "the win was found and lost to an
unspecified oracle contract", which is a stage defect rather than a cell defect.

Both corrections make the corpus's headline conclusion *stronger*, not weaker: the 1-core-reduction
class is the highest-yield opportunity in the corpus, and it was successfully realised in two cells out
of the three that had one.

---

## 8. Stage defects this analysis exposes

1. **The oracle contract is unspecified** — kind, bar, and bar provenance are all left to the cell, and
   the strictest available reading discarded the corpus's largest measured win. Fix in
   [`ORACLES`](ADVCHAL-V2-ORACLES.md) §"Fix for the stage".
2. **The ceiling misprices in-chain re-grids at zero**, so the accounting can read zero on a cell with a
   13 % win available. §4.2.
3. **`reconcile.py` never fills the verdicts the gate demands**, and the skill forbids editing its
   output — an impossible contract that three cells solved three different ways (rewriting the tool, editing
   the JSON, and adding a `record_decisions.py`). Whether a cell ended up tagged reflects *which
   violation it chose*, not the quality of its work.
4. **Soft name/position pairing produces false boundaries.** qwen's IR inspection found claimed removed
   boundaries the IR explicitly retains; north-mini B disproved several as rotary-input layout
   conversions. Both cells caught it; a cell that trusted the tool would have over-attributed.
5. **No warm-up floor scaled to model size.** Two cells (phi B, gemma-4-26B onA) hit first-repeat
   settling signatures and had to re-freeze their controls with 20 warm-ups.
6. **The profiler and the timing protocol conflict.** north-mini onA overflowed device profiler buffers
   at 250 replays; gemma-4-26B FN found trace-replay rows carry no host markers between signposts, so
   the bounded window came back empty. Both had to add a separate profile-only path. The skill should
   prescribe that path rather than leaving each cell to discover it.
