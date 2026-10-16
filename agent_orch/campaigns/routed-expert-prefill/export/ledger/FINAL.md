# routed-expert-prefill campaign: final report

Op: `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`, 9 production prefill shapes, one Blackhole chip (bh-qbge-12, 2× p300c, FW 19.7), bf4 weights with the routed-expert LoFi compute config, as in the benchmark.

**Metric:** mean `DEVICE KERNEL DURATION [ns]` of `UnifiedRoutedExpertFfnDeviceOperation` over the 10 measured calls per shape (3 warmup calls dropped). The score is the geomean over shapes of baseline µs ÷ attempt µs.

**Accuracy gate:** relative to the baseline, per shape: mean_abs within +1% and PCC within −1e-4. No precision reduction: same fidelity, data formats, fp32 accumulation and approximation modes.

## Result

Best valid attempt: `r08-b04-a01` (tag `dream/routed-expert-prefill/n/r08-b04-a01`), score **1.7812**.

| case | baseline µs | best µs | speedup |
|---|---|---|---|
| deepseek-v3 h7168 i2048 | 801.3 | 478.5 | 1.675× |
| kimi-k2.7 h7168 i2048 | 1162.0 | 675.5 | 1.720× |
| kimi-k3 latent h3584 i3072 | 2570.4 | 1307.4 | 1.966× |
| glm-5.3 h6144 i2048 | 704.1 | 419.9 | 1.677× |
| mistral-small-4 h4096 i2048 | 261.8 | 160.5 | 1.631× |
| deepseek-v4-flash h4096 i2048 | 525.2 | 286.9 | 1.830× |
| deepseek-v4-pro h7168 i3072 | 1604.1 | 956.1 | 1.678× |
| minimax-m3 h6144 i3072 | 477.8 | 316.9 | 1.508× |
| gpt-oss-120b h2880 i2880 | 572.0 | 227.3 | 2.517× |

PCC and mean_abs are identical to the baseline on every shape. The code-only version of this attempt is on `opgen_hackathon/dream_rsi_v1_routed_expert_r08-b04-a01`. It re-measured at 1.7777 (two runs), and that branch's `ROUTED_EXPERT_OPTIMIZATION.md` explains each change.

## Progress per round

Each round after the first started from the best valid attempt so far (`root_mode: best`).

| round | policy | W × R | round root | attempts | best |
|---|---|---|---|---|---|
| r01 | v0 | 4 × 4 | campaign root | 16 | 1.5011 |
| r02 | v0 | 4 × 4 | r01-b04-a04 | 16 | 1.5697 |
| r03 | v0 | 4 × 4 | r02-b02-a04 | 16 | 1.6103 |
| r04 | v1 | 4 × 4 | r03-b02-a03 | 9 | 1.6275 |
| r05 | v2 | 4 × 1 | r04-b02-a01 | 4 | 1.6667 |
| r06 | v2 | 4 × 1 | r05-b02-a01 | 4 | 1.6931 |
| r07 | v2 | 4 × 1 | r06-b04-a01 | 4 | 1.7387 |
| r08 | v2 | 4 × 1 | r07-b02-a01 | 4 | 1.7812 |

73 attempts, 69 valid. The 4 invalid ones were 2 accuracy failures, 1 runtime error (program too large for the kernel config buffer) and 1 hang (recovered by the eval tool's device reset). No attempts were lost, and no round tripped the baseline drift check. The run took about 10 hours unattended, driven by `agent_orch/tools/run_campaign.sh`.

## What made it faster

The best attempt's lineage, in order:

1. Row-group expert parallelism (different experts on different M-row groups at the same time).
2. Row-shared x tilize, deeper input rings, and all 10 worker rows.
3. A two-phase expert schedule with mixed-height leftover layouts.
4. Cheaper SFPU activations and a PACK-thread activation pipeline.
5. x handling moved off the weight-sender cores.
6. Weight DRAM read-ahead gated on the sender's own ring slot.
7. A 60/40 NoC 0/NoC 1 split of weight reads.
8. Wrap-free down and gate/up weight multicasts.
9. Cross-group weight-stream pacing through a per-group progress board.

The per-round `summary.md` files under `ledger/rounds/` have the details for each step.

## Dead ends

- Read-ahead of weights issued before the current multicast (before step 6's gating).
- `in0_block_w_gu` 16→32 and fused plain-SiLU-GLU: both are precision changes and failed the gate.
- Streaming x through SrcB / reuse-B matmul.
- Per-sender NoC request VCs and a deeper down look-ahead: neutral.
- A grid-wide multicast progress board for pacing (every group got 29–55% slower).
- Start-up staggering, row-tilted NoC splits, and moving all gate/up reads to the reader.

## Policy evolution (dreaming)

- **v0** (r01–r03): the hand-written parallel refine policy: 4 branches × 4 attempts, never closes a branch.
- **v1** (after r03): adds a frontier prune (close a branch more than about 1.1% below the round best after its 2nd attempt), a collapse close (more than about 8% below the branch's own best), and recover tags. Replay: 1.531 vs v0's 1.520.
  - In r04 it closed the leader's own branch after a dip. The frontier rule compares against the round best, which can be that branch's own earlier attempt.
- **v2** (after r04): one-step rounds after the first (4 independent attempts from the round root), so each round costs 4 attempts instead of 16. Replay: 1.570 vs v1's 1.551. Dreaming after r05, r06 and r07 found nothing that beat v2 by more than one attempt's cost.

## Known limits of this setup

- **The root rule is greedy.** Every round after r01 re-rooted on the best node, so all later work descends from r01-b04-a04's row-group design. Competing designs were only kept alive when workers ported them. A rerun of the rmsnorm campaign with every round starting from the campaign root (the paper's setup) did better than re-rooting there: 1.795 vs 1.570.
- **Single eval runs pick lucky winners.** Each attempt is measured once, with about ±1% noise. r04's root, `r03-b02-a03`, was a measurement-only attempt whose edge over its parent was within noise.
- **The replay objective only rewards the best score inside a round.** It can't value discoveries that pay off in later rounds, so each dreaming phase added pruning.
