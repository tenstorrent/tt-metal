# r02-b03-a01 result: 1.4973 (ok), neutral vs parent r01-b04-a04 (1.5011)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
printed digit). No hang. So the two-phase mechanics work:
- the phase-A semaphore pair,
- ring alignment at the phase switch (equal units per phase-A group),
- writer slot counters that count every block,
- the per-expert (count, rows) mailbox pair in compute.

The plan fired on exactly the four predicted shapes (rows 2 -> 5). Per shape vs the parent (noise
±1%):

| shape | parent µs | this µs | Δ | model predicted |
|---|---|---|---|---|
| dsv4-pro | 1128.0 | 1083.6 | **-3.9%** | -9% |
| kimi-k2.7 | 764.3 | 744.8 | **-2.6%** | -9% |
| dsv3 | 547.5 | 565.0 | **+3.2% (slower)** | -10% |
| glm | 487.6 | 492.5 | +1.0% | -10% |
| kimi-k3 | 1378.8 | 1396.6 | +1.3% | 0 (single phase) |
| mistral / dsv4-flash / minimax / gpt-oss | | | +1.6 / +1.5 / -0.1 / +0.5% | 0 (single phase) |

## Why (profiler evidence)
TRISC1 kernel-end per physical row, mean of the 10 measured calls (µs from op start;
/tmp/r02b03a01/rows.py on reports/<node>/.logs/profile_log_device.csv):

| shape | parent rows 0..9 | this rows 0..9 |
|---|---|---|
| kimi-k2.7 | 741 741 754 753 588 589 552 551 585 585 | 741 740 740 740 740 732 731 731 731 731 |
| dsv4-pro | 1089 1088 1123 1123 888 890 875 871 833 832 | 1075 ... 1073 (all rows) |
| dsv3 | 537 537 527 527 512 512 318 317 257 256 | 562 x5, 461 x5 |
| glm | 468 468 464 465 459 459 281 281 242 242 | 489 x5, 391 x5 |

- **The tail is gone**, but each phase-B expert runs much slower than modelled. On kimi-k2.7 and
  dsv4-pro all rows now finish together.
  - Phase A (2 waves on 2-row groups) ends where the parent's light groups ended, ~560-590 µs on
    kimi-k2.7 and ~830-890 µs on dsv4-pro.
  - Each phase-B expert (pcM 1, two 5-row groups streaming) then takes ~150 µs on kimi-k2.7 and
    ~195 µs on dsv4-pro: 2.7 c and 2.3 c, against the modelled 2.0 c (c = one expert's weights at
    ~443 GB/s).
  - Two 5-row groups together pull only ~330-380 GB/s. A lone 5-row group, measured on dsv3's
    second phase-B expert, pulls ~250 GB/s (~100 µs = 1.8 c).
- **The model's flaw.** It assumed any set of concurrent streams can reach the full DRAM rate.
  In fact a group's stream rate is capped by its 11 column senders (issue-bound reads plus a
  multicast to rows-1 receivers). Aggregate bandwidth scales with the number of streaming groups,
  and saturates around 5 groups of 2 rows. Fewer, larger groups trade compute for weight-sender
  parallelism, and that trade is far less favourable than the model said.
- **dsv3 / glm lost** because 3 leftover experts on 2 phase-B groups take two rounds on group 0:
  ~140 µs for the first expert (2 streams), then ~100 µs for the second (alone). The parent's
  one-round tail at pcM 3 on three 2-row groups was shorter. Rows 5-9 (group 1, one phase-B expert)
  end at 461 / 391, ~100 µs before group 0.
- **Single-phase shapes** are +1-1.6% on kimi-k3, mistral and dsv4-flash. That is within the known
  run-to-run spread of the 1-expert-per-group shapes. kimi-k3's rows end ~10 µs later on every
  row, which could also be the extra plan arithmetic at kernel start (28 experts, about 3 x
  model_phase on every RISC before the first DRAM read). Not proven.

## Classification
neutral (within noise overall, -0.25%). The idea half works: there are real wins on the shapes whose
leftover fits in one phase-B round (dsv4-pro -3.9%, kimi-k2.7 -2.6%). It loses where the leftover
needs two phase-B rounds (dsv3 +3.2%, glm +1%). The implementation is correct and general
(bit-identical, no hang, single-phase path unchanged). The plan selection model is miscalibrated.

## What a child of this node should try next
1. **Cheapest fix, likely a net win (~+0.7% score):** only take a two-phase plan when the leftover
   fits in ONE phase-B round (leftover experts <= kGridY / rows_b). Also add a per-group stream
   cap to `model_expert`: a group of `rows` rows streams at most ~1.8 c per expert alone and
   ~2.5 c with two such groups (5-row data from this node). With that, dsv3/glm go back to the
   parent's schedule, and dsv4-pro / kimi-k2.7 keep their -3..-4%. Everything is in
   `adaptive_chunk::make_plan` / `model_expert` (kernel-only change, no host rebuild).
2. **Better phase B for odd leftovers:** groups of mixed size in phase B. For example dsv3's 3
   leftovers on groups of 4/3/3 rows (pcM 2/2/2, 3 streams), or 2-row groups for the larger
   leftover plus a 5-row group. This needs host topology sets for those sizes. The phase machinery
   here already handles any per-phase topology index, so a set is just more runtime args plus a
   group_rows_index mapping.
3. **The real limiter exposed here is per-group weight-stream rate.** A 5-row group's 11 senders
   deliver only ~250 GB/s. If one group's stream rate could rise, every tail (and phase B) would
   shrink: e.g. use 2 senders per column in big groups, splitting the K-rows of each block between
   two rows' readers, which is like DOWN_SPLIT across cores. That is a different lever from
   anything tried so far.
4. Don't retry "move the tail onto bigger groups" with the old uncapped model. Bigger groups are not
   DRAM-bound at pcM 1.
