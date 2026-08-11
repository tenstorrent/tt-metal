# skillexp — machine a

generated 2026-08-11T14:15:22+00:00 on qb2-120-p05t03

tt-metal HEAD `9289ca97ad0` on
`mvasiljevic/qb2/skillexp/run/advchal-v3/exp17/meta_llama_llama_3_2_1b_instruct`

## Phase 1 — functional decoder (owned by this machine)

| model | goal | gate | fd-ready tag |
|---|---|---|---|
| `microsoft_phi_3_5_mini_instruct` | complete | - | yes |
| `qwen_qwen3_6_27b` | complete | - | yes |
| `coherelabs_north_mini_code_1_0` | not-started | - | yes |
| `google_gemma_4_26b_a4b_it` | not-started | - | yes |

## Phase 2/3 — optimize, this machine's arms

| arm | model | goal | gate | done tag |
|---|---|---|---|---|
| nofuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | yes |
| nofuse-advise | `qwen_qwen3_6_27b` | complete | pass | yes |
| nofuse-advise | `coherelabs_north_mini_code_1_0` | complete | pass | yes |
| nofuse-advise | `google_gemma_4_26b_a4b_it` | complete | pass | yes |
| fuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | yes |
| fuse-advise | `qwen_qwen3_6_27b` | complete | pass | yes |
| fuse-advise | `coherelabs_north_mini_code_1_0` | complete | pass | yes |
| fuse-advise | `google_gemma_4_26b_a4b_it` | complete | pass | yes |

## Device
```
                      All available boards on host (UMD):                       
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ UMD Chip   ┃            ┃            ┃            ┃ Device      ┃ Board      ┃
┃ ID         ┃ PCI BDF    ┃ PCI Dev ID ┃ Board Type ┃ Series      ┃ Number     ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0          │ 0000:01:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 1          │ 0000:02:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 2          │ 0000:03:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 3          │ 0000:04:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
└────────────┴────────────┴────────────┴────────────┴─────────────┴────────────┘
                        Boards that can be reset (UMD):                         
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ UMD Chip   ┃            ┃            ┃            ┃ Device      ┃ Board      ┃
┃ ID         ┃ PCI BDF    ┃ PCI Dev ID ┃ Board Type ┃ Series      ┃ Number     ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0          │ 0000:01:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 1          │ 0000:02:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 2          │ 0000:03:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
│ 3          │ 0000:04:0… │ /dev/tens… │ Blackhole  │ p300c       │ 000004613… │
└────────────┴────────────┴────────────┴────────────┴─────────────┴────────────┘
```

## Cells taken over from the other machine (claimed)

| cell | claimed by | at | FD | done tag |
|---|---|---|---|---|
| `advchal-v2/fuse-noadvise/coherelabs_north_mini_code_1_0` | machine=a host=qb2-120-p05t03 | 2026-08-03T10:47:29+00:00 | `6375656b892` | claimed, queued |
| `advchal-v2/fuse-noadvise/microsoft_phi_3_5_mini_instruct` | machine=a host=qb2-120-p05t03 | 2026-08-03T06:44:11+00:00 | `0cd94765fd9` | tagged |
| `advchal-v2/nofuse-noadvise/google_gemma_4_26b_a4b_it` | machine=a host=qb2-120-p05t03 | 2026-08-03T08:50:46+00:00 | `f9e28e24de3` | claimed, queued |
| `nofuse-noadvise-onA/qwen_qwen3_6_27b` | machine=a host=qb2-120-p05t03 | 2026-07-30T21:22:42+00:00 | `528c4d45ea1` | claimed, queued |

A claim is one ref per cell; claiming is a **non-force** push of a new ref, so the remote decides
the winner and two machines cannot both proceed. The claim is released only once the cell is
tagged — a cell that ends untagged keeps its claim, so the owning machine does not blindly re-run
a failure someone else already hit.

## Rejected / contaminated cells and re-runs pending

| cell | why | parked at | state |
|---|---|---|---|
| `fuse-advise-microsoft_phi_3_5_mini_instruct` | inherited the sibling arm's artifacts (own advisor capture not run) | `mvasiljevic/qb2/skillexp/parked/CONTAMINATED-fuse-advise-microsoft_phi_3_5_mini_instruct` | superseded: clean re-run tagged (capture differs from the parked copy) |
| `NOTFRESH-challenger-fuse-noadvise-microsoft_phi_3_5_mini_instruct` | parked | `mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-fuse-noadvise-microsoft_phi_3_5_mini_instruct` | tag retracted -- **re-run queued** |
| `NOTFRESH-challenger-fuse-noadvise-qwen_qwen3_6_27b` | parked | `mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-fuse-noadvise-qwen_qwen3_6_27b` | tag retracted -- **re-run queued** |
| `NOTFRESH-challenger-nofuse-noadvise-microsoft_phi_3_5_mini_instruct` | parked | `mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-nofuse-noadvise-microsoft_phi_3_5_mini_instruct` | tag retracted -- **re-run queued** |
| `NOTFRESH-challenger-nofuse-noadvise-onA-microsoft_phi_3_5_mini_instruct` | parked | `mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-nofuse-noadvise-onA-microsoft_phi_3_5_mini_instruct` | tag retracted -- **re-run queued** |
| `OLDMETHOD-challenger-fuse-noadvise-microsoft_phi_3_5_mini_instruct` | parked | `mvasiljevic/qb2/skillexp/parked/OLDMETHOD-challenger-fuse-noadvise-microsoft_phi_3_5_mini_instruct` | tag retracted -- **re-run queued** |
| `PREV-cell-advchal-v3-fuse-noadvise-coherelabs_north_mini_code_1_0` | parked | `mvasiljevic/qb2/skillexp/parked/PREV-cell-advchal-v3-fuse-noadvise-coherelabs_north_mini_code_1_0` | tag retracted -- **re-run queued** |
| `nofuse-noadvise-onA-google_gemma_4_26b_a4b_it-not-from-scratch` | failed validate_cell before tagging | `mvasiljevic/qb2/skillexp/parked/REJECTED-nofuse-noadvise-onA-google_gemma_4_26b_a4b_it-not-from-scratch` | tag retracted -- **re-run queued** |
| `nofuse-noadvise-onA-qwen_qwen3_6_27b-not-from-scratch` | failed validate_cell before tagging | `mvasiljevic/qb2/skillexp/parked/REJECTED-nofuse-noadvise-onA-qwen_qwen3_6_27b-not-from-scratch` | tag retracted -- **re-run queued** |
| `onA-qwen-attempt3-not-from-scratch` | failed validate_cell before tagging | `mvasiljevic/qb2/skillexp/parked/REJECTED-onA-qwen-attempt3-not-from-scratch` | tag retracted -- **re-run queued** |
| `RUN2-challenger-nofuse-noadvise-google_gemma_4_26b_a4b_it` | parked | `mvasiljevic/qb2/skillexp/parked/RUN2-challenger-nofuse-noadvise-google_gemma_4_26b_a4b_it` | tag retracted -- **re-run queued** |
| `RUN3PARTIAL-challenger-phiFN` | parked | `mvasiljevic/qb2/skillexp/parked/RUN3PARTIAL-challenger-phiFN` | tag retracted -- **re-run queued** |
| `nofuse-advise-microsoft_phi_3_5_mini_instruct` | clean start **unprovable** -- the stage ran while the isolation preflight was a no-op (P37). NOT proven contaminated; retracted because it could not be verified either way | `mvasiljevic/qb2/skillexp/parked/SUSPECT-nofuse-advise-microsoft_phi_3_5_mini_instruct` | superseded: clean re-run tagged (capture differs from the parked copy) |
| `rerun1-nofuse-advise-phi-publish-failed` | a re-run attempt whose measurement was real but whose **publish** failed -- **not a cell**, kept so its work is not lost | `mvasiljevic/qb2/skillexp/parked/rerun1-nofuse-advise-phi-publish-failed` | archived, no tag expected |
| `run-fuse-advise-with-contaminated-phi` | snapshot of a whole run branch before surgery -- **not a cell**, kept so the pre-surgery state is recoverable | `mvasiljevic/qb2/skillexp/parked/run-fuse-advise-with-contaminated-phi` | archived, no tag expected |
| `run-nofuse-advise-with-suspect-phi` | snapshot of a whole run branch before surgery -- **not a cell**, kept so the pre-surgery state is recoverable | `mvasiljevic/qb2/skillexp/parked/run-nofuse-advise-with-suspect-phi` | archived, no tag expected |

Re-runs are queued behind the live cell; a rejected cell is parked, never deleted, and gets no
tag until it passes `validate_cell` (own advisor capture, FD ancestor, no byte-identical
artifacts shared with the sibling arm).

### `nofuse-advise/phi` — corrected timeline (supersedes earlier status pages)

The monitor's reading was closer than machine A's own earlier report. What actually happened:

1. **2026-07-29T13:24:50** — the original cell WAS published and tagged. It did **not** fail to publish.
2. Later it was **deliberately retracted** by machine A after the contamination review, and parked at
   `parked/SUSPECT-nofuse-advise-microsoft_phi_3_5_mini_instruct` (36 optimize files preserved).
   Reason: its stage ran while the isolation preflight was a **no-op** (P37: `STAGE=2` never matched
   `case 02)`), so a clean start could not be **proven**. It was never shown to be contaminated.
   The 20-minute duration was the trigger for looking, but duration turned out to be a **weak**
   signal — see step 4.
3. **Re-run #1**, 08:45:57 → 09:27:26 (41 min, 37 optimize files). Clean start, verified preflight.
   Its measurement was real but its **publish genuinely failed** ("could not assemble the run
   branch"): `run/nofuse-advise` still carried the 36 retracted phi files, so the merge conflicted.
   Parked at `parked/rerun1-nofuse-advise-phi-publish-failed`. **This is the failure the monitor saw.**
4. **Re-run #2**, 09:27:34 → 09:44:53 (17 min). It **cherry-picked re-run #1's commits** at 09:29:53 —
   it RESUMED rather than re-optimizing from scratch, which is why 17 min is not an outlier. It then
   **regenerated** the advisor capture (its `report.json` differs from both re-run #1's and the
   suspect original's). Net agent work across both launches: ~58 min.
5. **Now published and tagged**: `run/nofuse-advise` @ `4f5a676d8ff`, tag
   `skillexp/done/nofuse-advise/microsoft_phi_3_5_mini_instruct`. `validate_cell` PASS — 81 optimize
   files, own capture, FD ancestor, **0** byte-identical blobs against the sibling arm (6 shared paths),
   and capture distinct from both parked copies. Correct arm ancestry, no cross-arm ancestry.

**For the analysis agent:** the **−50.3% batch-1 figure from the ORIGINAL cell must not be used.**
Take `nofuse-advise/phi` numbers from the tag above only.

**Method note worth carrying:** a short run is not evidence of contamination on its own. Re-run #2
reproduced a short duration from a *verified* clean tree, and duration also varies with whether a
run resumed prior commits. Judge cells on the byte-identity and ancestry checks, not the clock.
Also: an agent can satisfy a clean-working-tree preflight and then import a previous attempt's work
from git history — the preflight cannot see that, only the artifact-identity checks can.

## MACHINE A: what it is doing now (2026-07-30) -- read this before interpreting new tags

**Machine A has STOPPED replicating machine B's cells.** It is now running a different job.

### 1. The B-replicate job (`nofuse-noadvise-onA`) -- 1 of 4 done, PAUSED by operator instruction

Purpose was a machine/agent-variance control: re-run B's own cells on A with **identical inputs**
(same `skillexp/fd-ready/<md>` tag, same skills branch, same DECODE_BATCH), so A-vs-B can be
compared on the same cell. Published under `nofuse-noadvise-onA` so it can never collide with B's
`nofuse-noadvise` tags.

| cell | batch | state |
|---|---|---|
| `nofuse-noadvise-onA/microsoft_phi_3_5_mini_instruct` | 32 | **tagged** 2026-07-30T12:23, wall 3235s (53m55s) |
| `nofuse-noadvise-onA/qwen_qwen3_6_27b` | 32 | withheld -- not started |
| `nofuse-noadvise-onA/google_gemma_4_26b_a4b_it` | 1 | withheld -- not started |
| `nofuse-noadvise-onA/coherelabs_north_mini_code_1_0` | 1 | withheld -- not started |

The three withheld cells are **deliberately paused, not failed** -- a stop sentinel holds them
between cells. They resume after the audit below. **Machine B keeps ownership of its own cells;
A claimed only the `-onA` replicate slots.** Nothing under `nofuse-noadvise` (no suffix),
`run/nofuse-noadvise`, `wip/nofuse-noadvise/*`, `claim/nofuse-noadvise/*`, `status/b` or
`skillexp-heartbeat/b` was written, moved, deleted or retagged.

### 2. NOW RUNNING: stage 02b advisor-challenger -- `skillexp/done/challenger/<arm>/<md>`

```
!! Tags under skillexp/done/challenger/** are NOT 2x2 arm results.
!! Do NOT mix them into any advise-vs-noadvise comparison.
```

What it is: take each **finished no-advise optimized decoder** and run `$shard-advise` on it
*afterwards*, as a new stage 02b. **It SHIPS A NEW `optimized_decoder.py`** -- measured to be the
best of the incumbent and everything the advisor suggested -- not just a report. It also answers
the thing the 2x2 never measured --
**given a finished no-advise decoder, how much does a post-hoc advisor pass add?** It is cleaner
than the 2x2 on this question: the incumbent is real and finished, the search is over so nothing
can be biased, and the capture sees the shipped precision by construction.

It is **not** a counterfactual for the advise arms. Those used the advisor as a seed and diverged
from the first decision, so there is no path back.

Targets, in value order (5 incumbents, 3 models):

| # | incumbent | tag SHA | why |
|---|---|---|---|
| 1 | gemma `nofuse-noadvise` | `a3e826aa6f74` | 11 RMSNorms on 1 core = 24-30% of decode window, found by NEITHER arm |
| 2 | phi `nofuse-noadvise` (B) | `c2331f8bccfb` | independent optimize run 1 of 3 |
| 3 | phi `fuse-noadvise` (A) | `6e04e475cf41` | independent run 2 -- different arm and machine |
| 4 | phi `nofuse-noadvise-onA` (A) | `7b050e2281f7` | independent run 3 |
| 5 | qwen `fuse-noadvise` (A) | `c5c4223d83cb` | only the full-attention kind is capturable (16 of 64 layers) |

Skill + stage live on `mvasiljevic/qb2/skillexp/challenger-skill` (branched off `base`, so none of
the five arm branches move -- RUN-PLAN 6 would invalidate every completed run). Each cell's tree is
deliberately MIXED: `.agents/` from that skill branch plus `models/autoports/<md>/` from the
incumbent tag. That is why these are not arm results.

Per-cell wall clock and gate outcome accumulate in `skillexp-logs/challenger-timing.tsv` on
machine A; a cell that fails the gate is pushed to `wip/challenger/<arm>/<md>`, never tagged.

**Why phi three times:** three independent optimizations of the same model turn the advisor's
"poor precision, good recall" claim into a **reproducibility test**. If recall is durable the three
disagreement sets should largely coincide; if they diverge, recall is agent-dependent too.

Gate conditions (a cell cannot be tagged otherwise) -- each maps to a documented past failure:
1. `incumbent.json`: 3 repeat decode measurements taken BEFORE any advisor artifact exists
   (frozen baseline + noise floor; observed same-config spreads 0.2-31 us, so ties are real)
2. one capture per layer kind; `report.json` + `final_ir.mlir` parse and contain a matmul
3. the capture's traced weight dtypes EQUAL the incumbent's shipped dtypes -- 3 of 4 existing
   capture scripts construct with no policy argument, which is why north traced bf16 and shipped bfp8
4. `dram_sharded_considered` recorded, and a 0 classified as EITHER a wrong-precision capture OR
   the bf16-DS-off-by-default eligibility gate -- OPT-015 names only the first
5. `reconciliation.json`: every disagreement carries a measured number or an explicit
   below-threshold reason. No prose-only rejections -- that is how qwen's 152 us/layer norm win was lost
6. `final_ms <= incumbent_ms`, incumbent wins ties -- the stage cannot ship a slower decoder, only
   spend time. A NO-CHANGE outcome (nothing beat the incumbent) is a valid, publishable result.
7. the shipped decoder passes the incumbent's own correctness oracle -- a faster decoder that fails
   its oracle is a regression with a good number
8. iteration is allowed but capped at 3 captures per layer kind, each with a recorded trigger

Advice is consumed as **recall, not geometry**: the set difference of ops the advisor puts in
sharded L1 that the incumbent leaves interleaved or on <=2 cores. Its geometry is ignored or swept
in BOTH directions -- the north sweep that appeared to vindicate its block widths only tested
points below them on a monotone axis. `compute_config`/`math_fidelity` is traced state, never
advice (taking it literally cost north 30.8 us, -10.7%).

#### 2026-07-30 ~15:00 — STAGE EXTENDED AGAIN: combination search added. Third run.

The operator asked whether the stage finds the **best combination**. It did not, and that is a
third method change, so the five cells are being run once more under the full method. No
challenger tag currently exists — the slate is clean, and nothing partial is being published.

What was missing: steps 4-5 screened each advised chain **independently** against the frozen
incumbent and accepted or rejected it on its own. **Chains interact** — they share L1 and the
conversion boundaries at their edges — so two changes that each beat the incumbent alone can lose
together, and a cumulative set could ship while being worse than one of its own members. The old
invariant only guaranteed "not worse than the incumbent", never "best of what was measured".

Three additions (`challenger-skill` @ `67da647d5eb`, also on `challenger-skill-pipeline`):

1. **COMBINE** (new step 5b): after screening, measure the cumulative winner set plus pairwise
   combinations of the top material chains. Every measured set is recorded with its ms and oracle
   result; `best_set` must be the best **measured** set, never an inferred one. The invariant is now
   `final_ms <= incumbent_ms` **AND** `<= best_single_ms`, so shipping a combination worse than a
   single change already measured is refused rather than left to judgement. Deliberately
   cumulative+pairwise rather than the power set: the cost is bounded by how many chains survived
   screening, which is small exactly when the advisor had little to say.
2. **Per-candidate op-level evidence**: every kept candidate records its own `tt-perf-report` CSV as
   `perf_report`. Previously a candidate needed only an end-to-end latency number, which shows that
   a change helped but not *where* the time moved — and cannot reveal a change that wins overall
   while making its own target op slower.
3. **Re-rank from a fresh profile**: the chain ranking comes from the incumbent's op-level CSV, and
   once the graph changes that distribution no longer exists. Every iteration after the first must
   re-profile, re-run `reconcile.py` on the new CSV, and record it as `reranked_from`.

Also folded in: the gate no longer needs any orchestrator environment, so stage 02b now runs as an
ordinary pipeline stage (it previously required a `CHALLENGER_DECODE_BATCH` env var only this
experiment's driver exported — a gate that cannot pass in the pipeline is a wall, not a gate), and
it asserts `captured_at` is after `measured_at`, i.e. the incumbent really was frozen before the
advisor ran. Prior artifacts are parked at `parked/OLDMETHOD-*` and `parked/RUN2-*`.

#### 2026-07-30 ~14:30 — the two defects that triggered the SECOND run (still applicable history)

```
!! Any challenger result you may have already read is RETRACTED. Do not use it.
!! The old artifacts are preserved, unchanged, on
!!   parked/OLDMETHOD-challenger-<arm>-<md>   and   parked/OLDMETHOD-run-challenger
!! The tags skillexp/done/challenger/** were DELETED, not overwritten, so nothing stale can be
!! mistaken for a current result. The single shared run/challenger branch was deleted too, and is
!! NOT recreated: results now live on one run branch PER CELL -- see WHERE TO LOOK FOR RESULTS.
```

The operator asked two questions that turned out to expose real defects. Both bias the stage
toward "no change", i.e. toward under-crediting the advisor, so every number produced before the
fix is unusable and all five cells are being re-run from scratch.

**Defect 1 — the batch instruction never reached the agent.** `run_stage.sh` delivers the decode
batch by substituting the bare token `DECODE_BATCH` into the prompt. When the prompt was shortened
to fit Codex's 4000-char objective limit, the only occurrence of that token was removed, so the
substitution had no target. phi then captured **and** measured at batch 1 while the driver asked for
32. Because capture and measurement agreed with each other, nothing detected it — but it answers the
batch-1 question on a model whose serving batch is 32, and the same advice has measured **+12.3% at
b1 versus -8.8% at b32**. gemma was unaffected (batch 1 requested, captured and measured — correct
for MoE). The gate now asserts `capture_batch == incumbent decode_batch == requested DECODE_BATCH`;
two of the three agreeing is precisely what hid this.

**Defect 2 — the materiality bar was per-op, which shreds chains.** The advice is a *chain*: an op
resharded in isolation pays the layout conversions at both its edges, and only the whole L1-resident
chain pays off (OPT-003). Thresholding per op therefore discards exactly the advice class with the
best track record. Measured on phi: one RoPE chain arrived as **9 rows of 0.46-0.97% each**, every
one under the 1% per-op bar, **all dropped unmeasured** — while their summed share was **5.86% of the
decode window**. `reconcile.py` now groups ops into chains and thresholds on the chain's SUMMED
share; the gate refuses a material chain with no number, and refuses a set of same-chain rows
dropped while their sum clears the threshold.

**Freshness enforcement added for the re-runs.** A clean-tree preflight is not sufficient on its own
— earlier in this project a re-run passed exactly that check and then cherry-picked its
predecessor's commits out of the object store. Each re-run must now also pass: every commit touching
`doc/advisor_challenger/` authored inside its own stage window (cherry-pick and rebase preserve
author dates, so inherited history is visible); no cherry-pick/rebase in the work branch reflog; and
no byte-identical artifacts shared with ANY parked copy of that cell -- `OLDMETHOD-*`, `RUN2-*` and
`RUN3PARTIAL-*` alike; a glob covering only the first would have let a re-run inherit the rest.
Failing any
of these parks the cell to `parked/NOTFRESH-challenger-<arm>-<md>` instead of tagging it.

Skill + stage + gate fixes are on `mvasiljevic/qb2/skillexp/challenger-skill` @ `e0a56a11064`.
Gate re-verified after the change: the all-good control and the legitimate no-change outcome pass,
and 8 new negative cases fire including the three-way batch mismatch and the exact 9-row chain shape.

#### RESULTS: ALL 9 CHALLENGER CELLS TAGGED (2026-07-30)

| cell (tag under `skillexp/done/challenger/`) | batch | wall | gate |
|---|---|---|---|
| `fuse-noadvise/microsoft_phi_3_5_mini_instruct` | 32 | 1445s | pass |
| `nofuse-noadvise/google_gemma_4_26b_a4b_it` | 1 | 1145s | pass |
| `nofuse-noadvise/microsoft_phi_3_5_mini_instruct` | 32 | 905s | pass |
| `nofuse-noadvise-onA/microsoft_phi_3_5_mini_instruct` | 32 | 965s | pass |
| `fuse-noadvise/qwen_qwen3_6_27b` | 32 | 1745s | pass |
| `exp17/meta_llama_llama_3_2_1b_instruct` | 32 | 905s | pass (strict) |
| `exp17/meta_llama_llama_3_1_8b_instruct` | 32 | 786s | pass (strict) |
| `exp17/microsoft_phi_3_5_mini_instruct` | 32 | 1145s | pass (strict) |
| `exp11/google_gemma_4_12b` | 32 | 1925s | pass (strict) |

#### WHAT THE ADVISOR ACTUALLY DID, PER CELL (read from the tagged artifacts)

Every cell captured cleanly: `dram_sharded_considered` 4-5 and `advised` 4-5 everywhere, no
`dram_sharded_zero_cause` anywhere, batch three-way agreement on all nine. So no result below is
"the advisor was blind" -- it saw the graph and made recommendations in every case.

| cell | material chains | share of decode window | outcome |
|---|---|---|---|
| `exp17/microsoft_phi_3_5_mini_instruct` | 3 (mlp, rope, norm_residual) | **54.8%** | all rejected, -0.02% |
| `exp11/google_gemma_4_12b` | 4 (o-projections, per-head norms) | **37.9%** | all rejected, 0.0% |
| `nofuse-noadvise/google_gemma_4_26b_a4b_it` | 2 (attn projections, qkv) | 24.7% | all rejected, 0.0% |
| `fuse-noadvise/qwen_qwen3_6_27b` | 2 (rope, head norms) | 5.0% | all rejected, 0.0% |
| `exp17/meta_llama_llama_3_1_8b_instruct` | 1 (norm_residual) | 3.2% | rejected by 0.007 ms, 0.0% |
| `exp17/meta_llama_llama_3_2_1b_instruct` | 0 -- empty set difference | 0% | nothing to test |
| `fuse-noadvise/microsoft_phi_3_5_mini_instruct` | 5 | - | **kept, -7.40%** |
| `nofuse-noadvise-onA/microsoft_phi_3_5_mini_instruct` | 3 | - | **kept, -6.01%** |
| `nofuse-noadvise/microsoft_phi_3_5_mini_instruct` | 2 | - | **kept, -5.20%** |

#### CORRECTION (2026-07-31): "all rejected" was too strong, and one cell's baseline is defective

I re-checked whether the baseline was freshly re-measured and whether candidates were measured the
same way. Two things came out of it that change how the table above must be read.

**The baseline IS fresh and in-session on every cell.** Each stage re-measured the shipped decoder
itself before running the advisor -- 3 to 5 repeats, `incumbent_ms` = the BEST repeat, timestamped
minutes after the stage started. No published or cross-machine number was reused as a baseline.

**But candidates are NOT measured symmetrically.** Every candidate records a single `measured_ms`;
**no cell records repeats or samples for any candidate**. So the incumbent gets best-of-3-to-5 while
a challenger gets one draw, and must beat the incumbent's FASTEST sample to win. That bias runs
AGAINST the advisor: wins are conservative and trustworthy, rejections are not symmetric.

**Read DELTA %, not the noise-floor ratio.** An earlier version of this page ranked results by
|delta| / noise_floor. That was wrong: the ratio measures how precisely a cell happened to measure, not
how much the change matters. Two candidates make the point: phiFN mlp_gate_up at **+9.09%** and
phi-exp17 rope at **+1.48%** have nearly identical ratios (42.4 vs 43.4) for effects six times apart,
purely because one cell had tighter repeats. And gemma-4-12B sliding_attention_o_proj is a genuine
**-0.96%** yet reads "unresolvable" at ratio 0.7, only because that cell had three repeats scattered by
18.28 us. The floor belongs as a RESOLVABILITY FLAG, never as a ranking.

| cell | candidate | delta % | delta us | floor us | resolvable? |
|---|---|---|---|---|---|
| phiFN | mlp_gate_up | **+9.09%** | +73.38 | 1.73 | yes |
| phiFN | **SHIPPED** | **-7.40%** | -59.70 | 1.73 | yes |
| phiFN | norm_residual (kept) | -7.33% | -59.14 | 1.73 | yes |
| phiA | rope = **SHIPPED** | **-6.01%** | -39.42 | 0.68 | yes |
| phiB | **SHIPPED** | **-5.20%** | -40.91 | 1.12 | yes |
| phi-exp17 | norm_residual | **+4.54%** | +49.94 | 0.38 | yes |
| phiFN | attention_qkv | +3.75% | +30.23 | 1.73 | yes |
| phi-exp17 | rope | +1.48% | +16.30 | 0.38 | yes |
| phiA | mlp | +1.25% | +8.19 | 0.68 | yes |
| phiFN | attention_output | +1.24% | +10.03 | 1.73 | yes |
| llama-3.1-8B | norm_residual | +1.13% | +7.02 | 6.28 | marginal (1.1x) |
| gemma-4-12B | sliding_attention_o_proj | -0.96% | -12.85 | 18.28 | **NO - floor too coarse** |
| gemma-4-12B | 3 other chains | -0.37% .. +0.16% | -4.95 .. +2.13 | 18.28 | **NO** |
| phi-exp17 | **SHIPPED** | -0.02% | -0.27 | 0.38 | tie |
| llama-3.2-1B | best candidate | -0.20% | -0.81 | 4.36 | tie |

**Stated at the right level:**

* **Cell outcomes:** phiFN **-7.40%**, phiA **-6.01%**, phiB **-5.20%** shipped faster decoders.
  phi-exp17, gemma-4-12B, llama-3.2-1B and llama-3.1-8B shipped unchanged.
* **Candidate outcomes:** several advisor suggestions were HARMFUL and correctly discarded - phiFN
  mlp_gate_up **+9.09%**, phi-exp17 norm_residual **+4.54%**, phiFN attention_qkv **+3.75%**. That is
  the advisor being wrong and measurement catching it. These are NOT wins; an earlier version of this
  page conflated cell-level outcomes with candidate-level ones.
* **Within one cell the advice splits both ways:** phiFN KEPT norm_residual (-7.33%) and REJECTED
  mlp_gate_up (+9.09%). Per-chain screening is what separates them - a single accept/reject over the
  whole advice set would have shipped a +9% regression alongside the -7% gain.
* **gemma-4-12B measured nothing usable:** its floor (18.28 us from 3 scattered repeats) is LARGER than
  every effect it tested (-12.85 .. +2.13 us). That answer is "unmeasurable", not "no gain".

**The floor is weak evidence in itself:** max-min of 3-5 samples, not a standard deviation and not a
confidence interval. With n=3 the range is crude, unstable, and understates variability, so "34x the
floor" is a sanity check and not a statistical claim.

**`fuse-noadvise/qwen_qwen3_6_27b` -- BASELINE DEFECTIVE, do not use its numbers.** Its
`incumbent.harness` is `tests/traced_synthetic_pcc.py`, a CORRECTNESS/PCC test rather than a perf
harness, which is why its "latency" is **937.344 ms** for a decode step every other qwen measurement
puts near 1.2 ms. Its combination sets are all ~937 ms (self-consistent with that baseline) while its
reconciliation chains are ~1.2 ms -- two different measurement scopes in the same field. The invariant
was evaluated on the 937-scale so nothing invalid shipped, and the cell reports 0.0%, but that 0.0% is
NOT a decode-latency result and its chain verdicts are not verifiable from the artifacts. Needs
re-running against a real perf harness.

**Gate gap this exposes:** the gate requires a `measured_ms` per candidate but does not require
candidate REPEATS, nor that the candidate used the SAME harness as the incumbent freeze, nor that the
harness is a perf harness at all. All three should be enforced before further challenger cells.

**Three things in that table are worth the analysis agents' attention:**

**(a) The zero-deltas are NOT "nothing to say".** On four cells the advisor's recall reached
**25-55% of the measured decode window**, those chains were built and measured, and the incumbent's
existing choices won every time. That is a much stronger statement than a null result, and it is
only visible in `reconciliation.json` -- `final.json` alone shows `0.0%` and looks empty.

**(b) The same model split on the INCUMBENT, not the model.** phi-3.5-mini ran four times with the
same advisor, same batch, same gate: **-7.40% / -6.01% / -5.20%** on the three skillexp incumbents,
and **-0.02%** on the experiment-17 incumbent. Whatever drove the difference, it was a property of
the decoder being challenged, not of the model. This is the one axis the current corpus can see
and is the reason a same-model / different-builder pair would be informative.

**(c) llama-3.2-1B is a complete overlap, not a silence.** The capture traced all 21 ops
(`uncapturable: false`), advised **DRAM-sharded weights on all 5 linears plus L1 width-sharded
RMSNorms**, with `allow_bf16_dram_sharded_matmul=true` (its norm weights are bf16). The cell'\''s own
words: *"Empty set difference: both RMSNorms already execute L1 width-sharded on 32 cores and all
five linears already execute with DRAM-sharded weights."* The no-advise decoder had independently
reached the advisor'\''s entire recommendation. The only residual difference was CORE GEOMETRY, and the
cell deliberately did not treat that as advice -- consistent with the corpus finding that the
advisor'\''s op SET is reliable while its geometry is not (rejected on phi at -34us, tied on qwen,
and on north it only "won" because nothing above its value was ever tested).

**Batch caveat on the four exp cells.** DECODE_BATCH=32 was chosen by me on the dense-model
convention, NOT read from each snapshot: three of the four record no batch in their own optimize
docs (phi-exp17 mentions batch 1). Each cell is internally consistent -- freeze, capture and
evaluate all at 32 -- but 32 may not be the batch those incumbents were tuned at. Re-running any of
them at batch 1 is cheap if the matched design point matters.

**Selection guidance for any further advisor testing.** Counter-intuitively, the NEWEST and best
incumbents are the WORST test material: the better the decoder, the emptier the set difference
(llama-3.2-1B is the limit case). Good targets are weaker/older incumbents with headroom, and dense
models. Poor targets: granite-4.0-h-tiny (hybrid Mamba -- SSM ops are terminal in the tracer, so the
advisor is structurally blind) and qwen3-30B-A3B (MoE experts invisible; the qwen cell showed only
5% of its window reachable at all).

**FOUR CAVEATS THE ANALYSIS MUST NOT MISS:**

1. **Two gate versions.** The five 2x2-arm cells ran `challenger-skill` @ `c30d4115b45`; the four
   dense cells ran the stricter `challenger-skill-pipeline` @ `f463d26e584`. Consequence: in the
   first five, `combination.measured_sets` entries carry a number and an oracle result but
   **`chains: null`** -- you can see that N configurations were measured but not WHICH. The four
   dense cells name every set. Totals and winners are sound in both; per-set attribution exists only
   in the dense four. The gate was deliberately NOT tightened mid-batch, because changing method
   inside a batch is what forced three earlier re-runs.

2. **"Advisor found nothing" is not "advisor lost".** On `exp17/meta_llama_llama_3_2_1b_instruct`
   the only reconciliation row is `advisor_recall_set_minus_shipped`, **below_threshold** -- so there
   was no material chain to test and the combination search had nothing real to combine. The one set
   that beat the incumbent (`static_capture_cleanup`, 0.41107 vs 0.41188 ms) is INSIDE the 0.00436 ms
   noise floor and looks like capture scaffolding, not advisor advice; the tie rule correctly kept the
   incumbent. Both this and "chains were tested and lost" record as `outcome: no_change`. They are
   different findings. Read `reconciliation.json` before concluding the advisor was measured and beaten.

3. **`exp17/microsoft_phi_3_5_mini_instruct` is a SECOND, INDEPENDENT phi incumbent.** Do not pool it
   with the three skillexp phi cells -- different incumbent, different provenance. It was also
   published BY HAND: its advisor decision trace was 112.56MB, over GitHub's 100MB per-file limit, so
   the driver's commit was rewritten with the trace gzipped (112.6MB -> 1.1MB, `gunzip` restores it
   byte for byte) and the gate re-run after the rewrite. `dense-timing.tsv` records
   `yes-manual-after-blob-surgery` for that row.

4. **The dense four had no shipped perf harness.** None of those snapshots ships
   `tests/optimized_decoder_perf.py`, so each stage located or wrote its own profiling path and
   recorded it in `incumbent.json.harness` (llama-3.2-1B, for instance, wrote
   `doc/advisor_challenger/perf_harness.py`). Check that field before comparing wall-clock or
   latency across dense and 2x2 cells.

#### PAUSED 2026-07-31 by operator instruction - no further cells will launch

Stop sentinels are in place for all three drivers (.onA-STOP-AFTER-CELL, .challenger-STOP,
.dense-STOP). Nothing is on the device, no driver is running, no stage lock is held. The repo was left
clean: remote.origin.fetch back to +refs/heads/*, tagOpt unset, and all hidden-ref save files restored.
Resuming is: remove the relevant sentinel and relaunch that driver.

**COMPLETED WORK, all tagged and verified on github.com/tenstorrent/tt-metal:**

| set | cells tagged | notes |
|---|---|---|
| challenger 02b, 2x2 arms | 5 of 5 | phiFN, gemma-26B, phiB, phiA, qwen |
| challenger 02b, dense snapshots | 4 of 4 | llama-3.2-1B, llama-3.1-8B, phi-exp17, gemma-4-12B |
| nofuse-noadvise-onA replicates | **3 of 4** | phi 54m, north 2h03m, gemma-26B 2h19m; **qwen outstanding** |

#### THE ONE OUTSTANDING CELL: nofuse-noadvise-onA/qwen_qwen3_6_27b

Rejected **three times**, each time for reusing a previously-built optimized decoder rather than
building one. Nothing was published on any attempt; each is parked and the claim is retained.

| attempt | wall | what happened |
|---|---|---|
| 1 (2026-07-30 21:22) | 1685s | cherry-picked 4 commits at 21:24:01, 2m after a clean start |
| 2 (2026-07-31 05:05) | 245s | cherry-picked 6 commits at 05:07:52 |
| 3 (2026-07-31 08:08) | 2286s | cherry-picked 1 commit at 08:10:56, then ~36m of genuine work on top |

Attempt 3 ran with protections verified engaged BEFORE launch: 46 local refs carrying a prior qwen
optimized decoder were hidden, remote.origin.fetch was narrowed to a non-existent refspec and tag
following disabled so a fetch inside the stage could not restore them, and a scan confirmed **0 refs
exposed prior qwen work**. It still cherry-picked. The protections self-restored correctly afterwards.

**So hiding refs is NOT sufficient, and an earlier version of this page overstated the fix.** A
cherry-pick mints a new SHA but preserves the original author date, and the source objects stay
reachable without any ref - through the work-branch reflog, dangling objects, or the packfile directly.
Refs were never the only path.

**What the protections DID achieve:** gemma-26B launched at 05:09:55, after the same protections were
fully in place, and built from scratch cleanly - 0 cherry-picks, 2h19m, 185 optimize files. The qwen
attempt that failed at 05:07:52 did so seconds BEFORE the protections were applied. So they help, and
they are now applied automatically by the driver around every stage and restored after; they simply do
not stop this particular cell.

**Pattern worth noting:** three of four models built from scratch without difficulty (phi, north,
gemma-26B - the latter twice, once with and once without protections). qwen is the only cell that reaches
for prior work, and it does so within ~3 minutes on every attempt.

**The next thing that would actually change the outcome** (proposed, NOT done): give the stage an object
store that physically cannot contain a prior decoder. The arm branch and the FD tag contain ZERO optimize
work for this model (verified: doc=0 code=0 on both), so: move .git aside, init a fresh repo, fetch only
the arm branch and the FD tag, run the stage there, then restore the full history to run the
freshness/identity checks and publish (those checks need every parked copy, which only the full history
has). It deletes nothing, and the build survives because build_Release and python_env are untracked.

Until that is run, qwen should be recorded as **not reproducible from scratch on this box under the
current nofuse-noadvise arm prompt**. The arm prompt itself is machine B's and must not be edited - doing
so would invalidate every cell already measured against it.

#### CURRENT ACTIVITY — refreshed every ~15 min, so this line is live

**No stage on the device right now.**

Last 3 driver events:
```
2026-08-11T07:19:28+00:00   GATE PASSED
2026-08-11T07:19:34+00:00   PUBLISHED + TAGGED skillexp/done/advchal-v3/fuse-noadvise/coherelabs_north_mini_code_1_0
2026-08-11T07:19:54+00:00 === challenger driver done ===
```

#### TOOLCHAIN — for comparison with the other machine

| what | value |
|---|---|
| codex model | **gpt-5.6-sol** |
| reasoning effort | **not set** (codex default; no explicit effort/verbosity in `~/.codex/config.toml`) |
| codex-cli | 0.144.4 |
| runner | `app-server`, `approval_policy=never`, `sandbox=danger-full-access` |
| advisor | tt-mlir pinned `618cd4e75d` (`ttnn-advise` from `/opt/ttmlir-toolchain/venv`) |
| host | qb2-120-p05t03, 4 devices |

**Verified PER STAGE**, by mapping each stage's `thread_id` from its `manifest.txt` to that
thread's own session rollout — not generalised from one file:

| stage group | model | effort | coverage |
|---|---|---|---|
| skillexp optimize (2x2 arms) | `gpt-5.6-sol` | unset | 16 threads / 10 stages |
| `nofuse-noadvise-onA` replicate | `gpt-5.6-sol` | unset | 1 thread / 1 stage |
| challenger 02b | `gpt-5.6-sol` | unset | 9 threads / 5 stages |
| functional decoder | `gpt-5.6-sol` | unset | 2 threads / 2 stages |

So the challenger stage is on the same footing as the optimize cells it judges. This mattered to
check: across ALL sessions on this box there are **three** combinations — `gpt-5.6-sol/None` (122),
`gpt-5.6-sol/low` (48) and `gpt-5.5/xhigh` (29) — but the `low` and `gpt-5.5` sessions are other
work on the same machine, not experiment stages. Quoting the aggregate would have reported a mix
that applies to no measured cell.

**Caveat for cross-machine comparison:** `effort=unset` means no explicit effort is configured, so
the value actually applied is the app-server default for `gpt-5.6-sol`, which the rollout does not
record. All stages here are configured IDENTICALLY; the absolute level is not a number I can quote.
If the other machine sets effort explicitly, that is the discrepancy to look for — unset-vs-explicit
looks like "same config" on both sides while differing in practice.

Read from the live session rollout, not assumed: `model=gpt-5.6-sol`,
`reasoning_effort=None`, `approval_policy=never`. If the other machine differs on either of the
first two, cell-to-cell comparisons between machines carry an agent-capability term, not just a
machine term.

#### CHANGE OF PLAN (2026-07-30 late) — scope extended to dense models

After the five challenger cells, the stage also runs on the **four dense models** that exist
outside `forge_experiments` in the agentic-research repo. Rationale: dense is where the advisor can
actually see the graph — `sparse_matmul` (all MoE experts) and SSM/gated-delta ops are terminal in
the tracer, so on MoE models it is structurally blind to most of the time.

| # | model | snapshot | arm label | policy provenance |
|---|---|---|---|---|
| 6 | llama-3.2-1B | `experiment-17/.../evidence-final-20260615T2202Z` | `exp17` | shipped artifacts present (9 json) |
| 7 | llama-3.1-8B | `experiment-17/.../evidence-final-20260615T2301Z` | `exp17` | **re-measured** (README+work_log only) |
| 8 | phi-3.5-mini | `experiment-17/.../evidence-final-20260615T2202Z` | `exp17` | **re-measured** |
| 9 | gemma-4-12B | `experiment-11/.../completed-20260609T074007` | `exp11` | **re-measured**, dir rename needed |

Ordered best-evidenced first so the mechanism is validated on the strongest cell before the weak
ones. Where no shipped-policy artifact exists the policy is **re-measured** and
`shipped_policy_source` says so — the gate's ban on `constructor_defaults` stays intact, and the
weaker provenance is visible in the artifact rather than hidden.

**Excluded and why:** llama-3.1-70B (already advisor-seeded AND multi-device); mistral and dense
qwen (exist ONLY under `forge_experiments/`, which is out of scope); the MoE snapshots
granite-4.0-h-tiny / qwen3-30B-A3B / qwen3.6-27B / gemma-4-26B-A4B / north-mini (MoE, and four of
them duplicate models already covered by the skillexp cells).

**`phi-3.5-mini` note for the analysis:** cell 8 is a SECOND, INDEPENDENT incumbent for a model the
skillexp cells already cover. Do not pool it with them — different incumbent, different provenance.
Its `exp17` arm label keeps the refs distinct.

After the dense cells, machine A returns to the paused `nofuse-noadvise-onA` replicates of machine
B's cells (qwen b32, gemma b1, north b1).

#### 2026-07-30 17:30 — FOURTH RUN. All five cells lost to a bug in my own freshness check.

```
!! The five cells that ran 15:43-17:29 produced NO results. Do not look for them.
!! Cause was machine A's driver, NOT the models and NOT the gate.
```

Every cell was rejected with `FRESHNESS REJECT: no commit ... touches doc/advisor_challenger at
all`. The stage writes its output into the **working tree** and does not commit it — the driver
commits at publish time — so "commits touching the artifact path" is always **zero**, and the check
read the normal case as fraud. It then parked a work branch that contained no artifacts, and the
next cell's purge deleted the working-tree files. **~103 minutes of device time across five cells,
unrecoverable** (untracked files removed by `git clean` are not in the object store).

Two fixes, both in the driver:
1. the stage's output is now **committed the moment the stage exits**, before any check runs, so the
   artifacts are durable before anything can reject or purge them;
2. zero artifact-touching commits is treated as **normal**, not fraud. Freshness now requires that
   artifacts EXIST, and rejects only genuinely inherited history (a commit authored before this
   stage started, a cherry-pick/rebase in the reflog, or blobs byte-identical to a parked copy).

A third fix, in the gate: the runner calls `<prompt>.check.sh` with **no arguments** and scopes it
via the `MODEL_DIR` env var. The gate required `$1`, so it died instantly on every in-stage check
(twice per cell, all five cells) and the agent never received one actionable gate message — it could
not self-correct in its own remediation loop, and the stage reported `advisory-fail` for a reason
unrelated to its evidence. Now resolves `$1` → `MODEL_DIR` → `HF_MODEL`.

Skill/stage/gate at `mvasiljevic/qb2/skillexp/challenger-skill` @ `c30d4115b45`.

#### WHERE TO LOOK FOR RESULTS — all on `github.com/tenstorrent/tt-metal`

Every ref below is keyed by **arm AND model_dir**, because three of the five cells are the same
model (`microsoft_phi_3_5_mini_instruct`, on arms nofuse-noadvise / fuse-noadvise /
nofuse-noadvise-onA) and therefore write the same `doc/advisor_challenger` path. An earlier layout
merged all five cells into one shared `run/challenger`, where each phi overwrote the previous one's
evidence. **There is no single `run/challenger` branch any more — do not look for one.**

For each cell, three refs, all three carrying the same tree:

| what | ref | means |
|---|---|---|
| **the result** | `skillexp/done/challenger/<arm>/<model_dir>` (tag) | gate PASSED; this is the only thing that counts as a result |
| the run branch | `mvasiljevic/qb2/skillexp/run/challenger/<arm>/<model_dir>` | the published tree the tag points into |
| the cell branch | `mvasiljevic/qb2/skillexp/cell/challenger/<arm>/<model_dir>` | that cell's own work branch, full commit history |
| gate FAILED | `mvasiljevic/qb2/skillexp/wip/challenger/<arm>/<model_dir>` | ran but did not pass; **never** a result |
| not fresh | `mvasiljevic/qb2/skillexp/parked/NOTFRESH-challenger-<arm>-<md>` | reused a prior attempt's artifacts; **never** a result |

The five cells, spelled out so nothing has to be guessed:
```
skillexp/done/challenger/fuse-noadvise/microsoft_phi_3_5_mini_instruct        (phiFN, b32)
skillexp/done/challenger/nofuse-noadvise/google_gemma_4_26b_a4b_it            (gemma, b1)
skillexp/done/challenger/nofuse-noadvise/microsoft_phi_3_5_mini_instruct      (phiB,  b32)
skillexp/done/challenger/nofuse-noadvise-onA/microsoft_phi_3_5_mini_instruct  (phiA,  b32)
skillexp/done/challenger/fuse-noadvise/qwen_qwen3_6_27b                       (qwen,  b32)
```

Inside a tagged cell, the evidence is at `models/autoports/<model_dir>/doc/advisor_challenger/`:

| file | what to read it for |
|---|---|
| `incumbent.json` | the frozen baseline: `incumbent_ms` (best of >=3 repeats), `noise_floor_ms`, `decode_batch`, and the shipped precision policy with the artifact it came from |
| `final.json` | the outcome: `final_ms`, `delta_pct`, `changed`, `best_single_ms`, `combination.measured_sets` (every set actually measured), `oracle`/`oracle_passed`, `iterations` |
| `reconciliation.json` | what the advisor said vs what shipped, grouped into **chains** with each chain's summed share of the decode window, its verdict and its measured number |
| `shard_advise/<layer_kind>/report.json` | per capture: `dram_sharded_considered`/`advised`, `traced_weight_dtypes`, `capture_batch` |
| `shard_advise/<layer_kind>/final_ir.mlir` | authoritative for what was advised (program configs, block widths, required input layouts) |
| `tracy/` | tt-perf-report outputs, incl. the per-kept-candidate CSVs referenced as `perf_report` |
| `README.md` | advised / kept / rejected-with-a-number / uncapturable, in prose |

A cell shipping `outcome: no_change` is a **real result**, not a failure: it means nothing the
advisor suggested beat the frozen incumbent, and the invariant forbids shipping a change that does
not. Read `reconciliation.json` to see what was offered and measured before concluding anything.

**Parked refs are NOT results** — they are superseded attempts, kept so nothing is lost:
`parked/OLDMETHOD-challenger-*` and `parked/OLDMETHOD-run-challenger` (per-op thresholding, batch
never delivered), `parked/RUN2-challenger-*` (second method), `parked/RUN3PARTIAL-challenger-*`
(work branches polluted by untracked leftovers). Per-cell wall clock and gate verdicts accumulate in
`skillexp-logs/challenger-timing.tsv` on machine A, which is host-only and not in any repo.

Skill, stage prompt and gate: `mvasiljevic/qb2/skillexp/challenger-skill` @ `67da647d5eb`
(same content on `challenger-skill-pipeline`, which is the branch to reuse outside this experiment).

#### Live per-cell state (rendered from the runner's own records, not from prose)

Rows are in the driver's ACTUAL execution order, so row numbers here match the order cells ran.
phiFN is first because its stage was already live when the queue was rebuilt, not because it ranks
highest; the original value ordering put gemma first.

| # | cell | incumbent | batch | wall | gate | tag |
|---|---|---|---|---|---|---|
| 1 | `challenger/fuse-noadvise/microsoft_phi_3_5_mini_instruct` | `6e04e475cf41` | 32 | 24m05s | pass | **tagged** |
| 2 | `challenger/nofuse-noadvise/google_gemma_4_26b_a4b_it` | `a3e826aa6f74` | 1 | 19m05s | pass | **tagged** |
| 3 | `challenger/nofuse-noadvise/microsoft_phi_3_5_mini_instruct` | `c2331f8bccfb` | 32 | 15m05s | pass | **tagged** |
| 4 | `challenger/nofuse-noadvise-onA/microsoft_phi_3_5_mini_instruct` | `7b050e2281f7` | 32 | 16m05s | pass | **tagged** |
| 5 | `challenger/fuse-noadvise/qwen_qwen3_6_27b` | `c5c4223d83cb` | 32 | 29m05s | pass | **tagged** |

`gate` is the verdict of `02b-advisor-challenger.check.sh`, run by the DRIVER after the stage
exits — a cell is tagged only if the gate passes, so "not tagged" with a wall time means the stage
finished but failed the gate and its work is on `wip/challenger/<arm>/<md>`.

### 3. Also running: a direct norm-sharding sweep on gemma, independent of the advisor

The advisor is not needed to test the norm hypothesis -- qwen already showed that op class going
**85 us -> 9 us** by sharding decode norms. So gemma's norms are swept directly as well, which
separates "is the win there" from "does the tool find it". Published under the audit namespace.

Known ceiling, stated up front: none of this reaches gemma's largest remaining win -- the routed
expert-down grid (8 cores vs 44, **+117.7 us/layer** at identical BFLOAT8_B, all 30 layers) is
`ttnn.sparse_matmul`, terminal in the tracer. No re-sequencing reaches it.

## Blocked / critical (audit these before calling them real blockers)
- none
