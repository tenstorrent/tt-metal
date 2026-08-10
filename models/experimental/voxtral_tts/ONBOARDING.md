# Voxtral-TTS on Tenstorrent — start here

You are picking this up with no context. Read this top to bottom once — about ten minutes — and you
will be able to run everything and change something without breaking it.

**Three files, three jobs:**

| file | what it is | when to read it |
|---|---|---|
| **`ONBOARDING.md`** (this) | how to run things, how to prove you didn't break them, the method | first, once |
| **`STATUS.md`** | the running log, `§1`–`§6.38`. Every experiment with its numbers, **including the rejected ones** | before trying anything, to check it isn't already settled |
| **`tt/NOTES.md`** | the prose that used to live in the code. Grep-able IDs `[gpt-04]`, `[flow-10]`, `[codec-12]`, `[pipe-02]` | when a line of code carries a `NOTES.md [id]` pointer |

The `tt/*.py` files are deliberately thin — one-line pointers, no essays. **If you find yourself
writing a paragraph in a `.py` file, it belongs in `NOTES.md` under a new ID.**

### Quick reference

```bash
cd /localdev/lserbedzija/repos/tt-metal
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate    # NOT the repo's python_env — see §2
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD

V=models/experimental/voxtral_tts
python -m pytest $V/tests/ -q --noconftest                 # 122 tests, ~45 s   (--noconftest REQUIRED)
python $V/tests/tt_gates.py --gate codes                   # blocks 1+2, integer codes
python $V/scripts/generate_quality_set.py --tag mychange   # audio; NOTE: writes results{tag}.json
python $V/scripts/score_quality_set.py $V/generated/resultsmychange.json
```

Current: **long-form RTF 0.57–0.64, ~48 ms/frame, 0 WER errors.** Beat that without breaking it.

---

## 1. What this model is

Text + a voice preset in, 24 kHz audio out. Three stages per utterance:

| block | what it does | size | file | cost/frame |
|---|---|---|---|---|
| **Block 1** | autoregressive backbone. Prefills the prompt, then emits one hidden state per audio frame | 3.4B, 26 layers, DIM 3072 | `tt/ttnn_voxtral_gpt.py` | ~23 ms |
| **Block 2** | flow-matching acoustic transformer. Hidden state → 36 acoustic codes, by solving an ODE in 7 Euler steps over 3 layers | 390M | `tt/ttnn_voxtral_flow.py` | ~20.8 ms |
| **Codec** | codes → waveform. Once per utterance, not per frame | | `tt/ttnn_voxtral_codec.py` | ~3.5 ms total |

One frame is **80 ms of audio**, so real-time is 80 ms/frame and we are at ~48, i.e. RTF ~0.6.

`tt/ttnn_voxtral_pipeline.py` wires the three together. `reference/` is a pure-fp32 PyTorch
implementation — **it is the ground truth, not the device.**

**Hardware: one Wormhole N150.** 64 Tensix cores (8×8), 12 DRAM banks, measured DRAM ceiling
**194–202 GB/s**. That last number is the single most useful fact in this project (§6).

> **Every tuned constant in `tt/*.py` was measured on this card.** The norm grids (8×4), `_QKV_GRID_X`,
> `_WO_PRG`'s `in0_block_w=2`, `_SDPA_PRG`'s `k_chunk_size=512`, the core-count minimum at 32 — all of
> it. On different silicon they are starting guesses, not answers, and the sweeps in STATUS are the way
> to re-derive them. Note the code hardcodes `8` rather than querying
> `device.compute_with_storage_grid_size()`, which returns `8×8` here.

---

## 2. Setup — the trap that will cost you an hour

**There are two virtualenvs and the obvious one is wrong.** Use
`/localdev/lserbedzija/repos/xtts_ref_venv`. The repo's own `python_env/` is a **docs** environment
with no `torch` and no `ttnn`. Two symptoms of getting it wrong:

- `ModuleNotFoundError: No module named 'torch'`
- `AttributeError: module 'ttnn' has no attribute 'L1_MEMORY_CONFIG'` — this one means `PYTHONPATH`
  picked up the repo's `ttnn/` *source directory* as an empty namespace package instead of the built
  module.

`PYTHONPATH=$PWD` is required for `models.*` imports and must be set from the repo root.

---

## 3. How to run things

### Tests — 122 tests, ~45 s

```bash
python -m pytest models/experimental/voxtral_tts/tests/ -q --noconftest
```

`--noconftest` is **required**: the repo's `conftest.py` imports `torchvision`, which isn't
installed, and without the flag every test errors at collection.

### Gates — device, seconds to minutes

```bash
G=models/experimental/voxtral_tts/tests/tt_gates.py
python $G --gate wiring      # does anything run at all
python $G --gate prefill26   # Block 1 prefill vs fp32, 15 prompts
python $G --gate decode      # Block 1 decode vs fp32, 15 prompts × 22 frames   (~4 min)
python $G --gate flow        # Block 2 velocity + codes vs fp32
python $G --gate codec       # codec vs fp32
python $G --gate codes       # blocks 1+2 end to end, INTEGER codes  ← the one that predicts audio
python $G --gate decode --cases 0,2 --verbose    # quick subset while iterating
```

`--gate decode` runs all 15 prompts **by default, deliberately.** A 2-prompt default previously let a
regression through: per-case mean ranges ~0.45 pp, so an aggregate over a *different* prompt set is
not comparable to a recorded one. Only ever compare a paired A/B — same cases, same session, one
change.

### End-to-end audio + WER — ~20 min for all 15 prompts

```bash
python .../scripts/generate_quality_set.py --tag mychange
python .../scripts/score_quality_set.py .../generated/resultsmychange.json
```

**The tag has NO underscore.** `--tag mychange` writes `resultsmychange.json`. `generated/` holds ~40
stale `results*.json` from past sweeps; every one scores cleanly and none announces that it is the
wrong file. **Cross-check the frame counts in the scorer's output against the generation log** — that
catches a stale file instantly (`§6.32` is the time it didn't).

WAVs land in `generated/`, gitignored — CC BY-NC derived, so they stay on the box. They are the only
way to actually *hear* the model, and no metric substitutes for that. A ready-made sampler of the
current build is `generated/SAMPLER_current_build.wav` (13 clips, 136 s); play with
`ffplay -nodisp -autoexit`.

---

## 4. Where we are now — the numbers to beat

| | current |
|---|---|
| long-form RTF | **0.57–0.64** |
| per frame | **~48 ms** — Block 1 ~23, Block 2 ~20.8, host 0.2 |
| long-form WER | **0 wrong words** (298 over all 15 cases; 274 over the 3-case subset) |
| Block 1 worst-sample mean / p90 | 0.92% / 1.28%, min PCC 0.999040 |
| Block 2 velocity PCC | 0.99998480 |
| `[END_AUDIO]` natural termination | 15/15 |
| listening pass | "sounds good", `§6.38`-era build |

**"Long-form" means ≥100 frames — currently cases 2, 3 and 10.** Quote only those, always with the
case list. **Case 0 includes kernel compilation (RTF ~1.8) and must be excluded.** Short cases are
seed noise: the same code at seeds 0/1/2 swung that bucket 0.88–2.06%.

Reproduce these before changing anything, so you know the ground you stand on.

---

## 5. The gate ladder — how to prove you didn't break it

Weakest to strongest. Every claim in STATUS names which rungs it cleared.

| rung | cost | what it establishes |
|---|---|---|
| `pytest tests/` | 45 s | structure, shapes, invariants. 122 tests |
| `--gate flow` / `--gate codec` | ~1 min | one block vs the fp32 reference |
| `--gate codes` | ~2 min | integer codes, blocks 1+2. **Read the REAL-prompt block, not the synthetic one** (`§6.40`) |
| `--gate decode` | ~4 min | Block 1 precision, 15 prompts × 22 frames |
| **frame-count A/B** | ~90 s | **bit-exactness** — see below |
| WER, ≥3 seeds | ~10 min | output quality |
| listening pass | minutes | what no metric catches |

### The frame-count A/B is an EXACTNESS gate and NOT a quality gate

```bash
python .../generate_quality_set.py --cases 0,2,3 --tag before   # old build
python .../generate_quality_set.py --cases 0,2,3 --tag after    # new build
# compare 'frames' per case in generated/resultsbefore.json vs resultsafter.json
```

Generation is autoregressive — each frame's output is the next frame's input — so **any** divergence
compounds and moves where the model stops. Reproducing 461 and 487 frames exactly is real proof of
bit-identity, for ~90 s of work.

**But a CHANGED frame count proves nothing about quality.** The control (`§6.38`) — same shipped
code, same prompts, only the `x_0` noise seed changed:

```
case 0:  66 /  76 /  70 frames   (seeds 0/1/2)   swing 10
case 2: 461 / 470 / 438                          swing 32
case 3: 487 / 508 / 449                          swing 59
```

A non-bit-exact change *will* move frame counts by tens, for free. Two findings cited a moved count as
evidence of degradation (`§6.37` sdpa 461→445, w2-BFP8 487→523) when both sit **inside** the seed-only
swing. Read it asymmetrically:

| observation | means |
|---|---|
| frame counts **identical** | the change is bit-exact — strong, and it has never falsely fired |
| frame counts **moved** | the change is not bit-exact. **Nothing more.** |

For quality on a non-exact change the only usable instrument is **WER across ≥3 seeds.** That is
stable on the shipped build — 0 wrong of 274 at seeds 0, 1 and 2 — so the clean baseline is
reproducible rather than lucky, and a word appearing is real signal. Never judge it on one seed.

---

## 6. How to find an optimization

### The floor method

```
op time  =  bytes it must move / 194 GB/s   +   overhead
            \______ irreducible ______/         \__ the only part you can win __/
```

Measure an op isolated, compute its floor from the bytes it genuinely touches, and the difference is
your budget. Rank by `overhead × calls per frame`. **`§6.27` is Block 1's map, `§6.36` is Block 2's
(with line numbers).** Redo the map after any structural change — `§6.36` supersedes `§6.29` for
exactly that reason.

What the maps found: **all ten weight matmuls across both blocks are at the roofline.** Some measure
*faster* than a 194 GB/s floor, which is how we know the true ceiling is ~202. There is no matmul work
left. Everything remaining is small-op overhead, and a small op on a `[1,6,3072]` tensor costs
**~20 µs just to be launched** — genuine device time, not host dispatch (`§6.38` proved that: eager
19.145 ms vs traced 19.230, dispatch is 0%).

That reframes the problem: you are not making math faster, you are **removing work that doesn't need
to happen here.** The three wins in `§6.31` are all that shape — a constant recomputed per step, a
layout conversion avoidable by reading the original layout, and a reduction sitting on the chip when
its result was already being shipped off it.

### Nine rules, each of which cost real time to learn

0. **`N/288` MEANS TWO DIFFERENT THINGS IN THIS PROJECT.** `7/288`, `9/288`, `21/288` in NOTES and
   `§6.8`/`§6.10` are **8 frames x 36 codes on ONE REAL fixture**. `gate_codes`' old `97/288` was **8
   SYNTHETIC frames x 36**. Same denominator, incomparable — and the synthetic one is non-monotonic in
   precision, so it cannot even rank configs (`§6.40`). Current real number: **24/864 (2.8%), 0
   semantic, 100% off-by-one**.

1. **ALWAYS GATE ON REAL PROMPTS.** Random activations read PCC 0.892 where real prompts gave 0.9994
   (trap #12). Written down, and *still* violated in `§6.31` — a bf16 change scored on 64 random
   Gaussian draws, which for an argmax over a vocabulary has no power at all.
2. **Isolated measurements do not decide. The whole block decides.** Six times an isolated result
   failed to survive: `§6.18`, `§6.19`, `§6.27`, `§6.30` (1.543× isolated → *zero* whole-block),
   `§6.33` (**the wrong sign**), `§6.37` (1.14–1.20× → +0.001 ms).
3. **Always report spread next to mean.** A single number with no spread is not a measurement. If the
   effect is smaller than the spread, say INCONCLUSIVE and go measure the whole block. `§6.33` is the
   case where a ~10 µs effect was reported from a measurement whose own spread was 5.8–27.6 µs.
4. **Compare numerics against fp32/fp64 truth, not against the current default.** `§6.25` nearly
   discarded a real 1.2× because it differed from the *default* by 5.3e-03, when against fp64 every
   config was 6.6e-04 from truth.
5. **A config that fails to BUILD tells you nothing about whether it is fast.** `§6.28` records
   writing "the rejection holds" on the strength of an assertion about a missing `memory_config`.
6. **Compare like for like on memory config.** `§6.31`/`§6.33`: a hand-rolled path timed writing to
   DRAM against a fused op writing to L1, read as a tie. Where q/k/v live is worth 2.5 ms/frame
   downstream. Check `t.memory_config().buffer_type` in probes.
7. **Never put a `||` fallback in a gate.** One that silently substitutes a different input is worse
   than one that fails, because it returns a plausible number (`§6.32`).
8. **`git checkout <commit> -- <file>` STAGES the old version.** `git status` shows `MM`, and a later
   `git commit` silently reverts your change. `git restore --staged <file>` after any A/B using it.
9. **Frame counts in a multi-case run depend on the preceding cases.** An hour went into a "moved from
   207 to 220" that reproduced identically when the case was run alone.

### The order to work in

1. reproduce §4's baseline numbers
2. build or refresh the overhead map — measure, never guess where the time is
3. take the largest `overhead × calls` item and form a hypothesis
4. **split the line item before proposing anything.** `§6.30`'s repack theory and `§6.31`'s argmax
   finding both came out of splitting one row into its pieces, and the first killed its own hypothesis
5. probe candidates isolated *only to rank them*, numerics against fp32
6. measure the survivor on the whole block — interleaved A/B, ≥6 rounds
7. climb the §5 ladder as far as the change's numerics demand
8. commit with the numbers **and the rejected alternatives**; add a `§6.x` to STATUS
9. if your hypothesis was wrong, **record that too.** The rejected list is the most valuable part of
   STATUS because it stops the next person re-running a dead end

---

## 7. What is open

**Ranked by size of prize.**

- **The 31 unused tile rows — measured, `§6.35`, and the biggest lead.** Decode uses 1 row of every
  32. In Block 2, **4 utterances cost 1.18× the time of 1 — 3.4× throughput** (844.9 → 995.1 µs per
  `_block`). The 32-row ceiling is **our own constant**, not the hardware: `_NORM_SHARD` is hardcoded
  `(32, 96)` and raises at 48 rows while `wqkv` handles 48 fine. **The fix has a reference
  implementation in-tree — see §9.** This is *throughput, not latency*: per-utterance RTF is unchanged.
- **w2 in BFP8 — a genuine open decision, `§6.38`.** Worth **2.644 ms/step** (5.5% of a frame), the
  largest single win left. Upstream cost is real and reproducible (mean 0.92→1.16%, p90 1.28→1.68%).
  But the output evidence is **1 wrong word in 822 against 0 in 822** — which no test calls
  significant. To settle: all 15 cases × 3+ seeds on both arms.
- **bf16 semantic head** — `§6.31` candidate E, 2.079× on `semantic_code`, another 0.265 ms/frame. Held
  back because it moves the *semantic* token, which feeds Block 1's next input, so one flip redirects
  the rest of the utterance. Needs a broad real-prompt gate, not 8 frames.
- **Block 2's 7 Euler steps → 5** — ~28% of Block 2, but a listening call, not a metric one. No gate
  can tell you whether it still sounds right.
- **`nlp_create_qkv_heads` upstream issue is unwritten.** ~97 µs floor; worth 1.233 ms/frame to
  hand-roll around in Block 2 (`§6.31`). Block 1's decode variant has the same shape. The
  `halo_gather` out-of-range NOC write also deserves one.

**Unexplained, logged rather than resolved:**

- **The 1.233 ms mechanism (`§6.33`).** The hand-rolled head split is *not* faster in isolation, yet
  the block is 1.233 ms/frame faster. Candidate causes listed; none verified. The tracy op profiler
  (§9) is the tool for this.
- **`[flow-10]`'s 158 µs vs today's ~112–127 µs** for what should be the same hand-rolled split.
- **`§6.8`'s absolute levels don't reproduce.** Ruled out gate code, model code, reference, ttnn build,
  prefill rows, prompt selection.

---

## 8. What is settled — do not re-run these

Full numbers in STATUS; this is the index so you don't spend a day on a closed question.

| tried | verdict |
|---|---|
| BFP4 weights | 8.4× the error for 12% of the time |
| fusing w1+w3 into one 3072×18432 matmul | **4× slower** — matmul bandwidth collapses past N≈9216 |
| ttnn's fused q+k RoPE | wrong convention (interleaved vs our half-split); 0.236 ms to adopt |
| device tracing **as a shipping strategy** | +0.35 ms and three silent failure modes. (As a *measurement* tool it is fine and confirmed dispatch is 0% — `§6.38`) |
| lower math fidelity (HiFi2 / LoFi) | Block 2: **slower and 9× worse** (`[flow-03]`). Block 1: ~4 ms for 10–20× the code errors |
| DRAM-sharded matmul for the norm output | 1.66× slower even with blocking tuned; closed three ways (`§6.28`) |
| one-op interleaved `rms_norm` in Block 2 | 2.4× **slower** than three sharded ops — sharding is the fast path |
| folding CFG + Euler into a weighted reduce | 1.543× isolated, **zero** whole-block, flips an FSQ boundary |
| permuting straight from `av` in the unfold | 1.77× faster and **returns garbage** |
| project-then-duplicate in `_solve` | 0.785× isolated. The redundant matmul is ~1 µs; duplicating a 3072-wide result rather than a 36-wide input costs +12 µs (`§6.34`) |
| `ttnn.repeat` instead of `ttnn.concat` to duplicate | 1.8× worse |
| **`sdpa` for Block 2's attention interior** | 5.9× on the op, **+0.816 ms/frame** on the block — but 6.48× the error vs fp64 truth and **cost one WER word**. Reverted (`§6.37`). Needs `scale=1.0` if retried, plus a multi-seed study to settle worse-vs-different |
| **in-place ops, EVERYWHERE in the decode path** | dead end four ways (`§6.39`): Block 2 `_block` +0.001 ms, Block 1's three sites **0.063 ms SLOWER**, Block 2 `_solve` tail within noise. Op count is unchanged and at these sizes the cost is the launch, not the allocation. Also makes functions mutate their arguments |
| **`inplace=True` on the norm program config** | **INERT** (`§6.39`) — Block 1 identical to 3 dp with a 0.006 ms spread and bit-identical output; Block 2 inside noise. Both blocks ship `inplace=False`; changing it does nothing |
| residual-as-bias, **Block 1** | w2's add is already free |
| residual-as-bias, **Block 2** | **not expressible** — ttnn `bias` is per-output-column, our residual differs per row |
| `ttnn.swiglu` | `TT_THROW`s on a concatenated pair; would need w1/w3 fused, which is 4× slower |
| **w1 program config to genuinely fuse silu** | `activation="silu"` is **NOT fused** — it costs the same as a separate `ttnn.silu` (+10.3 µs, 0.485 ms/frame over 47 calls). Only `fused_activation` in a program config folds it in, and on 64 cores no legal grid wins: best 8×6 at 0.975× isolated, −0.057/−0.078 ms whole-block (`§6.41`). **Reachable on P150** (12×6 of 130 cores, 2.42 ms/frame) — not here |
| `_solve` tensors moved into L1 | neutral-to-worse, monotonically (`§6.37`): 20.916 → 20.946 → 20.974 → 21.046 ms |
| **eliminating CFG** | costs only **1.8%** (0.322 ms/frame), because 3 rows and 6 rows both pad to one 32-row tile and the row fold reads each weight once. `p2`'s known-zero half is also free (65.1 µs either way). "CFG doubles the work" does not hold here (`§6.35`) |

---

## 9. What to reuse from elsewhere in this repo

Surveyed 2026-08-06 (`§6.38`). This model was built standalone; four things elsewhere are worth
knowing, and the first unblocks §7's biggest lead.

- **`models/common/modules/rmsnorm/rmsnorm_1d.py` → `_create_sharded_norm_program_config(dim, grid,
  tile_padded_batch_rows, tile_size)`** — builds the config from the **row count**:
  `block_h = tile_padded_batch_rows // tile_size`, with `tile_padded_batch_rows = 32*ceil(batch/32)`,
  plus a `subblock_w` search. **This is exactly what our hardcoded `_NORM_SHARD (32, 96)` /
  `block_h=1` cannot do, and it is the stated blocker on §6.35's 3.4× batching win.**
  `models/common/modules/mlp/mlp_1d.py` applies the same pattern to shard shapes *and* matmul configs.
  **Start here if you pick up the batching lead.**
- **`models/demos/gemma4/tt/spec_decode.py`** — verify step "runs ONE batched forward over
  `[anchor, d1, …, dK]` … candidates in the batch dim", precisely the mechanism for our unused tile
  rows. Its docstring also reasons through worse-vs-different: a batched forward differs by ~1e-5,
  "flips only target near-ties", giving "an equally-valid greedy trajectory thereafter". **Do not
  borrow that conclusion** — their correctness is guaranteed *by construction* (committed tokens always
  come from the target verify); a precision change has no such guarantee.
- **`models/common/tests/modules/sampling/test_sampling_1d.py`** — classifies index disagreements as
  TIE-BREAK (acceptable) vs TRUE-MISMATCH (kernel bug); notes `ttnn.argmax` returns uint32 vs torch's
  int64. Bears on `[flow-08a]`: we moved the semantic argmax to the host, so ties would now be broken
  by `torch.argmax`. fp32 logits plus an additive mask make exact ties effectively impossible — that
  is *why* it is safe, and it wasn't checked when the change shipped.
- **`models/common/tests/modules/attention/profiling/`** — trace capture for device-side timing.
  Checked against our method and **ours holds**: dispatch is 0% of what these probes measure. The full
  tracy profiler (`tools/tracy/process_ops_logs.py`, `python -m tracy -r -m …`) would give per-op
  device times — the tool for §7's unexplained 1.233 ms — but needs `websockets` (absent from the venv)
  and probably a profiler-enabled rebuild.
- **`models/tt_dit/`** (`pipelines/qwenimage`, `pipelines/ltx`) and `models/demos/z_image_turbo` — flow
  matching / DiT pipelines, the nearest in-tree neighbours to Block 2 if its structure is revisited.
  Not examined in depth.

**Checked and not reusable:** `deepseek_v3_b1/fused_ops/lm_head_sampling/` is a bespoke multi-device
kernel on CCL/MoE infrastructure, and our semantic matmul is already at 182 of ~194 GB/s.
`ttnn.sampling` / `ttnn.topk` exist but the host argmax is simpler and 1.439× faster.
`models/experimental/speecht5_tts` and `models/demos/audio/whisper` are different architectures. Two
shared-library *choices* differ from ours and are already settled here — see §8 for HiFi2, and
`[gpt-20]` for the `in0_block_w` divisor heuristic where we swept and found an exactness cliff.

**Also relevant:** `ign/voxtral_p150_qb2` is a separate Voxtral-TTS effort targeting Blackhole P150,
measured in `§6.5` — its build lives at `/localdev/lserbedzija/ign_build` with a venv at
`/localdev/lserbedzija/ign_venv`. On *our* card it runs at 598.1 ms/frame against our 77.6-era number,
but that measures their code on the wrong silicon and says nothing about their P150 work. Two things it
does settle: their comment independently calls the acoustic FM core "the 78%/step bottleneck", which
reproduces our Block 2 finding on a different implementation, and their Block 1 runs **all-BFP8**
without the hang that blocked us — evidence the hang is a tt-metal-version property.

**This survey sampled by relevance, not exhaustively** — the repo has hundreds of models. Covered:
audio/TTS, speculative decoding, continuous/paged batching, sharded-norm configs, sampling/argmax,
flow-matching and DiT, shared matmul helpers, math-fidelity defaults, profiler tooling. Untouched:
vision/CNN, multi-device CCL, training.
