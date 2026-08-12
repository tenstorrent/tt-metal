# Voxtral-TTS on Tenstorrent — start here

You are picking this up with no context. Read this top to bottom once — about ten minutes — and you
will be able to run everything and change something without breaking it.

**Three files, four jobs:**

| file | what it is | when to read it |
|---|---|---|
| **`ONBOARDING.md`** (this) | how to run things, how to prove you didn't break them, the method | first, once |
| **`STATUS.md`** | the running log, `§1`–`§6.74`. `§1`–`§6.38` are N150; `§6.39+` is this fork. Every experiment with its numbers, **including the rejected ones** | before trying anything, to check it isn't already settled |
| **`§8` of this file** | the **ledger**: every shipped optimization, every reversal, every rejection, one line each, pointing at its `§` | to see the whole decision history at a glance |
| **`tt/NOTES.md`** | the prose that used to live in the code. Grep-able IDs `[gpt-04]`, `[flow-10]`, `[codec-12]`, `[pipe-02]` | when a line of code carries a `NOTES.md [id]` pointer |

The `tt/*.py` files are deliberately thin — one-line pointers, no essays. **If you find yourself
writing a paragraph in a `.py` file, it belongs in `NOTES.md` under a new ID.**

### Quick reference

> **THIS IS THE BLACKHOLE p150 FORK.** Everything below that carries a number was measured on a
> Wormhole N150 unless it says otherwise; `STATUS.md §6.39+` is the p150 re-derivation. The port
> runs on Blackhole with zero source changes, but **the tuned constants do not transfer** — see
> §1's hardware note and §6.39/§6.40 for two that have already reversed.

```bash
cd /localdev/lserbedzija/repos/tt-metal
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD          # all three — see §2, $PWD alone is WRONG

V=models/experimental/voxtral_tts
python -m pytest $V/tests/ -q --noconftest                 # 132 tests, ~70 s   (--noconftest REQUIRED)
python $V/tests/tt_gates.py --gate codes                   # blocks 1+2, integer codes
python $V/scripts/generate_quality_set.py --tag mychange   # audio; NOTE: writes results{tag}.json
python $V/scripts/score_quality_set.py $V/generated/resultsmychange.json
```

Current on the **p150**: **26.9 ms/frame on the long-form cases, RTF 0.357**. Long-form WER
**0 wrong of 894**, MOS long-form 4.61. Beat that without breaking it.

> **Quote ms/frame, not RTF, when comparing builds.** ms/frame is repeatable to 0.390 ms; RTF also
> carries prefill, the codec and the trace capture, which amortise differently as frame counts
> change, so two runs of IDENTICAL code have read 0.4559 and 0.4415. `--tier audio`'s
> `ms_per_frame` is the gate for that reason (§6.63).

> **These exclude case 0, and any RTF you quote from this harness must too.** It is the first
> utterance in each process and pays one-time program-cache compilation — 3.3 s over 5.4 s of
> audio, RTF 1.346. Leaving it in reads as 0.759 / 0.694 and will not reconcile with anything
> (§6.52).

---

## 1. What this model is

Text + a voice preset in, 24 kHz audio out. Three stages per utterance:

| block | what it does | size | file | cost/frame |
|---|---|---|---|---|
| **Block 1** | autoregressive backbone. Prefills the prompt, then emits one hidden state per audio frame | 3.4B, 26 layers, DIM 3072 | `tt/ttnn_voxtral_gpt.py` | ~15.9 ms |
| **Block 2** | flow-matching acoustic transformer. Hidden state → 36 acoustic codes, by solving an ODE in 7 Euler steps over 3 layers | 390M | `tt/ttnn_voxtral_flow.py` | ~14.2 ms |
| **Codec** | codes → waveform. Once per utterance, not per frame | | `tt/ttnn_voxtral_codec.py` | ~3.5 ms total |

One frame is **80 ms of audio**, so real-time is 80 ms/frame and we are at ~26.9, RTF ~0.36.

`tt/ttnn_voxtral_pipeline.py` wires the three together. `reference/` is a pure-fp32 PyTorch
implementation — **it is the ground truth, not the device.**

**Hardware: one Blackhole p150b.** 130 Tensix cores (13×10), 8 GDDR6 banks, measured DRAM ceiling
**367 GB/s** (§6.41/§6.53) and a **~68 µs per-op floor** (§6.45). Those two numbers together are the
single most useful fact on this fork, and they invert the N150's: bytes are cheap and launches are
expensive, so **deleting ops wins where the N150 wanted fewer, bigger kernels**. Seven N150 decisions
have already reversed here — §6.39, §6.40, §6.43, §6.44, two in §6.45, and §6.52. **One of those
has since reversed BACK**: §6.72 restores the hand-rolled head split §6.45 removed, because §6.65
traced away the per-op launch cost the reversal was made on.
**But the chip's real limit is §6.53: this workload uses 0.37% of its compute and ~49% of its
DRAM, so single-stream latency is nearly done and batching is the only order-of-magnitude lever.**

> **Every tuned constant here has now been re-derived on the p150, and most of the N150's did not
> survive.** Gone: both width-sharded norms (§6.39/§6.40), `_WO_PRG` (§6.43), `_V_SHARD` and the
> fused cache write (§6.44), Block 2's row fold (§6.45; its head-split half is BACK, §6.72). Changed:
> `_QKV_GRID_X` 8 → 1. Kept and re-verified: `_SDPA_PRG`'s `k_chunk_size=512` on 8×2, the only
> config exact at all 13 positions — and **a position sweep, not a gate run, is what makes an sdpa
> config safe** (§6.27, reproduced here on a different config).
>
> The code still hardcodes grid numbers rather than querying
> `device.compute_with_storage_grid_size()`, which returns **13×10** here.

---

## 2. Setup — the trap that will cost you an hour

**`PYTHONPATH=$PWD` IS NOT ENOUGH, and this section used to say it was.** The `ttnn` package root
is `$PWD/ttnn/ttnn`, so `$PWD` alone resolves `ttnn` to the *outer* directory — a namespace package
with no `__init__.py` — and you get:

- `AttributeError: module 'ttnn' has no attribute 'L1_MEMORY_CONFIG'`, or, once a real build
  exists, `ttnn.__file__ is None`.

Note `pytest.importorskip("ttnn")` does **not** protect against this: the shadowing package imports
fine, it is merely empty, so `test_tt_defaults.py` errors at collection instead of skipping. The
working setting is all three entries, `tools` included (it holds the `tracy` module, without which
`import ttnn` raises `ModuleNotFoundError: No module named 'tracy'`):

```bash
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
```

**Python env, on this p150 box:** `/opt/venv` already carries torch, transformers and the Whisper
caches; no separate venv is needed and `xtts_ref_venv` does not exist here. One package must be
added — `ttnn.graph` imports `graphviz` unconditionally (`uv pip install graphviz`). **Do NOT
install `torchaudio`**: the wheel's ABI is broken against this torch, and merely having it importable
breaks `transformers` too, which takes `scripts/score_quality_set.py` down with it — that script
needs torchaudio only to resample 24 kHz to 16 kHz, so scipy's `resample_poly` is the way round it.

**A fresh clone needs a build**, and `git submodule update --init --recursive` first: `./build_metal.sh
--release --enable-ccache` is ~13 min on 8 cores here and picks the clang-20 toolchain by itself.

---

## 3. How to run things

### The one command — run this around every change

```bash
Q=models/experimental/voxtral_tts/scripts/quality_report.py
#   THE BEFORE-TAG ALREADY EXISTS: `quality_head_5641f04`, a clean audio tier on HEAD
#   (§6.72). Compare against it rather than re-running 18 minutes. Older tags predate the
#   current head split and are NOT valid baselines.
python $Q --tier fast  --tag before      # ~3.5 min   pytest + flow + codes
python $Q --tier full  --tag before      # ~20 min    + wiring, prefill26, codec, decode
python $Q --tier audio --tag before      # ~50 min    + generation, WER, artifacts, MOS
#   ... make the change ...
python $Q --tier fast  --tag after
python $Q --compare before after         # exits 1 if anything is worse beyond tolerance
```

**⚠ A TRACE MAKES EVERY LATER DEVICE ALLOCATION UNSAFE (`§6.64`).** Once `begin_trace_capture`
has run, any `from_torch(..., device=)` or op that allocates can be corrupted when the trace
executes — ttnn warns, then it hangs, then the board needs `tt-smi -r`. It happened here. All
per-frame inputs must be preallocated and written with `copy_host_to_device_tensor` first.

**⚠ A BLOCK A/B IS A SCREEN, NOT A VERDICT (`§6.63`).** Timing a block in a tight loop measures
device time with dispatch fully overlapped. The real loop syncs **10 times per frame** at host
round-trips and spends **2.8 ms/frame** drained, which can absorb a device saving completely —
`§6.62` won 2.124 ms on the blocks and **zero** on the frame. Screen with a block A/B if you like;
decide on `--tier audio`'s `ms_per_frame`, which is repeatable to 0.390 ms.

**It takes TWO TAGS on purpose.** Nothing on this fork is judged against a number recorded in
another session — `§6.15` and `§6.52` are both cases where that produced a regression that did not
exist, and the codes gate's "10/288 vs 86/288" cost a session's worth of doubt for exactly this
reason. Run the tier on the base commit, change something, run it again, compare the two.

Tolerances are the branch's own measured noise floors (`§6.15` decode spread, `§6.52` timing floor,
`§6.7` short-bucket WER). **A metric that fails to parse is reported as `null` and exits 2** — a
gate whose output drifts breaks loudly instead of quietly reporting success.

`--tier audio` needs the MOS venv once: `bash tests/probes/mos_setup.sh`. It installs DistillMOS
into `/tmp/mosvenv` — **never the main venv**, because it pulls `torchaudio`, which §2 records as
breaking `transformers` and taking the WER scorer with it. Without it, MOS is skipped and says so.

The sections below are the same checks run by hand, for when you want one of them in isolation.

### Tests — 132 tests, ~70 s

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
#   ^ prints TWO blocks: synthetic (a pessimistic proxy, reads ~30%) and REAL PROMPTS
#     (~4%, and 100% of those off by one FSQ level of 21). Quote the real-prompt one.
#     Both print n/288 and they are NOT comparable -- STATUS.md 6.54.
python $G --gate decode --cases 0,2 --verbose    # quick subset while iterating
```

`--gate decode` runs all 15 prompts **by default, deliberately.** A 2-prompt default previously let a
regression through: per-case mean ranges ~0.45 pp, so an aggregate over a *different* prompt set is
not comparable to a recorded one. Only ever compare a paired A/B — same cases, same session, one
change.

### End-to-end audio + WER — ~20 min for all 15 prompts

```bash
python .../scripts/generate_quality_set.py --tag mychange
python .../scripts/score_quality_set_scipy.py .../generated/resultsmychange.json
#   ^ the _scipy one. The original needs torchaudio, which cannot be installed here (2).
```

**The tag has NO underscore.** `--tag mychange` writes `resultsmychange.json`. `generated/` holds ~40
stale `results*.json` from past sweeps; every one scores cleanly and none announces that it is the
wrong file. **Cross-check the frame counts in the scorer's output against the generation log** — that
catches a stale file instantly (`§6.32` is the time it didn't).

WAVs land in `generated/`, gitignored — CC BY-NC derived, so they stay on the box. They are the only
way to actually *hear* the model, and no metric substitutes for that. The sampler **for this build**
is `generated/SAMPLER_shipcheck.wav` (15 clips, 140.0 s), built by `tests/probes/make_sampler.py`
from the `quality_shipcheck` run on `035983fef2`; play it with `ffplay -nodisp -autoexit`.

It has been listened to — `§6.73`. Three of its clips are the deliberately adversarial fixture
prompts and the index labels them so; `STATUS.md §3.2` records the fp32 CPU reference collapsing on
the same text, so they are a model limitation, not a port defect. **`make_ref_ab.py` renders the
fp32 side and `make_ab_sampler.py` interleaves the two, level-matched** — that pairing is what turned
`§6.73` from an impression into a measurement.

`make_sampler.py TAG` builds one from any tagged run — it only STITCHES existing wavs, so the tag
must name a generation you actually ran. `SAMPLER_p150_HEAD.wav` is the older `§6.58` one on
`1e778bc297`, which predates `§6.65` and `§6.67`; `SAMPLER_current_build.wav`, named in
`STATUS.md §3`, is the N150-era pass and is no longer on the box at all.

---

## 4. Where we are now — the numbers to beat

| | current (p150, `§6.72`) |
|---|---|
| per frame | **26.928 ms** — Block 1 ~15.9, Block 2 ~14.2, host ~2 |
| long-form RTF | **0.3567** |
| long-form WER | **0 wrong words of 894** (3 seeds, `§6.71`/`§6.72`) |
| MOS long-form / min / mean | 4.6050 / 2.6597 / 3.9972 |
| clicks / natural `[END_AUDIO]` | 52 / 30 |
| utterance length ceiling | `max_seq_len`, holding prompt + frames; 2048 ≈ 136 s. Costs DRAM, not RTF (`§6.69`) |
| listening pass | **DONE on this build** (`§6.73`) — `generated/SAMPLER_shipcheck.wav` vs `SAMPLER_FP32REF.wav`, plus the interleaved `SAMPLER_AB_hs_hand_vs_fp32.wav`. One finding, on an adversarial fixture, traced to the model. **Still not a naturalness eval** — one listener, no raters (`§3`) |

**"Long-form" means ≥100 frames — currently cases 2, 3 and 10.** Quote only those, always with the
case list. **Case 0 includes kernel compilation (RTF ~1.8) and must be excluded.** Short cases are
seed noise: the same code at seeds 0/1/2 swung that bucket 0.88–2.06%.

> This table read **48 ms/frame, RTF 0.57–0.64, WER over 298 words** until 2026-08-11 — the N150
> numbers, ~30 sections stale, and contradicting §1 of this very file. If you are editing here,
> the headline lives in THREE places (this table, the top of this file, `STATUS.md §1`) and they
> must move together.

Reproduce these before changing anything, so you know the ground you stand on.

---

## 5. The gate ladder — how to prove you didn't break it

Weakest to strongest. Every claim in STATUS names which rungs it cleared.

| rung | cost | what it establishes |
|---|---|---|
| `pytest tests/` | 70 s | structure, shapes, invariants. 132 tests |
| `--gate flow` / `--gate codec` | ~1 min | one block vs the fp32 reference |
| `--gate codes` | ~1 min | integer codes, blocks 1+2, 8 frames |
| `--gate decode` | ~4 min | Block 1 precision, 15 prompts × 22 frames |
| **frame-count A/B** | ~90 s | **bit-exactness** — see below |
| WER, ≥3 seeds | ~10 min | output quality |
| listening pass | minutes | what no metric catches — and `§6.73` is the proof: cases 4/10/11/14 are bucketed out of WER, so a human ear is the ONLY instrument covering them |

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

**Ranked by size of prize. Rewritten for the p150 — three of the N150's open items are closed
here and are in §8 instead.**

- **The 31 unused tile rows — `§6.35`, still the biggest lead, and now easier.** Block 2's decode
  uses 6 rows of every 32; 4 utterances cost 1.18× the time of 1, i.e. **3.4× throughput**. §6.38
  named the blocker as `_NORM_SHARD`'s hardcoded `(32, 96)`, which raises at 48 rows — **that
  constant no longer exists on this fork (§6.40)**, so the stated obstacle is gone and the lead
  should be re-costed. Still *throughput, not latency*: per-utterance RTF will not move.
- **Block 2's `_trunk` concat — now the largest non-matmul item in Block 2, and re-measured
  (`§6.74`).** Line 174's 3-way concat costs **165.6 µs eager / 271.4 µs traced**, i.e.
  **1.16–1.90 ms/frame** over its 7 calls — well above `§6.36`'s recorded 0.707. **Line 176's
  reshape is a GHOST**: 9.8 µs traced against a recorded 0.742 ms/frame, so half of what this item
  used to claim does not exist. The obvious 2-way rewrite is closed (`§6.30`: `p2` changes per
  frame, so the concat moves rather than vanishing). Unexplored: the attention has no RoPE and no
  mask, so token ORDER is free — see `§6.74`.
- **`wo` and Block 2's `w2` run at ~40% of the ceiling** (§6.41) — both N=3072, and §6.41 shows the
  p150 penalises *narrow* N. §6.43 proved blocking cannot reach it and the isolated gap is
  overlapped away, so this needs a different attack: §6.28's DRAM-sharded matmul was rejected
  against a 194 GB/s ceiling that no longer holds.
- **Block 2's 7 Euler steps → 5** — ~28% of Block 2, but a listening call, not a metric one.
- **bf16 semantic head** — `§6.31` candidate E, 2.079× on `semantic_code`. Never re-measured here.
  Held back because it moves the *semantic* token, so one flip redirects the rest of the utterance.
- **Tracing is worth +1.34 ms/frame (3.4%) here against the N150's 0.7%** (`§6.49`) and is NOT
  shipped — §6.26's failure-profile argument, at 5x the stakes. The probe is kept, and its
  0.003 ms spread makes it the best instrument available for small effects.
- **Block 3 has never been touched on this chip.** It is 0.4% of wall (§6.10) so the prize is
  small, but every constant in it is a Wormhole number.
- **Two upstream issues still unwritten**: the `halo_gather` out-of-range NOC write (`[pipe-02]`,
  still live in ttnn — we only stopped calling it), and, if it reproduces here,
  `nlp_create_qkv_heads`' floor.

**Unexplained, logged rather than resolved:**

- **The microbenchmark noise degraded ~40× mid-session** (§6.43) — 0.005 ms spread early, 1.9 ms
  later, with host load 0.69, no other users, AICLK steady at 800 MHz, no thermal throttle. Every
  step-level number taken after that resolves ~±0.3 ms at best. **Re-establish the noise floor
  before trusting a small effect.**
- **Why `_SDPA_PRG` survives the whole step and `_WO_PRG` did not** (§6.43/§6.46). Plausibly `wo`
  sits between two large matmuls that hide it; unverified.
- **`§6.8`'s absolute levels don't reproduce** — an N150-era open item, inherited, never chased.

---

## 8. The ledger — every optimization and every rejection

**This is the whole decision history in three tables.** Full numbers and reasoning live in the
`§` each row names; this is the map, so you can see what was tried without reading 4,600 lines
chronologically. A rejection here is worth as much as a win: it is the reason you should not
spend a day re-deriving it.

### ⚠ Three things that cost the most time to learn

**1. `activation="silu"` IS NOT FUSED, and was never fused on either chip.** It measures 98.8 µs
against a plain matmul's 85.5 — the same +14.9 as writing `ttnn.silu()` yourself, because that is
what it does. Only a program config's `fused_activation` folds it in (88.1 µs), and it is slightly
*more* accurate besides. Worth **2.42 ms/frame** across the 47 w1 calls. Both blocks use
`DECODE_PRG` now (`§6.52`, `[gpt-26]`). **If you write a new matmul with an activation, do not use
the kwarg.**

**2. AN ISOLATED SWEEP MEASURES PIPELINING; A BLOCK MEASURES DISPATCH (`§6.52`).** A tight loop of
*identical* ops pipelines and a real block of *differing* ops does not, so isolated microbenchmarks
understate op cost by ~4× — the silu op is 12.2 µs isolated and ~54 µs in-block. That is why `w2`
and `wo` posted 2.4× isolated wins and delivered **exactly 0.00 ms**, and why `§6.47`'s estimate
missed by 48×. **And a block A/B is itself only a screen (`§6.63`): `--tier audio`'s `ms_per_frame`
decides.**

**3. DECODE IS ONE TILE OF ROWS; PREFILL IS MANY.** The matmul program configs (`§6.52`),
residual-as-bias (`§6.62`) and the sharded norm (`§6.67`) are all decode-only, and **two of the
three would be SILENTLY WRONG on prefill** rather than raising.

### 8.1 Shipped — what the 26.9 ms is made of

Largest first. "block" means it was measured on the block and did not survive to the frame.

| change | worth | § |
|---|---|---|
| sharded decode RMSNorm, restored once tracing removed the reshard cost | **−5.399 ms/frame** | `§6.67` |
| decode matmul program configs, incl. real silu fusion | **−5.06 ms/frame** end-to-end | `§6.52` |
| the whole per-frame graph traced as one capture | **−4.244 ms/frame** | `§6.65` |
| sdpa for Block 2's attention interior | −2.555 ms/frame | `§6.45` |
| residual as matmul bias, both Block 1 sites | −1.918 ms/step, **0 on the frame** | `§6.62` |
| in-place elementwise, Block 1 | +0.929 ms/step | `§6.47` |
| two plain KV writes + 1-core qkv shard | +0.907 ms/step | `§6.44` |
| in-place elementwise, Block 2 (needs the L1 concat) | +0.790 ms/frame | `§6.48` |
| hand-rolled 9-op head split, restored | **−0.775 ms/frame**, bit-exact | `§6.72` |
| `_SDPA_PRG` — the one N150 config that survived | +0.197 ms/step | `§6.46` |
| `out_subblock_w` candidate list was missing 3 | inert, fixed for correctness | `§6.61` |

Inherited from the N150 branch and still shipping: the CFG batch fold into rows (2.23×), the qkv
weight fusion, `SCALE` folded into wqkv's q rows, `_trunk` projecting before it narrows, the
semantic argmax on the host, the codec's gather-based pad and its matmul output projection, BFP8
on FF+attn with w2 in bf16. See `§6.6`–`§6.31`.

### 8.2 Reversed — decisions that flipped, and why

**Nine.** Five because `§6.65` traced the frame and removed the per-op launch cost their reasoning
rested on; the rest because the chip is different. **A rejection is stale when its premise is a
cost someone has since removed** — and `§6.72` is the case where that rule was applied with the
wrong op's number and a reversal went unnoticed.

| was | is now | why it flipped | § |
|---|---|---|---|
| width-sharded decode norm (N150) | dropped, then **restored** | interleaved parallelises over ROWS and decode has one; the two reshards cost launch, which tracing removed | `§6.39`/`§6.40` → `§6.67` |
| hand-rolled head split (N150) | fused, then **hand-rolled again** | ops cost 67.7 µs, so fusing won; tracing made ops cheap and the fused op is 90.5 µs traced vs nine ops' 48.6 | `§6.31` → `§6.45` → `§6.72` |
| tuned matmul configs measure SLOWER | **shipped** | the N150 result was measured on `wq`, already at 94% of its floor | `§6.24` → `§6.52` |
| tracing rejected, "0.7% for three silent failure modes" | **shipped** | `§6.52` made the device work faster and exposed 13.3% dispatch in Block 2 | `§6.26`/`§6.49` → `§6.65` |
| residual-as-bias rejected at +0.069 ms | **shipped** at −1.918 | the add hid in a 92.7 µs matmul's shadow; `§6.52` made it 40.3 and exposed it | `§6.47` → `§6.62` |
| sdpa for Block 2 rejected, 6.48× the error | **shipped**, 1.57× | different chip; codes do not move and WER improved | `§6.37` → `§6.45` |
| fused KV write + `_V_SHARD` | **two plain writes** | the fused one is 0.687 ms/step slower here | `§6.20`/`§6.22` → `§6.44` |
| `_QKV_GRID_X = 8` | **1 core** | the grid never reaches the consumers; filling fewer cores is cheaper | `§6.19` → `§6.44` |
| `wo` needs a tuned program config | **no config** | inert on the step, and removing it is bit-exact | `§6.25` → `§6.43` |

### 8.3 Rejected — measured, and not taken

| idea | verdict | § |
|---|---|---|
| BFP4 weights, Block 2 | 8.4× the differing codes for 1.139× | `§6.17` |
| fusing w1+w3 into one 18432-wide matmul | rejected on BOTH chips for OPPOSITE reasons — 4× slower on N150, faster there but loses the free silu and pays a split here | `§6.24`, `§6.42` |
| DRAM-sharded matmul for the norm output | 1.66× slower with blocking tuned; re-opened and still rejected | `§6.28`, `§6.68` |
| `rotary_embedding_llama_fused_qk` | wrong rotation convention — ours are permuted to half-split at load | `§6.23` |
| lower math fidelity (HiFi2 / LoFi) | Block 2 slower AND 9× worse; Block 1 ~4 ms for 10–20× the code errors | `[flow-03]` |
| fp32 KV cache | `sdpa_decode` rejects the dtype; hand-rolling around it is **44.7× slower** | `§6.57` |
| higher-precision prefill | fp32 weights buy nothing; fp32 activations work and the gain is gone by decode step 1 | `§6.55`, `§6.56` |
| bf16 weights through decode | +29% for no measurable accuracy, non-monotonically | `§6.57` |
| moving the three host steps on device | device is 7–29× slower; the whole host tail is 82 µs | `§6.50` |
| project-then-duplicate in `_solve` | 0.785× — moving duplication downstream widens the tensor 48× | `§6.34` |
| eliminating CFG | costs only **1.8%** — "CFG doubles the work" does not hold | `§6.35` |
| 2 command queues | a vision-only idiom here; our per-step host input is 6 KB | `§6.51` |
| bf16 semantic head | 2.079× and **deliberately held back** — one flip redirects the whole utterance | `§6.31` |
| fewer Euler steps (7→5) | reaches RTF 0.411 but is a MODEL change; the reference uses 7 | `§7` |
| sdpa in Block 3 | 1.44–2.27× faster, 3.3× worse worst-case error, failed 11 tests | `§4.1` |
| smaller codec slab / batched chunks / unchunked attention | all three reduce FLOPs and all three are slower | `§4` |
| MCD as a quality metric | failed its own self-test by ~10×; **no MCD number is reported** | `§6.59` |

### 8.4 Corrections — claims this project made and later disproved

The most useful rows in the whole file, because each was believed and acted on.

| claim | correction | § |
|---|---|---|
| `activation="silu"` fuses | it never did, on either chip | `§6.52` |
| "the fused head split costs 6.2 µs traced" | that is Block 1's op; Block 2's is **90.5** | `§6.72` |
| the codes gate's 29.5% is a real accuracy number | synthetic-input artefact; real prompts read 3.9%, 100% off-by-one | `§6.54` |
| prefill needs chunking past ~1024 tokens | never measured, never true; clean to 4096 | `§6.69` |
| "duration agrees within 2 frames on symbol text" | one draw each; with 3 seeds the ranges are disjoint | `§6.73` |
| the norm's core count has an interior minimum at 32 | true on N150, monotone on Blackhole | `§6.18` → `§6.39` |
| a single-seed WER comparison can rank two builds | the same code spans 0.88–2.06% across seeds | `§6.7` |
| worst-sample MAX is a usable statistic | unstable order statistic; use mean and p90 | `§6.8` |

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
