# Voxtral-TTS on Tenstorrent — start here

You are picking this up with no context. Read this file top to bottom once; it should take ten
minutes and leave you able to run everything and change something safely.

Three files matter and they have different jobs:

| file | what it is | when to read it |
|---|---|---|
| **`ONBOARDING.md`** (this) | how to run things, how to not break things, the method | first, once |
| **`STATUS.md`** | the running log. Numbered findings `§6.1 … §6.33`, every experiment with its numbers, including the rejected ones | when you want to know whether something has already been tried |
| **`tt/NOTES.md`** | the prose that used to live in the code. Grep-able IDs `[gpt-04]`, `[flow-10]`, `[codec-12]`, `[pipe-02]` | when a line of code has a `NOTES.md [id]` pointer and you want the why |

The `tt/*.py` files are kept deliberately thin — one-line pointers, no essays. If you find yourself
writing a paragraph in a `.py` file, it belongs in `NOTES.md` under a new ID.

---

## 1. What this model is

Text + a voice preset in, 24 kHz audio out. Three stages run per utterance:

| block | what it does | size | file | cost/frame |
|---|---|---|---|---|
| **Block 1** | autoregressive backbone. Prefills the prompt, then emits one hidden state per audio frame | 3.4B, 26 layers, DIM 3072 | `tt/ttnn_voxtral_gpt.py` | ~23 ms |
| **Block 2** | flow-matching acoustic transformer. Turns that hidden state into 36 acoustic codes, by solving an ODE in 7 Euler steps over 3 layers | 390M | `tt/ttnn_voxtral_flow.py` | ~20.8 ms |
| **Codec** | decodes the codes to a waveform. Runs once per utterance, not per frame | | `tt/ttnn_voxtral_codec.py` | ~3.5 ms total |

One "frame" is **80 ms of audio**. So real-time is 80 ms/frame; we are at ~48 ms, i.e. RTF ~0.6.

`tt/ttnn_voxtral_pipeline.py` wires the three together. `reference/` holds a pure-fp32 PyTorch
implementation that everything is measured against — **it is the ground truth, not the device.**

Hardware: one **Wormhole N150**. 64 Tensix cores in an 8×8 grid, 12 DRAM banks, measured DRAM
ceiling **194–202 GB/s**. That number is the single most useful fact in this project (see §5).

---

## 2. Setup — and the trap that will cost you an hour

**There are two virtualenvs and the obvious one is wrong.**

```bash
cd /localdev/lserbedzija/repos/tt-metal
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate    # <-- THIS one
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD                        # from the repo root
```

The repo's own `python_env/` is a **docs** environment. It has no `torch` and no `ttnn`. If you use
it you get `ModuleNotFoundError: No module named 'torch'`, or worse, `AttributeError: module 'ttnn'
has no attribute 'L1_MEMORY_CONFIG'` — that second one means `PYTHONPATH` found the repo's `ttnn/`
*source directory* as an empty namespace package instead of the built module.

`PYTHONPATH=$PWD` is required for `models.*` imports and must be set from the repo root.

---

## 3. How to run things

### Tests (122 tests, ~45 s)

```bash
python -m pytest models/experimental/voxtral_tts/tests/ -q --noconftest
```

`--noconftest` is **required**: the repo's `conftest.py` imports `torchvision`, which is not
installed. Without the flag every test errors at collection.

### Gates (device, seconds to minutes)

`tests/tt_gates.py` is the gate harness. Six gates, each answering a different question:

```bash
G=models/experimental/voxtral_tts/tests/tt_gates.py
python $G --gate wiring      # does anything run at all
python $G --gate prefill26   # Block 1 prefill vs fp32, 15 prompts
python $G --gate decode      # Block 1 decode vs fp32, 15 prompts x 22 frames  (~4 min)
python $G --gate flow        # Block 2 velocity + codes vs fp32
python $G --gate codec       # codec vs fp32
python $G --gate codes       # Blocks 1+2 end to end, INTEGER codes  <- the one that predicts audio
python $G --gate decode --cases 0,2 --verbose   # quick subset while iterating
```

`--gate decode` runs all 15 prompts **by default, deliberately**. A 2-prompt default is what
previously let a regression through: per-case mean ranges ~0.45 pp, so an aggregate over a
different prompt set is not comparable to a recorded one. Only ever compare a paired A/B: same
cases, same session, one change.

### End-to-end audio + WER (~20 min for all 15 prompts)

```bash
python models/experimental/voxtral_tts/scripts/generate_quality_set.py --tag mychange
python models/experimental/voxtral_tts/scripts/score_quality_set.py \
  models/experimental/voxtral_tts/generated/resultsmychange.json
```

**The tag has NO underscore.** `--tag mychange` writes `resultsmychange.json`, not
`results_mychange.json`. `generated/` holds ~40 old `results*.json` files from past sweeps; every
one of them scores cleanly and none of them will tell you it is the wrong file. **Always check the
frame counts in the scorer's output against the generation log** — that single cross-check catches a
stale file instantly. See §6.32 for the time this was got wrong.

WAVs land in `generated/` (gitignored — audio is large and CC BY-NC derived). They are the only way
to actually *hear* the model, and no metric substitutes for that.

---

## 4. Where we are now — the numbers to beat

| | current |
|---|---|
| long-form RTF | **0.57–0.64** |
| per frame | **~48 ms** (Block 1 ~23, Block 2 ~20.8, host 0.2) |
| long-form WER | **0 wrong words of 298** |
| Block 1 worst-sample mean / p90 | 0.92% / 1.28%, min PCC 0.999040 |
| Block 2 velocity PCC | 0.99998480 |
| `[END_AUDIO]` natural termination | 15/15 cases |

**"Long-form" means cases with ≥100 frames — currently 2, 3 and 10.** Quote only those, and always
with the case list. Case 0 includes kernel compilation (RTF ~1.8) and must be excluded. Short cases
are seed noise: the same code with seeds 0/1/2 swung that bucket by 0.88–2.06%.

Reproducing these is your baseline check before you change anything.

---

## 5. How to find an optimization

### The floor method

Every op's time splits into two parts:

```
time  =  bytes it must move / 194 GB/s   +   overhead
         \_______ irreducible _______/       \__ the only part you can win __/
```

So: measure an op in isolation, compute its floor from the bytes it genuinely has to touch, and the
difference is your budget. Rank candidates by `overhead × times per frame`. `§6.27` is Block 1's
map, `§6.29` is Block 2's. Redo the map after any structural change.

What the maps found: **all ten weight matmuls across both blocks are already at the roofline.**
Some measure *faster* than a 194 GB/s floor, which is how we know the real ceiling is ~202. There is
no matmul work left in this model. Everything remaining is small-op overhead, and a small op on a
`[1,6,3072]` tensor costs **~20 µs just to be launched**, regardless of what arithmetic it does.

That reframes the whole problem: you are not making math faster, you are **removing work that does
not need to happen here**. The three wins in `§6.31` are all that shape — a constant being
recomputed per step, a layout conversion that could be avoided by reading the original layout, and a
reduction that was on the chip when its result was already being shipped off it.

### The rules, each of which cost real time to learn

1. **ALWAYS GATE ON REAL PROMPTS.** Random activations reported PCC 0.892 where real prompts gave
   0.9994 (trap #12). This is written down and was *still* violated in `§6.31` — a bf16 change was
   scored on 64 random Gaussian draws, which for an argmax over a vocabulary has no power at all.

2. **Isolated measurements do not decide. The whole block decides.** Five times now an isolated
   result failed to survive: `§6.18`, `§6.19`, `§6.27`, `§6.30` (1.543× isolated → *zero*
   whole-block), and `§6.33` — where the isolated number had **the wrong sign**. The effect was
   ~10 µs and the run-to-run spread of the measurement was 5.8–27.6 µs.

3. **Always report spread next to mean.** A single number with no spread is not a measurement. If
   the effect is smaller than the spread, say INCONCLUSIVE and go measure the whole block.

4. **Compare numerics against fp32/fp64 truth, not against the current default.** A variant that
   differs from the shipped path is not thereby wrong — `§6.25` nearly discarded a real 1.2× because
   it differed from the default by 5.3e-03, when against fp64 *every* config was 6.6e-04 from truth.

5. **A config that fails to BUILD has told you nothing about whether it is fast.** `§6.28` records
   writing "the rejection holds" on the strength of an assertion about a missing `memory_config`.

6. **Compare like for like on memory config.** `§6.31`/`§6.33`: a hand-rolled path was timed writing
   to DRAM against a fused op writing to L1, and read as a tie. Where q/k/v live is worth 2.5
   ms/frame downstream. Check `t.memory_config().buffer_type` in probes.

7. **Never put a `||` fallback in a gate.** One that silently substitutes a different input is worse
   than one that fails, because it returns a plausible number (`§6.32`).

8. **`git checkout <commit> -- <file>` STAGES the old version.** `git status` shows `MM`. A later
   `git commit` will silently revert your change. `git restore --staged <file>` after any A/B that
   uses it.

9. **Frame counts in a multi-case run depend on the preceding cases.** An hour went into a "moved
   from 207 to 220" that reproduced identically when the case was run alone.

### The strongest exactness gate, and it is cheap

If you are claiming a change is exact, do this rather than trusting an 8-frame code comparison:

```bash
python .../generate_quality_set.py --cases 0,2,3 --tag before   # on the old build
python .../generate_quality_set.py --cases 0,2,3 --tag after    # on the new one
# compare 'frames' per case in generated/resultsbefore.json vs resultsafter.json
```

Generation is autoregressive — each frame's output becomes the next frame's input — so **any**
divergence compounds and moves the frame at which the model decides to stop. Reproducing 461 and 487
frames exactly is real proof of bit-identity. Two cases, ~90 s. See `§6.32`.

### The order to work in

1. reproduce the baseline numbers in §4, so you know the ground you stand on
2. build or refresh the overhead map — measure, do not guess where the time is
3. pick the largest `overhead × calls` item and form a hypothesis
4. **split it before proposing anything.** `§6.30`'s repack theory and `§6.31`'s argmax finding both
   came from splitting one line item into its pieces, and the first killed my own hypothesis
5. probe candidates in isolation *only to rank them*, with numerics checked against fp32
6. measure the survivor on the whole block, interleaved A/B, several rounds
7. gate: `--gate flow` + `--gate codes` + frame-count A/B, then WER if numerics moved at all
8. commit with the numbers *and the rejected alternatives* in the message; add a `§6.x` to STATUS
9. if you had a wrong hypothesis, **record it** — the rejected list is the most valuable part of
   STATUS, because it stops the next person re-running a dead end

---

## 6. What is open

- **`nlp_create_qkv_heads` upstream issue is unwritten.** ~97 µs floor, and it was worth 1.233
  ms/frame to hand-roll around it in Block 2 (`§6.31`). Block 1's decode variant has the same shape.
- **The `halo_gather` out-of-range NOC write** also deserves an upstream issue.
- **bf16 semantic head**: `§6.31` candidate E, 2.079× on `semantic_code`, another 0.265 ms/frame.
  Held back because it moves the semantic token, which feeds Block 1's *next* input, so one flip
  redirects the whole rest of the utterance. Needs a broad real-prompt gate, not 8 frames.
- **`[flow-10]`'s 158 µs vs today's ~112–127 µs** for a hand-rolled head split is unreconciled — two
  measurements of what should be the same construction. Not chased down.
- **The 1.233 ms mechanism is not established.** The op is *not* faster in isolation, yet the block
  is. Candidates in `§6.33`; none verified.
- **`§6.8`'s absolute levels do not reproduce.** Ruled out gate code, model code, reference, ttnn
  build, prefill rows and prompt selection. Still unexplained.
- **Structural, not yet attempted**: decode uses 1 of every 32 tile rows, so there is 32× free
  capacity for concurrency or speculative decoding. And Block 2's 7 Euler steps → 5 would be ~28%
  of Block 2, but it is a listening call, not a metric one.

---

## 7. Things that are settled — do not re-run these

Full detail and numbers in STATUS; this is the index so you do not spend a day on a closed question.

| tried | verdict |
|---|---|
| BFP4 weights | 8.4× the error for 12% of the time — no |
| fusing w1+w3 into one 3072×18432 matmul | **4× slower**; matmul bandwidth collapses past N≈9216 |
| ttnn's fused q+k RoPE | wrong convention (interleaved vs our half-split); 0.236 ms to adopt |
| device tracing | +0.35 ms but three silent failure modes; and fewer launches means less to recover |
| residual-as-bias | w2's add is already free |
| DRAM-sharded matmul for the norm output | 1.66× slower even with blocking tuned (`§6.28`), three ways closed |
| one-op interleaved `rms_norm` in Block 2 | 2.4× **slower** than three sharded ops — sharding is the fast path |
| folding CFG + Euler into a weighted reduce | 1.543× isolated, **zero** whole-block, flips an FSQ boundary |
| permuting straight from `av` in the unfold | 1.77× faster and **returns garbage** |
