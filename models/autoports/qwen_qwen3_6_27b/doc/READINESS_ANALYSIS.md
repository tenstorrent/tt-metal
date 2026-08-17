# What blocks this model, how far it is, and what has never been checked

Operator analysis, 2026-08-17. Written from committed artifacts on this branch.

## Status: stages 04–10 complete, stage 11 blocked on infrastructure

Everything through optimized-vLLM passed its gate. Stage 11 ran the release
workflow repeatedly with the external-server topology, served at
`max_model_len=262144`, completed the ten-point benchmark sweep, ran
single-sample IFEval and GPQA, passed a non-aligned 32,768-token request, and
made TTI-side repairs. It blocked on **`terminal_bench_2`**, which runs through
Harbor and executes its tasks **in Docker**; this container has neither a docker
CLI nor `/var/run/docker.sock`.

That is a real environment limit, not a model defect. Unblocking needs a docker
socket in the container, SSH to the physical loudbox host, or an explicit
decision to waive the agentic eval. Worth knowing: the sequence-length audit
records this model scoring **0.0 on both `terminal_bench_2` and
`swe_bench_verified`** in its earlier Exp19 TTI run, so the eval may well run and
score zero.

## Is the PCC good? Yes, including at long prefill

| gate | value |
|---|---:|
| multichip prefill→decode PCC at **S32769**, BF16 | **0.99999997** |
| official-weight layer PCC, full / linear | 0.999741 / 0.999906 |
| AIME24 prefill top-1 / top-5 / top-100 | 92 % / 100 % / 100 % |
| AIME24 teacher-forcing top-1 / top-5 / top-100 | **97 %** / 100 % / 100 % |
| replayed decode, cache-reset identical | PCC 1.0 every step |

Unlike the other two ports, this one has a genuine **long-prefill** correctness
number: PCC 0.99999997 on a full S32769 prefill→decode. That covers the prefill
path at length. It does **not** cover long generation — see below.

## What has never been checked: long generation

| what | length |
|---|---:|
| longest correctness-measured generation | **~100–128 tokens** (AIME24 100-token window; 128 measured trace replays) |
| prefill correctness | up to **S32769** ✓ (and capacity probes to 262,143) |
| advertised context | 262,144 at batch 1; **82,432 at batch 32** (measured, C=82,496 fails) |

So the asymmetry on this branch is the reverse of the usual one: prefill is
verified at length, generation is not. No artifact here demonstrates correctness
after thousands of consecutive self-fed decode steps.

**Why that matters concretely:** the sibling Gemma-4-26B port on
`mvasiljevic/fmf/google-gemma-4-26b-a4b-it` has healthy layer PCC and passes at
100- and 1,280-token generations, then **fails `meta_gpqa_cot` 4/10 against an HF
control's 10/10 at a 32,768-token generation budget**. Nothing on this branch
would have caught that class of defect either, because its longest
quality-measured generation is also ~100 tokens. If stage 11 is unblocked and the
mandatory evals run, `meta_gpqa_cot` is the row to watch first, and a failure
there should not be read as a reference problem before the long-generation path
is checked.

## Other open items on this branch, in priority order

1. **TTFT is this model's weak axis and remains unaddressed.** 3,784 ms served
   after stage 10 (down from 4,139), against Falcon3-7B's 183 ms. Stage 07 left
   it untouched and stage 08's datatype selection made it **worse by 502 ms**
   while ranking on a non-serving metric — see `doc/datatype_sweep/TTFT_SELECTION.md`,
   which names two already-measured configs that return ~500 ms at equal or
   better accuracy.
2. **Batch-32 serving is not viable** and the numbers say so plainly: TTFT P50
   ~162.6 s and 17.05 tok/s aggregate at `max_num_seqs=32`. The linear mixer
   costs 9.55× more per layer at batch 32 than batch 1. TTI's own prod spec pins
   this model to `max_concurrency: 1`, which looks like necessity rather than
   caution.
3. **The full-norm hidden-sharded boundary was never measured** across stages 05
   and 07 — see `doc/optimized_multichip_decoder/BOUNDARY_CANDIDATE.md`. Two
   64-layer models shipped it; six models have now rejected only the weaker
   fractured variant.

## How far from "fully working"

Functionally complete and serving at full context. The blocker is environmental
and one decision away. The substantive engineering debt is TTFT and the
untested long-generation path — the second of which is now known to be where a
sibling port actually broke.

---

## CORRECTION: the Docker blocker is scoped to the agent, not to this machine

I recorded stage 11 as a "genuine infrastructure blocker". That is right for the
stage agent and wrong as a statement about what is possible here. Tested
2026-08-17:

| test | result |
|---|---|
| Docker server reachable from the host | **yes**, 29.2.1 |
| `mvasiljevic-ttxla` network | bridge, IP **172.17.0.2**, no published ports |
| host -> container port 8000 (the vLLM port) | **HTTP 200** |
| sibling container with `/var/run/docker.sock` mounted can drive Docker | **yes** (listed running containers) |

So the accurate statement is: **the stage agent runs inside
`mvasiljevic-ttxla`, which has no docker CLI and no `/var/run/docker.sock`, so
from inside that container Terminal-Bench genuinely cannot run.** Its diagnosis
was correct for its own scope. But the operator is not confined there, and
`terminal_bench_2` is satisfiable on this machine without any new hardware or
access:

1. Serve from `mvasiljevic-ttxla` exactly as stages 09–11 already did (the TT
   devices stay where they are).
2. Run the Harbor / Terminal-Bench client **outside** that container — either on
   the host or in a sibling container started with
   `-v /var/run/docker.sock:/var/run/docker.sock`, which lets Terminal-Bench
   spawn its own task containers.
3. Point `api_base` at **`http://172.17.0.2:8000`** instead of loopback. This is
   precisely what the stage's own AutoFix plan asked for — "changing only
   `api_base` from loopback to the verified host-reachable address. Harbor is
   client-side and must not mount or access TT devices" — and the address is now
   verified reachable.

Setup cost is real, not zero: TTI's `.workflow_venvs` live inside
`mvasiljevic-ttxla`, so a sibling container needs either a bind-mount of
`/home/mvasiljevic/tt-inference-server` (visible on the host at that same path)
or its own install, plus the HF token. That is work, but it is work rather than a
wall.

**What this does not change:** the audit records this model scoring **0.0 on both
`terminal_bench_2` and `swe_bench_verified`** in its earlier Exp19 TTI run, so the
eval may well run and score zero. The value of unblocking is a real number in
place of an unknown, not a likely pass.

---

## CORRECTION 2: stage 11 got much further than recorded, and the accuracy question is open

Investigated 2026-08-17 by reading the committed stage-11 artifacts. Two things
above need revising: how far this stage got, and — more importantly — my framing
of this port as "functionally complete, blocker is environmental".

### How far it actually got

Not "blocked with the release unstarted". It completed almost the whole workflow:

- **Benchmark sweep across long inputs**, committed under
  `doc/tti_release/release_cache/.../llm/`: ISL **128, 1024, 2048, 4096, 8192,
  16384** with OSL 128 and 1024.
- **Both standard evals executed**, with per-sample records retained:
  `results_2026-08-15T09-56-41.json` + `samples_ifeval_*.jsonl` and
  `results_2026-08-15T10-01-21.json` + `samples_gpqa_diamond_cot_zeroshot_*.jsonl`
  under `release_final3_cache/`.
- **It reached the agentic stage**: `release_final4_cache/.../agentic/
  eval_Qwen__Qwen3.6-27B/terminal_bench_2_harbor_config.json` was generated
  before Docker stopped it, and the Terminal-Bench token budget was repaired
  (176K input + 80K output = 262,144 exactly, 35 tests passing).

So the block is at the last of six required evals, not at the start.

### The scores, and why they are not yet a verdict

| eval, 5 % subset | result |
|---|---:|
| `ifeval` prompt-level strict | **17.86 %** (5/28) |
| `ifeval` inst-level strict | 34.88 % |
| `gpqa_diamond_cot_zeroshot` flexible-extract | **0.3 → 3/10** |
| `gpqa_diamond_cot_zeroshot` strict-match | 0.0 |

**No HF control was ever built on this branch** — zero `hf_reference*` artifacts,
where the Falcon3-7B and Gemma-4-26B ports both built paired CPU controls. So
these are uncontrolled numbers, and the honest status of this port's accuracy is
**unassessed**, not acceptable.

Two comparisons that make the direction hard to ignore:

- On the **identical task and the identical ten documents**, the sibling
  Gemma-4-26B port scored **4/10 and that was graded a FAIL** against an HF
  control's 10/10. This port scored **3/10**.
- `ifeval` prompt-level strict of 17.86 % is base-model territory. The
  instruction-tuned Gemma port scored **82.62 %** on the same metric against an
  HF control of 87.04 %.

I am deliberately not calling this a failure: different models legitimately score
differently, and without a same-command reference there is nothing to divide by.
That is precisely the defect — the stage blocked on the agentic eval before doing
the reference work that would have settled it.

### Revised priority for finishing this port

1. **Build the paired HF control first.** `ifeval` and
   `gpqa_diamond_cot_zeroshot` on the same 5 % subsets, same snapshot, chat
   template, seed 42, deterministic generation. CPU is sufficient — on the Gemma
   port the equivalents took about 32 min (IFEval) and 50 min (GPQA). **This
   needs no Docker at all** and converts two floating numbers into pass/fail.
2. **Verify the chat template was actually applied.** A 17.86 % prompt-strict
   IFEval is what a base model *or* a template-less path produces. The shared
   readiness runner only switched its qualitative check to
   `/v1/chat/completions` during the Gemma run, and an operator attempt on Gemma
   using the raw `/v1/completions` backend returned HTTP 400 for exactly this
   reason. If this port's evals ran through raw completions, the number is an
   artifact rather than a result.
3. **Then the agentic evals** — `terminal_bench_2` and `swe_bench_verified`, via
   the sibling-container route in Correction 1 above.

### What this means for the earlier framing in this document

The section above describing this port as functionally complete with the
substantive debt being TTFT and the untested long-generation path understates the
position. Long generation is not merely untested here: the two evals that
exercise it have already produced low uncontrolled scores, and the sibling port
that *did* build a control failed on the same task at a higher score than this
one achieved. Treat accuracy as the open question and TTFT as second.

---

## RESOLVED: both low eval scores are generation-budget truncation, not model quality

Investigated 2026-08-17 by reading the per-sample outputs this run *did* retain
(`release_final3_cache/.../samples_*.jsonl`). The scores recorded above are
harness artifacts. **This port's accuracy remains unmeasured — but it is not
established as bad, and the numbers above must not be quoted as quality.**

### The mechanism, from the task definitions

| task | `max_gen_toks` in the lm-eval task YAML | effective budget here |
|---|---|---|
| `ifeval` | **1280** | 1280 |
| `gpqa_diamond_cot_zeroshot` | **not set** | **256** (lm-eval API backend default) |

The stage passed `gen_kwargs = {'stream': False, 'seed': 42}` with no override, so
GPQA inherited a 256-token cap. Chain-of-thought GPQA cannot finish in 256 tokens.

### What the retained samples show — GPQA (20 rows = 10 docs x 2 filters)

| measure | value |
|---|---|
| raw response length | min 93, **median 120**, max 142 words |
| ends with terminal punctuation | **0 / 20** |
| contains `boxed` (the string the extractor needs) | **0 / 20** |
| `filtered_resps == "[invalid]"` | **16 / 20** |

Tails are severed mid-formula, e.g. `"$\Gamma_2 \approx \frac{\hbar}{"` and
`"1. 1,"`. The model never reached an answer, the extractor therefore returned
`[invalid]`, and that produced the 0.3. **The 3/10 measures the cap, not the
model.**

### And IFEval — 28 rows

Median 759 words (≈1000 tokens) against its 1280 budget, but only **9 of 28 end
with terminal punctuation**: 19 responses are cut mid-sentence, e.g.
`"...He successfully"`. Unfinished responses fail instruction-following format
checks, which is a sufficient explanation for 17.86 % prompt-level strict without
invoking any model defect.

### What to run, corrected

1. **GPQA with an explicit budget** — this is the decisive test. Gemma's stage used
   `max_gen_toks=32768` for the same task, having diagnosed this exact issue
   ("lm-eval API backend's implicit 256-token default truncated GPQA reasoning").
2. **IFEval above 1280** — 4096 or higher. Re-running *at* 1280 changes nothing,
   because 1280 is what it already had; that was an operator error worth recording.
3. Only then, if scores are still low, build the paired HF control. Building a
   control against truncated generations would have measured the cap on both
   sides.

### Why this does not transfer to the sibling Gemma port

Gemma's GPQA ran at `max_gen_toks=32768`, so its 4/10 against an HF control's
10/10 is **not** this defect. Its own open questions are different and are
recorded on that branch: `strict-match` scoring 0.0 in *both* arms (so the strict
extractor never matches even a known-good reference), `log_samples` unset in both
arms (so its failures cannot be inspected at all), and `until: ['</s>']` being a
Llama/Mistral stop string that can never fire on a model whose stops are `<eos>`
and ids `[1, 106, 50]`.

### The general lesson for this pipeline

Three of the four "quality failures" seen across these three ports were harness
defects, not model defects: Falcon's IFEval graded against the wrong metric
variant of a model card (apparent 0.544, actually parity), and now both of this
port's scores. Establish that the harness let the model finish and that the
extractor can parse its output **before** attributing a low score to the port.
