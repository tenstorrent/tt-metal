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
