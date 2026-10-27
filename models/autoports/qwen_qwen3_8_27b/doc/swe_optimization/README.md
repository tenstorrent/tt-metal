# Qwen3.8 SWE optimization — 27 October 2026

Status: investigation active; no accepted optimization or SWE100/500 result yet.

Target: useful SWE-bench Verified throughput on one QB2 (four Blackhole chips),
with correctness and small scored samples established before expansion to 100
and 500 tasks. The user authorizes local device tests and QB2 dispatch CI,
including changes to the autoport, vLLM integration, TTI, and sampler experiments.

Source baseline: Metal `ed3804e8afd3475a1fb74054a346d662cc8b4844`,
autoport `models/autoports/qwen_qwen3_8_27b`, all 64 layers, native 262144 context.
The separate `models/demos/qwen38_27b_qb2` is not the implementation under test.
Worktree: `/home/mvasiljev/build/tt-metal-qwen38-swe`, branch
`mvasiljevic/qwen38-swe-opt`; the existing Granite checkout is preserved.

## References and existing evidence

- [Experiment 26 snapshot](https://github.com/tenstorrent/agentic-research/tree/mvasiljevic/experiment-26-qwen38-autoport/experiment-26),
  research commit `817b96e7`.
- [Model card](https://huggingface.co/Qwen/Qwen3.8-27B), pinned checkpoint
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: SWE-bench **Pro 61.7**,
  not a reference for Verified. No matched Verified GPU reference has been found.
  The TTI provisional Pro-as-Verified reference is removed on the experiment ref.
- Historical B32/C8 SWE8 repeats
  [40395063128](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40395063128)
  and [40395139503](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40395139503):
  4/8 and 5/8 raw verifier successes, 2/8 and 5/8 normal endings, respectively;
  1.769823/1.768001 evaluation hours. Nine of sixteen attempts hit the agent
  deadline. Prior trajectory reviews disclose upstream-patch exposure in some
  successes. These are contextual measured controls, not accepted accuracy parity.
- Archived 32-task SWE score 0/32 used a 900-second outer agent deadline;
  every task timed out. Do not interpret it as an uncensored model score.
- Gemma experiment 29 supplies the measurement method: hold task IDs, harness,
  verifiers, sampling and deadlines fixed; report scored outcomes, normal endings,
  malformed/capped responses, generated work, cache reuse, startup and tails.
  A cached-token fraction or sampler replay gain is not a SWE score improvement.

## First scored candidate

Hypothesis: fixed B32 decode wastes work at C8; B8 should lower decode latency
and allow more of the same tasks to complete, preserving model/policy/context.
This changes resident capacity, including its derived sampling CCL default;
it is a whole serving configuration comparison, not isolated padding evidence.

TTI `6264049eb089ac94988c7279110420c859b9035d`, published experiment branch
`mvasiljevic/qwen38-swe-b8-20261027`. This retains the immutable image
`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:182a91996c8104cb1373503bf9326deb5d98e01347655bb1305840165262950e`
and plugin input `1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf` (image includes
the archived autoport deployment patch). No model source overlay is applied.

Tasks: astropy 13398, 13453, 14309, 7166; scikit-learn 13135; sphinx 8459;
sympy 18698, 21612. Frozen 500-task registry, mini-swe-agent 2.2.8,
seed1234, temperature1/top-p.95/top-k20, 16384 output, 6300-second agent,
3600-second request, 7200-second suite, eight clients, prefix caching,
shared recurrent snapshots and 256-token checkpoint geometry are preserved.
Server resident capacity changes 32→8. No claim of supported-context reduction.

Forecast before collection: 1–2.4 CI hours, depending on task tails; prior B8
samples used shorter deadlines and cannot guarantee completion at this protocol.
Do not extrapolate a full-suite go from censored eight-task wall time.

First dispatch 40516088399 failed during source checkout because GitHub checkout
does not resolve abbreviated commit `6264049e`. No model ran. Corrected dispatch
[40516531644](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40516531644)
uses the published experiment branch, whose immutable HEAD is recorded above.
Collect the resolved TTI commit, runtime spec and all outcomes before comparison.

Host validation: 11 existing audit/mount tests passed; importing the actual TTI
evaluation registry confirmed only SWE, C8, 6300 seconds and the intended policy.

## Sampler experiment

User-requested comparison: current Sampling1D versus TTSampling. The standalone
`tests/compare_samplers.py` uses full Qwen vocabulary 248320 split over TP4,
identical BF16 input logits, explicit per-slot seeds, k20/p.95/temp1, matching
power-of-two padding and collective configuration. It records five warmed
traced replay windows, 32 matched stochastic draws, and six global vocabulary
boundary maxima. Batch8/32 and baseline/async gathers are separate strata.
No model weights or hidden dimensions are reduced in the sampler operation.
This is a sampler microbenchmark, not a model quality or SWE measurement.

Local runtime: owned container `qwen38-swe-local`, Ubuntu24 dev image
`8767a9a60bed`, compiled TTNN from the retained experiment checkout
`/data/mvasiljev/tt-metal-exp26-20261023-0822` at `3fbf3dbf1e88`,
Python packages from the existing host Metal venv, model/sampler Python source
from this worktree. Record this difference from the exact CI image.
Added container-local libnsl2; original Ubuntu22 image could not load the newer
binary's GLIBCXX requirement. Four local devices are discoverable.

Probe development failures are retained outside the repository under
`/home/mvasiljev/build/qwen38-swe/`: initial parameter vector length32 failed
the B8 sampling contract; corrected to B8. A missing plus_one warmup failed
trace capture; warmup now includes every captured operation. Neither failure
is a sampler performance measurement.

### Measured sampler decision

Final common-source repeats (five 200-replay windows per row):

| Logical batch / collective | Sampling1D µs | TTSampling µs | Result |
|---|---:|---:|---|
| 8 / existing gather | 474.435 | 557.730 | TTSampling 17.56% slower; pads to32 |
| 8 / existing gather, compact TT offsets/masks | 474.435 | 521.506 | TTSampling 9.92% slower; actual8 rows |
| 32 / async gather | 463.940 | 533.531 | TTSampling 15.00% slower |

All three comparisons have matching source and input hashes. The256/256 active
draws for each B8 candidate and1024/1024 B32 draws exactly match Sampling1D.
Every row passes six unique-maximum vocabulary-boundary cases on every active
slot. The compact variant stages TTSampling's offsets and greedy mask for B8
before capture, avoiding its stock fixed32-row offset broadcast failure. No
sampler replacement is selected. These are traced sampler-only host wall times;
they establish neither model-task accuracy nor a SWE runtime gain. Independent
process repeats of the stock rows agree closely (see local raw artifacts).
The retained final JSON and comparison are in [samplers](samplers/comparison.json).

The earlier failed capture left its process alive in teardown; a subsequent
probe waited on its device lock. Evidence was retained, triage attempted (installed
0.3.32 versus required tt-exalens0.4.1), and a version-bypassed diagnostic saved.
That mismatched diagnostic's core warnings are not a model defect diagnosis.
Stopped/restarted only the owned Qwen container, reset the four local devices once
with tt-smi, and verified all four were discoverable. All final sampler processes
then opened, ran and closed the mesh successfully. Capture cleanup now ends the
capture even when a captured operation raises.

## Separate reasoning policy pilot

The model card and pinned chat template support medium reasoning. Its instruction
differs from default xhigh, so this is explicitly a policy experiment with its
own measured score, not a kernel speedup. Thinking remains enabled and preserved
by the checkpoint defaults. Same eight tasks, B8/C8, generation sampling and
all deadlines as the first candidate; output budget remains16384.

TTI `a6258f1645e649c82a76e8a2cf63b7f9ed6d0617`, experiment branch
`mvasiljevic/qwen38-swe-b8-medium-20261027`,
[CI40518335859](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40518335859).
Its optional SWE-only control forwards reasoning_effort via chat_template_kwargs;
public validated policy selectors are recorded by the audit middleware without
logging arbitrary template values. Twenty host tests plus four subtests pass,
including actual catalog task/deadline/policy preservation and request observation.
Prediction: potentially less per-turn reasoning; overall task latency and solve
rate may worsen, as the model card cautions. Retain the same1–2.4 CI-hour planning
window until actual completion, output work, caps, failures and verifier scores
are collected. No default-policy promotion or full-suite decision yet.

Raw run artifacts and local logs stay under `/home/mvasiljev/build/qwen38-swe/`.
Copy only compact evidence and provenance into this report after validation.
