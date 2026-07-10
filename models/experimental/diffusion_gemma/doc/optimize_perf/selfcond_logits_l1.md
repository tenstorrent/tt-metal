# Self-conditioning logits L1 chain (dg-08)

Date: 2026-07-10 UTC
Target: DiffusionGemma 26B-A4B-it, QB2 `P150x4`, TP=4, BF16, 256-token canvas
Build limitation: `ENABLE_TRACY=OFF`; synchronized component timing and warmed Metal trace replay are used.

## Result

The selected path keeps each 8K-vocabulary logits slice, its immediate
`subtract -> exp`, denominator reduction, and ordered denominator accumulator
in interleaved L1. The 32 chunk matmuls, ordered numerator accumulator, and
final divide remain in DRAM. The operation order is unchanged. No dtype, value,
chunk boundary, matmul shape, or diffusion decision changes.

`DG_SELFCOND_LOGITS_L1` now defaults to `chain`; `off` is the diagnostic
control. The required final process with the variable unset reproduced:

| Canonical run | @48 steady block | @48 throughput | Full generation | Derived warmed step |
|---|---:|---:|---:|---:|
| prior selected default | 13.6817 s | 18.711 t/s | 153.3410 s | 258.631 ms |
| final reviewed unset L1-chain default | **13.5849 s** | **18.844 t/s** | 153.9791 s | **257.575 ms** |
| delta | **-0.71%** | **+0.71%** | +0.42% regression | **-0.41%** |

This final-default reproduction is the selected headline, not the faster
explicit-candidate samples.

## Component and traced evidence

The synchronized two-layer probe isolated the intended change:

| Placement | Soft embedding | Full two-layer step |
|---|---:|---:|
| DRAM control | 18.213 ms | 73.524 ms |
| only dynamic slice in L1 (intermediate) | 17.721 ms | 72.891 ms |
| selected logits + denominator chain in L1 | **16.038 ms** | **71.359 ms** |

The selected chain reduces the component by **11.94%**. At canonical @48,
three fresh independent controls measured 13.6284, 13.6161, and 13.6051 s per
block. Two explicit chain processes measured 13.4969 and 13.5253 s. Their
medians are 13.6161 -> 13.5111 s (**-0.77%**) and 18.801 -> 18.9475 t/s
(**+0.78%**). All runs retained the established `a9f0d18709b07d1e` committed
token digest.

The first-block trace-capture cost varied enough that the independent-process
full-generation medians did not improve. No full-generation win is claimed;
the final reviewed unset-default process regressed 0.42% against the prior
selected default despite the warmed steady-block improvement.

At @12, the one explicit chain sample regressed 4.2752 -> 4.2981 s. An earlier
unset-default process measured 4.2647 s; the required fresh paired final
measured 4.3122 s / 59.366 t/s. The final @12/@48 pair implies 257.575 ms per
warmed traced denoise step.

One same-model sequential A/B also contradicted the independent-process result:
the second session moved 13.6456 -> 13.7841 s. This is retained as a real
limitation. Production constructs one fresh session, and two independent
candidate processes plus the final unset-default process all reproduced the
@48 improvement. A later 8K/default control during the rejected chunk sweep
measured 13.6321 s; after removing that selector completely, the final fresh
process first measured 13.5120 s. The final review-followup process measured
13.5849 s. The spread is reported rather than interpreted as a wiring
difference, and the latest process is the conservative headline.

Stage review found that TTNN reduction/binary output inheritance also keeps the
denominator reduction and accumulator in L1. Making those inherited memory
arguments explicit produced a materially slower final run at 13.6380 s /
18.771 t/s and was removed. The selected inherited behavior is now
source-commented, covered by a memory-aware placement test, and accurately
described above.

Machine-readable rows and the final-default policy are in
`selfcond_logits_l1_e2e.json`.

## Correctness and capability gates

- RUN-first argmax: 48/48 steps exact for argmax, sampled IDs, BF16 entropy,
  accept mask, renoised canvas, commit candidate, and final commit.
- Production `ChunkedGumbelNoise`: the same six fields and final commit are
  exact for all 48 steps with identical injected noise descriptors.
- Prompt-correct production-Gumbel A/B: three outputs and committed tensors
  exact at 16 steps.
- 256K capability: 30 layers, `max_seq_len=262144`, non-aligned 24-token
  prompt, 48 steps, one 256-token block, production chunked-Gumbel, clean exit.
- Trace-safe canvas feedback: separate watcher run attached/detached all four
  devices and matched zero error signatures.
- Proportional unit tests: 41 passed.

Artifacts:

- `selfcond_logits_l1_decisions.json`
- `selfcond_logits_l1_gumbel_decisions.json`
- `selfcond_logits_l1_gumbel_qualitative.json`
- `selfcond_logits_l1_256k_chunked.json`
- `selfcond_logits_l1_watcher.json`
- `selfcond_logits_l1_watcher_summary.json`

## Exact commands

Every hardware command was preceded by a process-ownership check. Shared
environment:

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn
export TT_METAL_HOME=/home/zni/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARCH_NAME=blackhole
export DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
```

Component control (the subsequently rejected split experiment was disabled;
this predates the L1 selector and therefore resolves to the DRAM path):

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_TRACE_REGION_SIZE DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 DG_SELFCOND_PRECHUNK_EMBED=1 DG_SELFCOND_SPLIT_LOGITS=0 python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py --num-layers 2 --iters 5
```

Selected component candidate:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_TRACE_REGION_SIZE DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 DG_SELFCOND_PRECHUNK_EMBED=1 DG_SELFCOND_LOGITS_L1=chain python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py --num-layers 2 --iters 5
```

Representative explicit candidate and required final unset-default @48:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED -u DG_SELFCOND_LOGITS_L1 DG_TRACE_REGION_SIZE=10737418240 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers selfcond_l1 --budgets 48 --blocks 3 --out /tmp/dg_selfcond_l1_chain_e2e_candidate48_repeat.json
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED -u DG_SELFCOND_LOGITS_L1 DG_TRACE_REGION_SIZE=10737418240 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers baseline --budgets 48 --blocks 3 --out /tmp/dg_selfcond_l1_postreview_inherited_final48.json
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED -u DG_SELFCOND_LOGITS_L1 DG_TRACE_REGION_SIZE=10737418240 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers baseline --budgets 12 --blocks 3 --out /tmp/dg_selfcond_l1_postreview_inherited_final12.json
```

Exact argmax and production-Gumbel decision A/B:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT -u DG_TRACE_REGION_SIZE -u DG_SELFCOND_PRECHUNK_EMBED DG_SELFCOND_LOGITS_L1=off python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --gumbel-mode argmax --out /tmp/dg_selfcond_l1_argmax_control.json
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT -u DG_TRACE_REGION_SIZE -u DG_SELFCOND_PRECHUNK_EMBED DG_SELFCOND_LOGITS_L1=chain python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --gumbel-mode argmax --out /tmp/dg_selfcond_l1_argmax_candidate_review.json
python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --compare /tmp/dg_selfcond_l1_argmax_control.json /tmp/dg_selfcond_l1_argmax_candidate_review.json --out models/experimental/diffusion_gemma/doc/optimize_perf/selfcond_logits_l1_decisions.json
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT -u DG_TRACE_REGION_SIZE -u DG_SELFCOND_PRECHUNK_EMBED DG_SELFCOND_LOGITS_L1=off python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --gumbel-mode chunked --out /tmp/dg_selfcond_l1_gumbel_control.json
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT -u DG_TRACE_REGION_SIZE -u DG_SELFCOND_PRECHUNK_EMBED DG_SELFCOND_LOGITS_L1=chain python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --gumbel-mode chunked --out /tmp/dg_selfcond_l1_gumbel_candidate_review.json
python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py --compare /tmp/dg_selfcond_l1_gumbel_control.json /tmp/dg_selfcond_l1_gumbel_candidate_review.json --out models/experimental/diffusion_gemma/doc/optimize_perf/selfcond_logits_l1_gumbel_decisions.json
```

256K and watcher gates:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT -u DG_TRACE_REGION_SIZE -u DG_SELFCOND_PRECHUNK_EMBED DG_SELFCOND_LOGITS_L1=chain python -u models/experimental/diffusion_gemma/demo/serving_smoke.py --num-blocks 1 --canvas-length 256 --max-denoising-steps 48 --max-seq-len 262144 --gumbel-mode chunked --disable-eos-stop --local-files-only --metrics-json models/experimental/diffusion_gemma/doc/optimize_perf/selfcond_logits_l1_256k_chunked.json
env -u DG_SELFCOND_PRECHUNK_EMBED -u DG_DENOISE_TRACED_MULTISTEP -u DG_DENOISE_EARLY_HALT DG_SELFCOND_LOGITS_L1=chain DG_DENOISE_TRACED=1 DG_TRACE_REGION_SIZE=1073741824 TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_DISABLE_ETH=1 python -u models/experimental/diffusion_gemma/demo/serving_smoke.py --num-layers 2 --max-seq-len 1024 --num-blocks 1 --canvas-length 256 --max-denoising-steps 4 --gumbel-mode argmax --disable-eos-stop --local-files-only --metrics-json models/experimental/diffusion_gemma/doc/optimize_perf/selfcond_logits_l1_watcher.json
```

Proportional tests:

```bash
pytest -q models/experimental/diffusion_gemma/tests/test_tt_self_conditioning.py models/experimental/diffusion_gemma/tests/test_denoise_forward.py
python -m py_compile models/experimental/diffusion_gemma/tt/self_conditioning.py models/experimental/diffusion_gemma/demo/serving_smoke.py models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py models/experimental/diffusion_gemma/doc/optimize_perf/qualitative_prechunk.py
git diff --check -- models/experimental/diffusion_gemma
```

## Rejected adjacent work

Replacing the 32 logits slices with one multi-output `ttnn.split` was a traced
regression and was removed; see `selfcond_logits_split_rejection.md`.

Extending L1 placement through the numerator matmuls/accumulator and final
divide saved only 0.375 ms beyond `chain` in the component probe. Its
traced attempt is not evidence: an external hardware test started after
preflight and overlapped the run. The speculative mode was removed, and no
number from that contaminated run is used.

Increasing the ordered vocab chunk from 8K to 32K was a real +0.95% warmed @48
win, but changed the canonical clean-commit digest. It was removed on decision
fidelity; see `selfcond_vocab_chunk_rejection.md`.

Making TTNN's inherited denominator placement explicit via output
`memory_config` arguments regressed the required final run from the earlier
18.946 t/s sample to 18.771 t/s. Those arguments were removed; the inherited
placement is retained and tested.
