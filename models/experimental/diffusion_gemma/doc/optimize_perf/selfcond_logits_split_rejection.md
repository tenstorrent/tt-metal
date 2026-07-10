# Self-conditioning logits split — measured rejection

Date: 2026-07-10 UTC. This is a rejected candidate, not selected-current
evidence. The selected path retains sequential logits slices; the later
placement-only L1 chain is documented in `selfcond_logits_l1.md`.

## Candidate

Replace the 32 sequential `ttnn.slice` operations over the dynamic
`[1,1,256,262144]` BF16 logits with one
`ttnn.split(..., 8192, dim=-1)`. Embedding chunks, matmul shapes, arithmetic,
and ordered accumulation were unchanged. The experiment was opt-in through
`DG_SELFCOND_SPLIT_LOGITS=1`.

## Result

| synchronized component, 2 layers × 5 iterations | slices | split | delta |
|---|---:|---:|---:|
| soft embedding | 18.2129 ms | 18.2116 ms | -0.007% |
| full step | 73.5245 ms | 73.9067 ms | +0.52% |

| canonical traced full model, @48, 3 blocks | slices | split | delta |
|---|---:|---:|---:|
| warmed steady block | 13.6284 s | 13.6445 s | +0.12% |
| throughput | 18.784 t/s | 18.762 t/s | -0.12% |
| full generation | 152.1866 s | 151.4487 s | -0.48% |
| committed SHA | `a9f0d18709b07d1e` | `a9f0d18709b07d1e` | exact |

The targeted component was unchanged and warmed traced throughput regressed.
The candidate's lower full-generation total came from variable first-block
trace capture (123.0523 versus 123.8355 s), not warmed execution. This is not
a verified speedup. The candidate code and selector were removed.

Machine-readable evidence is in
`selfcond_logits_split_rejection.json`.

## Exact commands

Before every command below, device use was checked with:

```bash
ps -eo pid,ppid,pgid,tty,stat,etime,args | rg -i 'cursor-agent|/agent |pytest|serving_smoke|prof_step_breakdown|bench_lever|verify_selfcond|qualitative_prechunk|tt-smi|python.*diffusion_gemma|python.*models'
```

Component control/candidate (run once each with the final variable set to `0`
and `1`):

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate && env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_TRACE_REGION_SIZE PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn TT_METAL_HOME=/home/zni/tt-metal TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARCH_NAME=blackhole DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 DG_SELFCOND_PRECHUNK_EMBED=1 DG_SELFCOND_SPLIT_LOGITS=0 python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py --num-layers 2 --iters 5
```

Canonical traced control/candidate (run once each with the final variable set
to `0` and `1`):

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate && env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn TT_METAL_HOME=/home/zni/tt-metal TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARCH_NAME=blackhole DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it DG_TRACE_REGION_SIZE=10737418240 DG_SELFCOND_SPLIT_LOGITS=0 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers baseline --budgets 48 --blocks 3 --out /tmp/dg_logits_split_e2e_control.json
```

For the candidate process, `DG_SELFCOND_SPLIT_LOGITS=1` and output
`/tmp/dg_logits_split_e2e_candidate.json` were used.
