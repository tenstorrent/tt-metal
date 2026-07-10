# Self-conditioning vocab chunk size — faster but decision-inexact

Date: 2026-07-10 UTC. This candidate was rejected and its selector was removed.
The selected 8192-vocabulary grouping is unchanged.

## Candidate and result

The production soft embedding evaluates the 262144-vocabulary online softmax
as 32 ordered chunks. Larger chunks reduce slice, exp, matmul, reduction, and
ordered-add launch count without changing dtype or persistent bytes, but they
also change each matmul/reduction grouping.

| vocab chunk | soft embedding | full two-layer step |
|---:|---:|---:|
| 8192 selected control | 16.038 ms | 71.359 ms |
| 16384 | 15.554 ms | 71.810 ms |
| **32768** | **15.322 ms** | **70.994 ms** |
| 65536 | 15.223 ms | 72.447 ms |

The 32K candidate advanced to the canonical traced @48 gate:

| path | steady block | throughput | full generation | committed SHA |
|---|---:|---:|---:|---|
| 8K control | 13.6321 s | 18.779 t/s | 151.9821 s | `a9f0d18709b07d1e` |
| 32K candidate | **13.5042 s** | **18.957 t/s** | 152.1255 s | `f224bc72c06ce5a0` |

The candidate is a real warmed throughput win (+0.95%), but it changes the
canonical clean commits. That is a decisive diffusion-decision failure under
the identical prompt, seed, step count, BF16 policy, and traced workload. A
full per-step acceptance campaign cannot make a final-commit mismatch
eligible, so it was not spent. `DG_SELFCOND_VOCAB_CHUNK_SIZE` and all
speculative code were removed.

## Exact commands

Every hardware command was preceded by the campaign process-ownership check.
Shared environment:

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn
export TT_METAL_HOME=/home/zni/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARCH_NAME=blackhole
export DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
```

The component sweep used the following command with
`DG_SELFCOND_VOCAB_CHUNK_SIZE` set successively to `16384`, `32768`, and
`65536`:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u DG_DENOISE_TRACED -u DG_TRACE_REGION_SIZE DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 DG_SELFCOND_PRECHUNK_EMBED=1 DG_SELFCOND_LOGITS_L1=chain DG_SELFCOND_VOCAB_CHUNK_SIZE=32768 python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py --num-layers 2 --iters 5
```

Canonical control:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED -u DG_SELFCOND_LOGITS_L1 -u DG_SELFCOND_VOCAB_CHUNK_SIZE DG_TRACE_REGION_SIZE=10737418240 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers baseline --budgets 48 --blocks 3 --out /tmp/dg_selfcond_chunk8k_control48.json
```

Canonical 32K candidate:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_APPEND -u TT_METAL_WATCHER_DISABLE_ETH -u TT_METAL_WATCHER_DUMP_ALL -u TT_METAL_WATCHER_NOINLINE -u TRACY_NO_INVARIANT_CHECK -u DG_SELFCOND_PRECHUNK_EMBED -u DG_SELFCOND_LOGITS_L1 DG_TRACE_REGION_SIZE=10737418240 DG_SELFCOND_VOCAB_CHUNK_SIZE=32768 python -u models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py --levers baseline --budgets 48 --blocks 3 --out /tmp/dg_selfcond_chunk32k_candidate48.json
```

Machine-readable rows are in `selfcond_vocab_chunk_rejection.json`.
