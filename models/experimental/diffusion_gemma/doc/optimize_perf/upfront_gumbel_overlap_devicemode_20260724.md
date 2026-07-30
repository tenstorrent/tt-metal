# Up-front denoise: the Gumbel shape trap and the device-Gumbel win (2026-07-24)

Status: provenance-only for the host arm — `DG_VLLM_GUMBEL_MODE=host`, `DG_HOST_GUMBEL_PREFETCH`
and the whole host-Gumbel-prefetch mechanism were deleted 2026-07-28 — and for the op table, taken
on the token-gather MoE deleted 2026-07-29. The shape trap and its fix are current.
Owns: the 8 GiB TILE_LAYOUT padding OOM in the permuted-vocab full-vocabulary Gumbel draw and its
2-D reshape fix; the post-change per-step op distribution.
See also: [refuted list](../REFUTED.md), [optimize-perf hub](README.md),
[device Gumbel restored](../decision_fidelity/device_gumbel_restored.md).

## REUSABLE SHAPE TRAP — the 8 GiB TILE_LAYOUT padding OOM

The permuted path built `ttnn.rand([vocab, 1, canvas, 1])`. `TILE_LAYOUT` pads the trailing size-1
axis to a full 32-tile, inflating the `[1,1,256,262144]` buffer **256 MiB → 8 GiB**:

```
TT_FATAL bank_manager.cpp:462 — allocate 8589934592 B
```

**Fix:** draw a 2-D `[vocab, inner]` rand with all non-vocab dims collapsed into one tile-aligned
inner axis (vocab stays outermost, so still not the correlated innermost axis), then permute
vocab→innermost and reshape. Buffer back to 256 MiB; the vocab-outermost distribution property is
preserved, exact values change. Pinned by
`test_permuted_vocab_gumbel_noise_deallocates_pre_permute_tensor`. The `ttnn.rand` PRNG defect that
made the permuted draw necessary is owned by
[device Gumbel restored](../decision_fidelity/device_gumbel_restored.md).

## What was measured

**DIAGNOSIS that motivated the work:** the up-front path was **host-bound, not device-bound** under
the host-Gumbel contract — every replay step regenerated `torch.rand((1,256,262144))` (256 MiB,
~313 ms host CPU) and replicated ~1 GiB H2D per step, with a redundant per-step
`synchronize_device` foreclosing any overlap.

Removing that per-step `ttnn.synchronize_device` in `tt/traced_denoise.py` is safe because the
following `read_halt_scalars` `to_torch` is CQ0-ordered and blocking, so ordering is preserved.

**Throughput record:** host-serial ~771 ms/step → host+prefetch ~658 ms/step (decode block 9.98 vs
11.48 s = 1.15×) → device ~428 ms/step (~1.8× vs host-serial, ~1.54× on top of prefetch). Gumbel-max
terminal sampling costs +16 ms over argmax (29 → 45 ms). The full device-mode step summed to ~496 ms
eager-synchronized against a ~450 ms traced step, i.e. **traced ≈ eager on this path**.

**BYTE-IDENTICAL CRITERION:** because the Gumbel is per-`(block, step)` privately seeded, a prefetch
change alters only *when* it is generated and never the value, so no decision re-gate is required.
The shared-generator renoise-token stream was deliberately left untouched.

**Device-mode quality re-gate — NOT a clean pass at n=2.** GPQA @3072, samples 0/1, same-samples
A/B: doc 0 (target C) host `em=1 \boxed{C}` vs device `em=0` — correct reasoning reaching C but the
`\boxed{}` wrapper dropped; doc 1 (target A) both `em=0` with `\boxed{C}`. Accuracy host **0.5** vs
device **0.0**. The owed sub-40 host-vs-device re-gate is now permanently unrunnable (the host arm
was deleted), so the device default still rests on this n=2 result, which the original doc itself
called "inconclusive-to-negative" — see [refuted list](../REFUTED.md). The current
`DG_VLLM_GUMBEL_MODE` default and the full host/device history are owned by the
[optimize-perf hub](README.md).

The up-front validator still rejects `chunked` and `argmax` loudly because they are not materialized
full-tensor Gumbel sources (confirmed live in `tt/generator_vllm.py`).

## Per-step 30-layer op distribution — provenance only

Reduced-layer synchronized per-component device time (approved substitute, Tracy OFF), 2-point fit
L=2/L=6, measured on the deleted token-gather MoE:

| component | 30L ms | % step |
|---|--:|--:|
| backbone layers | 418.7 | 84.3 |
| — MoE (router+gather+experts) | 282.6 | 56.9 |
| — attention + norms + RoPE | 135.6 | 27.3 |
| terminal (gumbel-max sampling + sort/cumsum/scatter accept) | 45.0 | 9.1 |
| soft-embedding (262k self-cond input) | 16.0 | 3.2 |
| Gumbel RNG (device permuted-vocab) | 10.9 | 2.2 |
| LM head | 4.3 | 0.9 |
| self-cond gated MLP | 1.6 | 0.3 |
| commit (per block, separate) | ~960 | — |

**REFUTED LEVER:** at ~2.2% (~10.9 ms) of the step, the 2-command-queue RNG-overlap lever is not
worth doing. The table's "MoE 6-D gather-Permute fusion" next-lever ranking describes a path that no
longer exists.

> **OPEN CONTRADICTION (unexplained):** the denoise per-step cost is stated as ~428 ms traced /
> ~496 ms eager-sync here (MoE 56.9%), ~465–540 ms in
> [context speed sweep](context_speed_sweep_20260722.md), ~0.9 s with MoE ~89% in
> [official sampler](official_sampler_earlyhalt_20260722.md), and 4–5.6 s in the deleted
> `ttft_ts_sweep.md`. Each was measured on a different, now-superseded MoE path and nothing in the
> tree reconciles them. Not explained.

## Repro

All runs: 4× Blackhole p300c, no Tracy or watcher. Env: see [plan](../../plan.md).

- Quality/serving vehicle: `run_upfront_gpqa.sh smoke` with `DG_VLLM_GUMBEL_MODE` set and
  `MAX_GEN_TOKS=3072`.
- Op-breakdown vehicle: `doc/optimize_perf/prof_step_breakdown.py`.
- The throughput A/B used `bench_gumbel_mode.py` and `bench_upfront_prefetch.py`, **neither of which
  is present in the repo**, so those numbers cannot be re-measured as recorded.
