# Shared Q/K full-model comparison and reference-eval queue

Completed at 13:10:41 UTC on Oct 8, 2026, after launch at 09:09:38 (about
4 hours 1 minute). All eight fresh-process runs passed, with clean device
shutdown and identical generated tokens between control and candidate at every
geometry. Three measured repetitions plus warmup each re-prefilled the complete
prompt and generated 128 tokens, including beyond EOS. Precision is unchanged:
BFP8 KV, BF16 activations, FP32 recurrent state and existing BFP4 weights.

| Context | Batch per TP4 | Control output tok/s | Shared-Q/K output tok/s | Uplift | Eightfold projection |
|---|---:|---:|---:|---:|---:|
| 32K | 32 | 350.416 | 373.695 | +6.64% | 2989.6 |
| 16K | 32 | 402.164 | 432.929 | +7.65% | 3463.4 |
| 128K | 16 | 163.357 | 168.739 | +3.29% | 1349.9 |
| near256K | 8 | 89.832 | 91.303 | +1.64% | 730.4 |

The measured output rates are one TP4 replica, all 64 layers. Eightfold figures
are extrapolations, not physical Galaxy throughput. HTTP overhead and prefill
are excluded. Prefill input rates stayed effectively unchanged at about 5,185
(32K/B32), 6,027 (16K/B32), 4,044 (128K/B16) and 3,687 (near256K/B8) tok/s.
The smallest uplift, 1.64%, should receive a repeated/interleaved comparison
before treating its magnitude as a stable deployment guarantee.

Each geometry has raw sweep receipts, JUnit and a `comparison` directory with
PNG/SVG/PDF/CSV/JSON. Local reanalysis validated raw sample accounting, output
hash equality, workload, source/precision parity and all comparison summaries.
This is performance and repeatability evidence, not reference-eval accuracy.

## Physical Galaxy and GPQA follow-through

- **15:46:56 UTC:** launched `qwen38-shared-qk-g0-v1-20261008.service` using
  the immutable passing model source. Eight physical TP4 replicas, short-prompt
  B1 output agreement and isolated/concurrent TPOT <=3% regression. B1 uses the
  policy's fused normalization fallback; this G0 run alone does not measure
  shared-Q/K at high batch or long-context eight-replica throughput.
- **15:52:31 UTC:** launched persistent `qwen38-shared-qk-gpqa-v1-20261008.service`.
  It waits for successful G0 process completion, exact model-source/precision
  hashes, all eight distinct physical chip groups and passing JUnit before
  starting the isolated vLLM endpoint on loopback port 8078.
- GPQA uses all 198 Diamond questions, concurrency 128 across eight engines,
  32K maximum output tokens, temperature 1.0, top_p 0.95, top_k 20, seed 42 and
  thinking enabled. The threshold remains 89.2% (at least 177/198). This matches
  the prior run's budget/sampling rather than changing them at the same time
  as the model path. The benchmark's dataset bytes and harness are pinned.
- The official [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)
  was refreshed Oct 8: it reports GPQA-D 89.2% and recommends those thinking
  sampling parameters. It does not specify a complete GPQA protocol or a
  GPQA-specific output budget, so our run is not described as an exact official
  harness reproduction.
- Serving now accepts an isolated source root; discovery, child working
  directory and imports use that source, preserving the original checkout.
  Prefill is explicitly bounded to the validated 32,768-token chunk budget.
  `--exit-after-eval` stops owned workers after GPQA so the hardware lock is
  released for kernel experiments. No resident endpoint or HTTP sweep is
  claimed by that mode. A dirty-device marker is retained for the next safe
  runner's reset; process shutdown is not mislabeled device qualification.
- 350 CPU tests plus 40 subtests and shell syntax checks passed. G0 has a
  3-hour bound; the dependent evaluation service a 9-hour bound, 256-GiB host
  memory cap and eight-CPU quota. Both survive SSH/session disconnect, not a
  reboot. Checkpoint, native install, firmware and NFS remain unchanged.

G0 and GPQA were still active at publication; these launch snapshots are not
pass receipts. Latest historical full GPQA remains 171/198 = 86.36%. Overall
plan completion, physical long-context scaling, reference accuracy and serving
promotion remain unproven.
