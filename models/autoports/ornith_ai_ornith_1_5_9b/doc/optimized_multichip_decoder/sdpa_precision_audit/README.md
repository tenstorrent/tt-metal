# Decode-SDPA precision isolation

Source-only candidate; no hardware compilation, correctness, latency or watcher results claimed. Production remains unchanged.

Current production passes shared `self.compute` into private/B1 paged SDPA (`tt/multichip_decoder.py:875–885`) and pooled batched paged SDPA (`:1197–1207`). That config is HiFi4, exact math, FP32 destination, whereas the public paged wrapper's omitted-config default is HiFi2, approximate math, BF16 destination (`ttnn/.../transformer/sdpa_decode/sdpa_decode.cpp:147–148`). The current target SDPA costs211.747 us; this is a distinct material precision family, not covered by projection fidelity or SDPA grid sweeps.

`tests/multichip_sdpa_precision_candidate.py` wraps only `ttnn.transformer.paged_scaled_dot_product_attention_decode` for this process. It captures the original callable and restores it in `finally`, including failed gates. Each call receives a new compute-config object with only three fields changed; `self.compute` is never mutated. Q/K norms, partial RoPE, GDN, projection math, BF8 KV, all operands, positions/tables, scale, output layout, ordered core map, chunk geometry and chunked prefill SDPA remain unchanged. Unaligned continuation that legitimately uses the decode kernel also receives the candidate policy; padding/masking/slicing stay owned by production.

| Policy | Fidelity | Approximate math | FP32 destination |
|---|---|---|---|
| control | production unchanged | unchanged | unchanged |
| hifi2_exact_fp32 | HiFi2 | false | true |
| hifi2_approx_fp32 | HiFi2 | true | true |
| hifi2_approx_bf16 | HiFi2 | true | false |
| lofi_exact_fp32 | LoFi | false | true |

Packer L1 accumulation, destination synchronization and throttle settings are copied from the original config, rather than silently adopting unrelated library defaults. Therefore the BF16-destination candidate matches the three advertised default precision settings, not necessarily every default compute-config field. The installed WormholeComputeKernelConfig constructor accepts all six preserved/selected fields; host tests verified this without a device. The SDPA factory consumes these arguments (`sdpa_decode_program_factory.cpp:219–220,947–950`) and changes DST capacity from4 to8 tiles when FP32 destination is disabled (`:428`). These are real kernel changes, not metadata-only policy labels.

The wrapper emits `<output_stem>_sdpa_policy.json` even on failure, recording original/selected fields and actual query shapes/dtype. Its host invocation count intentionally excludes device trace replays. A successful harness run must have observed at least one patched operation. This evidence verifies policy reach without adding tensor readback or host work to the captured replay.

## Jobs and independent HF evidence

`../sdpa_precision_jobs.json` has22 jobs:

- Production-policy control at B32 short plus recorded-HF B32 control.
- For each of four policies: B1 and B32 16-step state/benchmark, B32 prefix4095/1024-answer, B32 prefix65535/128-step, and independent recorded-HF full-attention B32.

Stack jobs retain the same frozen references, .995 output/contribution/state gates, rank checks, immutable program-cache tracking and traced warmed benchmark. The64K job is stress/regression evidence, not a replacement target for the4K→5K optimization roofline.

`--recorded-batch` uses existing `multichip_recorded_batch.py` on layer3 with `generated/multichip/recorded_precision_inputs.pt`. Its decoder binding is replaced only in this process with a subclass inheriting **all current production methods**; this avoids unintentionally benchmarking the harness's historical GDN candidate base. No policy methods override production. Existing independent HF output/contribution PCC >=.995 and rank agreement >=.99999 remain unchanged; both corpora and recorded prefix lengths are retained. The original harness binding is restored afterward. The control uses the same routing. A recorded-HF pass cannot override an evolving-state failure, or vice versa.

`../sdpa_precision_watcher_jobs.json` contains four separate B32 watcher/state jobs, one per candidate. **Select only policies that passed the numerical matrix before queuing watcher.** This is a separate selection list because run_stage5_jobs has no conditional dependency feature; do not automatically run all four as if their accuracy gates had passed. Watcher jobs omit benchmarking and are not performance evidence.

Parent's serialized launch command (not run here):

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/sdpa_precision_jobs.json
```

After passing-policy selection, submit those watcher entries through the same existing protocol with fresh job labels retained. If a dtype/layout/kernel assertion occurs, inspect the reached policy sidecar and adapt the isolated config/geometry before rejecting the whole precision family. Source defaults demonstrate applicability, not proof that these particular paged B32 shapes pass.

Host validation passed: both stack/recorded `--help` paths before mesh opening; ten policy/config combinations including preserved packer=true/false and dst synchronization; operand/geometry identity; original-config immutability; operation restoration after an injected exception;26 unique fresh job labels and all reference/input paths. Artifacts: `host_validation.json`, `source_manifest.json`. No device was opened. Final selection still requires cumulative-default reruns, final-position profile and context/trace/watcher checks.
