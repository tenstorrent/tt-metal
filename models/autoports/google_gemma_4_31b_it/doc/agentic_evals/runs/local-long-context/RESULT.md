# Long-context result

Both two-client synthetic growing-history phases passed on the same full60 B8 server, including the final262136-token prompt plus eight generated tokens at the advertised262144-token boundary. This confirms admission and cache reuse for these request shapes, not GPQA accuracy, long free-running output quality or general real-agent capacity.

The server registration event resolves `models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM` from this execution checkout, with layer indices0–59 and devices0–3 in a1×4 mesh. Current14 source files match setup/source_publication.json SHA256 values; no source change was made during this effort. Registration proves the live class/layers, while the hash comparison is a filesystem check, not an in-process code hash attestation. The original context and serving defaults remain intact.

| Measurement | Through131072 | Advertised boundary |
|---|---:|---:|
| Prompt sequence | 4096,32768,65536,98304,131072 | 131072,196608,262136 |
| Clients / captured batch | 2 / 8 | 2 / 8 |
| Completed requests | 10/10 | 6/6 |
| Output tokens per request | 8 | 8 |
| HTTP workload elapsed | 140.414945s | 443.724616s |
| Prefix hit fraction of attainable ceiling | 0.998585178 | 0.999566669 |
| Late hit fraction | 0.999133375 | 0.999638888 |
| Prefilled tokens | 262656 | 524528 |
| Final prompt-token sum | 262144 | 524272 |
| Prefilled/final prompt-token ratio | 1.001953125 | 1.000488296 |
| Preemptions | 0 | 0 |
| New trace captures after warmup | 0 | 0 |

Evidence: growing.json and advertised_context.json. All existing prefix, late-prefix, trace-lifetime and trace-timing checks passed without modifications. The final round's two prompt+output histories total524288 logical tokens, whereas the artifact denominator524272 counts prompts only. No preemptions were observed at that demand. This does not measure peak allocator usage or prove capacity beyond these two histories. The second phase used unique prefixes without clearing prior evictable cache state; it is not a clean-cache performance comparison. The final prompt's non-block-aligned length also completed successfully.

First-phase sampling counts are126 host-compatible intermediate-prefill steps, six device-prefill steps and64 device-decode steps. Second-phase counts are254 host-compatible intermediate-prefill, four device-prefill and42 device-decode. This accepted stock-plugin host-sampling limit remains visible; decode uses the device sampler.

The initial estimate was0.15device hours for the original phase including startup. The extension's plan first guessed300seconds for phase2; before launch the coordinator revised it to600seconds using conservative approximately4× scaling of the measured140.415second phase, with a1000second client timeout inside the original1800second total box. Actual443.725seconds exceeded the preliminary300second guess but fit the launch prediction by156.275seconds. Long full-attention cost scales nonlinearly, so the early linear estimate was insufficient; these synthetic observations do not replace measured suite-time projections.

The coordinator queue records start05:48:33.328934UTC and measurement_end06:03:47.779193UTC:914.450259seconds, or0.254014device hours, within the0.5hour box. This includes startup and coordinator intervals up to the recorded measurement end; the two HTTP workloads total584.139561seconds. The server remains temporarily idle for a separate host protocol investigation. Queue closure is pending, and the idle gap after measurement_end is explicitly excluded from this effort.

No optimization or numerical change was proposed; the requested synthetic confirmation succeeded. Remaining gaps are scored suite trajectories, real working-set distributions, long free-running quality, full task limits and CI release execution. These host-wall measurements are not device timing or headline4096/128/B1 performance. No profiling was performed and no device-time roofline values may be inferred.
