# Native router score projection

Direct FP32 scaled input × original BF16 router weights, FP32 accumulation/output. Centering and native gate unchanged; prefill unchanged. All ten actual4096/128 trials pass. These are traced whole-layer host timings; final device rows and cumulative contracts remain pending. Exact commands: `direct_router_commands.json`.

| Candidate | Minimum decode PCC | Median host us |
|---|---:|---:|
| actual_router_direct_k22_hifi4_layer0.json | 0.995341957 | 988.225 |
| actual_router_direct_k22_hifi4_layer5.json | 0.995214121 | 1077.508 |
| actual_router_direct_k44_hifi2_layer0.json | 0.995381194 | 986.606 |
| actual_router_direct_k44_hifi2_layer5.json | 0.995200366 | 1076.483 |
| actual_router_direct_k44_hifi4_layer0.json | 0.995341957 | 988.473 |
| actual_router_direct_k44_hifi4_layer5.json | 0.995214121 | 1077.733 |
| actual_router_direct_k44_lofi_layer0.json | 0.995309726 | 987.371 |
| actual_router_direct_k44_lofi_layer5.json | 0.995145877 | 1076.352 |
| actual_router_direct_k88_hifi4_layer0.json | 0.995341957 | 990.273 |
| actual_router_direct_k88_hifi4_layer5.json | 0.995214121 | 1078.147 |

## Extended selection

`router_selected_stress_commands.json` records selected512-step controls. SlidingK44HiFi2 fails one actualposition1459 at.9943880922; `router_repair_commands.json` tests K22HiFi4 and passesall512 atminimum.995523605, preserving the broadcast minimum. Keep slidingK22HiFi4 (headline988.225us). FullK44LoFi passesall512 and the isolated2049-token request-reuse failure window; keep fullK44LoFi (headline1076.352us). Final default public contracts/watcher/profiling are still pending.
