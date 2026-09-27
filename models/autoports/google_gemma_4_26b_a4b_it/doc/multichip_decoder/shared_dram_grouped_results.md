# Shared DRAM and grouped-MoE reduction measurements

Both DRAM-sharded shared-MLP candidates pass the paired real-weight 4096/128 gate, but are slower than the geometry1 control. Grouping shared/routed BF16 outputs into one all-reduce passes and improves traced host decode latency for both layer kinds. Geometry1 plus grouped reduction is the better measured candidate for this replicated Linear layout. Defaults remain unchanged; these measurements do not complete stage acceptance.

All four new runs use hybrid EP prefill/TP decode, fused tail and optimized shared decode precision. They retain replicated residuals and Linear topology, with no fused AGMM. Shared DRAM uses geometry0; grouped reduction uses geometry1. Each paired run compares TP4 against TP1 with 4096 prefill tokens, 128 teacher-forced traced decode positions and KV-cache checks. No device-profiler or Watcher environment is enabled.

## Correctness and traced host timing

| Kind / candidate | Minimum PCC | Minimum cache PCC | TP4 prefill median, us | TP4 decode median, us | Decode change vs geometry1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [sliding shared geometry1](sliding_shared_geometry1.json) | 0.9998746421 | 0.9999999987 | 94886.508 | 769.428 | control |
| [full shared geometry1](full_shared_geometry1.json) | 0.9994611280 | 0.9999715022 | 80722.176 | 761.406 | control |
| [sliding shared dram](sliding_shared_dram.json) | 0.9999102726 | 0.9999999987 | 94888.004 | 783.622 | +1.84% |
| [full shared dram](full_shared_dram.json) | 0.9994750345 | 0.9999715022 | 80511.527 | 774.580 | +1.73% |
| [sliding grouped shared geometry1](sliding_grouped_shared_geometry1.json) | 0.9998746421 | 0.9999999987 | 95257.136 | 750.404 | -2.47% |
| [full grouped shared geometry1](full_grouped_shared_geometry1.json) | 0.9994611280 | 0.9999715022 | 80875.047 | 741.927 | -2.56% |

All new runs exit 0, exceed the unchanged 0.995 PCC gate, pass the device-only fallback guard and verify exact equality across replicas and repeated trace replay. The complete 129-element output-PCC arrays and eight-element cache-PCC arrays for grouped reduction are identical to their geometry1 controls. This is equality of reported correlation values; tensors were not saved for a cross-run bitwise comparison.

Timings are synchronized host intervals around traced replay, with input uploads and output reads outside the interval. They are not native device times. Each decode median uses 128 positions. Prefill medians use three warmed host calls. TP1 decode medians in the new runs remain 825.027–825.747us sliding and 875.933–876.223us full. Native profiles are still needed for attribution.

## Source and implementation

- DRAM runs use runtime `e9b7dd05c6f1dd3201c869756860dbc9e3050bd7f2eaa2af68c584e68b233d20`, preserved in [runtime_shared_dram.py.txt](runtime_shared_dram.py.txt). Runner SHA is `7c9e3fde951ee9ca2429546b95e5a4a8218ae2e4fa34a4e6b6539b57d20b47c8`, preserved in [runner_shared_dram.py.txt](runner_shared_dram.py.txt). Exact records: [shared_dram_applied_provenance.json](shared_dram_applied_provenance.json).
- Grouped passing runs use runtime `20151de9802f92af989cdad8f27cb764c79c8b8fad6c80939f5392edb8575cd4`, preserved in [runtime_shared_dram_grouped.py.txt](runtime_shared_dram_grouped.py.txt), with the same runner. Exact records: [shared_dram_grouped_provenance.json](shared_dram_grouped_provenance.json). The only intervening runtime change converts `shared.shape` to a tuple before slicing its first two dimensions.
- The existing geometry1 controls use runtime `be27d72163e05ac9ff294624e2475412ebc2bf9ba5f5a1afaa022816cac32426`. The DRAM subclass is inactive in grouped/control geometry1 runs. The shape-check correction affects only the opt-in grouped branch.
- The DRAM packing, 1088-to-1280 local padding, one-core down storage geometry, requested precision and capacity additions are described in [shared_dram_audit.md](shared_dram_audit.md). There were no DRAM API adaptations. The source-supported one-storage-core down path executes correctly on both layer kinds.
- Grouped reduction concatenates `[shared,routed]` along dimension 1, performs one all-reduce, then slices the two tensors apart. `_fused_tail` independently normalizes shared and routed outputs before combining them. No pre-normalization branch addition was introduced.
- The runtime retains both opt-in candidates for parent evaluation; no defaults changed. Shared DRAM is unselected because whole-layer traced decode is slower than geometry1 in both measured kinds. Stack, batch 32, maximum-context, native-profile, Watcher and independent-review gates for any final selected combination remain parent-owned work.

## API failure and recovery

The first grouped sliding run exited1 during prefill, before the grouped collective, because `ttnn.Shape.__getitem__` supports integer indexing but not slicing. [The preserved failure log](sliding_grouped_shared_geometry1_api_failure.log) records `tuple(shared.shape[:2])`; the correction is `tuple(shared.shape)[:2]`. The model loop closed its devices normally. No hang occurred, no process was killed, and no lock was removed.

The required recovery sequence ran serially and all commands exited0:

```sh
timeout 60 python_env/bin/tt-smi -ls --local
timeout 180 python_env/bin/tt-smi -r
timeout 60 python_env/bin/tt-smi -ls --local
timeout 60 python_env/bin/python - <<'SMOKE'
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK', flush=True)
SMOKE
```

Logs: [before list](grouped_api_list_before.log), [reset](grouped_api_reset.log), [after list](grouped_api_list_after.log), [mesh smoke](grouped_api_smoke.log). The list shows four P300c ASICs and the smoke contains `MESH_SMOKE_OK`. No second reset was required. Triage-before-kill was not applicable because the process returned a Python error rather than hanging.

## Reproduction and handoff

Exact commands, source hashes, medians and comparison percentages are also in [shared_dram_grouped_results.json](shared_dram_grouped_results.json). The four new commands were:

```sh
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --fused-tail --hybrid-experts --optimized-shared --shared-dram --layer 0 --length 4096 --steps 128 --trace --check-cache --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_shared_dram.json > models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_shared_dram.log 2>&1
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --fused-tail --hybrid-experts --optimized-shared --shared-dram --layer 5 --length 4096 --steps 128 --trace --check-cache --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_shared_dram.json > models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_shared_dram.log 2>&1
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --fused-tail --hybrid-experts --optimized-shared --shared-geometry 1 --grouped-moe-reduce --layer 0 --length 4096 --steps 128 --trace --check-cache --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_grouped_shared_geometry1.json > models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_grouped_shared_geometry1.log 2>&1
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --fused-tail --hybrid-experts --optimized-shared --shared-geometry 1 --grouped-moe-reduce --layer 5 --length 4096 --steps 128 --trace --check-cache --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_grouped_shared_geometry1.json > models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_grouped_shared_geometry1.log 2>&1
```

Both changed Python files pass `python -m py_compile`; no C++ or CMake changed. Final full-grouped session 51782 exited0. A process audit found no remaining model, tt-smi, triage or Tracy jobs. Hardware, runtime and runner ownership were explicitly released to the parent before any next experiment; subsequent work here is documentation only.
