# Llama 3.1 8B Instruct full-mesh decoder

The completed path is a real tensor-parallel decoder layer for the four-chip
Blackhole P300c ring on this machine.  It derives from `OptimizedDecoder`, keeps
the optimized BFP4/LoFi projection policy, and targets only `MeshShape(1, 4)`.
It does not start the full-model or vLLM stages.

## Selected mesh plan

The model is dense, so MoE/expert parallelism is not applicable.  TP=4 uses all
four detected devices and divides every model-parallel dimension exactly; no
feature padding is necessary.  Logical sequence padding remains internal.

| Role | Global shape | Per-device shape / ownership | Final placement |
| --- | --- | --- | --- |
| residual and norms | `[...,4096]`, gamma `[4096]` | full replica | BF16; decode width-sharded over 16 L1 cores |
| Q/K/V | Q8/KV2 heads per rank | packed weight `[4096,1536]` | column TP, BFP4 DRAM-sharded, 16-way storage/N partition |
| attention context | `[...,4096]` | `[...,1024]` | head-local until O projection |
| O | `[4096,4096]` | transposed weight `[1024,4096]` | row TP, BFP4 DRAM-sharded, 16-way storage/N partition |
| gate and up | `[...,14336]` each | `[...,3584]` each | separate column TP, 16-way storage/N partition |
| down | `[14336,4096]` runtime | `[3584,4096]` | row TP, BFP4 DRAM-sharded, 16-way storage/N partition |
| contiguous cache | `[batch,8,context,128]` | `[batch,2,context,128]` | KV-head sharded, BF16 or BFP8 |
| paged cache | `[blocks,8,64,128]` | `[blocks,2,64,128]` | KV-head sharded; final-page padding is private |
| page table / position | `[batch,pages]` / `[batch]` | full replica | INT32 DRAM |

O and down produce full-width partials.  Two trace-safe asynchronous BF16 sum
reductions per layer restore the replicated residual, using the shared
per-mesh `TT_CCL` owner, ring topology, and two links.  Prefill returns a DRAM
replica and decode returns a 16-core width-sharded L1 replica; both have the
logical shape `[1,batch,seq,4096]` and feed the next decoder directly.

The detailed pre-code calculations, shard specs, collective byte ledger,
memory budget, and rejected alternatives are in [mesh_plan.md](mesh_plan.md).
The final policy came from the complete sweep in
[candidate_results.csv](candidate_results.csv).  The exact topology comparison
is in [topology_results.csv](topology_results.csv), with the fused row-boundary
API audit in [fused_mm_rs_audit.md](fused_mm_rs_audit.md).

## Correctness and runtime gates

The reference is the single-chip TTNN `OptimizedDecoder`, not a host framework.
The tests instantiate representative layer 16; all 32 Llama decoder layers
have the same dense attention/MLP kind and tensor shapes.  The
fallback-forbidden final run passed both reference and TP4 tests.

| Gate | Result |
| --- | --- |
| non-aligned prefill, logical length 7 | PCC `0.9999993671` |
| decode at position 7 | PCC `0.9999886729` |
| second-layer prefill boundary | PCC `0.9999982966` |
| second-layer decode boundary | PCC `0.9999547372` |
| local key/value cache heads | PCC `0.9999870894` / `0.9999805259` |
| three adversarial paged decode runs | PCC `0.9999871259`; bitwise deterministic |
| non-aligned prefill length 31 | PCC `0.9999996415` |
| decode positions 63 / 64 / 65 | PCC `0.9999704508` / `0.9999710440` / `0.9999695770` |
| physical page/head reconstruction | minimum PCC `0.9998642446`; unwritten suffix exactly zero |
| warmed page-1 decode trace | five replays PCC `1.0`, bitwise equal |
| host fallback audit | pass with `throw_exception_on_fallback=true` |
| watcher stress | four devices, ten deterministic trace replays, clean exit |

Correctness and fallback command:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
timeout 300 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k multichip_correctness
```

Watcher command:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
RUN_MULTICHIP_DECODER_WATCHER=1 timeout 300 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k multichip_watcher_stress
```

ETH watcher instrumentation is the only controlled disable.  Archived
same-host TP2 evidence identifies an active-ETH Blackhole firmware teardown
false positive after successful fabric execution; compute, dataflow, and NoC
watcher coverage remained enabled.  See [watcher_summary.txt](watcher_summary.txt).

## Warmed performance

The final run used batch 1, logical prefill length 18, 50 warmed prefill
iterations, and 1,000 warmed decode trace replays.  Measurements are host wall
time after synchronization.

| Path | Single chip | TP4 | Speedup | TP4 efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill | 1.243475 ms | 0.733909 ms | 1.694317x | 42.3579% |
| decode | 0.581578 ms | 0.320058 ms | 1.817101x | 45.4275% |

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
MULTICHIP_DECODER_PREFILL_REPEATS=50 MULTICHIP_DECODER_TRACE_REPLAYS=1000 \
timeout 300 pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

The canonical log is [logs/perf_final.log](logs/perf_final.log).  Candidate
provenance and the logs for O/residual core counts, CCL dtype/link count, and
packed gate/up are linked from `candidate_results.csv`.

## Profiler acceptance

The retained losslessly compressed Tracy op CSV is analyzed per signpost with
`tt-perf-report`.

| Region | Dominant observations |
| --- | --- |
| TP4 prefill | matmul 40.19%; norm 23.51%; reduce-scatter 11.15%; all-gather 6.63%; modeled DRAM 13.6% / 70 GB/s |
| TP4 decode | matmul 36.20%; reduce-scatter 20.76%; all-gather 12.67%; explicit DM+TM about 14.46%; modeled DRAM 15.5% / 119 GB/s |
| single prefill/decode | matmul 76.33% / 81.59%; modeled DRAM 17.9% / 38.8% |

The profile shows that TP4 decode is communication-sensitive rather than
DRAM-saturated.  BF16/two-link CCL is retained: BFP8 CCL and one link both
regressed decode.  The fractured boundary adds distributed-norm traffic and
measured 1.066% slower; the generic fused all-gather-matmul API is unsafe on
TP4 because its implementation has a TP8 transfer ledger.  The source-level
evidence is in [AUTOTRIAGE.md](AUTOTRIAGE.md), and the bounded repair decision
is in [AUTOFIX.md](AUTOFIX.md).

The generic fused matmul+reduce-scatter API was separately adapted to exact
TP4-local BFP4/LoFi O and down shapes. It was correct but 26.6342% and
14.5637% slower, respectively, than matching separate interleaved matmul+RS
paths. The minimal-strided fused API has a repository-documented Blackhole
race (`#46181`) and was not launched unsafely. See
[fused_mm_rs_audit.md](fused_mm_rs_audit.md).

The retained filtered matmul rows report `Input 1 Datatype=BFLOAT4_B` and
`Math Fidelity=LoFi` for the measured projection path.  The final decode inner
blocks (QKV/O/gate/up/down = 8/2/8/8/7 tiles) are the full K width of their
16-core input shards, so a larger block is not legal for those shard specs.
The DRAM factory internally selects real `1x6`, `1x8`, `1x7`, `1x7`, and
`1x8` output subblocks even though its public config does not serialize them
for `tt-perf-report`.  Its 80-core kernel bounding launch must not be confused
with the configured 16-way storage/N partition.  The exact source derivation,
`SLOW` warning classification, eight-way comparison, and full interleaved 1-D
comparison are in [profiler_geometry_audit.md](profiler_geometry_audit.md).

See [tracy/README.md](tracy/README.md) for exact capture/report commands and
all human-readable and CSV artifacts.  Tracy instrumentation changes absolute
timing, so the long uninstrumented run above is the speedup authority.

## Capacity and limitations

The advertised 131,072-token context remains unchanged.  With two KV heads per
device, a full 32-layer BF16 cache is 4,294,967,296 bytes/device.  The complete
conservative plan is 13,866,369,024 bytes/device versus 34,178,731,008 bytes of
allocator-visible DRAM.  [../context_contract.json](../context_contract.json)
contains the exact formulas and BFP8 alternative.

This file intentionally supports only the available four-chip P300c ring.  It
does not promise smaller meshes.  Full-model weight loading, 32-layer runtime,
generation, and vLLM serving are downstream goals.  The host has a 64 MiB
`/dev/shm`, which emits MPI capacity warnings but did not invalidate any gate.
The repo-local `generated/inspector` directory is not writable by this user;
watcher evidence is clean, while external tt-triage worker callstack capture
is limited by the installed ttexalens/UMD `noc_read` ABI mismatch.

## Artifact index

- [final_gate_results.txt](final_gate_results.txt): compact acceptance ledger.
- [hardware_inventory.txt](hardware_inventory.txt): final `tt-smi -ls --local`
  inventory showing Blackhole p300c IDs 0 through 3.
- [work_log.md](work_log.md): chronological commands, failures, and decisions.
- [logs/fallback_audit_final.log](logs/fallback_audit_final.log): final full
  correctness, stack, cache, trace, and fallback gate.
- [logs/pytest_final.log](logs/pytest_final.log): complete default stage test
  file, `5 passed, 5` expected opt-in skips.
- [logs/static_final.log](logs/static_final.log): current TP4 plus archived TP2
  static contracts, `6 passed`.
- [logs/watcher_final.log](logs/watcher_final.log): watcher-clean stress log.
- [candidate_results.csv](candidate_results.csv): complete warmed sweep.
- [profiler_geometry_audit.md](profiler_geometry_audit.md): source-backed
  output-subblock/core mapping and same-policy projection alternatives.
- [fused_mm_rs_audit.md](fused_mm_rs_audit.md): exact O/down fused-MM+RS
  measurements and API-by-API disposition.
- [topology_results.csv](topology_results.csv): replicated versus fractured
  topology measurement.
- [tracy/](tracy/README.md): retained profiler CSV, report tables, and provenance.

`logs/correctness_final.log` and `logs/fallback_audit.log` are retained
pre-stack checkpoints and are superseded by `logs/fallback_audit_final.log`.
