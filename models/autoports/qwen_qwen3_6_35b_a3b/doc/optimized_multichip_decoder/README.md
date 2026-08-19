# Qwen3.6-35B-A3B Optimized Multichip Decoder

This directory records the optimized-multichip-decoder state for
`Qwen/Qwen3.6-35B-A3B`. Scope is the repo-local TTNN multichip decoder layer
only. No full-model or vLLM work was started.

The measured final path is the default `MultichipDecoder` path on the local
`2x2` Blackhole p300c mesh. Single-chip rows in the profiler run are
comparators only; they are not the stage signoff path.

## Final Path

| Contract | Final setting |
| --- | --- |
| Mesh | `2x2` local Blackhole mesh |
| Tensor parallelism | TP=2 across mesh columns, `cluster_axis=1` |
| Expert parallelism | EP=2 across mesh rows, `cluster_axis=0` |
| CCL default | public `ttnn.all_reduce`, `topology=Ring`, `num_links=2`, BF16 payload |
| Layer boundary residual | replicated BF16 DRAM/interleaved hidden tensor `[1, batch, seq, 2048]` |
| Inter-layer collectives | none; no gather, reshard, all-reduce, reduce-scatter, or all-gather between decoder layers |
| Packed projections | retained full-attention `q/k/v`, linear-attention `qkv/z/b/a`, shared MoE gate/up, and routed MoE gate/up packing |
| MoE execution | gate-selected active-expert sparse path; no dense all-expert runtime path selected |
| Candidate knobs | `QWEN36_MULTICHIP_NUM_LINKS`, `QWEN36_MULTICHIP_CCL_MODE`, and `QWEN36_MULTICHIP_CCL_DTYPE` remain for reproducing rejected candidates |

The final inter-layer residual layout contract is written in
`operation_topology_audit.md`. Full-model bringup should preserve this
replicated boundary and must not add layer-to-layer collectives.

No `doc/context_contract.json` change was made. This pass does not change
KV-cache dtype, KV-cache layout, cache block size, public tensor shapes,
activation-state capacity, or per-device context capacity. Valid non-aligned
logical sequence lengths remain supported; internal padding/masking/slicing is
owned inside the decoder path.

## Correctness

Acceptance bar is the completed multichip decoder baseline. Final default
correctness passed with real weights enabled:
`logs/final_correctness.log` (`10 passed, 4 deselected`).
After the fused AGMM candidate hang/reset, the same final default correctness
selection passed again in `logs/final_correctness_post_fused_recovery.log`
(`10 passed, 4 deselected`).

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| synthetic linear layer 0, seq 5 | 0.9999484088 | 0.9999441730 |
| synthetic full layer 3, seq 33 | 0.9999434563 | 0.9999454953 |
| synthetic linear layer 0, non-aligned seq 65 | 0.9999464360 | 0.9999427555 |
| synthetic full layer 3, non-aligned seq 33 | 0.9999434563 | 0.9999454953 |
| synthetic linear layer 0, batch 2, seq 5 | 0.9999495000 | 0.9999499536 |
| synthetic full layer 3, batch 2, seq 33 | 0.9999451874 | 0.9999451794 |
| real weights linear layer 0, seq 1 | 0.9999731323 | 0.9999286757 |
| real weights full layer 3, seq 1 | 0.9999452009 | 0.9998278764 |

`logs/final_runtime_fallback_audit.log` passed the same selected path with
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`
(`10 passed, 4 deselected`). The same dynamic fallback audit also passed after
fused-candidate recovery in
`logs/final_runtime_fallback_audit_post_fused_recovery.log`
(`10 passed, 4 deselected`). `tests/test_multichip_decoder.py` also keeps a
source audit for the measured runtime functions.

Watcher evidence is in `logs/final_watcher_correctness_disable_eth.log`
(`10 passed, 4 deselected`) and `watcher/final/generated/watcher/`. The watcher
run used `TT_METAL_WATCHER=10` and `TT_METAL_WATCHER_DISABLE_ETH=1`; this is the
same scoped active-Ethernet watcher limitation recorded by the completed
multichip decoder stage on this p300c host. `logs/final_watcher_failure_scan`
only matched pytest timeout text and normal Fabric messages, and
`logs/final_watcher_failure_scan_filtered.log` records no watcher failure
patterns after filtering.

## Performance

Before is the completed multichip decoder default from pre-stage code
(`num_links=1`). After is the final no-env default path (`num_links=2`).
Warmed decode rows are traced. `tt-perf-report` tables were regenerated from
Blackhole-normalized raw Tracy CSVs with `DEVICE ARCH=blackhole` and
`AVAILABLE WORKER CORE COUNT=110`.

| Multichip window | Before profiled wall ms | After profiled wall ms | Before device us | After device us | Before CCL us | After CCL us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear prefill seq 5 | 22.126 | 25.747 | 6234.459 | 6192.940 | 269.703 | 228.146 |
| full prefill seq 33 | 39.712 | 39.202 | 19200.155 | 18951.233 | 1535.556 | 1318.758 |
| linear traced decode | 1.400 | 1.346 | 1165.495 | 1152.415 | 89.982 | 75.970 |
| full traced decode | 1.203 | 1.152 | 982.739 | 968.534 | 93.984 | 77.437 |

Final default no-Tracy screen values are in `logs/final_perf_screen.log`:

| Multichip window | Final default screen wall ms |
| --- | ---: |
| linear prefill seq 5 | 20.733 |
| full prefill seq 33 | 32.504 |
| linear traced decode | 1.281 |
| full traced decode | 1.096 |

The no-Tracy one-link reproduction is in
`logs/baseline_num_links_1_perf_screen.log`: `19.408`, `31.912`, `1.327`, and
`1.067` ms for the same four multichip windows. Small wall screens are noisy;
the selected default is based on the final Tracy device tables showing lower
total device time and lower CCL time in every multichip window. The final
screen values above are still the headline final default path numbers.

Performance accounting for the final default run:

| Decode window | Modeled DRAM roofline from report | Device time us | End-to-end traced wall ms |
| --- | ---: | ---: | ---: |
| linear traced decode | 5.2%, 27 GB/s | 1152.415 | 1.346 profiled / 1.281 screen |
| full traced decode | 4.4%, 23 GB/s | 968.534 | 1.152 profiled / 1.096 screen |

## Candidate Results

| Candidate family | Result |
| --- | --- |
| `num_links=2` public all-reduce | accepted; CCL time dropped in all four multichip profiled windows and became the final default |
| explicit async reduce-scatter plus all-gather | correctness matched baseline, but screen perf was slower or tied: `19.619`, `34.567`, `1.540`, `1.095` ms; rejected |
| BF8 CCL payload | failed accepted-baseline PCC despite passing broad 0.995; rejected before perf |
| lower-movement residual layout | a stack-compatible width-sharded residual probe ran through real RMSNorm, real attention/linear mixer, residual add, real MoE/MLP boundary, and final residual; it was slower for both meaningful layer kinds (`3.455 -> 4.485 ms` linear, `2.556 -> 3.239 ms` full) and kept needing DRAM/interleaved mixer/MLP restores, so replicated DRAM remains selected |
| DRAM-sharded decode matmuls | sharded-only full qkgv was faster, but whole-contract variants with current-boundary conversion/restore or linear padding/slicing were slower; stack-compatible residual probing showed the lower-movement family does not recover those costs |
| fused matmul-CCL | `all_gather_minimal_matmul_async` exposes `cluster_axis`; after adapting output-sharded weights, legal 2-link TP axis 1 topology, and the worker grouping rule (`num_workers_per_link=4`), both persistent and non-persistent fused AGMM probes hung in fabric/router state and required triage/reset; rejected with adapted runtime evidence |
| persistent CCL buffers | public `ttnn.all_reduce` exposes no persistent output/intermediate buffer API; the buffer-bearing explicit RS/AG probe ran on TP and EP axes with preallocated DRAM buffers and failed public all-reduce correctness (`~0.949` PCC), while the model-level nonpersistent explicit RS/AG path was already slower than public all-reduce |
| packed vs separate projections | kept inherited packed projections; no repeated same-input projection matmuls remain to remove |
| inherited weight precision/fidelity | kept optimized-decoder policy; this multichip pass changed only CCL/topology defaults. Final rows still show BF16 activations, BFP8 dense/shared weights, BFP8 or BFP4 routed MoE per layer kind, and LoFi sparse expert rows as inherited from `doc/optimized_decoder/` |

Detailed evidence and rejected-option reasoning are in
`operation_topology_audit.md` and `work_log.md`. No stage-local multichip
optimization from the prompt or `$optimize` checklist is deferred.

## Artifacts

- Implementation: `tt/multichip_decoder.py`
- Tests: `tests/test_multichip_decoder.py`
- Operation audit: `operation_topology_audit.md`
- Work log: `work_log.md`
- Final checks: `logs/final_correctness.log`, `logs/final_correctness_post_fused_recovery.log`, `logs/final_runtime_fallback_audit.log`, `logs/final_runtime_fallback_audit_post_fused_recovery.log`, `logs/final_watcher_correctness_disable_eth.log`
- Perf logs: `logs/tracy_perf_baseline_summary.log`, `logs/tracy_perf_final_summary.log`, `logs/final_perf_screen.log`
- Source and CCL probe audits: `logs/fused_ccl_api_source_audit.log`, `logs/candidate_fused_agmm_bf16_nonpersistent_probe.log`, `logs/candidate_persistent_rsag_probe.log`
- Summary CSVs: `logs/perf_report_family_summary.csv`, `logs/perf_report_top_ops.csv`, `logs/perf_wall_summary.csv`, `logs/perf_screen_wall_summary.csv`
- Human and CSV report tables: `tracy/baseline_reports/*_perf_report.{txt,csv}` and `tracy/final_reports/*_perf_report.{txt,csv}`
- Raw report input provenance: `tracy/baseline_raw/reports/2026_08_19_06_31_27/*.csv.gz.parts/` and `tracy/final_raw/reports/2026_08_19_07_11_25/*.csv.gz.parts/`
- Watcher output: `watcher/final/generated/watcher/watcher.log.gz`, `watcher/final/generated/watcher/kernel_names.txt.gz`, and compact inspector YAML under `watcher/final/generated/inspector/`
- Device health: `logs/tt_smi_initial.log`, `logs/mesh_smoke_initial.log`, `logs/tt_smi_post_watcher.log`, `logs/tt_smi_final_post_recovery_gates.log`

The multi-GB profiler internals under `tracy/*_raw/.logs/` and
`profile_log_device.csv` are transient byproducts. The committed provenance is
the compressed raw op CSV parts, compact report tables, summaries, logs, and
compressed watcher output.
