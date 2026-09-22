# Profiler interpretation

Final measurements are summarized in `profiler_summary.json`. Each final
profile folder contains human-readable `device{rank}_{prefill,decode}_perf_report.txt`
and matching CSV tables. Per-device filtering preserves signposts and avoids
false gaps caused by merging unsynchronized device clocks. Uninstrumented
warmed timings in `performance_summary.json` are the latency claims; eager
Tracy host gaps are not substituted for them.

## Native projection geometry

The selected decode projections are BF16 activations × BFP4 weights, LoFi,
FP32 destination accumulation. The DRAM-sharded program uses eight native
weight readers/compute workers. The configured 10 or8 input cores describe
activation multicast ownership, not the total chip worker count (110).

The native DRAM program does not expose output-subblock fields in its Python
config, so those CSV cells are blank. This is not an untuned manual subblock:
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:169-191`
selects it using `get_matmul_subblock_params`, then pads width as needed.
At B1, per-core M=1 and FP32 destination accumulation limits the subblock to
four tiles. The selected projections use1×4 output subblocks. In particular,
output/down N=5120 gives160 N tiles /8 readers =20 tiles, divisible by4.
Linear attention and packed gate/up pad the per-reader compute widths to20
and36 tiles respectively; logical outputs still have4160 and8704 columns.

The output projection was swept at the selected BFP4/LoFi policy with input
cores/block K tiles24/2 (control),12/4,8/6, and4/12. The whole-layer medians
were0.5540,0.5292,0.5234, and0.5272ms in the same historical family; see
`tune_control_l0.json` and `tune_output{4,6,12}_l0.json`. The final choice8/6
therefore has measured support even if its small matmul remains tagged `SLOW`
by the utilization heuristic. A wider native weight-reader scheme hits the
unit-mesh assertion documented in `AUTODEBUG_dram_mesh.md`. All-core minimal
matmul and adapted AGMM/MMRS families were measured slower. The down34/4
candidate hit the exact L1 CB overlap documented in the work log.

## Collective and movement controls

Decode uses two direct all-reduces per layer, with a shared L1 workspace and
two Ring links. Prefill uses RS/AG after each row projection. The RS CSV's
first `UINT8` output is its opaque workspace; the semantic reduction result
is the BF16 second output. Fused AGMM/MMRS and hidden-sharded residuals were
measured through their consuming distributed norms, with packed MLP and
BF16/BFP8 payload controls. The coherent direct-AR BFP8 control also passes
PCC but loses both paired timings; see `direct_ar_dtype_comparison.json`.

The batched public DRAM boundary fixes a physical TILE-padding lifetime issue,
not a host fallback. The shape remains replicated `[B,1,5120]`; B1 remains
compact L1. `AUTOFIX_stacked_batch_layout.md` records the failing stacked B32
allocation and the corrected 100-step control.

## Final measured rows

Device0 summaries below are microseconds. Per-rank CSVs retain all four chips.
Each decode contains exactly two direct all-reduces; profiler rows confirm
BF16×BFP4/LoFi in all four dominant projections.

| Layer / phase | Kernels | Gaps | Matmul | CCL | Data movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear / prefill | 1017.367 | 1452.637 | 416.132 | 121.931 | 46.718 |
| Linear / decode | 425.601 | 49.185 | 188.322 | 31.583 | 82.841 |
| Full / prefill | 870.440 | 1312.183 | 409.743 | 118.602 | 51.684 |
| Full / decode | 315.055 | 40.875 | 180.946 | 31.641 | 39.348 |

Decode QKV takes44.081us (linear) or36.458us (full), packed gate/up82.197/82.137us,
output18.751/19.024us, and down43.293/43.327us. Reported DRAM utilization is
47–49% for QKV,53% for gate/up,50% for down, and40–41% for output; compute
utilization is70–73%,78%,74%, and60–61% respectively. These percentages are
tt-perf-report's architecture-based estimates, not independent bandwidth probes.
The small output matmul remains `SLOW`; the precision-matched input geometry
sweep above selected8/6 and earned rejection of wider/alternative paths.

Direct AR totals31.6us/layer versus roughly49–51us for the prior RS/AG decode
family. Linear attention retains more movement (82.8us versus39.3us) because
of convolution, head rearrangement and recurrent-update packing. Adapted
sharded residual and fused CCL/matmul measurements include those consuming
boundaries and were slower.

Prefill QKV takes122.670/116.591us, packed gate/up146.430/146.026us, and minimal
output/down44.5/102.6us. All use BFP4/LoFi. Prefill 2D/L1-placement
controls produced dispatch-variable, overlapping timing ranges; the selected
baseline prefill kernels are retained. The1.3–1.45ms instrumented eager gaps
are why the final20-repeat uninstrumented timings are the reported latency.

## Non-failing log diagnostics

- Motherboard discovery uses PCI bus IDs as tray IDs for this unregistered
  motherboard. Both baseline and TP4 open successfully; this is topology
  metadata fallback, not host model computation.
- Unit-mesh baseline open warns about remote UMD traffic. The baseline uses
  one MMIO device and no remote collective; reference transfers are outside
  timing. Device trace rows independently support the baseline latency.
- Tracy's optional WASM viewer copy looks in the default profiler directory
  despite this run's custom output directory (`tools/tracy/__main__.py:446-474`).
  Actual `.tracy` captures exist in each timestamped report directory and are
  archived with hashes. CSV enrichment and all requested tables complete.
- Pandas reports mixed metadata column types during enrichment. Required
  operation/duration/dtype/fidelity rows are present and inspected. No final
  run reports dropped markers, TT_FATAL, or failed enrichment. Compiler command
  lines containing `-Werror` are compiler options, not runtime errors.
