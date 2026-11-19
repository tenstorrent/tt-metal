# Terminal row assembly selection

Selected 2026-11-14 after isolated clean and allocation-tracked hardware proof:
`QwenGenerator._decode_body` uses native in-place BF16 fill of the existing
`sample_logits` to -100, followed by the existing `write_token_chunk` for each
active row, through24 active slots. Above24 it retains the original
logical-row concat/copy. The threshold is static physical geometry, not an
exception fallback. Preparation already warms every supported decode width.

The fill path leaves the persistent output address, TP4 vocabulary partition,
BF16 TILE DRAM layout, inactive-row values, and common split-greedy sampling
unchanged. It adds no persistent DRAM tensor. `full_like(...,
optional_tensor=sample_logits)` dispatches BF16 TILE DEVICE input to native
`ttnn.fill` in `creation.cpp`; this branch performs no host tensor fill/upload.
The existing tiled writer copies valid BF16 row bytes directly to their physical
slot. No decoder arithmetic, fidelity, cache, residual layout, or CCL policy is
changed.

Evidence: [terminal_rows_v2.json](terminal_rows_v2.json) and
[terminal_rows_v2.log](terminal_rows_v2.log) are clean no-model timings;
[terminal_rows_tracked_v1.json](terminal_rows_tracked_v1.json) and
[terminal_rows_tracked_v1.log](terminal_rows_tracked_v1.log) are separate tracked
correctness, with no timings. Both passed30 traces/39 prepared programs and12
case selections (nine distinct physical cases plus revisits). All inputs,
outputs, poison tensors, sampler buffers and programs were prepared before any
capture. Post-capture reads finish with another replay. Program count stays
fixed. The tracking run includes program-cache allocations in validation.

Every correctness replay begins after copying a preallocated NaN/Inf poison
tensor into the destination. All248320 vocabulary entries for each of32 rows
then equal the original CPU arrangement exactly, including inactive -100 rows.
Both common sampler replicas agree across all ranks and select a global maximum;
returned IDs equal the original concat strategy. Cases include slot31 alone,
sparse/reordered B2, B8/16/24/28/32, and small active sets afterB32.

| Active slots | Original concat (ms) | Native fill + tiled rows (ms) |
|---|---:|---:|
| b1_slot0 | 0.537668 | 0.047810 |
| b2_mixed | 0.537574 | 0.067042 |
| b8 | 0.537487 | 0.182429 |
| b16 | 0.537544 | 0.336477 |
| b24 | 0.537737 | 0.490270 |
| b28 | 0.537589 | 0.567541 |
| b32 | 0.537739 | 0.644717 |

Measurements are30 warmed trace submissions with boundary synchronization;
all row assembly work is included. Template-copy plus tiled writes was also
correct and within about a microsecond of native fill, but would need another
physical32 local-vocabulary buffer:3,973,120 tensor bytes, or3,981,312 bytes/rank
with eight-bank tile allocation padding. Native fill avoids
that allocation. Row writes lose to concat at28/32, so those shapes retain
concat. Widths25..27 conservatively retain concat as well; they were not used to
infer an unmeasured crossover.

The initial profile attributed473.651us to32 post-head untilizes,78.191us to
concat,28.452us to tilize and28.039us to copy. This isolated candidate removes
that repeated layout work for small active sets. These eager per-op sums and
isolated component timings are not full-model token throughput. Fresh integrated
all48 timing/accuracy remains required after this source selection.

CPU actual-method validation adds exact physical-row and nonfinite-clearing
checks at widths1,2,24,25,32, sparse/reordered slots and B32-to-small transitions.
Together with trace selection and input validation,21 CPU tests pass:

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_trace_signatures_cpu models.autoports.qwen_qwen3_8_flash_next.tests.test_decode_validation_cpu models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_validation_cpu -q
```

Reproduce the isolated benchmark in separate clean/tracked processes, preserving
new artifact names:

```bash
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto timeout 240 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly --fill-candidate --output <clean.json>
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto timeout 240 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly --fill-candidate --tracked-correctness --output <tracked.json>
```

Watcher must be absent. Profiler/perf counters are off in both regimes; allocation
tracking and diagnostics are off for clean timing. CLI preflight checks these
conditions. LM-head DRAM sharding is a separate selection documented below, not part of
this assembly change.


## LM-head DRAM projection selection

Selected after isolated clean and tracked validation: eight activation storage
cores, three readers per DRAM bank, K block1, BF16 weights/activations/output,
HiFi4, accurate math, FP32 DEST and packer accumulation **off**, exactly matching
the prior full-model head's compute flags. The accepted decoder projection
helper normally defaults packer accumulation on; the LM-head constructor
explicitly replaces that helper config with the existing `loader.compute`.
Decoder projection policies are unchanged.

Artifacts: [lm_head_dram_v1.json](lm_head_dram_v1.json),
[lm_head_dram_v1.log](lm_head_dram_v1.log),
[lm_head_dram_tracked_v2.json](lm_head_dram_tracked_v2.json), and
[lm_head_dram_tracked_v2.log](lm_head_dram_tracked_v2.log). Both completed and
passed. The earlier tracked invocation without `--tracked-correctness` was
rejected by clean preflight before device access and is not passing evidence.

| Geometry / trace window | Prior interleaved head | DRAM block1 | DRAM block2 |
|---|---:|---:|---:|
| Q1 projection incl. layouts/copy |0.868638 ms|0.718163 ms|0.716828 ms|
| Q1 projection + common split greedy |1.347678 ms|1.193434 ms|1.190575 ms|
| Q32 projection incl. layouts/copy |0.847203 ms|0.693952 ms|0.693952 ms|
| Q32 projection + common split greedy |1.326696 ms|1.170319 ms|1.170938 ms|

Block1 and block2 are effectively tied in this short component measurement;
select block1 for the smaller K block and simpler L1 requirement. Block2 is
recorded as rejected without claiming a statistically significant latency loss.
For block1, minimum PCC versus the original TT head is0.9999999977 atQ1 and
0.9999999951 atQ32. Top1/top5/top100 membership is100% and common greedy token
IDs agree at every tested row. Max absolute differences are0.0078125 and0.015625;
outputs are not bitwise identical. Inputs were seeded normalized BF16 synthetic
terminal activations with the real checkpoint head. These are isolated
TT-versus-TT numerical results, not HF accuracy or full-model acceptance.

Production `QwenModel` now constructs the selected prepared weight directly
from HF vocabulary partitions. It never allocates the original interleaved TT
head, so there is no duplicate resident head or transient native+prepared TT
weight overlap. Identical decode/rows policies share one prepared group, and
`PhaseDramProjection` preserves internal chunking for larger row counts. The
public output remains interleaved BF16 per-rank vocabulary logits. Input
resharding, output restoration, valid-vocabulary slicing, physical32 padding
and persistent copy were included in the isolated timing windows.

Per-rank persistent weight arithmetic: original317,849,600 bytes; selected
318,504,960 bytes; delta655,360 bytes due to per-bank padding. The probe's
`weight_memory` field deliberately includes its co-resident reference weight;
that636,354,560-byte comparison footprint is **not** the production footprint.
The final context/capacity accounting must include the655,360-byte delta and
validate real allocation headroom, without reducing advertised context.

CPU tests AST-extract the actual constructor and logits method, covering tied
and untied HF heads: only token embedding calls the native uploader; the head
partitions enter one prepared group; compute flags are shared with the prior
head config; logits call the prepared head after the final mixer. Both pass:

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_lm_head_setup_cpu -v
```

No hardware was run by the source-editing agent. The device owner ran the
isolated artifacts. Fresh full48 HF accuracy, resume, traced replay and final
clean performance remain required for the integrated model.
