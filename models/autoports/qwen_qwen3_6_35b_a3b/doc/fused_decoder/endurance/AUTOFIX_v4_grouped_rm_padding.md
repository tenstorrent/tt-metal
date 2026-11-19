# MoE graph audit and padded native RM down projection

Prepared candidate `v4_grouped_rm_padding.py.snapshot` is based on compact serial `v4_serial_l1_compact.py.snapshot` (`deb1fa8f…`), with SHA256 `d2b10cbcd146068bbb43a441e6a51e8c1b859e179dc8463930afd829fed3110b`. Only grouped-down setup packing and its prefill output adapter change. Production, prior probes and native sources remain untouched. The parent subsequently completed the exact device control below; complete-MoE timings are mixed and no selection is claimed.

## Measured bottleneck and existing blockers

The accurate qualified-v3 profile, not a new serial-candidate profile, attributes 6.488 ms of linear S256 prefill and 6.309 ms of full-attention S256 prefill to the eight grouped expert matmuls. Full-attention down matmuls account for 1.948 ms; the four immediately following untilizes cost 66.614, 67.821, 78.615 and 82.341 microseconds (295.391 total). This is a concrete adapter target. Decode indexed sparse matmuls total 173.317 microseconds linear and 173.029 microseconds full attention. Those source-bound observations do not establish current serial-prefill operation timings.

The current graph already folds expert SiLU and shared scalar sigmoid into binary multiply input activations. Sparse gate/up output is packed; uniformly activating its entire output would also activate the up half. Splitting gate/up to apply native SiLU was previously measured and rejected (`matmul_activation_0.json`). Native grouped-matmul validation only accepts GELU_TANH activation, not this model's SiLU (`grouped_matmul_device_operation.cpp:126–133`). A fused activation there therefore has a concrete semantic/API blocker.

Indexed sparse matmul accepts a per-group additive bias but has no per-token expert-scale operand (`operations/matmul/matmul.hpp:102–120`, `SparseMatmulParams`). Folding routing weights into inputs would move scaling before the BF16 matmul-output boundary and is not the existing expression. The shared expert's dynamic sigmoid scale likewise cannot become a constant matmul postactivation; both its scalar value and the BF16 down-projection boundary are meaningful. No safe new scale fusion is established. Existing reshape-only wrappers do not themselves imply removable kernels; actual consumer shapes/batch axes must remain consistent. DRISC prefetch and low-precision native fused MoE blockers remain as previously documented; this audit does not propose a broader datatype/kernel optimization.

## New adaptation of the previously invalid RM output

Previously, directly setting grouped-down `output_layout=ROW_MAJOR` was blocked by uneven N distribution over the device grid. Native `grouped_matmul_device_operation.cpp:146–160` computes columns from the whole grid, then requires the last column width equal to every other column for RM output. With grid.x11, N2048 is64 tiles, ten columns of6 plus a final4. There is no public core-grid override in this API.

The candidate pads each expert's BF16 down matrix with64 zero output columns, making N2112 or66 tiles. Eleven columns now each own6 tiles, satisfying the native RM contract. It requests native RM output and crops the RM width to2048 immediately before unchanged `routed_combine`. Gate/up matmul, SiLU/multiply, routing metadata, shared expert and final additions remain unchanged. The padded columns are never consumed. The setup addition is `256 experts × 512 K × 64 columns × 2 bytes = 16,777,216 bytes`, exactly16 MiB; arithmetic and temporary output width grow3.125% for this down projection.

Native compute `device/kernels/compute/grouped_matmul.cpp:52–95` uses `pack_untilize_dest` on the RM branch instead of TILE `pack_block`. Its writer emits each row segment directly into the RM output. The new graph removes four separate untilizes, replacing them with four RM width crops; it does not treat either pack-untilize or cropping as free.

A source detail corrects the initial quick hypothesis: native maximum subblock width is4. With original per-column widths6 and4, selection yields subblock2; with all widths6, it yields subblock3. K traversal remains in increasing order, but subblock scheduling and packing differ. **Matching shape algebra or K order is not a proof of bitwise arithmetic equivalence.** The candidate must pass actual consumed-output comparisons, including all four bank outputs and the final MoE. The previous statement that both use subblock2 was incorrect and is superseded here.

## Complete actual-MoE probe

`tests/v4_grouped_rm_padding_probe.py` loads actual checkpoint weights for all256 experts, four64-expert banks, router and shared expert. It invokes each snapshot's complete actual `_moe` method on seeded BF16 hidden rows. This is an operator diagnostic, not a replacement HF model reference. Default lengths33/128/129/256/4097 cover non-aligned inputs and both native L1 (`S<=256`) and DRAM grouped-temporary branches. The selected grid.x11 is checked explicitly.

Each comparison captures and checks every bank's `routed_combine` output, complete final MoE output, and unchanged input, all bitwise. Native grouped output padding rows can be unwritten by contract; the probe correctly compares consumed combine outputs rather than arbitrary allocation contents. Both final output layout/shape and the complete actual operation sequence are preserved by the snapshots. Complete eager and traced timings include every metadata operation, dispatch, gate/up/down matmul, activation, RM crop or untilize, combine, shared-expert path and final add. First/last replay and post-eager outputs are checked again. No per-bank speed estimates are added to infer a full-path gain.

The comparison fixture holds shared gate/up weights plus both baseline and padded down copies, so its residency exceeds a single selected implementation; it does not prove final context capacity. A selected candidate would replace the old grouped-down copy and incur only the recorded16 MiB delta. CPU-only preparation reads checkpoint metadata and verifies source scope, without loading the large expert tensors or importing TTNN. `v4_grouped_rm_padding_cpu_v2.json` passes with explicit final shape/dtype/layout/DRAM assertions in the probe; the first CPU preparation artifact is preserved. Device execution loads bank-sized CPU chunks, records provenance, runs only physical chip0, preserves output files and closes in `finally`.

Parent-only command:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_grouped_rm_padding_probe
```

If exact and faster, test source-bound whole linear/full decoder prefill and unchanged decode, then normal stress, watcher and full-capacity gates before retention. The candidate is prefill-only; decode timing variation would not establish a new decode optimization. Stop Rule and stage closure remain open.

## Completed parent device control: exact with mixed performance

[v4_grouped_rm_padding_probe.json](v4_grouped_rm_padding_probe.json) has SHA256 `848fc61c280d3a93f8a9fe02433a3264d99285939a309aa42408261efb9a05b7`. All five lengths pass exact comparisons for each of the four consumed bank outputs, final MoE output, input immutability, shape/dtype/layout, and first/last replay outputs. The device closed successfully. No thresholds or oracle values changed.

| Rows | Traced baseline ms | Traced padded RM ms | Eager baseline ms | Eager padded RM ms |
|---|---:|---:|---:|---:|
| 33 | 5.943334 | 5.939249 | 6.095995 | 6.262764 |
| 128 | 8.204490 | 8.160717 | 8.315259 | 8.254593 |
| 129 | 8.308726 | 8.295007 | 8.485814 | 8.583141 |
| 256 | 8.822933 | 8.754535 | 8.917319 | 8.870089 |
| 4097 | 26.941862 | 27.110195 | 26.958534 | 27.084836 |

Traced short-case gains are only about0.07–0.78%; eager results regress at33 and129 while improving at128/256. At4097 both regimes regress (traced26.941862 →27.110195 ms, roughly0.62%). These small and mixed effects may include measurement variation; they do not justify declaring a general performance improvement. Whole-decoder S256/S4097 controls are parent-owned and remain the retention criterion. Source inspection did not substitute for the now-completed exact device checks.

## Possible adaptations without a second full weight copy

No adaptation is prepared in this follow-up. The current native API infers N from packed weight dimensions and expert count; it has no separate logical-N, per-expert storage stride or core-grid override. The N2112 K-major expert packing cannot simply be viewed as N2048: each subsequent expert starts64 columns later. Keeping both complete formats would add another roughly512–528 MiB copy, not just the16 MiB padding delta. Runtime cropping/repacking full weights would introduce large traffic and is not a free view.

One source-supported specialization could avoid the long branch without duplicate formats: apply padded native RM only to full-attention layers, while linear layers retain the original N2048 TILE weights and output. `prepare_prefill` clamps full-attention physical chunks to256, so that public path reaches the shorter L1 MoE branch; linear planning permits4096-token chunks and reaches the large DRAM branch. Each instance would still hold one grouped-down format. This gives up the small linear short-case gains, and remains contingent on an actual full-attention decoder win and16 MiB capacity validation. It is a proposed specialization, not a measured selection.

Alternatively, switching only output_layout to TILE above256 while retaining the same padded N2112 weights is native-legal and needs no extra weights. It still performs the extra columns and requires the RM conversion/crop, so it cannot promise baseline performance or eliminate the observed regression. Native subblock/layout scheduling and output traffic must be measured. Restricting generic chunk sizes solely to hide this result would also affect other operators and is not proposed.
