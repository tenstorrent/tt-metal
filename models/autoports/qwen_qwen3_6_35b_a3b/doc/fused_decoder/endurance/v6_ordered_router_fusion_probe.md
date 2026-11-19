# Ordered router terminal scan fusion: source-only proposal

The candidate in `tests/v6_ordered_router_fusion_probe.py` removes the unused cumsum prefix tensor and two row crops from the ordered router projection. It retains the existing native SFPU default cumsum arithmetic and cross-tile carry. No sequential CPU sum, alternate reduction tree, FMA substitution, precision relaxation, shared/native edit, production edit, or decoder integration is involved.

Source SHA256: `067595f30f1b7e29c65795748dddd9b594725d40190c8541c0863a9cf8dd5314`.

The source snapshot is `v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot`. The test extracts its already-selected `linear(... ordered_router=True)` branch. The candidate replaces exactly six cumsum/crop statements with `partial = self.terminal_cumsum(products); del products`. Reversing this substitution restores the complete selected branch AST exactly. Weight preparation, pair permutation, activation gather/layout transformations, FP32 products, four-block `fast_reduce_nc`, requested dtype cast, and output placement remain identical.

The compute kernel calls the same `copy_init`, `copy_tile`, `cumsum_tile_init<CumsumMode::Default>`, and `cumsum_tile<CumsumMode::Default>(0, first)` sequence as the native implementation. It uses direct FP32 unpack, FP32 destination accumulation, and double-buffered destination synchronization. Each of 32 cores scans one `(block, width_tile)` run of 16 height tiles in order. Only the last tile is packed. The writer copies its row31 FP32 words into row0 of a zero-filled tile without floating-point arithmetic. The resulting logical `(4,1,1,256)` tensor feeds the original block reduction.

The full prefix output is 2,097,152 bytes; the candidate emits 131,072 bytes of padded terminal-row tiles. Products still occupy 2,097,152 bytes. Each active core uses 16 KiB of local circular buffers. These are source-derived sizes, not measured speedups. The generic-op descriptor construction, 32-core distribution, scalar zero-fill/copy, and removal of native prefix writes may have competing costs. Only parent-owned device measurements can establish a benefit. No roofline or final-decoder performance claim is made.

## Evidence and commands

CPU/source setup was run without importing TTNN or opening a device:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_ordered_router_fusion_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_ordered_router_fusion_cpu_setup_v2.json
```

Result: `CPU_SOURCE_SETUP_PASS 12`. Inputs include saved actual positions 227, 738, 5799; eight seeded randomized BF16 vectors spanning scales 1/4 to 4; and zeros. Checkpoint weights are the actual layer34 gate weights. The setup checks raw physical row mapping, exact input-tile visitation, and AST restoration. It freezes reader, compute, writer, baseline branch, candidate branch, and product-only branch sources beside the JSON and saves all input/weight fixtures. Dependency/capture hashes are in the JSON.

The earlier `v6_ordered_router_fusion_cpu_setup.json*` artifacts retain the first source-only setup, before adding the baseline object's missing `mesh_device` field and the runtime host-operation audit. The v2 setup corresponds to the frozen source hash above. Neither setup emulates native SFPU arithmetic or constitutes device correctness evidence.

Parent-only proposed component run:

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_ordered_router_fusion_probe.py \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_ordered_router_fusion_device.json
```

The device path requires bit-identical FP32 terminal partials and bit-identical complete FP32 and BF16 router outputs for every case. It compares against the actual native cumsum/crop branch, not a CPU reduction model. The projection calls run under `runtime_audit.device_only`, rejecting Torch arithmetic and host/device conversions inside forward. Input upload and result download occur outside the measured path. Timing warms each complete projection three times, captures it separately, measures ten groups of ten replays, and checks the final output bits after all 100 replays. Output JSON includes per-case differences, timings, exceptions, and device-close status; tensor artifacts preserve both variants.

No device execution or C++ JIT compilation was performed by this subagent. The parent must validate compilation, native carry preservation when intermediate packing is skipped, exact output bits, watcher cleanliness, traced replay, and measured component/full-decoder performance before integration. A component pass does not replace the full decoder gates. This proposal changes only a fixed 2048-input router reduction; it introduces no public sequence-length constraint. Generic Q/Z or attention-output reuse is a hand-off idea and is not implemented here.
