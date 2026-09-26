# Direct router-score projection

The test-only `tests/probe_optimized_router_direct.py` isolates native linear
projection of router scores. Its copied `GeneralizedRouter.__call__` was
checked by AST comparison: only the broadcast product/reduction/transpose
was replaced. Normalization, learned scale, scalar multiplication, optional
score centering, BF16 score conversion, native generalized gate, scatter,
per-expert scale, and prefill delegation remain identical.

The implementation proposal is `router_direct.patch`; this agent prepared
it without changing the live runtime. The parent owns application, formatting,
and final contract runs. The selected automatic policy is:

| Kind | Grid | K block | Fidelity |
| --- | --- | ---: | --- |
| Sliding | `(4,1)` | 22 | HiFi4 |
| Full | `(4,1)` | 44 | LoFi |

`router_direct="auto"` resolves to `bool(generalized_router)` after that
setting resolves. Explicit `router_direct=False` restores broadcast scores;
explicit `generalized_router=False` and lane-router controls disable automatic
direct projection. Explicit contradictory settings are rejected. Grid, K
block, and fidelity remain independently overridable. Runtime metadata records
the effective backend, dtypes, memory, program, grid, block, and fidelity.

The integration preserves the original BF16 weight object without reloading
or quantizing it. The old FP32 row transpose is retained for the broadcast
control. Although the unchanged prefill projection uses the BF16 matrix,
releasing the 1,441,792-byte router transpose is unnecessary for this bounded
performance change.

## Source contracts

The activation is scaled FP32 `[1,1,1,2816]`; the original router weight is
BF16 `[1,1,2816,128]` in tiled interleaved DRAM. Output remains FP32
`[1,1,1,128]` in interleaved L1 before centering and BF16 score conversion.
The direct compute config uses FP32 destination accumulation, no packer L1
accumulation, no approximate math, and no destination full-sync override.

The tile problem is M=1, K=88, N=4. The explicit multicast-1D config uses
`per_core_M=1`, `per_core_N=4/workers`, and a 1x`per_core_N` output
subblock/block, so grids `(1,1)`, `(2,1)`, `(4,1)`, and `(2,2)` are legal.
Subblocks contain at most four tiles for FP32 destination accumulation.
K blocks 1,2,4,8,11,22,44,88 divide tiled K; the native validation is in
`matmul_device_operation.cpp::validate_matmul_block_and_subblock_configuration`.
No lane decomposition, sum over partial rows, or BF16 activation cast occurs.

For grid `(4,1)`, the dense multicast factory's operand circular buffers plus
one FP32 output/intermediate tile are 268 KiB at K22 and 532 KiB at K44.
K88 also uses 532 KiB: batch=1 and one K block disable double buffering,
whereas K44 buffers two blocks. These estimates follow
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` and exclude other live
allocations. The same factory marks FP32 intermediate reloads
`UnpackToDestFp32`. FP32 storage and destination accumulation still do not
guarantee FP32 multiplier precision; actual routing must be validated.

The probe includes explicit API adaptations: automatic program selection,
interleaved DRAM input/output, or an FP32 storage cast of the same BF16 weight
values. The simplest conservative API retry is
`--direct-router-program auto --direct-router-input-memory dram --direct-router-output-memory dram`.
The explicit program already ran successfully in the recorded controls.
HiFi4, HiFi3, HiFi2 and LoFi controls are available in the probe; each requires
actual-input validation. Prior compensated-lane or Gaussian failures do not
establish the result for this direct FP32-input path.

## Actual-input evidence

The parent ran all hardware controls. These probe records use runtime SHA
`2a55a06716a405ff96f9f29a22c4dfb169bf4a56e279904a440e161eb5df41c9`, real
weights, recorded text-derived layer inputs, traced decode, and unchanged
0.995 PCC thresholds.

| Candidate | 4096/128 median traced host time | 1025/512 minimum decode PCC |
| --- | ---: | ---: |
| Sliding K22/HiFi4 | 988.225 us | 0.9955236050 |
| Full K44/LoFi | 1076.352 us | 0.9961629472 |

Headline records are `actual_router_direct_k22_hifi4_layer0.json` and
`actual_router_direct_k44_lofi_layer5.json`. Stress records are
`router_repair_hifi4_stress_layer0.json` and
`router_selected_stress_layer5.json`; both passed all 512 decode steps with
exact repeat equality. Full K44/LoFi also passed the recorded 2049-token
request-reuse failure window at PCC 0.9981852456 in
`router_selected_reuse_window_layer5.json`.

Sliding K44/HiFi2 passed the headline but failed actual 512-step stress at
position 1459, PCC 0.9943880922 (`router_selected_stress_layer0.json`). It
was rejected in favor of K22/HiFi4. The selected sliding stress minimum is
identical to the preceding broadcast control. These measurements support
the selected candidates; the integrated-default contract suite is separate.

Patch provenance: base SHA `2a55a06716a405ff96f9f29a22c4dfb169bf4a56e279904a440e161eb5df41c9`;
raw candidate SHA `e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`;
host-Black-formatted candidate SHA `57afb2c7622470bf11ab73c452c8614aaefdb3ae97b7827645856b4f62324918`.
CPU checks covered syntax, changed-code formatting, patch applicability,
legal/invalid geometry, weight aliasing, automatic policy and explicit opt-outs,
and exact original-call AST restoration when direct projection is disabled.
