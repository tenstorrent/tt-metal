# Exact-order chunk64 triangular recurrence fusion

The next prefill candidate replaces the 63-row triangular recurrence inside the frozen `8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa` chunk64 graph with one device operation. Initial Gram/decay construction, addition of the identity, corrected matmuls, state updates, and output calculation remain outside this component. No decoder implementation, frozen helper, snapshot, context contract, or quality gate was changed.

`v6_triangular_recurrence_fusion_cpu.json` counts 8,720 tensor API calls per chunk in the existing loop: 503 zeros, 3,906 slices, 4,025 additions, 160 concatenations, 63 multiplies, and 63 transposes. These are source counts, **not profiler dispatch counts**; structural calls can be no-ops. The candidate keeps the 64×64 matrix local while updating rows, eliminating repeated tensor construction and matrix concatenation. Latency benefit is unmeasured. Existing corrected-matmul fusion addresses another part of the graph and does not eliminate this loop.

## Arithmetic evidence

The prior `v6_triangular_sum_order_probe` establishes the exact width-dependent reduction schedule. This candidate retains that schedule, including additions to positive zero. For each row, all old coefficients are copied before any column is updated. Each FP32 product rounds before the selected partial accumulator is added. Each addition rounds separately. Replacing the recurrence with a native triangular solve, reassociating sums, or introducing FMA is out of scope.

`tests/v6_triangular_recurrence_fusion_cpu.py` compiles a portable scalar C++ reference with `-ffp-contract=off -fno-fast-math`. It matches the proven canonical CPU sum at rows 1, 2, 31, and 63, and at the final matrix, for seven actual chunk fixtures (zero/nonzero incoming state; logical lengths 128, 127, 65, and 1), eight random matrices including mixed magnitudes, and signed zeros. Ordinary CPU arithmetic differs from the saved native device matrices near underflow: 1,233 final entries across the seven fixtures. For the first S128 chunk, the maximum absolute difference is 5.493376966078777e-38; propagation reaches entries of magnitude 2.32e-34.

`tests/v6_triangular_recurrence_ftz_cpu.py` independently controls subnormal input and result flushing. `v6_triangular_recurrence_ftz_cpu.json` records all controls:

| Input flush | Result flush | Preserve zero sign | Final differing entries | All saved row boundaries exact |
|---|---|---|---:|---|
| No | No | Yes | 1,233 | No |
| No | Yes | No | 5 | No |
| No | Yes | Yes | 0 | Yes |
| Yes | No | No | 16 | No |
| Yes | No | Yes | 16 | No |
| Yes | Yes | No | 5 | No |
| Yes | Yes | Yes | 0 | Yes |

Thus signed result flushing after **every product and addition** reproduces all four saved row boundaries for all seven actual fixtures. This bounded evidence does not distinguish result-only flushing from input-plus-result flushing, and does not establish arbitrary native DAZ behavior. The device comparison remains the primary gate.

## Isolated component

`tests/v6_triangular_recurrence_fusion_probe.py`, SHA256 `8229068839e3e06750353eb66fe720c45445e197b981d982dba1328ab5de4853`, is frozen for the first parent-owned device test. `v6_triangular_recurrence_fusion_setup.json` records source checks and exact source snapshots. No hardware was accessed by this agent.

The reader fetches four FP32 tiles per head into a mutable 16 KiB local matrix. One worker owns one head. Before each row update it preserves the original 64 raw coefficient words. It streams coefficient and previous-row operands to compute, receives each updated row segment, and changes only valid strict-lower entries. Rows 4–7 use separate prefix/suffix passes because their reduction schedules differ within one physical tile. Final output is the same interleaved DRAM TILE FP32 shape. There is no persistent L1 allocation or sequence-length-dependent scratch; temporary CB storage is 40 KiB/core plus a small scalar row copy.

Compute retains four partial accumulator tiles, two operand tiles, the old row, and the result in the eight FP32 DST tiles. It explicitly stores the product before adding it. Its signed-result flush maps subnormals to signed zero by raw-bit masking after each arithmetic step. No host floating-point calculation occurs in the device path. The same frozen native loop and `triangular_sum` AST are extracted as the comparison implementation, with only the loop stop bound parameterized. A CPU adapter verified the extracted native loop against the canonical reference. The source-only checks cover all 63 row ranges and all 4,096 physical tile offsets.

First hardware command (parent owns device scheduling):

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_triangular_recurrence_fusion_probe.py --run-device --heads 1 --cases actual0 --stops 1,2,31,63 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_triangular_recurrence_fusion_device.json
```

After the first actual fixture passes, use a fresh output path with `--heads 32 --cases actual0,actual1,random,zeros --stops 1,2,31,63`. Every output bit, including signed zero, must match the existing native path. Both paths execute inside the runtime host-computation/conversion guard. Cold wall times are diagnostic only; compilation and observer overhead make them unsuitable for speed claims. Initial compilation and synchronization are unverified until the parent runs the probe.

## Complete adapter verification before selection

1. Require the native component gates above, all actual row boundaries, and deterministic repeat results. Add explicit very small operands if evidence suggests the input-flush boundary needs broader coverage; do not infer it from the seven saved fixtures.
2. Produce a **new** whole chunk64 output adapter which replaces only this loop, with AST restoration against the selected corrected-matmul graph. Preserve the external identity addition and every other arithmetic/layout operation. Do not edit the selected decoder or existing snapshots.
3. Reuse the existing canonical operand harness with zero and nonzero incoming state and logical lengths 1, 31, 32, 33, 63, 64, 65, 127, and 128. Require unchanged FP32 output, FP32 final state, and BF16 output against the complete existing adapter; record all intermediate deltas rather than dismissing near-underflow discrepancies.
4. Measure warmed whole-adapter prefill latency and, where supported, traced repeated replay with observers disabled, including all input/output placement. Report actual device dispatch/op-count changes from profiler evidence. The component's local streaming/feedback loop could bottleneck; fewer host calls alone is not selection evidence.
5. Only after complete-adapter success, integrate into a new standalone decoder snapshot and rerun the existing representative layer, nonaligned length, long stream, watcher, determinism, and context-capacity gates. Decode remains structurally untouched, but fresh correctness/performance evidence is still required before promotion. The 40 KiB temporary allocation must be measured in the full prefill allocation context.

## Parent-owned device results and complete wrapper

The first component run passed every FP32 bit at rows 1, 2, 31, and 63 (`v6_triangular_recurrence_fusion_device.json`) and closed cleanly. The expanded run passed all **48 comparisons**: 32 heads; both actual chunk matrices, random matrices, and signed zeros; stopping rows 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, and 63 (`v6_triangular_recurrence_fusion_all_device.json`). Cold host times are not treated as performance evidence.

`tests/v6_chunk64_triangular_output_probe.py`, SHA256 `dc03fffc59ebf574c66eba0c7876ce4cb5178a59636485a8836faf856cff9022`, loads the frozen complete 8cda graph and uses its unchanged output method for both variants. The candidate subclass changes only the triangular loop in `state_call`; exact AST restoration is recorded. Original canonical CPU operand/oracle setup is reused. Every common intermediate and the BF16 output, FP32 output, and FP32 final state must match the native complete adapter bitwise. All three complete outputs must also match the canonical HF result bitwise. Row1/2/31 observer samples are omitted from the candidate whole adapter because the separate component run already covers them; recomputing those rows would distort timing. `A_row63` and all subsequent intermediates remain compared.

CPU setup passed lengths128/127/65/1 after pinned Black23.10.1, autoflake2.3.1, and isort5.13.2. The parent reports the S128 zero-state device gate passed all complete outputs and all common intermediates with zero differing bits (`v6_chunk64_triangular_output_zero_device.json`). Nonzero state and timing remain separate gates.

Complete adapter commands:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_chunk64_triangular_output_probe.py --run-device --lengths 128 --initial-state zero --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_triangular_output_zero_device.json
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_chunk64_triangular_output_probe.py --run-device --lengths 127 65 1 --initial-state prefix64 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_triangular_output_nonzero_device.json
```

A fresh timing output uses `--lengths 128 --timing --warmup 2 --iterations 3 --samples 3 --trace-replays 100`. Both complete adapters include the entire device output/state graph and input/output placement; only setup/uploads and readback are outside timing. Observer callbacks are disabled in measured calls. Trace capture reserves64MiB in this diagnostic. First/last traced outputs and warmed last output must match the native expected outputs in all three dtypes/roles.

## Standalone integration prepared, not yet selected

`tests/v6_build_triangular_snapshot.py` builds onto the selected frozen c71 decoder, SHA256 `c71ea01a4b38458f0f53808d288d9a39ca5563a13cb5e97f71e701391670c3f2`. It adds a nonallocating helper constructor, replaces only the state graph's triangular loop, and embeds the exact measured literal kernels/runtime class. The complete module AST restores exactly to c71 after removing the four new definitions and restoring the two changed methods. Independent loop-only restoration guards against changing other state equations. The generated snapshot passes pinned Black without changing its AST.

Candidate `v6_corrected_fma_full_v5_triangular_compact_silu_integration.py.snapshot` SHA256 is `85644b73bc1effb516c1fc7a5092cc443ef7d8a30a67a6635767a06f3eb7f858`. `v6_triangular_integration_build.json` records hashes and restoration evidence. Decode, MoE, full-attention/V5 paths, setup weights, context advertisement, and other math remain unchanged. The snapshot is a source candidate pending complete-adapter timing, whole-decoder correctness/performance, and capacity checks.

Nonzero incoming prefix64 state also passed logical lengths127/65/1: every shared intermediate and all complete outputs are bit-exact against the frozen native adapter and HF (`v6_chunk64_triangular_output_nonzero_device.json`, clean device close). `v6_triangular_adapter_transfer.json` verifies that the 8cda and c71 `Chunk64StateGraph`, `Chunk64OutputGraph`, `Chunk64MKLNegativeExp`, and `FusedCorrectedMatmul` class ASTs are identical, connecting the complete-adapter oracle to the selected integration base.

Read-only pinned formatting checks on856 found one inherited isort issue in `chunk64_mathematical_exp_table`: its three local imports needed reordering. Black and autoflake already passed. `tests/v6_format_triangular_snapshot.py` creates a **new** snapshot, leaving856 intact. `v6_triangular_formatted_integration_build.json` proves that reversing only this exact import permutation restores the entire AST to856; no import was removed or added and no arithmetic changed. The final formatted candidate is `v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot`, SHA256 `ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334`. All three pinned format checks pass on that candidate.

The first complete-adapter trace attempt exceeded the diagnostic64MiB reservation: the native graph required279,453,696bytes at end-capture. The device closed cleanly; failure evidence is retained in `v6_chunk64_triangular_output_timing_device.json/.log`. The parent reruns the **same frozen wrapper** with `--trace-region-size 536870912` and a fresh `v6_chunk64_triangular_output_timing_512m_device.json` path. This reservation belongs only to the complete-adapter timing experiment; it does not change the decoder's public context contract or normal trace configuration.

The512MiB complete-adapter timing rerun passed and closed cleanly (`v6_chunk64_triangular_output_timing_512m_device.json`). S128 warmed median fell from446.936410231ms to162.780442430ms. Three samples of100 traced replays fell from319.911764520ms to159.745997479ms. Warmed last outputs, traced first/last outputs, and replay determinism were exact for BF16 output, FP32 output, and FP32 state. These are complete **chunk64-adapter** measurements; the parent separately qualifies ab28 as a complete decoder.
