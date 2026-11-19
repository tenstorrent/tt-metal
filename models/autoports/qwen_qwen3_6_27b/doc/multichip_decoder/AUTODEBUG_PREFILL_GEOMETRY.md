# Source diagnosis: native prefill geometry and MMRS read-NoC selection

Source-only investigation, 2026-11-13. No TTNN import, hardware execution, reset, production edit, or shared native edit was performed. The production source supplied by the parent was SHA256 `de734a896dc879ef5e4d3ebfb2f41f4734d53e31af1cae52c48d2d8775e42f3e`. The candidate wrapper records the actual production hash and whether it still matches. All performance proposals below require device validation; none is an accepted production improvement.

## Evidence and API boundary

`perf/final_profile_l0_b1/prefill_perf_report.txt` reports 37 device operations, approximately 1819 us device time and 1256 us gaps. Primary and correction matmuls take 285 and 189 us; attention MMRS takes 111 us; fused MinimalMatmul SwiGLU takes 315 us; MLP MMRS takes 192 us. Ordinary primary/correction matmul policies are being investigated separately by the parent.

The report's missing `program_config` advice is not proof of an unconfigured native op. MinimalMatmul accepts **`config=ttnn.MinimalMatmulConfig`**, and MMRS has its own automatic planner with **no public block/grid/program_config argument**. See `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/minimal_matmul_nanobind.cpp` and `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/matmul_reduce_scatter_prefill_nanobind.cpp:79–123`.

## MinimalMatmul: worthwhile legal alternatives

For local gate/up weight `[5120,8704]`, K is 160 tiles and N is 272 tiles (136 complete gate/up tile pairs). The factory uses M on grid Y and N on X until physical M exceeds 8704; then it transposes the core-grid assignment. M tiles are uniformly rounded to the M core axis; N is partitioned in complete pairs. Default grid 11x10 and M512 assign two M tiles and 26 N tiles to each core, padding to M640 and N9152. Twenty-two of the 110 cores have no live input rows. This is padding overhead, not an assertion that configured M-block4 computes four rows when the current block has two: the compute kernel clamps partial blocks.

Sources: `minimal_matmul_program_descriptor.cpp:374–446`, `device/kernels/compute_metal2.cpp:618–690`, and `ttnn/cpp/ttnn/operations/common/shared_with_host/runtime_length.hpp`. These paths are relative to the MinimalMatmul operation directory where applicable.

Candidate tuple order is **M-block,K-block,N-block,subblock-height,subblock-width,grid-X,grid-Y**. Sizes are tiles except grid sizes. B32/S513 is inherited as 512-token and one-token chunks, so its flattened M values are 16384 and 32.

| Logical case / flattened M | Tuple | Grid cores | Per-core M/N tiles | M/N/K block counts | Useful tile-volume fraction | Main DFB bytes/core |
|---|---|---:|---:|---:|---:|---:|
| B1/S512, control | `4,4,16,1,4,11,10` | 110 | 2 / 26 | 1 / 2 / 40 | 76.08% | 598016 |
| B1/S512, first candidate | `2,8,26,1,2,11,8` | 88 | 2 / 26 | 1 / 1 / 20 | 95.10% | 837632 |
| B1/S512, exact partition | `2,4,34,2,2,8,8` | 64 | 2 / 34 | 1 / 1 / 40 | 100% | 746496 |
| B1/S512, smaller buffers | `2,8,16,1,4,11,8` | 88 | 2 / 26 | 1 / 2 / 20 | 95.10% | 540672 |
| B32/S254, M8128 | `4,8,16,1,4,11,10` | 110 | 26 / 26 | 7 / 2 / 20 | 92.91% | 802816 |
| B32/S254, M8128 | `4,4,26,2,2,11,10` | 110 | 26 / 26 | 7 / 1 / 40 | 92.91% | 930816 |
| B32/S513, M16384 chunk | `4,8,16,1,4,11,10` | 110 | 47 / 28 | 12 / 2 / 20 | 96.20% | 802816 |
| B32/S513, M32 tail | `1,8,26,1,2,11,2` | 22 | 1 / 26 | 1 / 1 / 20 | 47.55% | 645120 |

Useful tile-volume fraction is logical tiled MKN divided by uniform padded MKN, not measured compute utilization. Main DFB footprint conservatively uses BF16 A/output, BFP8 weights including 1088-byte tiles, and FP32 intermediate: `2*M*K*2048 + 2*K*N*1088 + M*N*(2048+4096)`. It excludes descriptors, kernel scratch, and other live L1 tensors. Every candidate therefore still needs actual allocation and watcher validation.

The native validator requires positive block sizes, block/subblock divisibility, legal destination subblock area, and a grid at least 2x2 within the available device grid. The wrapper conservatively restricts subblock area to four. Even N blocks preserve gate/up pairs; even subblock width retains the native last-K fused SwiGLU path. Odd widths can be legal but select a separate epilogue (`minimal_matmul_program_descriptor.cpp:579–590`), so they are not the first candidates. K blocks 4/8/16/20 divide 160 without padding. The 2-row minimum grid makes a one-tile M tail inherently wasteful; changing the public logical length is not necessary or acceptable.

## MMRS: supported controls, occupancy, and watcher workaround hypothesis

Public controls are `l1_headroom` in bytes (default 64 KiB), input/output interleaved DRAM or L1 placement, one/two links, topology and compute fidelity/approximation. FP32 destination and full synchronization are forced. `output_tensor` may preallocate the exact native output. There is no public knob for resident versus streaming, block size, orientation, grid, or read NoC.

For N-scatter on TP4, output width5120 becomes local1280 = 40 tiles. The planner reserves a fabric-port strip. At M512, an 11x9 remaining matmul region can select M-on-Y with 8x10 active matmul cores and per-core M2/N4, plus boundary fabric ports: the observed 82 cores is consistent with this design. A generic warning that it does not use all 110 cores is not sufficient evidence of an avoidable occupancy defect.

The planner first tries resident plans, using K divisors and buffer depths, then streaming; it also searches M/N chunks. See `device/planner.cpp:40–63,263–295` and the outer candidate search. Lowering L1 availability with `l1_headroom` can change the plan, but it can produce a smaller **resident** plan instead of a streaming plan. Actual budget is also capped by live L1 allocations and optional output space (`matmul_reduce_scatter_prefill.cpp:180–245`). Therefore headroom is a useful sweep knob, not a guaranteed NoC selector. Moving all resident full-model weights to L1 is not a stack-compatible remedy. Short-lived input/output L1 placement is exposed only as an experimental candidate and includes conversion costs and changed capacity.

The parent's watcher failure at B32/S33 (M1056) reports pending NoC reads at the writer's completion. Independent AutoDebug owns proof of the precise shared-kernel defect. A model-local source-backed workaround hypothesis follows directly from `device/matmul_reduce_scatter_prefill_program_factory.cpp:535–536`:

```cpp
w_noc0_thin_wide = resident && scatter_dim == -1 && !m_on_x && grid_m <= 4 && grid_n >= 10;
W_READ_NOC = (resident && !w_noc0_thin_wide) ? 1 : 0;
```

Externally split flattened M into at most128 rows = four tiles, retaining N-scatter. With local N40 tiles and default planning, this should select M-on-Y, grid M<=4/N10, activating resident thin-wide NoC0 weight reads. Resident A occupies only 98304 bytes/core for attention K1536 or 278528 bytes/core for MLP K4352 when core M is one tile; this leaves room for native weight buffers at default headroom. Actual native plans must confirm the branch; this is not a hard API guarantee. A plan with internal N chunking, a different orientation or a different grid must not be silently assumed safe.

`--mmrs-chunk-rows 128` implements this externally using tile-aligned slices, native fused calls, then concatenation along M. It preserves row order and trims any optional padding. The operation is linear and N-scattered, so this does not change logical sequence-length support, positions, page tables, or cache ownership. It adds ceil(M/128) native calls (9 for M1056,64 for M8128,128 for M16384), slice/concat work, and repeated weight reads. It may be significantly slower than ordinary row matmul plus reduce-scatter. The latter is a required whole-layer control, exposed as `--mmrs-disable` (forces policy `fused_mmrs=False`, preserving other supplied policy keys).

`--mmrs-row-multiple 512` instead pads M1056 to1536 and trims output, but larger M does not reliably select NoC0. It is a diagnostic shape knob, not an independently justified watcher fix. Neither workaround suppresses watcher, changes shared C++, or excuses an outstanding-read failure.

## Test-only runner and exact options

New file: `models/autoports/qwen_qwen3_6_27b/tests/probe_multichip_prefill.py`. It wraps the original `run_multichip_decoder.main` and leaves its single-chip oracle, PCC/cache comparisons, changed-input trace checks, and timing loop intact. The wrapper intercepts only the local gate/up fused SwiGLU call and the two native row MMRS calls. It records each unique physical shape, selected config, native MMRS last-plan dictionary, inferred weight-read NoC, candidate/production hashes, exact command, options, and failures in the normal output JSON under `prefill_candidate`.

Use each command in a fresh process with the normal hardware environment and a matching single-chip reference. Ordinary runner arguments, including activation-file, policy-file, policy-json, stress and trace, are forwarded. Generate missing references with the **original** runner; baseline mode in this wrapper is deliberately rejected. Examples use explicit placeholder reference paths that must be replaced by the matching existing artifact:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill --layer 0 --batch 1 --length 512 --reference /path/to/matching_l0_b1_s512_reference.pt --trace --repeats 20 --swiglu-config 2,8,26,1,2,11,8 --output models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/prefill_swiglu_b1_s512.json
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill --layer 0 --batch 32 --length 513 --reference /path/to/matching_l0_b32_s513_reference.pt --trace --repeats 20 --swiglu-large-config 4,8,16,1,4,11,10 --swiglu-tail-config 1,8,26,1,2,11,2 --output models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/prefill_swiglu_b32_s513.json
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill --layer 0 --batch 32 --length 33 --reference /path/to/matching_l0_b32_s33_reference.pt --trace --repeats 20 --mmrs-chunk-rows 128 --output models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/prefill_mmrs_chunk128_b32_s33.json
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill --layer 0 --batch 32 --length 33 --reference /path/to/matching_l0_b32_s33_reference.pt --trace --repeats 20 --mmrs-disable --output models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/prefill_mmrs_disabled_b32_s33.json
```

Additional controls: `--mmrs-role all|attention|mlp`, `--mmrs-headroom-kib 64|256|512` (one integer per run), `--mmrs-links 1|2`, `--mmrs-input-l1`, `--mmrs-output-l1`, `--swiglu-input-l1`, `--mmrs-row-multiple 32|512`, and a global `--swiglu-config`. Tail overrides apply at padded M<=32; large overrides apply at padded M>8704. Use a wrapper run with no overrides for timing control, since Python observation bookkeeping is present in all candidates. JSON record count measures intercepted calls; it is not a count of trace replay device invocations.

Required acceptance evidence: matched oracle PCC/cache/trace, correct branch in every MMRS plan, watcher clean under identical real workload, full-layer warmed prefill latency versus both untouched wrapper control and `--mmrs-disable`, then profiler evidence for promising choices. The inferential source analysis and AST/syntax checks here are not hardware gates. No production recommendation is made before those results.

Source-only wrapper checks passed: Python syntax compilation, Black formatting, and an AST-extracted mock of MMRS slicing/concatenation for M33/512/1056/8128/16384 with row multiples32/512 and chunk128. The mock verifies exact row ordering/trimming and tile-aligned chunk bounds; it does not validate device scheduling. The parent's first hardware invocation exposed a debug-export mismatch: the plain nanobind getter is not on `ttnn.experimental`. The wrapper was corrected to `ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan`, matching the native unit-test helper in `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_matmul_reduce_scatter_prefill_bh.py:193`. The original hardware failure and rerun belong to the parent's artifacts.
