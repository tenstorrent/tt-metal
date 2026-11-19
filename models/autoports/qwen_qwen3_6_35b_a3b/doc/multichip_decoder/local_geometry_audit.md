# Local projection geometry audit

This began as a CPU-only source audit before geometry measurement. The update
below incorporates the recorded device screens; the documentation audit
itself opened no devices. Measured controls are distinguished from remaining
unmeasured combinations. There is **no exhaustive optimality claim**.

Dense projections retain BFP8 weights, BF16 activations/output, HiFi2 and FP32
accumulation. Indexed experts retain BFP8/LoFi, FP32 accumulation, L1
intermediates, and gate-selected execution of eight experts. Changing geometry
can change accumulation order even when these precision settings remain fixed.

## Current shapes and candidate priorities

`R/C/Kb/Ns` means readers per DRAM bank / activation-storage cores / K-block
width in tiles / output-storage width in tiles per core. Tile size is 32x32.
Weight-bank dimensions below use the target's eight DRAM banks. Candidate
lists are a small prioritization set, not an exhaustive sweep.

| Per-device projection | Current geometry and physical storage | Controls and follow-up candidates |
|---|---|---|
| Full QKVG, K2048 to N2560 | `3/16/4/2`; input16 cores, shard32x128; output40 cores, shard32x64; each weight bank2048x320 | R2/3 with Kb4/8/16; then input C8/32; output Ns4 |
| Full output, K1024 to N2048 | `3/16/4/2`; input16 cores, shard32x64; output32 cores, shard32x64; each weight bank1024x256 | R2/3 with Kb4/8; input C8/32; output Ns1/4 |
| Shared packed gate/up/scalar, K2048 to N288 | `3/16/4/1`; input16 cores, shard32x128; output9 cores, shard32x32; each weight bank2048x64 | R1/2 versus3; Kb4/8; bank-tile-multiple3 as a separate packing control |
| Shared down, K128 to N2048 | `3/4/4/1`; input4 cores, shard32x32; output64 cores, shard32x32; each weight bank128x256 | R1/2/3; Kb1/2/4; input C1/2/4; output Ns2/4 |
| Indexed expert GU, K2048 to N256 per selected expert | DRAM input; 8x1 core grid; **Kb64**; per-core N/output-block width1 tile; subblock1x1; BF16 L1 output | Kb16/32 and 4x1 grid with N2/output-block2/subblock1x2 measured below; Kb8 unmeasured |
| Indexed expert down, K128 to N2048 per selected expert | L1 input/output; 8x8 core grid; Kb4; per-core N/output-block width1 tile; subblock1x1 | 8x4 with N2/output-block2/subblock1x2 and 8x2 with N4/output-block4/subblock1x4 measured below; Kb1/2 unmeasured |

The dense candidate column records the original search priorities. The global
reader/block/input/output controls listed below have now been measured; isolated
per-role combinations, input C32, shared bank-tile-multiple3 and shared-down
Kb1/2 remain unmeasured. Do not interpret the table as a measured Cartesian sweep.

Dense geometries are built at load time. The test geometry wrapper rebuilds
DRAM weights/configs before warmup/capture; dictionary edits after construction
do not change existing programs. Sparse geometry is now an explicit constructor
argument, with default `{"gu":{"block_w":64}}`. Explicit `{}` selects the
previous GU block16/down block4 control. The grid expression now uses
`(min(8,cores),(cores+7)//8)`, supporting the measured 4x1 GU candidate.

Selected indexed tensors per rank are GU weights `[1,256,2048,256]` and down
weights `[1,256,128,2048]`, both BFP8. BF16 GU input is
`[1,1,1,2048]` in DRAM; eight indexed results are `[1,8,1,256]` in L1.
Gate/up halves are each 128 wide; the SiLU/product feeds down in L1 and produces
`[1,8,1,2048]` in L1 before weighted expert reduction. All eight IDs come from
the gate; no dense all-expert computation was introduced. Local-only grouped
EP dispatch, selected separately for B>1/prefill, is documented with exact
capacity and zero-initialization contracts in [the mesh plan](mesh_plan.md).

## Exact restrictions and useful distinctions

- `_make_dram_projection` accepts R1/2/3, requires positive input core count
  dividing K/32, and requires positive Kb dividing K/32. R1 additionally
  requires Kb and activation-shard tile width to divide one another. Positive
  output Ns must also produce a storage-core count fitting the device grid.
  Shared down has only four K tiles, so Kb8/16 are rejected by the builder.
- R1 selects the legacy DRAM reader; R2/3 select the multicore pipeline for
  these tiled, bias-free, activation-free calls. The latter accepts ragged
  internal partitions, although the Python builder still imposes exact Kb
  divisibility. Do not apply the legacy factory's weight-bank-width divisibility
  restriction to the selected multicore pipeline.
- Dense DRAM configuration has **no public output-subblock parameter**.
  `per_core_N` controls output storage; it does not directly set compute column
  ownership. The selected pipeline internally uses an eight-tile Dest pass.
  A sparse-style subblock sweep is therefore not the same tuning operation.
- Shared packed is narrow: N/32=9, bank width2 tiles, five active banks.
  With R3 the pipeline computes `CN=min(3,2)=2`, `CK=3//2=1`, giving ten
  workers, not fifteen. R2 has the same column/K partition. A
  bank-tile-multiple3 control instead allocates three tiles per bank, activates
  three banks, and uses nine workers at R3. This changes storage padding and
  bank utilization and needs measurement; fewer workers is not automatically
  faster.
- Sparse factory requires K/32 divisible by Kb. Output block dimensions must
  be divisible by their subblock dimensions, and per-core dimensions by output
  block dimensions. Current GU/down each have one output tile per core, so
  widening the subblock without reducing the core count is invalid. Candidate
  subblocks above retain height1 and width at most4 for FP32 accumulation.
- Larger blocks need a fresh CB/L1 budget. The baseline's block64 allocation
  failure involved larger single-chip projection shapes; it does **not** prove
  a physical limit for these smaller local shapes. The local sparse GU
  block64 candidate has now passed; this does not validate dense block64 or
  other unmeasured configurations. Source-valid dimensions alone do not prove
  a candidate fits L1.
- B32 attention still runs singleton projections per request, so full QKVG and
  output reader tuning can matter there. Shared B32 projections have M32 and
  bypass the singleton DRAM-projection branch. B32 routed experts use grouped
  EP rather than indexed sparse. Their profiler rows need separate attention;
  a B1 sparse improvement is not evidence of a B32 improvement.
- Linear attention's sensitive ordered BF16/FP32 projection helpers are not
  ordinary entries in this DRAM projection table. Do not silently replace
  their arithmetic while applying these candidates.

## Measured local geometry screens

Both screens use full-attention layer 3, S32/B1, context512, two decode steps,
fixed BFP8/BF16 and fidelity policies, serial mesh-only timing. Their saved
single-chip baseline is a correctness control, not a paired latency measurement.
Every listed case records prefill PCC 0.9999727589322803 and minimum decode PCC
0.9999892501403449. Timings are whole-layer warmed trace decode, not isolated
kernel timings or sums across devices. JSON artifacts remain the provenance.

The sparse screen in [sparse_geometry/search.json](sparse_geometry/search.json)
ranked the twelve measured layouts as follows. Unspecified GU settings are
8 cores/block16/subblock1; down defaults are 64 cores/block4/subblock1.

| Rank | Candidate / input | Geometry override | Decode ms |
|---:|---|---|---:|
| 1 | `dram_block64` / DRAM **selected** | `{"gu":{"block_w":64}}` | 0.392621 |
| 2 | `dram_block32` / DRAM | `{"gu":{"block_w":32}}` | 0.396644 |
| 3 | `control` / DRAM | `control` | 0.405796 |
| 4 | `dram_gu4` / DRAM | `{"gu":{"cores":4,"block_w":32,"subblock_w":2}}` | 0.406983 |
| 5 | `dram_down32` / DRAM | `{"down":{"cores":32,"subblock_w":2}}` | 0.408286 |
| 6 | `dram_down16` / DRAM | `{"down":{"cores":16,"subblock_w":4}}` | 0.411432 |
| 7 | `l1_block64` / L1 | `{"gu":{"block_w":64}}` | 0.416677 |
| 8 | `l1_block32` / L1 | `{"gu":{"block_w":32}}` | 0.419698 |
| 9 | `l1` / L1 | `control` | 0.427067 |
| 10 | `l1_down32` / L1 | `{"down":{"cores":32,"subblock_w":2}}` | 0.428810 |
| 11 | `l1_gu4` / L1 | `{"gu":{"cores":4,"block_w":32,"subblock_w":2}}` | 0.431215 |
| 12 | `l1_down16` / L1 | `{"down":{"cores":16,"subblock_w":4}}` | 0.432227 |

DRAM GU block64 lowers screen latency by 3.25% versus the block16 control.
DRAM block32 is second. L1-input variants and the measured four-core GU or
32/16-core down variants are rejected by these local measurements. They are
not rejected on legality or inherited single-chip evidence. The winning B1
geometry does not alter B32 grouped EP kernels.

The earlier dense screen in [geometry/search.json](geometry/search.json) keeps
its independent control; these runs are not an additive speedup decomposition
with the later sparse screen.

| Candidate | Override | Decode ms |
|---|---|---:|
| `control` | `{}` | 0.406818 |
| `readers1` | `{"default":{"readers":1}}` | 0.437512 |
| `readers2` | `{"default":{"readers":2}}` | 0.407104 |
| `block8` | `{"default":{"block_w":8},"shared_down":{"block_w":4}}` | 0.408762 |
| `block16` | `{"default":{"block_w":16},"shared_down":{"block_w":4}}` | 0.410952 |
| `input8` | `{"default":{"input_cores":8},"shared_down":{"input_cores":4}}` | 0.406808 |
| `output4` | `{"default":{"per_core_n":4}}` | 0.406157 |

The selected dense configuration retains its control values. `output4` is
0.16% lower in this screen and `input8` is essentially tied; this screen does
not establish that the retained dense control is uniquely fastest. R1 and
larger measured K blocks are slower. Unmeasured per-role combinations remain
open candidates, not evidence-backed rejections.

Local-only grouped dispatch was measured separately on natural S32/B32 inputs:
linear/full decode latency fell 2.16%/5.89%, full prefill fell 7.42%, and linear
prefill increased 0.79%. See `local_dispatch_l{0,3}_{false,true}.json` and the
mesh plan for absolute whole-batch timings and PCC. The selected dispatch
reduces each rank's padded expert region bound from 256 to 64 experts; combine
still zero-initializes `[1,1,T,8,2048]`, so it does not remove all movement.

## Evidence that transfers, and evidence that does not

The optimized baseline supports preserving the current dtype/fidelity policy,
packed projections, L1 expert intermediates, and exact selected-expert
execution. Its measured reader/block and core/subblock controls inform the
candidate ordering. They do **not** reject local TP4 geometries because the
matrix widths, K lengths, and consumer/storage layouts changed.

Examples in the baseline evidence include R1/2/3 at Kb8, Kb4/8/16/32 controls,
sparse subblock/core sweeps, and packed-versus-separate expert controls. The
documented dense block64 failure requested2774976 bytes with1572864 available;
that is a source-shape-specific failure, not a blanket family rejection.
The local measurements above qualify the listed mesh configurations only;
they do not establish unmeasured combinations' speed or correctness.

Accept a candidate only after local whole-output/state PCC, warmed trace
replay, and paired whole-layer timing on the target mesh. Use profiler rows to
confirm the intended kernels, dtype/fidelity, communication and movement costs.
Keep whole-batch timing distinct from aggregate work across four devices.

## Source and evidence references

Paths are relative to the tt-metal checkout; line numbers identify the source
at this audit and may move as the stage is finalized.

- `models/autoports/qwen_qwen3_6_35b_a3b/tt/multichip_decoder.py`:
  `from_state_dict` local DRAM projection construction and selected defaults;
  `linear` singleton dispatch; `_moe` indexed sparse geometry; `_grouped_moe`
  local-only EP dispatch; `dispatch_capacity` and `prepare_moe` workspace bounds.
- `models/autoports/qwen_qwen3_6_35b_a3b/tt/optimized_decoder.py:510`:
  DRAM geometry builder and checks; `:650` inherited attention-output Ns2.
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1280`:
  multicore pipeline; `:1302` Dest schedule; `:1328` active banks;
  `:1347` CN/CK partition; `:1843` pipeline dispatch.
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1661`:
  DRAM input/output, tiling, shard and fallback-path validation.
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:542`:
  sparse block/subblock divisibility checks.
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:198`:
  sparse K-block divisibility.
- [Baseline performance conclusions](../optimized_decoder/perf_conclusions.md):
  advice decisions, geometry controls, and limitations.
- [Baseline work log](../optimized_decoder/work_log.md): reader/block controls
  around lines174-175 and split/geometry evidence around lines287-293.
- `projection_l3_dram_ordered_v2.json`: recorded local effective geometry.
  `context_l3_32k.multichip.dram.json`: eight-bank allocator census.

No devices were opened and no implementation files were changed for this
documentation audit. The measured rows cite existing device artifacts.
