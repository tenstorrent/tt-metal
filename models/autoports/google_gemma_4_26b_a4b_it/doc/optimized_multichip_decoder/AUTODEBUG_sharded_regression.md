# Carried-shard fused gather-QKV regression

Source-only AutoDebug, 2026-09-27. No hardware was opened and no runtime source
was changed by this investigation. The parent owns all serialized experiments.

## Finding

The optimized default applies the replicated QKV K44 configuration to the
fused gather-QKV backend. That backend receives 704 elements, or 22 tiles, per
device. Its native matmul receiver divides this 22-tile ready slice by the
44-tile block, yielding zero blocks per slice, and subsequently uses that zero
as a modulo divisor to decide when to advance source addresses and wait for
gather completion. This is a concrete invalid configuration, and it predicts
the observed rank-dependent corruption much more closely than the changed
semaphore grid. A K22-only hardware control is still required to establish
that it is the sole cause of this observed failure.

The exact failing runtime SHA256 is
`2f240a9f5434efa370fed4cc6a3d6d72ec2dabc2bfae840ba9fbbc820919c797`.

## Observations and controls already on disk

| Artifact | Result | Relevant configuration |
| --- | --- | --- |
| `historical_k44_failure.json` | Prefill PCC 0.999735512; all 128 decode positions fail; minimum PCC -0.170925348; cache PCC finite on rank 0 and NaN on ranks 1–3 | Optimized default; QKV BF8; expert gate BF8; persistent buffers; fused AGMM; Ring; carried residual |
| `sharded_bfp8_sliding.json` | Passed all gates | Same explicit carried-shard/AGMM flags and QKV BF8, before optimized-default K44 policy |
| `sharded_precision_sliding_persistent.json` | Passed; warmed decode median 798.21 us | QKV BF4; explicit persistence; historical K22 projection |
| `output_agmm_sliding_persistent.json` | Passed; warmed decode median 801.78 us | Historical K22 QKV plus fused WO AGMM |

The medians above are host-wall traced decode observations, not device time.
The failing report is 799.26 us but is not a valid performance candidate.
Historical runtime hashes and exact commands are retained in the corresponding
JSON and `.command.json` files. The historical source versions are not commits;
their recorded configurations and the unchanged constructor's K22 are
supporting evidence, not a substitute for the proposed single-variable A/B.

## Causal source chain

Paths below are relative to the repository root; line numbers describe the
source inspected during this report.

1. `models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py:128`
   constructs `_Projection.program` with `in0_block_w=22`. At lines 816–822,
   `optimized_decode` replaces it with 44 before constructing
   `_GatherProjection`. The wrapper shares that same program object at line
   184 and passes it directly to `all_gather_matmul_async` at line 203.
2. The fused wrapper computes `local_hidden=hidden//4` at line 169. Here
   hidden=2816, so its tile-layout input is `[1,1,1,704]`, padded to one physical
   row tile and 22 K tiles. It keeps the gather fused into QKV and does not
   first restore a replicated residual.
3. `ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp:596` sets the tile-layout
   slicer's `num_cols=input_shape[-1]/tile_width=704/32=22`, and line 620 sets
   the source-slice page stride to that same 22.
4. `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/`
   `all_gather_matmul_async_program_factory.cpp:67` passes this 22-tile slice
   width and rank-dependent offset into `MatmulFusedOpSignaler`. The matmul
   factory uses the supplied K block; it does not adapt it to slice width.
5. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/`
   `reader_bmm_tile_layout_in0_sender_padding.cpp:113` constructs the
   `MatmulOpReceiver` with `in0_block_w`; the weight reader
   `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:147` supplies
   `in1_block_h`, which is the same K block. Both readers call
   `update_current_block_start_tile_id` in their K-block loops.
6. `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:219`
   computes `num_blocks_per_slice=tensor_slice_shape_width/tiles_per_block`.
   K22 gives 1; K44 gives 0. Line 232 uses
   `block_idx % num_blocks_per_slice` to trigger all of source-slice pointer
   updates, ring-index wraparound and semaphore waits at lines 235–253.
7. The entire gathered K extent is 88 tiles, so K44 means two K blocks.
   On RISC-V, integer remainder by zero returns the dividend; at block zero
   the branch can select the local slice, but block one does not advance to
   the next ready ring slice. Continuing from local offset `22*rank`, the
   contiguous 88-tile read window is `[0,87]`, `[22,109]`, `[44,131]`, or
   `[66,153]` for ranks 0–3. Only rank 0 lies inside the actual 88-tile K
   extent. This is a source-derived prediction, not a captured NoC trace.
   The rank-specific finite/NaN cache observations match it. Incorrect QKV
   also contaminates K/V before any expert-gate operation runs.

Native op validation checks rank, gather dimension, matmul config type and
generic matmul requirements, but lacks the ready-slice divisibility condition:
`all_gather_matmul_async_device_operation.cpp:26–88`. Generic gathered K=88
is divisible by 44, so the ordinary matmul validation cannot catch this.

The source contract for this backend must therefore include a positive K block
which divides the **local ready slice**, not merely the full gathered K width.
For this shape, K22 and K11 meet that requirement; K44 and K88 do not.

## Competing explanations

- **Full-grid `_MeshCCLManager` interacting with fused AGMM:** unsupported by
  the direct ownership chain. `_GatherProjection` allocates its own AG and
  barrier semaphores on the full physical compute grid at lines 171–174; it
  does not use `decoder.ccl` semaphores. `_MeshCCLManager._init_subdevice`
  only chooses allocation cores and records SubDeviceId(0). Neither the new
  method nor the inherited helper installs a subdevice manager. The changed
  grid can affect the independent distributed-norm collectives, but it does
  not change fused AGMM's semaphore core set or its `(0,8)` worker offset.
- **One CCL worker selected by the new sliding default:** not forwarded on
  this path. The sharded `gather` and `reduce_scatter` methods, and fused AGMM
  call, do not expand `self.ccl_tuning`. Changing `--ccl-workers` is therefore
  not a useful direct test of this fused path.
- **Expert gate BF8 causes numerical drift:** cannot explain rank-selective
  cache NaNs from first decode. QKV/cache generation precedes the expert path;
  BF8 also increases weight precision. Leave it at the final policy in the
  first control.
- **Persistent gather/stats buffer alias:** remains a possible independent
  lifecycle issue if K22 still fails. Historical persistent carried-shard
  results passed, and the invalid K44 geometry should be corrected first.
- **Native debug assertion:** line 220 checks the product of the slice count
  and blocks per slice. The AGMM factory currently hardcodes `num_transfers=4`
  while the receiver multiplies it by two directions, making that assertion
  appear inconsistent for a four-device ring even at K22. This is separate
  from the zero divisor and may matter in an assertion-enabled AGMM run;
  no assertion-enabled control was run here. Do not treat the assertion's
  presence as proof that ordinary runs reject K44.

## Exact bounded experiments

`probe_sharded_kblock.py` changes only the TP4 fused projection's K block
after factory creation and before any forward call. It preserves all final
weights, dtype, semaphores, CCL buffers, residual layout and harness gates. It
adds its source SHA, command and effective K-block configuration to the normal
candidate JSON. It never changes TP1.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder.probe_sharded_kblock \
  --layer 0 --length 4096 --steps 128 --trace --check-cache \
  --ring --sharded-residual --no-grouped-moe-reduce --fused-agmm --sharded-moe-bfp8 \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/sharded_k22_control.json
```

Expected decisive result: prefill unchanged, all per-rank cache PCC finite and
all decode PCC >=0.995. Compare ordinary warmed timings only after that gate
passes. A paired `--probe-k-block 44` reproduction can use the same command
with a different output filename; the existing failing artifact already
provides its before evidence on this exact runtime.

If K22 passes, retain the optimized K44 only for non-fused QKV, and keep K22
for fused gather-QKV. Then rerun the actual candidate as the unwrapped runtime
default with `--output-agmm` and `--fused-mmrs` separately, and repeat the
full-attention family. Recheck source-hash-dependent final default gates after
the runtime edit. This repair preserves the carried residual contract; it
does not add an immediate replicated gather or use a fallback matmul.

If K22 fails, compare K22 `--no-persistent-ccl` and instrument the first
distributed input-normalization output, gathered activation and fused QKV
output before checking lower-priority precision hypotheses. Never restore
the old undersized semaphore grid globally: the replicated default already
has independent source and hardware evidence that it is incorrect.

## Investigation and verification provenance

The mandated repo-local `.agents/scripts/autodebug.sh` runner was attempted in
this directory. Its fresh Codex subprocess could not execute any filesystem
read because its sandbox launcher lacked `bubblewrap`; the runner was stopped
and this already isolated AutoFix subagent performed the source-only analysis.
The failed runner log is `/tmp/autodebug_sharded_runner.log`.

The only executable added is `probe_sharded_kblock.py`, SHA256
`a95aaf2fb3a4ffdd7cb8e5c2f069fd6fda44d4b09ecf1a085615f228394c6261`.
`python_env/bin/python -m black --check <wrapper>` and Python AST parsing
passed. No accelerator test, reset, benchmark, native build or runtime edit
was performed by this subagent.
