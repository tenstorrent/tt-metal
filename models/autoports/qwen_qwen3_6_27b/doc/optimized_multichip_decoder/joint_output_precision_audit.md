# Joint GDN output precision: applicability and bounded probe plan

2026-11-13, CPU-only source/evidence audit. No implementation edits, TTNN/Torch
imports, build, or device execution. This is a viable unmeasured candidate,
not an accepted optimization.

**BF16 joint matmul output can feed the existing FP32-add/BF16 combine directly,
with a format-correct reader and CB descriptor. It does not by itself permit
eight-tile matmul subblocks.** Eight tiles require a separate reduction of
matmul destination accumulation precision on the current 2D path. The combine
must continue to use FP32 destination arithmetic in either experiment.

## Existing path and evidence

- `tt/packed_correction_candidate.py:11` defines primary4160 + correction2624
  = logical joint6784, normally padded7168. Setup copies the **decoded BFP8**
  weights and verifies exact readback; tile-aligned concatenation preserves
  their quantization blocks. Do not regenerate weights from original floats.
- `PackedCorrection.project` at line157 writes FP32 width-sharded L1 output;
  `project_prefill` at line178 writes FP32 DRAM chunks. `_combine` at line234
  produces the existing BF16 DRAM corrected width4160 without slicing the
  source into material primary/correction tensors.
- `multichip_decoder.py:1452` documents local columns: QKV tiles[0,80),
  Z[80,128), beta/alpha[128,130). Joint correction is at tile offset130;
  correction tile is `col` for col<80 and `col-48` for col>=128. Z has no
  correction. Preserve runtime padded row stride, including alternate reader
  geometries such as physical6912; never hardcode224 tiles as the row stride.
- `_NARROW_GDN_COMBINE_COMPUTE` in `optimized_decoder.py:4409` copies two tiles
  into FP32 DST, applies `add_binary_tile`, explicitly rounds Float32 to BF16,
  and packs BF16; Z intentionally follows `primary + positive zero` as well.
- Actual profile `perf/fusions20_fused_l0_b32_s33/prefill_perf_report.csv`,
  ID1621, records **978.104 us** for 1056×5120×7168, LoFi BF16×BFP8→FP32,
  K8/subblock1×4, 56 active cores. This is the joint op itself, not the sum of
  all layer matmuls. Profiler timing is not final uninstrumented latency.
- Existing precision/geometry evidence does **not** test this output reduction:
  `precision_geometry5.json` records joint `output_dtype=float32` across its
  fidelity/readers/core choices. All joint rows in
  `prefill_chunks7_{1d,2d,minimal}_l0_joint.json` report FLOAT32 output.
  Joint L1/cap/grid and tiled-batch families retain FP32 output and FP32 combine.
  Weight/activation/CCL reductions are not this post-matmul precision change.

For one padded1056-row invocation, BF16 instead of FP32 saves 15,138,816 bytes
of joint output storage/write traffic per device. The combine reads130 primary
plus82 correction tiles per row, saving another14,327,808 bytes of reads:
**29,466,624 bytes (28.102 MiB) less explicit write+read traffic per device**.
The corrected BF16 output, inputs, weights, and any FP32 matmul partials remain.
This is a traffic estimate, not a latency or full-context capacity claim.

## Minimal phase-specific hook

Use opt-in setup options, for example `prefill_output_dtype="float32"` and
`decode_output_dtype="float32"`, accepting only float32/bfloat16, plus explicit
phase-specific matmul `fp32_dest_acc_en` options defaulting True. Record both
in candidate policy/provenance. Do not reuse `attention_activation_dtype`,
`attention_dtype`, or a mutable forward-phase flag: they govern different
contracts. Keep current fidelity, weights and input BF16 unchanged initially.

Resolve each phase into an immutable dtype/compute-config pair. `project_prefill`
currently calls `project(owner, flat)` when rows<=32; this shortcut must pass
the **prefill** pair explicitly to a common private projector, rather than
silently using decode precision. This matters for public short/nonaligned
prefill. Both ordinary large prefill and compact decode should call the same
format-aware `_combine`, without typecasting the complete joint tensor back
to FP32 and without a slice/concat restoration. Reject `combine="slice"` for
the new precision family until it has an independently adapted contract:
the existing local separate combine asserts FP32 inputs.

The default `owner.matmul_configs["attention"]` always sets FP32 accumulation
(`multichip_decoder.py:106–129`). Create a candidate-specific config if that
setting changes; mutating the shared attention config would also change other
matmuls/convolution and invalidate attribution. Record unchanged math fidelity,
math approximation, and packer-L1 setting separately from output dtype.

## Format-correct direct combine

Keep the specialized compute source/math and BF16 writer unchanged. Select
reader and CB format consistently from the actual packed dtype:

| Item | Existing FP32 source | Direct BF16 source |
|---|---:|---:|
| Primary/residual TensorAccessor page and NoC read bytes |4096|2048|
| CB0 primary, CB1 residual page / total bytes |4096 /8192|2048 /4096|
| CB3 positive-zero page / total bytes |4096 /4096|2048 /2048|
| Reader zero initialization bytes |4096|2048|
| CB0/1/3 data format |Float32|Float16_b|
| CB2 output page / total bytes |2048 /4096|2048 /4096|
| Combine `fp32_dest_acc_en` |True|True|
| CB0/1/3 unpack mode |UnpackToDestFp32|Default|

All source inputs, including zero, share their source format. Using Default
for BF16 lets `copy_tile` widen the exactly representable BF16 value into FP32
DST; there is no numerical need for a material FP32 input buffer. Keep the
FP32-source UnpackToDestFp32 modes to avoid passing FP32 through a lower
precision source-register route. See `api/compute/tile_move_copy.h:46,125`
and `tt_metal/jit_build/data_format.cpp:180–220`. Do not blindly retain the
old FP32 modes or4096-byte zero operation with a2048-byte CB.

Preserve both accessors pointing to the **same** joint allocation, correction
tile offset130, padded row strides, total130 output tiles per tile row, and
owner/disjoint output checks. Generate the alternate reader with a bounded
audited substitution or explicit source parameters; do not globally replace
4096 in unrelated kernels. The unchanged writer emits2048-byte BF16 tiles.
Retain source hashes for both compiled variants.

## Precision × geometry: the necessary distinction

`matmul_device_operation.cpp:660–681` validates subblocks against the actual
DST register count. It explicitly disables full-sync allowance for this 2D
factory; `dst_full_sync_en=True` is not a workaround. The DRAM-sharded decode
factory at line203 similarly chooses max width4 with FP32 DST and8 otherwise.
The 2D factory at lines363–370 retains **FP32 intermediate K-block partials**
when FP32 accumulation is enabled, even with BF16 output. Turning accumulation
off changes those partials to BF16 too, not merely the final pack.

Use a small orthogonal comparison, phase by phase:

| Arm | Joint output | Matmul FP32 DST | Subblock area | Meaning |
|---|---|---|---:|---|
| A |FP32|True|<=4|Current control |
| B |BF16|True|same <=4|Isolate final projection rounding and traffic reduction |
| C |BF16|False|same <=4|Isolate reduced accumulation/partial precision |
| D |BF16|False|8|Measure the larger-DST geometry after accounting for precision |

The combine remains FP32 in all arms. Do not label D simply “BF16 output”.
For1056 rows/7168 columns the inherited8×8 plan has per-core M5/N28 and an
effective height block1, so it cannot use2×4 or1×8 just by relaxing a cap.
Viable comparison geometries include **8×10: M4/N28, height block4,
subblock2×4**, or **7×8: M5/N32, height block1, subblock1×8**. Rebase the
<=4 control on each selected geometry; compare whole families as well as the
original8×8 baseline. Validate grid bounds, K-block divisibility, out-block
divisibility and actual padded/active cores per invocation. Extend the pure
planner's allowed widths/heights/area based on accumulation mode rather than
changing its fixed4-tile guard indiscriminately. Decode uses the native
DRAM-sharded factory's selected geometry and needs its own profile evidence.

## Focused probe, gates and adaptations

1. CPU source/descriptor proof over B1..32 and public sequence lengths
   1/17/31/32/33/513, both physical7168 and legal alternate padding: enumerate
   all primary/correction tile IDs, page byte bounds and output coverage.
   Include rows<=32 prefill dispatch and chunk cap boundaries; pad/mask/slice
   internally as today. Assert dtype/config/CB/unpack consistency.
2. Device microprobe should first isolate combine from matmul using varied
   per-rank/user FP32 values, cancellation, signed zero, BF16 rounding ties,
   and preserved poisoned padding. BF16-family oracle is **FP32 add of the
   independently BF16-rounded operands, then BF16 rounding**, with Z+0;
   it must match this oracle exactly. Separately compare to the original
   FP32-projection combine and report the numerical difference without
   pretending the new family is mathematically identical.
3. Run actual checkpoint joint weights and captured regression/real-text inputs
   through A/B/C/D, comparing complete projection+combine, not isolated matmul.
   Report per-rank and per-channel-group PCC/error for QKV, Z, beta and alpha,
   changed-input trace and pointer rebinding, warmed matched timing and trace
   allocation positive control. Include B1/3/17/32 and nonaligned prefill;
   expand accepted candidates to all batches and long queued runs with watcher.
4. Whole-layer prefill/decode/state gates remain unchanged, at least B1S512,
   B32S33, B32S513 and real-text B1/B32. Start **prefill-only BF16 output with
   decode FP32**, then decode-only and combined if warranted. Recurrence stress
   is necessary: small beta/alpha errors can change state over many steps.
   Frozen accepted TTNN output/state PCC is not an independent HF oracle;
   retain the separate HF contract gates.
5. If B passes but D fails, retain B as a distinct viable candidate. For
   reduced-accumulation failures, try larger legal K blocks (fewer BF16 partial
   reloads), appropriate block/grid/L1 adaptations and HiFi2/HiFi4 controls
   before rejecting a material win; record that fidelity does not eliminate
   BF16 partial rounding. If API/layout fails, repair descriptors/padding and
   retry rather than reject at the first error. Use AutoFix when indicated.
6. Profile actual lowered output dtype, FP32-DST flag, intermediate format and
   selected subblocks for each retained/rejected arm. Confirm there is no
   full-joint typecast/split/restore in ops.csv. Time uninstrumented warmed
   prefill and traced decode with watcher/profiler disabled; run watcher and
   trace allocation checks separately. Only final default-path reruns can
   supply final stage numbers.

The accepted formula is `BF16(P32 + C32)`; the pure output reduction changes it
to `BF16(FP32(BF16(P32)) + FP32(BF16(C32)))`. This can lose small corrections,
especially near cancellation, despite FP32 combine arithmetic. Z already has
no correction, but beta and alpha occupy corrected columns and do acquire the
new pre-add rounding. Keep their sigmoid/decay/softplus/FP32 math and recurrent
persistence unchanged and call out the input rounding explicitly. If those
columns alone fail, a separate narrow FP32 beta/alpha projection is a possible
adaptation only with an honest full-family measurement of added compute,
weights, movements and memory; do not silently reduce gate precision or claim
the original one-matmul traffic model for that different family.

The reduced temporary can improve headroom but does not alter KV-cache layout
or persistent state. Update the final context inventory if selected; no larger
maximum context follows from this source-level byte estimate alone.

## Implemented opt-in hook and focused probe (CPU-only)

Implemented only in `tt/packed_correction_candidate.py`; main/default policy
and other helpers are untouched. Constructor/install options are:

```python
prefill_output_dtype="float32"
decode_output_dtype="float32"
prefill_fp32_dest_acc_en=True
decode_fp32_dest_acc_en=True
```

The original reader/writer/compute strings, CB descriptors, unpack modes,
compute configuration, and phase output dtype remain identical at these
defaults. Opt-in metadata is added only for a changed precision contract.
The short-prefill compact route explicitly resolves the prefill phase.
Reduced accumulation clones the owner attention config's fidelity,
approximation and packer setting, changes only FP32 accumulation and fixes
half-sync; it never mutates the owner config. FP32 output with reduced
accumulation, non-boolean flags, other output dtypes, and BF16 output with the
legacy slice combine are rejected. The BF16 reader adapts exactly **five**
4096-byte literals (two accessors, two reads, one zero initialization).

`tests/probe_joint_output_precision.py` is the new standalone probe. Its
combine mode verifies finite output against an exact independently computed
FP32-add/BF16 oracle, including correction cancellation, BF16 ties, signed
zero, alternate physical6912/7168 widths, and verified NaNs in the unused
physical N tail. It does not claim to inject NaNs into logical row padding.
Projection mode uses real checkpoint BFP8 weights and the actual TP4 input
normalization of recorded text activations; it times the complete joint
projection plus combine and distinguishes the six arms:

- `control`, `bf16_output`: existing geometry, FP32 accumulation in both.
- `fp32_geometry`, `bf16_geometry`: matched selected geometry with four tiles.
- `bf16_acc4`, `bf16_acc8`: BF16 output and reduced accumulation, four/eight
  tiles on the same selected 2D grid.

Compact rows<=32 use native DRAM-sharded automatic subblocks; the probe skips
the misleading geometry duplicates and `bf16_acc4` there. It records this
limitation rather than labeling an automatically widened compact program as
a controlled four-tile arm. Projection precision checks retain PCC>=0.995 for
QKV, Z, beta and alpha groups on every rank/user; exact control/eager/trace/rebind
checks additionally cover all4160 columns. Inputs change at every step and in
50 queued replays, with exact final replay checks. Tracking runs require a
deliberate unsafe-allocation positive control; watcher, profiler and tracking
runs are excluded from timing. Whole-layer recurrence/state/HF gates remain
parent-owned and mandatory.

CPU validation performed without importing Torch/TTNN:

- `python3 -m models.autoports.qwen_qwen3_6_27b.tests.probe_joint_output_precision --cpu-check`
  passes3072 legal geometry plans,49920 tile-mapping checks and six expected
  invalid-contract/source-change rejections. It checks exact default reader
  preservation, both CB layouts and an admitted eight-tile BF16-accumulation
  plan.
- An independent AST/stub comparison against the immediate pre-edit source
  (`/tmp/packed_correction_before_output_precision.py`, SHA256
  `3aeb2d87e1e5a813076d63a0351d20f319915318370905ef5e2a1e366f04b0e7`)
  passes1680 default plan comparisons and60 complete default generic-op
  descriptor comparisons: B1/2/3/17/32 × S1/17/31/32/33/513 × widths6912/7168.
  The comparison includes source strings, compile/runtime arguments, ranges,
  CB sizes/formats, writer and compute configuration/unpack modes.
- Python AST parsing and Black formatting pass. No native or device result is
  claimed. The parent must adapt and retry any first API/descriptor failure.

Suggested parent-serialized smoke, preserving failed artifacts with new names:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 bash "$D/run_job.sh" joint_output_combine_smoke_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m \
  models.autoports.qwen_qwen3_6_27b.tests.probe_joint_output_precision \
  --mode combine --batches 1 3 17 32 --rows 31 33 --no-timing \
  --output "$D/joint_output_combine_smoke_v1.json"
```

For real-weight projection, use `--mode projection --policy-file <current
matched policy> --arms control bf16_output`, initially watcher/tracking enabled.
Then use a fresh artifact name with watcher/profiler **unset**, tracking0, and
without `--no-timing` for warmed timings. The default projection batches are
1/3/17/32 and prefill rows31/33/512/1056. Use
`--phase prefill --prefill-batch 32 --rows 1056` to exercise actual public B32S33
normalization and the helper's internal flatten/pad/slice contract; default
prefill batch is1. For the geometry cross add
`--arms fp32_geometry bf16_geometry bf16_acc4 bf16_acc8 --phase prefill
--rows 512 1056 --grid 7 8`; default physical7168 is divisible by this grid's
32 tiles/core. For physical6912 choose/adapt a legal grid (for example9×8 gives
24 N tiles/core). Do not reject the precision family on an incompatible grid.

Exact example **overlays onto the current matched whole-layer policy**, keeping
all other flags and existing prefill config unchanged unless shown:

```json
{"packed_correction_options": {
  "prefill_output_dtype": "bfloat16", "decode_output_dtype": "float32",
  "prefill_fp32_dest_acc_en": true, "decode_fp32_dest_acc_en": true
}}
```

That is output-only prefill reduction. Decode-only uses float32/bfloat16 in
the two dtype fields, with both accumulation flags true. To compare reduced
accumulation4 versus8, use the following common phase contract and explicitly
match the control's grid/cap/K block; change only subblock_w4→8 between C/D:

```json
{"packed_correction_options": {
  "prefill_output_dtype": "bfloat16", "decode_output_dtype": "float32",
  "prefill_fp32_dest_acc_en": false, "decode_fp32_dest_acc_en": true,
  "prefill_config": {"grid": [7, 8], "in0_block_w": 8,
    "out_block_h": 1, "subblock_h": 1, "subblock_w": 8, "row_cap": 4096}
}}
```

Use the same explicit geometry with subblock_w4 and FP32 accumulation true for
matched A/B. These are opt-in experimental policies, not selected defaults.
