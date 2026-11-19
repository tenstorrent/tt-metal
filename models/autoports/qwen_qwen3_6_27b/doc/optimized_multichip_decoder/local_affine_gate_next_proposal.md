# Next bounded candidate: local GDN affine, gate and tile repack

CPU-only proposal; no device imports/execution, production edits, or new kernel implementation. The best additional bounded experiment is to adapt the **existing frozen single-chip `_affinev3_gdn_affine_gate_flat`** to local 12-head GDN. It fuses the exact post-RMS affine/BF16 boundary, FP32 SiLU gate multiply, and the costly head-row-to-hidden-row repack. It leaves recurrent state and the RMS reduction untouched.

## Evidence and duplicate check

Profile4 GDN B32 decode reports ~594 us kernels plus ~28 us gaps. Recurrence is ~125 us. Immediately after its retained typecast (~2 us) and RMSNorm (~5 us), the proposed replacement region is:

| Existing operation | Merged profile4 duration, rounded |
|---|---:|
| BF16 norm-weight multiply | 3 us |
| head-to-flat reshape | 11 us |
| packed gate slice | 2 us |
| two FP32 typecasts | 2 + 2 us |
| FP32 SiLU + multiply, BF16 output | 4 us |
| Total region replaced by one generic op | about 24 us |

The region is ~3.9% of the instrumented 622 us kernel-plus-gap total before replacement cost; this is an opportunity ceiling, not a predicted gain. Source is `perf/interim_profile4_l0_b32_s33/decode_perf_report.txt`, specifically IDs7522,7571,7428,7477,7526,7575. The profile's rounded values should not be subtracted from uninstrumented latency as a forecast.

Checked existing candidate modules, stage candidate notes and `candidate_search.md` categories. Existing work covers convolution, packed correction, recurrent source variants/streaming/column partition/native pack, SwiGLU, AGMM, MMRS, flat prefill and prefill RoPE. None installs this frozen affine-gate helper in `MultichipDecoder._gdn`; that method still spells out the six operations above. Frozen `optimized_decoder.py` already uses `_affinev3_gdn_affine_gate_flat` for its own decode path (~line960), but the multichip override bypasses it. This is a TP4 adaptation, not a request to repeat the recurrent arithmetic variants.

Full-attention profile4 B32 has ~453 us kernels +36 us gaps. The two Q/K norms cost ~7 us each, preceded by two height-sharded-to-interleaved conversions (~3 us each). Hidden norms each have FP32 sharding/norm/BF16 unsharding (~3+9+3 us) and a subsequent BF16 input sharding (~2 us). Those are material, but simply retaining Q/K height sharding through native `rms_norm` is illegal: `layernorm_device_operation.cpp:223–231` rejects HEIGHT_SHARDED input unconditionally. A custom row-local norm or a new block/width layout must preserve the later rotary/cache contracts. That is a larger, less bounded next change than adapting the existing affine-gate reader. Existing hidden-norm candidates have already varied gather/distributed/fused-AG routes; avoid duplicating them under a new name.

Persistent recurrence output alone does not remove its ~125 us kernel or DRAM state traffic: the state is already updated in place, and replay reuses captured addresses. Persistence is an ownership/lifetime tool, not evidence of a traced speedup.

## Exact minimal adaptation

Create a new helper module; do not modify frozen sources. Adapt the frozen affine helper's reader and Python wrapper with checked, single-use anchors:

- `source_head=user*48+head` and the second `(user*48+head)%32` become local12. Keep all face/row byte formulas, alignment checks, scratch copies and barriers.
- Packed gate base `page_id=320+tile` becomes `80+tile`, because the local gate starts at channel2560. TILE page width remains32.
- Logical normalized input is `[1,B*12,128]` BF16 TILE DRAM; gamma is existing `[1,1,128]` BF16; compact corrected packed input is `[1,B,4160]` BF16 TILE DRAM.
- Output is `[1,B,1536]`, padded `[1,32,1536]`. There are48 output tiles and `workers=min(48,requested_cores or grid_area)`. Replace the frozen host wrapper's192 task constants accordingly. Reject B outside1–32 and noncompact/legacy packed layout initially; parent fallback retains those paths.
- **Compute and writer strings remain byte-identical.** Frozen compute explicitly rounds norm*gamma to BF16, then evaluates gate SiLU in FP32, multiplies and explicitly rounds output to BF16. Do not fuse gamma into native RMSNorm: that would remove an existing rounding boundary. Do not replace the RMS reduction, recurrence helper or state layout in this candidate.

The exact parent integration point is immediately after the existing unweighted `rms_norm(typecast(recurrent_out,BF16))` in `_gdn`'s `s==1` branch. Under `local_affine_gate` and compact `[1,B,4160]` packed shape, call `local_affine_gate(out,self.weights['linear_attn.norm.weight'],packed)`, then the unchanged `_linear(...,'linear_attn.out_proj.weight')`. The else branch retains all original multiply/reshape/gate steps. B1 flat decode also satisfies the helper shape; `batched_conv_b1` is not required for this post-recurrence operation. Legacy `[B,1,4160]` and flat-prefill tails fall back unless their layout is separately adapted/proven.

## Ownership, bounds and trace contract

Each worker owns whole output tile IDs in a disjoint contiguous interval. Writer writes a complete2048-byte tile, waits for its NoC write barrier, then retires its CB page. No two cores update one output page. Reader normalized rows map to `source_head=user*12+head`, page `(source_head//32)*4+column`, tiled row `source_head%32`. Each32-byte horizontal face maps to exactly one user's output face. The 64-byte aligned fallback read stays within a2048-byte source tile and uses the original per-user scratch slots. With B32, scratch offsets reach at most user31*128+face64 plus64; gamma staging begins4096, within the existing8192-byte scratch CB. All48 gate pages80–127 are inside the130-page local packed row.

CPU plan check (`local_affine_gate_plan_cpu.json`) passed all32 batches: every live output value maps exactly once, source/destination face bounds hold, aligned fallback reads stay in page, and partition sums hold for1/3/8/32/48/110 requested cores. This is indexing evidence only, not a device correctness result.

Keep `output=None` by default. Optional caller-owned output must match mesh, layout, dtype, exact logical/padded shape and DRAM memory; it cannot alias normalized/gamma/packed. If persistent scratch is tested, allocate per execution slot before capture and pass it explicitly. Do not place a closure bound to the original model onto a shallow-cloned resource instance. Cache ownership must follow the instance/slot, or concurrent traces can overwrite each other's result. Do not allocate new persistent buffers or compile new programs while an earlier trace is active.

## Bounded proof and benchmark plan

1. Implement only the local reader/wrapper adaptation and a focused probe; assert frozen compute/writer hashes and exact source replacement counts. No production default change.
2. Compare the full six-op original region against the helper with exact BF16 bit equality, finite values and every rank/user/head. Cover B1/2/3/17/31/32; distinct rank/user/head/channel markers; random, zero norm, zero gate, large positive/negative gate; default workers and B3 cores1 for repeated CB wraps. Poison normalized dead rows and packed dead batch rows; prove retained output stays finite and unchanged.
3. Preallocate both outputs and all changed-input banks, warm both implementations before either capture, and retain all buffers until both traces release. Check changed-input replay versus saved eager results. Run with allocation tracking1, then a separate watcher check. Exact gate failure remains a failure; do not silently weaken to PCC.
4. Only after exact probe passes, use an owner-explicit class implementation or setup-only source snapshot in the existing whole-layer runner. Preserve output/state/per-user/contribution PCC>=.995 and changed-input trace checks. Measure fresh matched uninstrumented L0 B1S512, B32S33 and B32S513 controls, same selected policy/source and alternating order. Full attention is unchanged; one L3 control catches accidental policy/hook reach.
5. Profile only the winning family. Require disappearance of the six old ops and count replacement GenericOp plus unchanged input typecast/RMSNorm and output-projection input sharding. Compare that complete boundary and whole decode latency; reject if the local gather reader offsets any kernel savings.48 output tile tasks limit parallelism even at B32, so lower per-device head width does not guarantee improvement.

No benchmark command is advertised as runnable until the helper/probe exists. This proposal preserves the current RNE recurrence default and is independent of the native-pack precision experiment.
