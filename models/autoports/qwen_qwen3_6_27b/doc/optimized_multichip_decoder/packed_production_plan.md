# Owner-safe joint GDN integration and capacity proposal

This is a CPU proposal, not a final-default selection or hardware acceptance. The accompanying [patch](packed_production_proposal.patch) is **not applied**. It changes the joint helper and class dispatch; it leaves the now-present `local_fused_conv` class hook intact. The inspected candidate [c4096/g8x8 result](joint_c4096_g8x8_l0_b32_s33.json) reports pass, 7.147624623 ms prefill and 0.664333818 ms decode; its [L1-input variant](joint_c4096_g8x8_l1_l0_b32_s33.json) also reports pass, 7.161283866/0.665340330 ms. These are prototype measurements and do not validate the proposed ownership refactor.

## Ownership problem and proposed dispatch

The prototype installs `decode_forward` bound to a shallow copy of its initial layer. `PackedCorrection.layer` still points to the original owner; prefill also installs instance-bound methods. `check_multichip_stack_resources.py` later shallow-copies the prototype and replaces each clone's CCL manager, `_ccl_buffers`, exchange tensor and convolution semaphores. Copied instance-bound methods still execute on the old view/original owner. Address uniqueness in the resource report therefore does not by itself prove those resources are actually used by decode. A later class-based local-convolution hook can also be bypassed by a previously installed instance hook.

The patch removes the decode view, `MethodType`, and all installed bound methods. The helper holds geometry, program configs, kernel source strings and combine configuration, but no owner reference. Its runtime entrypoints become `project(owner,x)`, `project_prefill(owner,x)` and `_finish_projection(owner,output)`. All weight/config accesses use the explicit owner; the fused combiner depends only on its input tensor and immutable kernel configuration.

`MultichipDecoder._linear` dispatches the GDN key through its own helper. Compact `[1,B,5120]`, B<=32, goes directly to `project(self,x)`, avoiding an extra DRAM roundtrip through the prefill flatten adapter. Other logical shapes use `project_prefill(self,x)`. The class `_prefill_packed_gdn` hook routes flat prefill explicitly through joint prefill; when the option is absent it reproduces the existing two-raw-FP32-matmul/single-BF16-combine branch. Legacy/list/resume prefill still reaches `_linear` through the existing scheduler. No runtime phase flag or layer mutation is introduced. `flat_prefill.py` already calls this hook and needs no edit.

Construction installs joint setup only for GDN when `packed_correction_options` is present. Production installation requires `prefill=True`; releasing the separate weights defaults to true and requires both phases. Old decode-only CLI options intentionally fail clearly under this proposed adapter. Previously collected prototype results remain unchanged. The ordinary runner can use the option after the patch; the candidate CLI no longer installs the helper twice.

Example option, retaining the measured integrated3 policy outside this nested object:

```json
{"packed_correction_options":{"combine":"fused","prefill":true,"release_separate":true,"prefill_config":{"row_cap":4096,"grid":[8,8],"out_block_h":4}},"local_fused_conv":true}
```

The patch does not add this option to defaults. Final selection belongs to the parent after combined evidence.

## Resource-clone contract

Sharing read-only weight tensors, matrix programs, and helper source/configuration between resource-test clones is safe provided those objects are never changed after setup. Forward methods resolve on each clone's class and pass that clone to the shared helper. The existing test must still give each clone its own CCL manager, `_ccl_buffers`, exchange and convolution semaphores; states are already supplied independently. Production model layers continue to own their own actual weights.

Before interpreting a resource run, verify `decode_forward.__self__`, `prefill_forward.__self__`, `_linear.__self__` and `_prefill_packed_gdn.__self__` resolve to the executing clone, and there is no `_packed_correction_decode_view`, instance-bound replacement or helper `.layer`. Observe the owner used by the first projection/collective in an instrumentation-only run. This checks dispatch, while the existing 64-owner replay/isolation test checks device resources. Sharing immutable matrix configs is not a substitute for independent mutable CCL/state ownership.

The current `local_fused_conv` class hook already takes `self.taps_rm` and the caller's history. It dispatches only compact shape `(1,B,2560)` with B in1..32; legacy-shaped prefill tails retain the original branch. `batched_conv_b1` remains a separate policy choice: enabling local convolution alone does not route B1 through it. The helper allocates output per captured/eager invocation, owns no persistent state, and writes the supplied history. Its source-string adaptation is host setup of a device program, not host tensor computation.

## Actual matrix inventory and projection evidence

Integrated3 inherits BF8 matrices, c8/r2 decode geometry, packed MLP, minimal-SwiGLU prefill, MMRS copies, and packed QKVG for full attention. The current joint result's `matrix_allocations` contains exactly the four GDN matrices below. Its `logical_shape` field reports the TT tensor's physical/padded width, so do not misinterpret7168 as mathematical N. The patch registers a `projection_info` entry with raw `logical_n=6784`, `corrected_output_n=4160`, role attention, and actual joint geometry.

| GDN resident matrix | Mathematical K×N | Physical K×N/device | BF8 bytes/device |
|---|---:|---:|---:|
| `packed_gdn_correction_joint` | 5120×6784 | 5120×7168 | 38,993,920 |
| `linear_attn.out_proj.weight` | 1536×5120 | same | 8,355,840 |
| `mlp.gate_up.weight` | 5120×8704 | same | 47,349,760 |
| `mlp.down_proj.weight` | 4352×5120 | same | 23,674,880 |
| Matrix total | | | 118,374,400 |
| Retained interleaved MMRS copies, attention output + MLP down | | | 32,030,720 |
| Retained paired minimal-SwiGLU copy | 5120×8704 | same | 47,349,760 |
| **GDN resident matrix total** | | | **197,754,880** |

Full attention remains178,257,920 resident matrix bytes/device:98,877,440 ordinary matrices +32,030,720 MMRS copies +47,349,760 paired-SwiGLU copy. Its packed QKVG is5120×3584,19,496,960 BF8 bytes. The full64-layer matrix sum becomes **12,344,360,960 bytes/device** (48 GDN +16 full), versus12,478,054,400 with separate GDN matrices.

Actual allocation reporting must include the MMRS and paired-SwiGLU copies, which are outside `model.weights`; count unique physical buffers once. MMRS copies use logical5120 output columns after cropping, even if the decode matrix is more widely padded by a geometry sweep. A blanket sum of the padded original row-parallel matrices overestimates those copies for such policies. The existing capacity helper currently makes that blanket assumption, harmless for the inspected integrated3 default geometry but requiring correction before arbitrary geometry accounting.

For role attribution, classify the joint key as GDN attention-input projection, then its FP32-to-BF16 correction combiner as a separate operation. A decode invocation performs one joint matmul and one combiner; prefill performs `ceil(align32(B*S)/row_cap)` matmuls inside each caller-owned temporal chunk, followed by combines/reassembly. Record actual input M, weight K/N, output dtype, program grid/blocks and selected compute fidelity from the call. The existing `run_multichip_boundaries.py` observer watches row-parallel MMRS only and cannot establish joint projection dispatch. Likewise, infer neither executed kernel geometry nor role percentages solely from selected policy. Match the final profiler CSV/table to the observed joint shape; keep its prefill and decode calls separate.

## Parameterized capacity change

Let `Q=32*lcm(8*readers,cores)`, `N_J=ceil(6784/Q)*Q`, and `T8(K,N)=ceil(K/32)*ceil(N/32)*1088` for these eight-bank-aligned BF8 matrices. Joint logical columns are4160 primary +2624 narrow correction. Primary/correction boundaries are complete tiles; setup copies existing quantized values and checks exact decoded-value equality after repacking.

For released separate matrices, per-GDN resident delta is:

`T8(5120,N_J) - bytes(actual separate primary) - bytes(actual separate residual)`.

With c8/r2 separate physical widths4608 and3072, old bytes=41,779,200. c8/r2 joint width7168 gives **−2,785,280 bytes/layer/device**, or **−133,693,440 across48 GDN layers**. c8/r1 or c8/r3 joint width6912 gives37,601,280 bytes, a −4,177,920 delta against that same c8/r2 control; do not mix separate geometry baselines when making this comparison. K block and prefill cap/grid do not change matrix capacity.

If old weights are retained, resident delta is instead **+joint bytes**: +1,871,708,160 bytes/device across48 GDN layers at c8/r2. Their presence does not make them active decode traffic. The current helper has no joint-policy branch; update it by actual retention and active-role inventory before using it as final accounting.

Setup temporarily holds original separate matrices, the new interleaved upload and new DRAM-sharded joint allocation. For this projection, those allocations total up to119,767,040 bytes/device while converting, before old matrices and temporary upload are released. This is a setup peak component, not persistent capacity. Host Torch/readback/repacking is permitted only during construction, before warmup/capture; no forward invokes it.

## Context, cache/state and scratch remain explicit

The context contract is **B1 full context262144 and B32 short contexts**, not32 simultaneous full-context requests. Joint packing and local convolution do not change KV/recurrent/history shapes or dtypes. Preserve262144; no inspected allocation proves a need to lower it.

Per-device full-model components at B1/context262144, integrated3 plus one warmed persistent BF16 CCL signature/layer and joint release:

| Component | Bytes/device |
|---|---:|
| 64-layer resident matrices | 12,344,360,960 |
| Future BF16 embedding/head/final norm | 1,275,985,920 |
| 16 layers K+V, BF8 page32/head256 | 2,281,701,376 |
| 48 BF16 recurrent states `[B,12,128,128]` | 18,874,368 |
| 48 row-major BF16 histories `[B,3,2560]` | 1,966,080 |
| Auxiliary tensors and padded page tables | 177,758,208 |
| Persistent CCL DRAM buffers | 68,157,440 |
| **Resident subtotal** | **16,168,804,352** |
| Existing trace/activation/metadata reserve | 4,294,967,296 |
| **Planned resident + reserve** | **20,463,771,648** |

The CCL term assumes one RS and one hidden AG signature/layer, DRAM collective outputs, and no extra stats-AG signature: per layer655,360-byte doubled linear-RS scratch +81,920-byte RS result +327,680-byte hidden AG result. Multiply by actual distinct warmed signatures. L1 placement removes the latter two from DRAM accounting; global semaphores remain L1 resources. No CCL count changes solely because of joint packing or local convolution.

Source-derived short-case totals below retain the4GiB reserve and future embedding/head. Pages use the numerical runner's `ceil((S+33)/32)` initial table and its existing >=16-column rounding, not just live S. Stress with a larger extension must substitute its actual table/pages.

| B | S | Allocated table columns | KV bytes, all16 layers/device | Integrated3 total | Joint-release total |
|---:|---:|---:|---:|---:|---:|
| 1 | 33 | 3 | 835,584 | 18,312,413,184 | 18,178,719,744 |
| 1 | 513 | 32 | 8,912,896 | 18,320,498,688 | 18,186,805,248 |
| 32 | 33 | 3 | 26,738,688 | 18,945,073,152 | 18,811,379,712 |
| 32 | 513 | 32 | 285,212,672 | 19,203,579,904 | 19,069,886,464 |
| 1 | 512 | 32 | 8,912,896 | 18,320,498,688 | 18,186,805,248 |
| 1 | 262144 | 8192 | 2,281,701,376 | 20,597,465,088 | 20,463,771,648 |

These are source-derived plans, not allocator-peak measurements. Full B32/context262144 alone requires73,014,444,032 KV bytes/device and is outside the existing batch/context contract; that observation does not justify reducing B1 context.

Joint prefill retains combined BF16 chunks before concat. For an invocation with padded rows R, cap C and joint width7168, explicitly include the potential `2*R*4160*2 + min(C,R)*7168*4` bytes for retained combined chunks/concat plus one FP32 projection, and any live flattened/padded input `R*5120*2`; these overlap with existing activation-reserve components and must not be blindly added twice. At R16384/C4096 those components are390,070,272 and167,772,160 bytes respectively. `_finish_projection` temporaries and allocation lifetimes still require device peak/trace-allocation evidence. Increasing cap from512 to4096 increases one FP32 projection from14,680,064 to117,440,512 bytes. The decode L1 per-program N also grows even though total stored weight padding shrinks.

## Traffic and roofline

The GDN one-read matrix floor becomes118,374,400 bytes/device; recurrent read+write adds `2*B*12*128*128*2`, or786,432 at B1 and25,165,824 at B32. Full-attention cache traffic still depends on each user's visible KV tiles, not reserved cache capacity. Minimal-SwiGLU/MMRS resident copies are not additional decode weight reads when that decode path does not execute them. Old retained separate matrices likewise contribute capacity only when joint projection executes.

The local-convolution device kernel requests8 BF16 row-equivalents read (three history, token, four taps) and4 written (three history, output): **61,440*B bytes**, excluding the preceding native tile-to-row-major conversion and bus/protocol overhead. Tap rows are reread per user by this implementation; repeated movement belongs in an actual-path traffic ledger, not an inflated useful-work roofline floor. Circular buffers are10×4096 bytes per active convolution worker, L1 only. The2560-channel final chunk transfers1024 useful bytes; unused CB padding never becomes DRAM payload.

For joint projection, charge the new physical matrix once per actual matmul invocation in an actual-path ledger. Prefill cap-induced rereads, output writes and combine reads must remain separate from the ideal weight-once lower bound. No reduction in kernel count licenses claiming the same percentage reduction in whole-decoder latency. Keep the user's >=70% roofline objective unchanged.

## CPU verification and remaining evidence

`git apply --check` passed against the inspected sources. Applied only to temporary copies, both resulting modules parsed. An AST-extracted dispatch test shallow-copied a fake layer, replaced its CCL/buffer owner, and verified both `_linear` and the prefill hook passed the clone. No helper `.layer`, `MethodType`, shallow decode view, Torch numerical call, TTNN host upload or readback remains in the helper runtime methods. Setup host conversions remain explicit in the constructor. No TTNN import or device operation was performed for this review.

Parent final validation should use the ordinary runner and frozen chosen policy/source, then the already-required correctness/trace, boundaries/long context, interlayer, watcher,64-owner resource and runtime-sentinel evidence. Re-run the reachable static runtime audit with both joint and local-convolution helpers included, alongside the authoritative runtime conversion sentinel. Earlier prototype passes do not transfer automatically to changed class dispatch. No additional stress requirement is introduced here.

Patch base hashes at the successful CPU check:

- `tt/multichip_decoder.py`: `fb2518f9aa80728ed236ec977e092836ad5ac9b78c7bae1f4470e34995f0c5ce`
- `tt/packed_correction_candidate.py`: `82e1541495149296e5b71778120971512a71c6fd87063eb1344535155595ab17`
- `tt/flat_prefill.py`: `b97e2db56e1c970fc39bc8805bf0fd9fe3f31b843c4a04fd50ce85305402e8df`
- `tt/recurrent_next_candidate.py`: `d3cea77daef56498bbc8ec044059252506cb8de4ad1c98d236dbddd3517389e3`

Review/apply from repository root only when the parent is ready to change runtime sources:

```bash
git apply --check models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/packed_production_proposal.patch
git apply models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/packed_production_proposal.patch
```
