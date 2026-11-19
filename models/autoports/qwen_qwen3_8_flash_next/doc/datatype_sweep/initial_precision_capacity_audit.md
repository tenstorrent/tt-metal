# Initial precision and capacity audit

CPU-only source/artifact inspection, 2026-11-16, initial HEAD `ccb44bb0cb36efb5eb26718a5520dbdf1ec41dc5`. No TTNN import, devices, or runtime edits. Read installed datatype-sweep, tt-device-usage, and model-bringup startup instructions. This is planning evidence, not a sweep result or review verdict.

## Physical BFP8 baseline obstruction

The four-rank Blackhole mesh has **34,178,731,008 allocatable DRAM bytes per rank**, eight banks. Full context remains **262144**. Source inventory: `doc/multichip_decoder/memory_capacity_plan.json`, `doc/full_model/capacity_plan.json`, `doc/full_model/final_capacity_reconciliation.md`; actual dimensions agree with `tt/multichip_decoder.py:MeshMoe` and target configuration.

There are 48 layers, 512 experts/layer, EP4 local 128 experts/rank, hidden 2560, expert intermediate 640. Each gate/up/down matrix has 80×20=1600 tiles. Thus each rank owns 48×128×3×1600=29,491,200 expert tiles. BFP4 tiles are 576 B; BFP8 tiles 1088 B, including exponents.

| Quantity | Bytes/rank |
|---|---:|
| All routed experts BFP4 | 16,986,931,200 |
| All routed experts BFP8 | 32,086,425,600 |
| Increase to all BFP8 | 15,099,494,400 |
| Existing packed PLE table | 7,680,065,536 |
| BFP8 experts + PLE alone | 39,766,491,136 |
| Excess over all allocatable DRAM, before other weights/state | 5,587,760,128 |
| Current complete known static inventory | 28,290,535,424 |
| Inventory with all experts BFP8 | 43,390,029,824 |
| Static excess with all experts BFP8 | 9,211,298,816 |

Consequently an all-BFP8 resident optimized full model cannot fit even at zero KV capacity. Changing KV dtype or reducing context cannot remove that obstruction. Do not attempt an expensive knowingly impossible load. The BFP8 starting point must be recorded as physically rejected; a required-to-fit BFP4 limitation is justified for some routed expert weights, not automatically for every layer/group. This is not permission to waive P1 silently.

A whole-layer expert upgrade costs **314,572,800 B/rank**; one semantic group across all layers costs **5,033,164,800 B/rank**. The actual stage-6 B1 maximum-context post-finiteness snapshot leaves **1,385,784,320 B/rank** free. An arithmetic-only upper bound therefore allows four whole-layer BFP8 upgrades before other transients/fragmentation; it does not prove four will construct. First/last layer upgrades cost 629,145,600 B/rank and are a sensible higher-precision fitting control. The stage-7 32769-token smoke retained full-capacity cache and leaves 1,533,149,184 B/rank, but use the stricter max-context snapshot for a conservative bound. Verify actual load/prepare/replay free memory on each measured fitting candidate.

The current fused expert op requires **all gate/up/down expert tensors within one call to share dtype**: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_device_operation.cpp:135-143`. Separate gate/up BFP4 versus down BFP8 is not a legal policy in this path without a kernel adaptation. Layer exceptions are legal because every layer is a separate invocation. One compute config is also applied to the complete fused operation; factory line601 consumes `compute_config.math_fidelity`. Therefore group fidelity distinctions cannot merely be written into JSON.

## Current measured-construction policy

| Group | Weight dtype / fidelity | Source |
|---|---|---|
| All routed gate/up/down, all 48 layers | BFP4 / LoFi, FP32 DEST off, packer off | `MultichipDecoder.from_state_dict`, `MeshMoe` |
| Shared gate/up | BFP8 / HiFi4, FP32 DEST on, packer on | `MESH_DRAM_POLICIES[shared]` |
| Shared down and scalar gate | BFP8 / loader HiFi4, FP32 DEST on, packer off | `MeshMoe` |
| QSA QGate/K/V and output | BFP8 / HiFi4, FP32 DEST on, packer on | `MESH_DRAM_POLICIES` |
| QSA attention | HiFi2 | `MeshQSA.build` |
| GR injection | BF16 / HiFi4 | DRAM role `gr` |
| GR mixer down/up and norms | FP32 exact projection/accumulation paths | `MeshResidual` |
| GDN Z/A/B | BF16 / HiFi4 | DRAM role `gdn_zab` |
| GDN QKV/output, router | BF16 / precise path | `MeshGDN`, `MeshMoe.router` |
| GDN recurrent state | FP32 | `_create_base_state` |
| PLE key/value | BFP8; exact norm/lookup special paths retained | `MeshPLE`, `MeshPackedEmbedding` |
| Token embedding / LM head | BF16 / head HiFi4, FP32 DEST on, packer off | `QwenModel` |
| Residual/public projection IO | BF16 (FP32 ordered internal reductions) | decoder operations and `DramProjection._PreparedGroup` |
| CCL | Input dtype; common BF16 payload; exact sum widens after gather | `Collectives` |
| K/V | BF16 tiled `[pages,1,32,256]` per tensor/rank | `_create_base_state` |

The head explicitly overwrites `head_group.decode.compute = self.compute`; changing its `DramProjectionConfig.fidelity` alone would be ignored. Correct policy plumbing must remove/parameterize this override while retaining the baseline DEST/packer flags.

## Candidate and plumbing plan

No distinct canonical implementation for this exact experimental Qwen3.8 architecture was found in the checkout. The accepted stage-5 DRAM defaults and stage-7 full-model defaults are the closest performance/accuracy controls. `doc/optimized_multichip_decoder/candidate_summary.csv` contains real-weight HiFi2/LoFi projection and attention screens, but these are not full-model acceptance or ranking evidence.

1. Pass a complete policy explicitly from readiness `--build-kwargs[-file]` into `tt/generator.py:build_generator`, then `QwenModel`, then each `MultichipDecoder`; today build kwargs reach only `QwenGenerator` after model construction. Resolve required selected artifact only when no explicit override exists.
2. Bind role policies to the per-layer `MeshLoader` or constructor, replacing `mesh_projection_group` process-global `MESH_DRAM_POLICIES` lookup. Existing `DramProjectionConfig` accepts BF16/BFP8/BFP4, LoFi/HiFi2/HiFi4 and BF16/BFP8 activation input; prepared weights and compute config actually consume these fields. Keep both decode and rows phases coherent. Outputs remain BF16.
3. Parameterize expert dtype and fused compute config per layer; start fitting highest-precision controls with layer0/47 BFP8, and the inherited required-to-fit BFP4 elsewhere. Compare BFP4 LoFi/HiFi2 for the fused three-group bundle, and BFP8 LoFi/HiFi2 where layer upgrades fit. Retain exact blocker if semantic per-group separation is not implemented, rather than claim it was measured.
4. Likely low-cost candidate wins: shared gate/up and QSA QKV/output BFP8+LoFi versus HiFi2; shared BFP4+LoFi, QSA BFP4+LoFi separately (inner-layer exceptions), with matching BFP4 HiFi2 comparisons for any kept groups. Add head fidelity/dtype comparison because its canonical geometry already exists. Preserve BF16/FP32 router and sensitive recurrence/GR exactness unless actual evidence justifies change.
5. Projection BFP8 activation is already an actual consumed field. Full residual or CCL BFP8 needs explicit typecast/restore boundaries; `Collectives.gather` currently has no policy. Preserve exact original input/output dtype for integer/non-floating gathers and FP32 paths. Quantizing every gather blindly changes sensitive exact GR/PLE semantics.

## KV candidate accounting and blocker

At full context, BF16 K+V all 12 QSA layers use **3,221,225,472 B/rank**. Identical tiled BFP8 cache geometry uses **1,711,276,032 B/rank**, saving **1,509,949,440 B/rank**. One full-stack page32 is 393216 B BF16 or 208896 B BFP8. Recompute context-contract candidate arithmetic using these sizes; they do not change required 262144 capability. These savings still cannot make all experts BFP8 fit.

KV BFP8 is not a constructor-only toggle. `LocalSelectedPagedKV` (`multichip_decoder.py:1385`) rejects non-BF16 caches and its source kernel performs raw BF16 row/tile copies using 2048-byte pages; a BFP8 cache requires unpacking/converted selected outputs or an adapted gather. `cache_pool.py` likewise accepts/allocates BF16 only. QSA updates must retain their BF16 update input where TTNN requires it; paged fill must match destination dtype. Include exact API/kernel diagnostics and AutoFix evidence if this candidate cannot be implemented. After any adaptation, rerun nonaligned prompts, short-prefix and B32 per-user cache checks, then selected-only maximum context if KV layout/dtype wins. A 32K–64K smoke is sufficient if selected KV remains unchanged.

All speed/accuracy selection still requires refreshed serial traced full-model AIME100 and natural-text runs, P1/limitation gate, runtime policy readback, and post-selection token-out measurement. This report provides no new accuracy, performance, device-fit or selection claims.


## Live implementation audit follow-up

CPU-only audit of newly added `precision.py` and modified construction paths, 2026-11-16. This is an implementation audit, not independent final stage review. Added `tests/test_precision_policy_cpu.py`, loaded by `importlib.util.spec_from_file_location` without importing the model package or TTNN. Run directly with:

```bash
python models/autoports/qwen_qwen3_8_flash_next/tests/test_precision_policy_cpu.py
```

Initial result: **7 tests run; 4 failing subcases across 2 tests**. Five tests pass (deep-copy/no mutation, selected default/override resolution, missing/unknown fields, invalid fixed contracts, invalid layer exceptions). The failures intentionally expose these defects; they are not expected-failure waivers:

1. **Global LM-head overrides accepted inside layer exceptions but ignored.** Both weight and fidelity overrides pass `load_policy`; QwenModel constructs its single head from the top-level policy. Reject `lm_head` in every decoder exception. Similarly a QSA-only group override on GDN layer, or `gdn_zab` on QSA layer, has no consumer: validate applicability against layer kind at construction or document/represent these exceptions in a way that cannot claim an applied change. The test covers the unconditional head defect, not architecture-dependent groups.
2. **Mixed fused expert dtype rejected too late.** `load_policy` accepts global or per-layer expert-down-only BFP8 changes even though `MultichipDecoder` later rejects them. Validate the fully resolved gate/up/down tuple for each layer before loading PLE or any weights. This is wasted setup rather than silently wrong execution; the new CPU test protects against that costly invalid-candidate path.

Additional source findings requiring attention before selection:

- `QwenModel(..., precision=None)` still selects `inherited_policy()` instead of the selected artifact, whereas `build_generator` loads the artifact. Direct full-model construction is an alternate default path and also bypasses validation of explicit mappings. Resolve/validate the policy in QwenModel against its autoport root, or explicitly require a validated policy and remove the implicit fallback. The measured harness must use the same selected default promised for later construction.
- Fixed-numerics metadata overstates `norm_weights: float32`: `MeshGDN.norm` is loaded by BF16-default `loader.tensor`, while GR/QSA/PLE norms use FP32. Similarly `exact_projection_weights: float32` is overly broad: GDN QKV/output and router precise projection weights remain BF16, while GR/final mixer exact projection weights use FP32. Split these into accurately scoped groups and report the fixed QSA attention HiFi2 and fixed loader HiFi4 compute assumptions. The current fixed-contract validation checks equality to declarations; it does not prove those declarations match tensors.
- `activation_dtype` is consumed by DRAM projection roles and head only. Shared-down, routed-expert, residual, and exact GDN/GR paths retain BF16 inputs. This is a legitimate narrower policy, but the artifact must name its scope explicitly instead of implying all activations changed.
- The new CCL dtype actually reaches top-level embedding/PLE, decoder reductions, and final mixer. It converts only tiled BF16 inputs and restores the original dtype after gather; FP32 and integer payloads stay unchanged. This preserves storage contracts but quantizes formerly exact inputs, so model accuracy must adjudicate it. `precision_summary` reports configured CCL dtype but should state this conditional scope.

### KV consumer audit

BFP8 private K/V allocation and QSA gather selection are now wired. The custom BFP8 gather returns BF16 selected rows, so downstream attention/output assumptions remain consistent. `_update` retains BF16 sharded update values and delegates quantization to `paged_update_cache`, matching the required TTNN API direction. Compressed index keys/raw history remain BF16. Prefill checkpoints copy only non-KV state with source dtypes; slot-state signatures already compare actual tensor dtype, so neither contains a hardcoded KV conversion.

Remaining concrete mismatches outside the new gather:

- `QwenGenerator.cache_spec()` still declares `paged_kv_dtype: bfloat16` unconditionally, and pending K/V descriptors claim BF16. A BFP8 private cache therefore reports contradictory metadata.
- `cache_pool.normalize_specs` and pool allocation/binding only permit/allocate BF16. A selected BFP8 model with shared-cache setup would either reject a truthful spec or bind BF16 storage to a BFP8-only gather. This is a full-model cache-construction contract issue; fixing it does not require beginning vLLM integration.
- `cache_budget.py` still fixes both static model bytes and BF16 bytes/block; `QwenGenerator.max_tokens_all_users` calls it without a policy. Higher-precision layer exceptions reduce available pool memory, while BFP8 KV changes bytes/block. Default capacity estimates therefore cease to describe the selected construction. Recompute candidate context arithmetic and make the chosen construction use matching budget inputs.
- Private cache reset uses `ttnn.multiply(value, 0.0, output_tensor=value)` and slot migration uses `ttnn.copy` on the actual cache dtype. They are dtype-generic in Python, but BFP8 support and no-capture-allocation behavior need the affected warmup/reset/migration runtime checks. A gather-only smoke cannot establish these paths.

The new BFP8 decompressor source was inspected for scratch overlap/addressing: exponent scratch ends at40000 bytes within the40960-byte CB; mantissa and output regions are disjoint; selected output tiles remain BF16. This inspection is not numerical verification. Validate real TTNN-produced cache tiles (including mixed signs, small magnitudes, repeated updates within one page, page permutation, nonaligned prompt rows) against TTNN BF16 conversion before full-model ranking.


## CPU verification after policy and KV plumbing fixes

Owner fixes were audited in place: rejected layer-head exceptions and incompatible expert triples now fail before loading; direct QwenModel uses `load_policy`; fixed-numerics labels distinguish BF16 GDN norm/projections from FP32 hyperconnection paths; shared-pool specs/allocation support BF16 and BFP8; generator allocator rejects cache dtype mismatches; cache-spec metadata reads selected dtype; default admission planning loads selected policy and charges `static_weight_delta` plus its KV page dtype.

Expanded tests in `test_precision_policy_cpu.py`, `test_cache_pool_cpu.py`, and `test_cache_budget.py`. Existing cache-pool test that classified BFP8 as unsupported now uses BFP4. Checks cover exact BF16/BFP8 page boundaries with null/output reservations, BFP8 exponent bytes, shared tensor-identity accounting, incompatible allocator/spec dtypes, static memory delta before pool admission, unsupported dtype rejection, baseline zero delta, first/last expert upgrades, all-expert upgrades, and independent bank-padding derivation for every configurable projection group.

Result: **49 passed, 40 subtests passed**, exit0; **TTNN import actively prohibited** with a meta-path guard and verified absent from `sys.modules`. Test plugins were disabled, repository conftests suppressed, and `TT_VISIBLE_DEVICES=0,1,2,3` set. Exact invocation:

```bash
TT_VISIBLE_DEVICES=0,1,2,3 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PYTEST'
import importlib.abc
import sys
class NoTTNN(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ttnn' or fullname.startswith('ttnn.'):
            raise AssertionError('CPU-only audit prohibits TTNN import: ' + fullname)
sys.meta_path.insert(0, NoTTNN())
import pytest
root = 'models/autoports/qwen_qwen3_8_flash_next/tests/'
status = pytest.main(['--noconftest', '--confcutdir=' + root, '-q',
                     root + 'test_precision_policy_cpu.py',
                     root + 'test_cache_pool_cpu.py',
                     root + 'test_cache_budget.py'])
assert 'ttnn' not in sys.modules
raise SystemExit(status)
PYTEST
```

`python -m black` formatted those three test files; only precision test formatting changed. No runtime edits, imports or device work were performed by this audit.

The `static_weight_delta` coefficients match current construction dimensions, including bank padding: shared gate/up1280 tiles/layer (K2560, two padded local160 outputs); shared-down400 tiles/layer (K160,N2560); GR5120 tiles/layer (two K10240,N1 matrices, each padded across8 banks); GDN ZAB4480 tiles on36layers (K2560,N1560 padded to56tiles across banks); QSA QKV8960 tiles and output3840 tiles on12layers; LM head155520 tiles (K2560,N62080, bank-padded N62208). Experts have614400tiles/layer/rank; BFP8 versus BFP4 increases one layer by314572800bytes and first/last layers by629145600bytes. KV BFP8 whole-stack max-context cost is1711276032bytes versus3221225472bytes BF16. These are persistent storage arithmetic, not device high-water measurements.

Remaining inspection notes: layer-kind-inapplicable overrides still have no consumer unless explicitly rejected or avoided by the candidate generator. Caller-owned `QwenGenerator(kv_cache=...)` checks capacity and slot consistency but does not explicitly compare B1 cache dtype to the model policy; the normal generated-cache and allocator paths do. Add an early selected-policy check if that caller-owned path is used for precision experiments. `cache_budget` docstring still describes BF16-only storage and its result object does not serialize chosen dtype/block cost; correct accompanying evidence must record them. These do not contradict the passing CPU arithmetic/ownership checks and are not new hardware claims.
