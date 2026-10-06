# Selected defaults

Current runtime SHA256 is `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`. **All current targeted and inherited correctness gates pass.** [The v8 manifest](validated_v8_validation_summary.json) validates12 primary commands plus five boundary controls, including current full public/max contracts, both headline/Watcher kinds, four pytest cases, both512-step stress kinds and full tight/short/tails. Unchanged sliding broad contracts retain original v5/v6 attribution through the v7/v8 source chain. Current native profiles and CPU accounting pass; independent final review remains separate.

[The v8 source proof](source_delta_v8.md) selects full minimal-QKV M2 and directs the original weighted input-normalization output into L1. Sliding remains M4 with its original placement. Full HiFi2/sliding HiFi4 and all decode precision/lifecycle/capacity behavior remain unchanged. Earlier patches are already applied and must not be reapplied.

| Choice | Sliding attention | Full attention |
| --- | --- | --- |
| Expert gate / down, both phases | BFP8 / BFP4 | BFP4 / BFP4 |
| Decode expert backend / slots | Indexed / 8 | Indexed / 8 |
| Decode fused GELU/up multiply | Enabled | Enabled |
| Decode expert gate / down K blocks | 44 / 22 | 44 / 22 |
| Decode expert activation cast | Unchanged BF16 | BFP8 |
| Shared decode gate / down | BFP8 / BFP8 | BFP4 / BFP4 |
| Shared DRAM readers per bank | 1 | 2 |
| Direct decode QKV fidelity | HiFi2 | LoFi |
| **Prefill QKV backend / K block / fidelity** | **minimal_matmul / 8 / HiFi4** | **minimal_matmul / 16 / HiFi2** |
| Prefill QKV input / weight / output | FP32 / BFP8 / FP32 | FP32 / BFP8 / FP32 |
| Prefill QKV M block / input memory | M4 / original DRAM | M2 / L1 directly from input normalization |
| Prefill QKV output / tied-KV boundary | DRAM | L1 output and slice; concat returns DRAM |
| Native decode SDPA fidelity | HiFi4 | LoFi |
| Prefill SDPA fidelity | LoFi | HiFi2 |
| Later full-prefill paged SDPA | Not used | Grid `(8,4)`, primary Q64/K256; Q64/K128 only at tight capacity |
| Native SDPA through output-projection input | BF16 | BF16 |
| Preferred caller decode RoPE layout | ROW_MAJOR | ROW_MAJOR |
| Sharded hidden-width norm sites | `all` | `all` |
| Direct router grid / K block / fidelity | `(4,1)` / 22 / HiFi4 | `(4,1)` / 44 / LoFi |
| Generalized router score centering | Enabled | Enabled |
| Decode output grid / fidelity | `(8,8)` / HiFi4 | `(11,10)` / LoFi |
| **Prefill output backend / K block / fidelity** | **multicast 2D / 16 / LoFi** | **minimal_matmul / 8 / LoFi** |
| Prefill output input / weight / output | BF16 / BFP8 / FP32 | BF16 / BFP8 / FP32 |
| Prefill output input / output memory | DRAM / DRAM | DRAM / DRAM |

Both kinds retain BFP8 caches, native decode SDPA, FP32 direct decode QKV,
and its setup cleanup. Decode QKV uses grid `(11,10)`, K11 and automatic legal
subblocks. Decode output retains K16 and FP32 accumulation/output into L1.
The residual path stays FP32 and the final decoder result stays BF16.

Minimal prefill projections use grid `(11,8)`, N8 and subblock1×4. Four configurations are built at setup. Full QKV caps M at2; sliding QKV and full output cap M at4, preserving arbitrary tile tails without forward-time config construction. Full QKV uses HiFi2; sliding uses HiFi4. The wrapper delegates M=1 to the exact existing decode callable and restores full-attention tied K/V after its packed projection. Full output keeps LoFi, its original head concatenation and DRAM placement. No global patch or host tensor read is present.

Sliding output retains the previous 2D program: `(11,8)`, K16, LoFi, per-core
M4/N8 and subblock 1×4 at 1,024 rows. Its 32 row-specific programs are still
created at setup. Full output now uses four minimal-matmul configs, K8,
FP32 destination and DRAM output. Approximation and packer accumulation remain
disabled. These projection settings are separate from prefill SDPA fidelity.

Indexed decode consumes the generalized router's existing device top-8 IDs,
gathers learned-scale routing values in that order, and produces eight expert
slots. All 128 experts remain resident. The merge remains matmul K1; accurate
GELU is fused into the gate/up product. Routed-expert prefill retains active
unions over 32-token chunks, LoFi and K11. Matching phase weight tensors alias.
Shared prefill retains its BF16 weights.

Sliding BFP8 prefill gate repaired the previous long-context row misses and
removed a separate BFP4 gate allocation. Resident expert payload remains
681,967,616 bytes sliding and 428,212,224 bytes full. The minimal backends
reference the existing packed QKV/output weights; persistent tensor/cache
payload is unchanged. See [memory accounting](final_memory_accounting.md)
for selected per-core circular-buffer estimates and exclusions.

Full later-prefill attention retains Q64/K256 on grid `(8,4)`, HiFi2, FP32
destination and full destination sync, with outer chunks of 1,024 tokens.
The first full chunk retains regular causal SDPA. Sliding prefill remains
windowed LoFi SDPA. These defaults assume no Q/K environment override.
All hidden-width sharded norm sites apply only to one-row decode; multi-row
prefill input normalization preserves arithmetic but directly writes its final weighted result to L1 for full attention; other prefill sites retain their original placement. Sliding precise head norms remain
selected after the actual native-headnorm stress failure.

ROW_MAJOR decode RoPE is a caller setup capability, with TILE decode tables
still accepted. Prefill tables remain TILE. Neither arbitrary table-content
caching nor a forward-time host conversion is introduced. Direct router
scores remain FP32 scaled activation × original BF16 weights → FP32, then
centered before BF16 top-8/softmax. Learned expert scaling is retained.

## Selection evidence and validation boundary

The [minimal matrix](minimal_selection.md) contains both attention kinds,
QKV/output K8/K16 and matched 4096/128 baselines. Sliding selected QKV K8 has
whole-prefill median 221,364.003 µs versus 222,538.749 µs baseline; full K16
has 187,922.907 µs versus 190,377.647 µs. These are three-sample synchronized
whole-decoder host medians, not isolated device times. Full QKV K8 completes
the workload but fails position 4211 at PCC 0.994989323531.

Selected QKV stress minima are 0.995555061903 sliding K8 and 0.996252903034
full K16. Individual maximum-context controls pass all 291 sampled rows,
minimum 0.995130377831 sliding and 0.996084490295 full, plus aggregate/tail
and boundary decode checks. These tests do not compare every prefill row.

Eight same-process alternating output pairs reverse the weak sliding result
from the initial matrix: baseline median 221,598.1425 µs versus minimal K16
221,729.5825 µs, with minimal losing seven pairs. Sliding therefore retains
2D multicast output. Full minimal K8 wins seven pairs: 189,143.1020 µs versus
189,347.9410 µs baseline. The 204.839 µs observed improvement is approximately
0.108%; no additive combined-runtime speedup is assumed.

All four commands in `minimal_acceptance_commands.json` return zero.
Candidate checks above ran on `3d51014f…` plus the named probe. Full maximum
context used minimal QKV with the previous output backend. The complete ancestor v5 suite is bound below, including 512-step stress.
Its native profiles are also complete at 169c0d97. Archived b585a21f affected-path regressions pass, with broader v5 coverage retained through its source-delta proof. Native profiles remain separately attributed; neither v4 nor v5 rows are relabeled as v6.

## Archived completed v5 evidence

The following artifacts record ancestor v5 runtime `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`, not current v8. The [final v5 summary](validated_v5_validation_summary.json) reports `all_final_default_correctness_gates_passed`: all 18 journal commands return zero. Both kinds pass B32, prefix continuation, request reuse, BF16-cache compatibility, headline, strict sampled maximum/near-maximum rows and separate Watcher checks; all four pytest cases pass. Integrated 1025/512 stress passes both exact-HF and optimized-versus-fused preservation gates. Optimized HF minima are 0.995555061903 sliding and 0.996252903034 full; direct preservation minima are 0.995158301497 and 0.996327176492. Both v5 native profiles are archived at that same source hash. The archived v6 manifest retained its targeted/current and inherited scope. V7 validation and native profiles remain archived at daa82. Current v8 repeats the full affected contracts, both stress kinds and short/tail controls; unchanged sliding broad coverage remains inherited.

| Artifact | Minimum decode PCC | Sampled prefill rows / minimum PCC |
| --- | ---: | --- |
| [validated_v5_long_262144_layer5.json](validated_v5_long_262144_layer5.json) | 0.9978274142 | 291 / 0.9960844903 |
| [validated_v5_long_262143_layer5.json](validated_v5_long_262143_layer5.json) | 0.9976669813 | 291 / 0.9960844903 |
| [validated_v5_headline_layer5.json](validated_v5_headline_layer5.json) | 0.9950717323 | — |
| [validated_v5_long_262144_layer0.json](validated_v5_long_262144_layer0.json) | 0.9995932677 | 291 / 0.9951303778 |
| [validated_v5_long_262143_layer0.json](validated_v5_long_262143_layer0.json) | 0.9997222924 | 291 / 0.9951303778 |
| [validated_v5_headline_layer0.json](validated_v5_headline_layer0.json) | 0.9952574859 | — |

## Inherited v6 review fixes and targeted evidence

Fresh prefill now requests a tail only when another chunk follows. Final
sliding chunks set `self.tail=None` after computing attention, skipping two
unused BF16 K/V clone allocations (up to 8 MiB together). Nonfinal history,
cache writes, outputs and all projection arithmetic are unchanged.

Paged full-prefill constructs its boundary config at setup. It retains the
primary Q64/K256 program whenever its rounded read fits the logical page-table
capacity; otherwise it selects Q64/K128. The actual S=1025/cache1152 test
records read end 1152, prefill PCC 0.999041244675 and decode PCC 0.999161852007.
A CPU-only check of the exact scalar source covers 6,144 default-geometry
length/capacity cases, all within bounds. See `source_delta_v6.json` for the
four changed methods, 36 unchanged method hashes and the exact inverse-diff
proof. The current validation manifest additionally verifies seven more tight-cache controls, both full-tracker reuse catalogs, both headline/Watcher runs and four pytest cases. Broader unchanged-path B32, prefix/BF16-cache, maximum-context and 512-step stress coverage is retained under the original v5 hash, rather than listed as mandatory unperformed v6 reruns.

## Retained v7 precision setup proof and opt-outs

The new factory argument `prefill_qkv_fidelity="auto"` resolves to HiFi4 sliding and HiFi2 full. Explicit `None` preserves the old source compute object; an explicit fidelity overrides only this minimal-prefill role. `prefill_qkv_minimal=False` leaves the old path unchanged. Exact CPU source checks cover10 auto/override cases and8 constructor clone/alias cases;38/40 methods remain identical to v6. The archived v7 factory had76 keyword arguments. See [source_delta_v7.json](source_delta_v7.json).

## Archived setup proof and other opt-outs

The archived v5 proof resolves the four minimal factory arguments by executing
only the exact three `"auto"` branches with CPU literal stand-ins. All ten
explicit override cases pass, and the source has 75 keyword arguments.
The v6 factory is AST-identical to v5. The earlier 71-argument proof,
alias/program/norm checks and v4/v5 validation remain preserved under their
original hashes in historical records.
The new CPU helper mock checks cover tied/untied tile tails, arbitrary M,
setup-only configs, object aliasing and unchanged M=1 delegation. Neither
source proof is accelerator validation.

`prefill_qkv_minimal=False` restores the prior configured prefill QKV path;
`prefill_output_minimal=False` restores the prior output family. Independent
`*_minimal_block_w` controls accept 4, 8 or 16. Default QKV blocks resolve to
8/16 by attention kind; default output minimal selection resolves to
False/True, with inactive sliding block16 and selected full block8.
To restore the inherited output implementation completely, also set
`prefill_output_grid=None`, `prefill_output_fidelity=None` and
`prefill_output_l1=False`. Disabling only the grid retains LoFi with automatic
program selection; disabling only fidelity retains the explicit program with
inherited compute when the minimal backend is disabled.

Existing native SDPA, cache dtype, direct QKV/router, indexed expert, fused
GELU, norm-site and precision overrides remain selectable. Indexed experts
follow generalized routing; fused GELU follows indexed experts. Conflicting
explicit combinations raise the existing validation errors. A selectable
override is not a correctness endorsement. Public `_forward` still delegates
to `FusedDecoder`.

## Current v8 M block and producer placement

`prefill_qkv_minimal_block_h="auto"` resolves4 for sliding and2 for full; explicit1–4 is accepted. `prefill_qkv_input_l1="auto"` resolves true only for full minimal QKV. Explicit false disables the producer change; disabling minimal automatically disables it. All four M-tail configs are built at setup. Full input normalization uses the original unweighted operation followed by the same learned-weight multiply with L1 output, avoiding a separate copy. One-row decode and other norm sites stay on their original branches.

The exact four-method inverse-AST proof,48 auto/override cases, eight setup cases and18 norm control-flow checks are in [source_delta_v8.json](source_delta_v8.json). The factory now has78 keyword arguments. [Producer selection](prefill_qkv_producer_v7_layer5.json) wins22/32 pairs at matched M2; [grid selection](prospective_qkv_advice_v7.md) retains88 cores. These historical probe timings are separate from current validation and current native profiles. Full reuse now has379 stable cache entries across all nine request lengths. No allocator-peak or additive isolated-speedup claim is made.

Current v8 native rows also prove full minimal-QKV output isL1 because the None output request inherits the input memory. Its tied-KV slice staysL1; the following concat returnsDRAM. Sliding QKV staysDRAM and both attention output projections retain their stated memory. The frozen grid/placement probe narrative that saidDRAM output is corrected in the advice ledger; matched measurements themselves used identical effective output placement.
