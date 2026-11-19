# Independent interpretation of the matched HF control

CPU-only artifact/source review, 2026-11-14. The parent executed the control; this review did not run another model or device job. No production source, rejection ledger, frozen threshold, or acceptance status was changed.

**The tested original-HF CPU backend is not invariant across B1/B2 under the frozen numerical gate. TT's forced-43 failure is nevertheless larger and includes a top-5 failure that this HF control does not reproduce. The TT gate remains failed.**

## Observed gate results

The HF control completed in 5.520792 seconds with peak RSS 5,077,660 KiB (4.842434 GiB). Both repeats give identical results. The table preserves the frozen scorer's reported values; it does not substitute a different threshold or a tie-aware score. “Top1/5/100” means agreement with the same implementation's B1 reference, not agreement of TT with HF.

| Implementation and comparison | Whole PCC | Minimum row PCC | Top1 / top5 / top100 | Max absolute difference | Frozen verdict |
| --- | ---: | ---: | --- | ---: | --- |
| HF natural B1 → mixed B2, prefill | 0.998572206 | 0.997263274 | 0.5 / 1 / 1 | 0.30078125 | Fail: top1 |
| HF natural B1 → mixed B2, forced 42 | 0.996641706 | 0.994326185 | 0.5 / 1 / 1 | 0.3857421875 | Fail: top1 |
| TT B1 → B2, forced 42 | 0.998997478 | 0.998263122 | 1 / 1 / 1 | 0.2265625 | Pass |
| HF natural B1 → mixed B2, forced 43 | 0.992546833 | 0.987764712 | 1 / 1 / 1 | 0.78125 | Fail: whole and minimum-row PCC |
| HF width-33 padded B1 → mixed B2, forced 43 | 0.992933947 | 0.988435452 | 1 / 1 / 1 | 0.765625 | Fail: whole and minimum-row PCC |
| TT B1 → migrated B2, forced 43 | 0.979649656 | 0.970201804 | 0.5 / 0.5 / 1 | 1.044921875 | Fail: both PCC, top1, top5 |
| HF natural B1 → mixed B2, forced 44 | 0.998464051 | 0.996832626 | 0.5 / 1 / 1 | 0.359375 | Raw fail: top1 tie-selection artifact, described below |
| TT solo request0 → B2-history survivor B1, forced 44 | 0.994277325 | 0.994277325 | 1 / 1 / 1 | 0.453125 | Fail: whole PCC |

Frozen thresholds are finite logits, whole PCC >=0.995, minimum row PCC >=0.99, top1 =1, top5 >=0.98, top100 =1. The TT forced-44 row has a **different active-width history** from the HF mixed-B2 forced-44 row: TT switches to one surviving row after earlier B2 decode, while the current HF control retains B2. It is included to preserve the outstanding TT failure, not to claim a matched HF explanation for it. The HF control has no B2→B1 survivor run.

For forced 43, TT's minimum row PCC is lower by about 0.017563 and whole PCC lower by about 0.012897 than natural HF's. More substantively, TT loses the B1 reference top token from one row's top five; HF retains it in both rows, and HF top1 IDs actually agree for both rows. It is unsupported to call these equal severity, to attribute the excess automatically to an HF limitation, or to turn the failed TT gate into a pass.

## Repeatability, permutation, migration and padding

All 28 repeat comparisons have bitwise-identical logits; all 160 captured hidden comparisons are exact. All eight initially reversed-row comparisons have exact logits and all 64 hidden comparisons are exact. All eight original-HF `cache.reorder_cache([1,0])` comparisons have exact logits and all 64 hidden comparisons are exact. These counts cover both repeats and all four phases. HF also asserts the visible QSA logical-position history at every step. This supports deterministic same-geometry behavior at the inspected output/hidden boundaries. It is not an exhaustive comparison of every HF cache tensor.

The TT v5 artifact independently records private migration versus uninterrupted B2 exact for forced 42, 43 and 44, plus pooled migration versus uninterrupted B2 exact and solo-repeat exact. Consequently, neither artifact's observed cross-geometry divergence can simply be labeled a demonstrated migration bug. These results do not waive unrelated lifecycle requirements.

Padding-only comparisons all meet the frozen gate. The 33-token request is already unpadded and all its comparisons are exact. The short request's prefill last-token logits are exact; its subsequent natural-versus-padded B1 decode logits differ slightly: PCC 0.999972975, 0.999964162 and 0.999968390 for inputs 42/43/44, with max differences 0.03125, 0.0390625 and 0.0390625. All top-k agreements remain one. Width-33 padded B1 versus B2 still fails both PCC requirements at 43. Thus this particular padding difference does not account for the observed HF batch-geometry failure.

## Frozen scorer tie defect, preserved rather than hidden

Seven same-geometry comparisons report `passed=False` despite `exact=True` and max absolute difference zero. At request0 after forced 44, mixed-B2 logits have equal global maxima 4.0625 for token IDs 83908 and 201617. The scorer obtains the reference ID with `argmax` (83908) but the candidate ID with `topk(100)[0]` (201617). Its top1 agreement is therefore 0.5 even when both full-vocabulary tensors are bitwise identical. This affects the forced-44 repeat/reversed/migrated raw verdicts. It also explains natural-HF B1 versus B2 forced-44's raw top1 failure: direct `argmax` IDs agree for both requests, and the B2 tied alternative is also a valid greedy maximum.

The existing source and raw results were left unchanged. This is a concrete scorer tie-policy inconsistency, not measured nondeterminism. A future correction must preserve thresholds and explicitly define consistent top1 semantics; it must not be used to erase the forced-43 PCC or TT top5 failures. Small PCC values above one in exact controls likewise come from the scorer's floating-point reduction, not nonidentical tensors. None is near the failing forced-43 margins.

Independent offline deterministic-ID measurement uses `reference.argmax(-1)` versus `actual.argmax(-1)` on the saved natural-B1/mixed-B2 tensors. It is an additional analysis measurement, not a replacement for the frozen verdict above:

| Phase | Reference IDs, requests 0/1 | Actual IDs, requests 0/1 | Direct argmax agreement | Raw frozen top1 |
| --- | --- | --- | ---: | ---: |
| Prefill | 56680 / 90261 | 27200 / 90261 | 0.5 | 0.5 |
| Forced 42 | 221888 / 67012 | 221888 / 137930 | 0.5 | 0.5 |
| Forced 43 | 71111 / 77693 | 71111 / 77693 | 1 | 1 |
| Forced 44 | 83908 / 206690 | 83908 / 206690 | 1 | 0.5 |

The concrete forced-44 request0 fixture has B1 logits 4.03125 at ID 83908 and 4.0 at ID 201617; B2 has 4.0625 at both IDs. B2 `topk(100)[0]` returns 201617 whereas B2 `argmax` returns 83908. All same-geometry repeated/reversed/migrated direct-argmax comparisons are one because their full tensors are exact. This leaves the recorded raw failures intact and provides old/new metric evidence for a separately reviewed scorer correction.

## What the observation establishes

The assertion “identical logical requests must always retain exact logits, identical chosen top1 IDs and the frozen PCC bounds when run as B1 or B2 in the original model” is refuted for this recorded original-HF CPU backend. The observations support a narrower statement: batch geometry can cause deterministic numerical divergence even with original checkpoint arithmetic, matched logical histories and positions, stable repeated runs, and exact row-permutation controls. The failed B1/B2 invariant is therefore not exclusive to TT.

They do **not** show the same producer or magnitude. At the captured HF prefill last-token boundaries, layer0 is exact for both requests and layer1 is already slightly different; this control does not tap every operation or every prompt row, so it cannot identify the first causal HF producer. The earlier real TT fixture instead isolates its first same-input divergence to layer0 routed reduction. The TT-specific singleton FP32 DEST versus batched native BF16 DEST explanation comes from that TT source/fixture evidence, not from the HF result. Different implementations' within-backend PCC errors cannot be subtracted into a certified error budget.

Outstanding work therefore includes the TT forced-43 top5 loss and larger PCC drift, the forced-44 survivor failure, and the scorer tie inconsistency. Any correction still has to preserve the selected dtype/fidelity policy and pass the frozen gate under consistent documented scoring. This addendum authorizes neither wider precision, singleton fallback, changed thresholds nor stage completion.

## Backend and provenance limits

This is the real reduced 0/1/3 stack, not full48. It uses PyTorch 2.11.0+cpu and Transformers 5.22.2, four intra-op/one inter-op threads, original eager attention, original DynamicCache, BF16 checkpoint parameters, no autocast/compile or enabled external kernel substitution, and MKLDNN enabled. `deterministic_algorithms=False` is recorded, while the actual two repeats are exact. PLE selects exact BF16 checkpoint rows with quantization disabled; all other layer operations are original installed HF code. The model is loaded from pinned revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. HF prefills each padded batch in one call; TT uses its selected chunk strategy. This is not a control for another CPU build, GPU HF backend, TT quantization, full-model quality, page-table ownership or trace behavior.

The independent review verified the `.pt` SHA256 and recomputed all four natural-HF combined comparisons from stored full-vocabulary tensors with the original four CPU threads; values match the JSON exactly. An initial recomputation at the interpreter's default thread count changed low-order PCC digits (about 5e-7), with no verdict change; matched-thread recomputation removed that difference. No model was rerun.

- HF JSON: `shared_pool_hf_control_v1.json`, SHA256 `7c71d154da302d3db82c4edc2c5a62a7015e0138aeb8931186535bef84b04d68`.
- HF tensors: `/home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v1.pt`, SHA256 `2b97088501dc639ef223ff3ee55ed13244ff1b8d8a2e748ae88c15a4d327fbed`.
- HF source: `tests/probe_shared_pool_hf_control.py`, SHA256 `9626977966535f18442411481ff1de66ca259043f1b94d8339cecc43d2220989`.
- TT comparison: `shared_pool_generator_v5.json`, SHA256 `7ed9b4a8ca5e40083fbf6109ded3e44f751808e90dd351872ffdcbd5c6776767`.

The scheduled command and preparation scope remain in `shared_pool_hf_control_plan.md`; that document's “not run” statement describes preparation time and is superseded by the observed execution recorded here.

## V2 addendum: matched B2-to-B1 survivor history

The parent subsequently ran `shared_pool_hf_control_v2`: 16 histories/64 forwards completed in 9.629750 seconds, peak RSS 5,084,500 KiB (4.848957 GiB). This independent review loaded the saved tensors on CPU, verified their hash, and recomputed all 32 recorded survivor comparisons with the original four CPU threads. Their frozen metrics match the JSON exactly. No model or device was rerun for this review.

The new history closes the V1 active-width gap for original request0: B2 prefill and forced 42 in order `[0,1]`; native HF reorder to `[1,0]` before forced 43; native HF subset selection `[1]` before forced 44, retaining original request0 as B1 at logical position 5. Recorded visible length is 6; the physical cache length 36 includes the original 30 masked prefix positions. Native HF cache selection retains conv/recurrent/KV/indexer/position state; no state reconstruction, TT page-table operation, or sampler is involved. Thus the logical history and B2→B1 transition match the TT44 survivor test, while the documented HF padding and prefill implementation differences remain.

At forced 44, raw frozen results are:

| Reference → actual survivor | Whole/minimum row PCC | Top1 / top5 / top100 | Max absolute difference | Frozen verdict |
| --- | ---: | --- | ---: | --- |
| HF uninterrupted natural B1 → HF B2-history survivor B1 | 0.995195317 | 0 / 1 / 1 | 0.546875 | Fail: top1 |
| HF uninterrupted width-33 padded B1 → HF survivor B1 | 0.994891819 | 0 / 1 / 1 | 0.5625 | Fail: whole PCC and top1 |
| HF uninterrupted mixed B2 row0 → HF survivor B1 | 0.994139507 | 0 / 1 / 1 | 0.609375 | Fail: whole PCC and top1 |
| HF migrated B2 row0 → HF survivor B1 | 0.994139507 | 0 / 1 / 1 | 0.609375 | Fail: whole PCC and top1 |
| TT uninterrupted B1 → TT B2-history survivor B1, v5 | 0.994277325 | 1 / 1 / 1 | 0.453125 | Fail: whole PCC |

These survivor HF top1 failures are **not the V1 tie-scoring artifact**. The actual survivor has a unique global maximum 3.8125 at ID 193953; both `argmax` and `topk(100)[0]` select 193953. The natural B1 reference has unique maximum 4.03125 at ID 83908. Padded B1 has a tie at 4.03125 between IDs 83908/201617, and continued B2 has the same pair tied at 4.0625; their deterministic `argmax` is 83908. Therefore direct argmax-to-argmax agreement is zero for all four survivor references, exactly matching the raw frozen top1 disagreement. The survivor repeats themselves select 193953 identically. The prior continued-B2 tie artifact still exists elsewhere in V2 and is not rewritten.

Before selection, the survivor runs' outputs and captured hidden tensors are bitwise equal to both uninterrupted and migrated B2 histories through forced 43. Independently checking both requests across the two repeats gives 24 exact full-vocabulary comparisons and 96 exact hidden comparisons. Across the two complete survivor repeats, all seven available request/phase output pairs and 28 hidden tensors are exact, including forced 44. All 80 full-vocabulary and 320 hidden tensors from the fourteen pre-existing V1 histories are also bitwise equal between V1 and V2. This excludes an observed repeat/setup-history variation as the explanation for this new difference.

Against the migrated-B2 continuation at44, the first *captured layer-output boundary* that differs is layer0 (max difference 0.000244140625, PCC 0.999999071); differences are larger at the terminal hidden output (PCC 0.993865851). These hooks are not an operation-level HF producer trace or a complete cache-state equality proof. They support observed deterministic width-transition sensitivity, not identification of a particular HF arithmetic operation.

V2 therefore establishes that this original-HF CPU backend can fail the frozen survivor gate under the matched logical B2→B1 history, even while the same-width history and repeat controls remain exact. It refutes an unconditional claim that such an active-width transition must satisfy the frozen gate in the original model. The HF and TT survivor failures differ: natural-HF fails top1 while its PCC narrowly clears 0.995; TT retains top1 but misses the whole-PCC bound. Neither supplies a scalar error allowance for the other.

This observation does not excuse TT43's larger PCC drift and top5 loss: the unchanged matched HF43 result retains top5=1, whereas TT43 remains 0.5. The TT43 excess and the TT44 frozen failure remain work under the existing contract. No threshold, scorer implementation, raw JSON/PT, production policy or acceptance ledger was edited. A tie-scoring repair remains separate from both observed numerical failures.

V2 provenance:

- JSON `shared_pool_hf_control_v2.json`: SHA256 `e71a47a861b4d9482592109cc18b18a1ebcd477411dbd07c4cf160f4c8257155`.
- Tensors `/home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v2.pt`: SHA256 `888fc0c6e24f8a611b91dc4d6e535425c4ef9a13cc2b43161d447b8f5dcfdc15`.
- Executed helper: SHA256 `559358ecd97438c17d24fe9adf6650f258209011ac123a2d2c8bd8d38b47f646`.
- Backend/checkpoint settings and source/argv hashes remain recorded in the V2 JSON; the exact command is in the V2 preparation section of `shared_pool_hf_control_plan.md`.

## Separate V3 rescore after the parent's comparator correction

After V2 capture, the parent changed only the diagnostic top1 measurement to compare `actual.argmax(-1)` against `reference.argmax(-1)`. The original scorer is archived in `sources/shared_pool_geometry_scorer_v1.py.txt`. Neither PCC thresholds, top5/top100 metrics nor raw V1/V2 artifacts changed. The preceding tables intentionally retain the raw frozen scores.

The authorized offline rescore is now saved separately as `shared_pool_hf_control_v3_rescore.json`. Its reproducible command is:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.rescore_shared_pool_hf_control --input models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v2.json --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v3_rescore.json
```

The rescorer verifies the input tensor hash, loads the original scorer function from its archived AST, and first reproduces every one of the 144 raw scores exactly using four CPU threads. It then applies the corrected scorer and asserts that only `top1_agreement` and the derived `passed` field can change. All 15 changed comparisons are continued-B2 forced 44 tie cases: eight natural/padded B1 cross-geometry aggregate/per-row records, four reversed/migrated records and three repeat records. Recorded failures fall from 55 to 40; these are comparison records including repetitions and overlapping aggregate/per-row checks, not 55 or 40 independent failures.

Every forced 43 and every survivor comparison is unchanged. In particular, survivor ID 193953 remains a unique maximum, so its disagreement with ID 83908 is a real deterministic-ID change and its failed verdict survives correction. No model was rerun. The corrected metrics remove the demonstrated tie-order inconsistency; they do not waive the remaining HF or TT numerical gates.

- V3 rescore JSON SHA256: `ebdceface661e327d422c345b3e309abd79799d962b5c151e7083050ce947fd9`.
- Rescorer source SHA256: `566cf81b969225417472e353e7fec9220b4fbfc35b4f902f3f73cd6f84be315c`.
- Original scorer source SHA256: `83a9079df74a105790adb0913f6aeb321b0e3bc384ac364f26fc850add77b6d9`.
- Corrected scorer source SHA256: `7238400e301405f2c5949453927941d2daba4fb8e912a77ee45d3eb0f011d861`.

The new offline rescorer was checked with the repository-pinned cached Black 23.10.1, isort 5.13.2 and autoflake 2.3.1. Its full-artifact old/new assertions passed. It imports no TTNN code and opens no device.
