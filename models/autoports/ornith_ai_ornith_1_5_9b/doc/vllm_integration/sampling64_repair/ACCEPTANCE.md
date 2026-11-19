# Integrated sampling capacity 64 acceptance

The canonical native sampling extension, common Python propagation and Ornith admission/reset integration are installed in the main checkout. The current model capability is 64 candidates; physical sampling rows remain 32. Native callers that omit the new option retain capacity 32, and greedy remains k1. Unrestricted stochastic sampling and k>64 still use the existing unsupported-device boundary.

This note records completed evidence as of 2026-11-13 and supersedes the *pending device/build* statements in the earlier preparation documents for the checks listed below. It does not close full-model serving acceptance or P2.

| Check | Result and exact evidence |
| --- | --- |
| Untouched 32 baseline and patched default 32 | The revised frozen fixture records 27 combinations (users 1/17/32, widths 64/128/256, ordered/equal/signed-zero values), three seeded draws each. Original record and isolated comparison each passed 1 case/22 deselected. See [original baseline](trace_capture_fixed/legacy32_original.json), [original record log](trace_capture_fixed/device_original-record.log), and [exact comparison log](trace_capture_fixed/device_isolated-compare.log). |
| Isolated native suite | **22 passed, 1 deselected**, including explicit 64, tail membership/distribution, ties, row identity, cache switching, parameter reload and trace replay. The deselected baseline is covered separately above. See [suite log](trace_capture_fixed/device_isolated-suite.log). |
| Main native rebuild | All three unity compiles and both links returned 0; total command time 76.775s. [Provenance](ornith_integration/main_rebuild_20261113T212253Z/provenance.json) includes native source hashes, commands and installed-library hashes. [Main rebuild/import log](../sampling64_main_rebuild.log) confirms both extensions loaded from main, not the private validation checkout. |
| Integrated TP4 sampler | [sampling_capacity64.json](../../../readiness_vllm/sampling_capacity64.json) reports pass with layers 0/3, logical B3, context 256, physical 32 rows and allocation tracking enabled. All four k40/64 × penalties-off/on cases pass, with 64 draws per user, observed rank 32+ candidates and exact eager/trace/resume outputs plus seed/history state. Primitive calls explicitly use capacity 64 with gathered shape [1,1,32,256]. This is synthetic sampling correctness evidence, not a full-model serving benchmark. |
| Adapter compatibility | [adapter_decode.json](../sampling64_adapter_checks/adapter_decode.json) passes reduced layers 0/3, B3 checks covering inactive-state preservation, pending submissions, remaps, scoped reset, host logits and rejection before mutation. [seed_decode.json](../sampling64_adapter_checks/seed_decode.json) passes all six continuity checks across first decode, resident advance, remap/reset, host interval and device return. |

The installed library SHA256 values recorded by the successful rebuild are:

- Main `build_Release/lib/_ttnncpp.so`: `663b4fb768ee4386ed16138579322b1cb33cecab58e3228cc648f5d3868cdfdd`.
- Main `ttnn/ttnn/_ttnn.so`: `eb263df517618f401e0ecdeffa0d3ab9a34be156a199e2d3a065140683d787af`.

The initial isolated run failed because its trace test warmed a different output signature; [TRACE_CAPTURE_FIX.md](TRACE_CAPTURE_FIX.md) preserves that diagnosis and the prior failure evidence. The successful `trace_capture_fixed/` run uses the corrected fixture SHA `1282decc7fe3ea5f176110c4cfeeaa89b51511ae19e57d5a732d102934134dd0`. No native arithmetic change was needed for that test repair.

Only `current_stage9` in [serving_contract.json](../../full_model/serving_contract.json) advances to capacity 64. Its historical full-model rows, capability mapping and 32-boundary evidence remain unchanged. Full 32-layer current-source tracked serving acceptance, P2 harness checks, tracking-off measurements, final gates and independent review remain pending. Neither this native repair nor the synthetic sampler result waives those requirements or the separately documented replay limitations.
