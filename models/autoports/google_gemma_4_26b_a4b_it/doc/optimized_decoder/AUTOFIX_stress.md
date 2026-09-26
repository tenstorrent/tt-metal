# AutoFix: optimized 1025/512 stress differences

Current interpretation: the repairs below restore **Gaussian diagnostic
preservation**. Subsequent actual-text fixtures show the original defaults
pass every HF check at both 4096/128 and 1025/512. Per the user's explicit
constraint, Gaussian failures cannot veto real-input precision wins, so these
repair controls do not mandate production BFP8 down or three-term QKV. See
the actual-input follow-up and `precision_evidence.md` for the current,
provisional selection evidence. Original failures remain recorded unchanged.

## Starting evidence

`AUTODEBUG_stress.md` investigates real-weight, seed-42 Gaussian inputs with
1025-token prefill and 512 traced decode steps. The initial optimized path
fails direct 0.995 equivalence against fused at five sliding positions and
two full-attention positions. Both paths also have exact-FP32-HF failures.
Original commands are in `final_stress_commands.json`; paired controls are
in `stress_pair_commands.json`. No failed positions or thresholds changed.

## Hypothesis experiments

| Hypothesis | Independent control | Result / verdict |
| --- | --- | --- |
| Sliding BFP4 expert down consumes the output-error budget | `--defaults --default-overrides '{"expert_down_dtype":"bfloat8_b"}'` | Removes direct misses 1066/1108/1175; 1428/1519 remain. Verified contribution; insufficient alone. |
| Sliding two-term QKV is sufficient for the longer stream | `--defaults --default-overrides '{"qkv_terms":3}'` | Repairs large 1428/1519 direct differences; small 1036/1108/1175 remain. Two-term policy is not sufficient for preservation on this stream. |
| Sliding sharded norms are the primary cause | `--defaults --default-overrides '{"sharded_norms":false}'` | Large differences persist and an additional small miss appears. Refuted as a sufficient repair. |
| Full post/common sharded norms change preservation | Norms off, then `--defaults --default-overrides '{"sharded_norm_site":"common"}'` | Both pass every direct comparison. Restoring post-attention norm alone suffices; sharded common norm remains. |

After the independent sliding effects were demonstrated, the combined
`{"qkv_terms":3,"expert_down_dtype":"bfloat8_b"}` control passes prefill and
all 512 direct comparisons, minimum decode PCC **0.998895223973025**.
Full attention's `{"sharded_norm_site":"common"}` passes every direct
comparison, minimum **0.9960610282676844**, while retaining two-term QKV and
BFP4 expert down. Both resulting HF failure sets exactly match fused.

Exact commands and exit codes: `stress_down8_commands.json`,
`stress_normoff_commands.json`, `stress_terms3_commands.json`,
`stress_combined_commands.json`. CPU per-output verification:
`stress_comparison_layer0.json`, `stress_comparison_layer5.json`, and the
parent's `stress_verified_comparison_layer{0,5}.json`. Source and artifact
hashes are preserved in the paired reports. The hardware commands still exit
on shared exact-HF failures; a passing direct comparison is not an HF pass.

## Final status

The new optimized-versus-fused stress differences are fixed experimentally
with the layer-specific controls above. Final real-text precision selection,
runtime-default integration and final headline/public-contract verification
remain the parent stage owner's work. This investigator changed only test
helpers and diagnostic reports, and opened no device.

`compare_decoder_outputs.py` was validated on every saved control;
`probe_optimized_stress_routes.py` is available for optional boundary/route
inspection and requires exact equality to uninstrumented outputs. Both pass
Black with target Python 3.10 and Python compilation. The route probe remains
hardware-unverified. The separate CPU-cache investigation explains a subset
of inherited HF outliers; no universal cache explanation is claimed.

## Durable default preservation runner

`tests/run_optimized_stress.py` serially runs fused and optimized defaults for
both layers at 1025/512, then calls the CPU output comparison helper. The
optimized child uses `run_optimized_contract --contract run_decoder`, which
forbids `FunctionalDecoder._forward`. The runner requires both prefill HF
checks, every direct output comparison, deterministic trace replay, clean
runtime audits, the program-cache guard, complete per-position checks, and
no optimized HF failure at a position where fused passes. Its
`preservation_passed` and `exact_hf_passed` fields remain separate. A child
exit 1 is only accepted when its complete report and final traceback agree
that the unchanged decode-HF assertion failed; unrelated errors fail the run.

Recorded inputs are explicit:

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_stress \
  --input-fixture-template 'models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer{layer}_1025_512.pt'
```

The template must point to the actual fixture location. Alternatively pass
`--input-fixture 0=PATH --input-fixture 5=PATH`. Omitting input arguments runs
the original Gaussian stream and explicitly labels it diagnostic evidence;
it does not silently substitute text inputs or decide production precision.
Every invocation keeps timestamped raw reports, tensors, logs, comparisons,
and an exact command journal under `doc/optimized_decoder/stress_default` by
default. Separate latest `gaussian_summary.json` and `recorded_summary.json`
preserve visibility of both input modes; source and fixture hashes are checked
for changes during the run.

This investigator validated the orchestration only on CPU with hardware child
processes mocked and the existing saved 512-step outputs. It accepts preserved
shared-HF failures, rejects the original optimized regressions, missing fallback
and program-cache guards, and an unrelated teardown exception. A separate mock
checks fixture-template expansion and per-layer fixture hashes. No TTNN import
or hardware execution occurred in these checks. Black, Python compilation and
`git diff --check` pass; actual default-run execution remains parent-owned.

## Actual-text follow-up: precision selection remains open

`actual_text_fixture_manifest.json` records actual repository-text tokens,
pinned HF/checkpoint provenance, genuine layer-0/layer-5 inputs, and exact
BF16 transport tensors. Both HF and TT consume the same transported inputs;
the continuations are teacher-forced corpus tokens. The following results
were read directly from completed JSON artifacts:

| File | Workload | Minimum decode HF PCC | Decode host median, us |
| --- | --- | ---: | ---: |
| `actual_text_original_default_layer0_4096.json` | Sliding 4096/128 | .995388274 | 2138.290 |
| `actual_text_original_default_layer5_4096.json` | Full 4096/128 | .998328714 | 2270.984 |
| `actual_text_original_default_layer0_1025.json` | Sliding 1025/512 | .995526104 | 1997.579 |
| `actual_text_original_default_layer5_1025.json` | Full 1025/512 | .998456538 | 1806.089 |
| `actual_text_stress_down4_layer0.json` | Sliding 1025/512, three terms/grid110/BFP4 down | .995533782 | Not recorded |
| `actual_text_stress_down8_layer0.json` | Same except BFP8 down | .996275731 | Not recorded |

All six pass prefill and every recorded decode HF check. The original policy
already has two-term QKV and BFP4 down; therefore a slower high-precision
policy is not required solely by its better Gaussian preservation.
`actual_text_repaired_layer0_4096.json` also passes (.999802708), but its
2210.008 us host median changes QKV geometry as well as precision and is not
an isolated down-dtype timing comparison.

Full-attention BFP4 gate/up now passes both the actual headline and all 512
stress steps (`actual_text_gate4_layer5_4096.json`, .995955897;
`actual_text_gate4_stress_layer5.json`, .996907513). Sliding BFP4 gate/up
fails actual positions 4130/4138 across the tested 11/22/44-worker and K11/K22
configurations; exact policies and results are tabulated in
`precision_evidence.md`. Both prefill gate/up BFP4 headline controls pass.
These are real-input distinctions, unlike a blanket veto from Gaussian data.

The newly completed native-SDPA full-sync and BFP8-cache actual headline
controls pass both layer kinds. The native host medians are 1384.416 us
sliding and 1495.726 us full, with minimum PCC .995161226/.998370139.
Cache-only BFP8 controls give .995293268/.998335777 at 2100.032/2253.695 us.
These are warmed host-wall measurements, not device timings. Longer actual
stress, combined policies, final contracts, and default selection remain
pending; earlier Gaussian rejection does not close these candidates.

The complete evidence ledger retains exact files, command journals, prefill
PCCs/timings, provenance limitations, and per-group pending decisions. No
Gaussian result was deleted, marked passed, or hidden when the actual-input
results changed the selection rationale.
