# AutoFix: match shared-MLP decode policy across TP1 and TP4

## Starting evidence

The independent source-only diagnosis is [AUTODEBUG_stack.md](AUTODEBUG_stack.md).
The unchanged stack gate failed with hybrid experts and the fused tail:
[stack_hybrid_tail.json](stack_hybrid_tail.json), `stack_hybrid_tail.log`.
The saved original runtime is `runtime_hybrid_tail.py.txt`, SHA256
`3f3fc2ed3ad8ff21bcc19cb37519438187dbac5fd1e4cfdcfcf6aacd66504a20`.

The test compares real-weight layers 0 and 5 with direct device handoff,
33 prefill tokens, and decode positions 33/34 in one advancing trace. Its
synthetic fixture and skipped intermediate layers remain unchanged. The
acceptance threshold remains PCC >= 0.995 for every recorded comparison.

## Hypothesis and isolated intervention

**Hypothesis:** TP4's shared-MLP decode numerical policy differs from the
optimized TP1 baseline sufficiently to cause this stack discrepancy.

Source confirmed that TP1 shared decode uses BFP8 weights for sliding layers
and BFP4 for full layers, explicit LoFi compute, FP32 destination accumulation
disabled, packer accumulation enabled, and BF16 outputs. TP4 instead used the
BF16 prefill weights with default `linear` kernels for decode as well.

The parent implemented one opt-in shared-decode policy change:

- `optimized_shared=False` remains the factory default.
- `_SharedMLP.configure_decode` prepares separate BFP8/BFP4 weights at setup.
  It preserves the loader's padding and per-rank `[up_i, gate_i]` ordering;
  down weights use the matching row partition.
- One-token shared-MLP calls use those weights with the explicit baseline
  compute policy and BF16 L1 outputs. Automatic matmul program selection is
  retained; the implementation does not claim identical TP1/TP4 geometry.
- Prefill continues to use the original BF16 projections. GELU, local
  intermediate splitting, reduction and the decoder tail remain unchanged.

Inspection of the diff from the frozen runtime confirms no other runtime
changes in this experiment. TP1 `optimized_decoder.py` retains SHA256
`5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.
Removing only the new flag's parser, forwarding and report lines from the
new stack harness reconstructs its exact original SHA256
`3c015c53fee4c049c29cea2ad43e541ff3fdf76ea6d2906ad2bf4a18aa8e0d14`.
That check was performed by this investigator without running target code.

## Experiment and results

Parent-run experiment, reported exit 0:

```sh
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.test_multichip_stack \
  --hybrid-experts --fused-tail --optimized-shared \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/stack_shared_policy.json
```

Evidence: [stack_shared_policy.json](stack_shared_policy.json) and
`stack_shared_policy.log`. The log records both TP completions, `passed=True`,
and normal device shutdown. Source hashes match the inspected implementation:

- Runtime: `5f75aa5f250b0424b93cfdd5a9fa036f9b322fca38bb7eab40b8c2f5c4764dff`.
- Harness: `ef3a5f297b197ea1c70eb7e5d97860f131ec53cd257381065d74d5b5de15c81b`.

| Comparison | Original PCC | Shared-policy PCC |
| --- | ---: | ---: |
| Layer 0 prefill | 0.9999606744 | 0.9999606744 |
| Layer 0 decode 33 | 0.9999570691 | 0.9999691440 |
| Layer 0 decode 34 | 0.9999312895 | 0.9999201119 |
| Layer 5 prefill | 0.9993982243 | 0.9993982243 |
| Layer 5 decode 33 | **0.9947994328** | **0.9998315790** |
| Layer 5 decode 34 | **0.9931027566** | **0.9997725434** |

Both recorded prefill PCC values are exactly unchanged. The report does not
save cross-run tensors, so this observation alone is not a cross-run bitwise
identity check. All six comparisons pass. Both reports retain exact output
replicas, deterministic repeated trace outputs, direct interlayer handoff,
independent caches and a clean device-only guard.

**Verdict: verified at the shared-decode policy boundary.** Matching that
policy is sufficient to fix the original failing gate. This evidence supports
retaining the opt-in implementation. It does not distinguish the individual
effects of weight quantization, explicit compute settings, output placement,
or their induced program selection within that branch. It also does not
establish whether the improvement arose inside layer 5 or partly through
changed layer-0 outputs. Those narrower claims are unnecessary for this fix.

Router rank amplification was a diagnosis hypothesis. No router IDs or score
margins were measured in this A/B, so no routing explanation is claimed.
Other attention/cache/backend differences identified by AutoDebug were left
unchanged; this passing control provides no reason to alter them for this
failure. The repair matches the baseline's lower-precision shared policy,
rather than relying on a blanket higher-precision change.

## Final status and remaining validation

The original stack failure is fixed with `--optimized-shared`; the option is
retained and remains opt-in. Runs without it retain the previous behavior.
This report does not claim a default change or completion of Stage 04.

The parent is responsible for paired single-layer 4096/128 validation for
both attention kinds on the retained candidate, plus the stage's remaining
required gates. The separate decode weights coexist with BF16 prefill
weights, so update full-stack residency accounting before capacity acceptance.
No performance improvement is inferred from this correctness experiment.

This investigator inspected source, hashes and artifacts only, created this
report, and ran no hardware checks or implementation edits. The parent owns
the stage work-log update and subsequent hardware verification.
