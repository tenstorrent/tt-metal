# AutoFix: sliding-prefill gate/up precision

## Starting evidence

`AUTODEBUG_sliding_prefill.md` is the fresh source-only diagnosis. The original
actual-text maximum-context run, `verified_long_262144_layer0.json`, passes its
aggregate .995 and endpoint/decode gates but has four sampled rows below .995:
32, 71679, 140287, and 262120. The exact-input fused control passes all 291 rows.
The parent requested that these real-input misses be repaired and covered by
an actual-input sampled-row gate. Synthetic-only misses are not a selection
veto and do not acquire this additional gate.

All optimized controls below identify runtime
`e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`,
fixture `actual_text_long/actual_text_layer0_262144_0.pt` with SHA256
`74cfb212f7f18fbc2bb7ebc3aa5f8f10619766a437b0ea93f86c8ad09f15c596`,
and the unchanged exact HF query oracle
`actual_text_long/actual_text_layer0_262144_reference.pt`.

## Hypothesis experiments

The parent executed all device controls. This investigator inspected the source
and saved results; it did not run hardware. Exact argument lists and return
codes for the four weight controls are in `sliding_precision_commands.json`.
Each command uses `run_optimized_contract --contract long_context --layer 0
--length 262144 --threads 4` with the fixture/reference above and exactly the
listed `--default-overrides` JSON.

| Hypothesis / isolated override | Result artifact | Aggregate PCC | Misses / 291 | Verdict |
| --- | --- | ---: | ---: | --- |
| Baseline BFP4 gate/up | `verified_long_262144_layer0.json` | .998877366858 | 4 | Reproduced diagnostic discrepancy |
| Prefill SDPA LoFi is sufficient cause: `{"prefill_attention_fidelity":"HiFi2"}` | `actual_long_sliding_hifi2.json` | .998962441965 | 5 | Refuted as sufficient repair |
| Prefill gate/up weight loss: `{"prefill_dtype":"bfloat8_b"}` | `actual_long_sliding_pref_gate8.json` | .999310065697 | 0 | Verified passing intervention |
| Prefill down weight loss alone: `{"prefill_down_dtype":"bfloat8_b"}` | `actual_long_sliding_pref_down8.json` | .999352150014 | 3 | Refuted as sufficient repair |
| QKV weight loss alone: `{"qkv_weight_dtype":"bfloat16"}` | `actual_long_sliding_qkv_bf16.json` | .998863544311 | 5 | Refuted as sufficient repair |
| Output weight loss alone: `{"output_weight_dtype":"bfloat16"}` | `actual_long_sliding_output_bf16.json` | .998875459104 | 4 | Refuted as sufficient repair |

All four weight-control processes returned zero because the existing enforced
gates were aggregate plus endpoint checks; that return code does not turn the
remaining sampled-row misses into passes. Down BFP8 even has a higher aggregate
PCC than the passing gate/up candidate while retaining three misses.

| Query | Baseline gate/up BFP4 | Isolated gate/up BFP8 |
| --- | ---: | ---: |
| 32 | .994265170838 | .995109737003 |
| 71679 | .994287855709 | .995689577980 |
| 140287 | .982345507834 | .997184060335 |
| 262120 | .993570005055 | .998259262439 |

The gate/up candidate's minimum is position 32 at .995109737003. Its complete
saved `precision_policy` differs from baseline only at `prefill_expert_gate`.
Its complete saved decode-result list is exactly identical. This verifies the
prefill gate/up intervention with unchanged cache, down dtype, attention math,
and decode policy. It does not establish an all-token or full-model guarantee,
and the worst sampled row has a small margin.

## Source-backed repair

`OptimizedExperts.__init__` constructs separate prefill gate/up tensors from
the original packed source weight (`tt/optimized_decoder.py:102–109`).
`_active_prefill` first projects that weight, then applies GELU/multiply before
the down projection (lines 224–237). Gate/up precision can therefore change
the routed expert result without changing the input, attention, router scores,
selected expert IDs, shared MLP branch, or K/V cache. Routing has already run
before the expert call (`tt/fused_decoder.py:179–180`). The controlled repair
does not require a speculative router-flip explanation.

`sliding_prefill_gate8.patch` contains two minimal changes. The parent has
**applied it** after the isolated controls completed:

1. Set `OptimizedDecoder.from_state_dict(prefill_dtype="auto")` to choose
   BFP8 for sliding attention and BFP4 for full attention. Explicit dtype
   overrides remain authoritative. Retain BFP4 prefill down, LoFi expert and
   sliding SDPA compute, BF8 QKV/output weights, BF8 cache, and all decode
   choices.
2. In `tests/long_context.py`, when a validated recorded real-input fixture
   is present, require every already-computed sampled row to meet .995. Keep
   the aggregate and endpoint checks, sample set, and oracle unchanged. Write
   `prefill_sampled_rows_passed` to the result. The existing final assertion
   consumes this combined pass value after decode completes. With no input
   fixture, preserve the prior aggregate/endpoint/decode behavior.

The fixture loader already rejects any source other than
`recorded_real_text_hf_layer_inputs`, so `input_fixture is not None` precisely
identifies the intended real-input path. The additional check covers all 291
queries at 262144, including all four original misses, and applies to both
attention kinds. It does not silently exclude diagnostic positions or lower
the gate.

Patch bases and proposed SHA256 values are in
`sliding_prefill_gate8_patch_checks.json`. Runtime base is the hash above;
the proposed runtime hash is
`0aabcac2109a35b436c78ca6322ba4e88331abdab39e8271e14ea5235af9938a`.
The applied runtime and test hashes were read back and match both proposed
values in the check artifact.

## Memory consequence

The existing sliding decode gate/up is already BFP8. Lines 104–106 assign
`prefill_gate = self.gate_up` when prefill/decode dtypes match, so the proposed
default reuses that existing tensor and eliminates the extra BFP4 prefill
gate/up copy. This is an object alias in setup, not a fresh BFP8 allocation.
Full attention keeps its matching BFP4 decode/prefill pair.

For the packed weight `[1,128,2816,1408]`, the BFP4 copy contains
`128 * 88 * 44 = 495616` tiles. At 576 bytes per standard BFP4 tile
(`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:126`), its
payload is **285474816 bytes (272.25 MiB)**. This is static tensor payload
accounting; allocator padding and measured live-device memory must be checked
in the parent's final accounting. Sharing changes if a caller explicitly
overrides decode gate dtype away from the selected default.

## Validation completed and remaining

Completed by this investigator without editing the implementation/test files:

- Compiled both proposed Python texts with `compile(..., "exec")`.
- `git apply --check models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/sliding_prefill_gate8.patch` passed.
- Replayed the proposed gate code against saved artifacts: it rejects original
  sliding, HiFi2, down8, QKV-BF16, and output-BF16 diagnostic misses; accepts
  gate/up BFP8, fused sliding, and both final full-attention maximum/near-maximum
  controls. A no-fixture replay preserves the prior passing baseline gate.
- Confirmed source hashes stayed unchanged during patch preparation. After
  parent application, confirmed they match the proposed hashes. These CPU checks are recorded in
  `sliding_prefill_gate8_patch_checks.json`; they are not device default-path
  validation.
- Standalone `python_env/bin/black --check --target-version py310 --line-length
  120` accepts the test file but requests a pre-existing nested-conditional
  reflow in `OptimizedDecoder.normalize`, outside this patch. No formatting
  edit was made. The repository pins Black 23.10.1; the parent should use its
  pinned pre-commit hook at final integration.

The parent has applied the patch. Rerun the same actual 262144 command on
defaults without the explicit gate/up override, actual 262143 sliding, and
the relevant short/nonaligned/public/stress checks. Recheck full attention's
unchanged BFP4 selection. Measure prefill performance and live memory on the
combined final runtime; the present accuracy controls do not establish a
speed improvement. The gate/up control proves accuracy for that explicit
configuration; patch integration and any concurrent compact-expert changes
still require their own validation.

## Final status

**A sufficient precision repair is verified; the parent has applied the
requested patch.** Competing single-boundary promotions are unnecessary for this
repair. Final default-path checks, performance measurements,
and stage review remain outstanding. No source implementation edit or hardware
job was performed by this investigator.
