# V7 minimal-QKV fidelity decision and integration guidance

This proposal was applied by the parent as runtime `daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`. The [exact proof](source_delta_v7.md) and [completed manifest](validated_v7_validation_summary.json) now bind the full-only policy. The candidate baseline remains `b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846`.
Its exact source is archived in
[minimal_pairs_v6/runtime_snapshot.py.txt](minimal_pairs_v6/runtime_snapshot.py.txt).

The [four paired controls](minimal_pairs_v6_summary.md) support retaining grid
11×8 and checking HiFi2 for both attention kinds. HiFi2 wins 7/8 sliding pairs
and 8/8 full pairs, with median paired whole-prefill changes of −473.456 and
−545.573 µs. Grid11×10 gives no demonstrated gain: sliding 4/8 wins with a
+4.303 µs median paired change, full 2/8 with +87.481 µs. These are synchronous
whole-prefill host observations; they do not establish isolated GEMM savings.
The paired runs pass all headline HF/repeat/cache gates. Both actual1025/512 HiFi2 stress controls also pass. Sliding HiFi2 is now **rejected** by the262144-token recorded-input check: sampled row32 has PCC0.9949930133358956, below the unchanged .995 bar, despite aggregate PCC0.9992968877546407. See [the exact result](minimal_hifi2_acceptance_v6/long_layer0.json). [Full HiFi2 maximum-context acceptance](minimal_hifi2_acceptance_v6/long_layer5.json) now passes all291 sampled rows, minimum0.996029643684647, plus its tail/decode gates. Select the candidate for full attention only; production integration and source-bound validation are now complete; native profiles/review remain separate.

## Smallest isolated runtime change

Add a factory keyword dedicated to minimal prefill QKV, for example
`prefill_qkv_fidelity="auto"`, resolving to HiFi4 for sliding and HiFi2 for full. Forward it only to `MinimalPrefillQKV`. A constructor keyword default of `None` can preserve the
old inherited-compute behavior for direct class callers; an explicit factory
`None` should retain the original source compute config. Disabling minimal QKV
must continue to leave its old projection path unchanged.

In `MinimalPrefillQKV.__init__`, construct a new architecture-appropriate
compute config for `self.projection` only. Copy the existing source config's
`math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`, `dst_full_sync_en` and
`throttle_level`; change only `math_fidelity` to the resolved policy (HiFi4 sliding, candidate HiFi2 full). Do not mutate
`source.compute`, the generic `MinimalPrefillProjection` class, or the wrapped
`decode_source`. Both paired candidates used the following exact boundary:

| Property | Baseline | Candidate |
| --- | --- | --- |
| QKV prefill activation / weight / output | FP32 / BFP8 / FP32 | Identical |
| Fidelity | HiFi4 | HiFi2 tested both; rejected sliding, accepted full candidate |
| FP32 destination | True | True |
| Approximate math / packer L1 accumulation / full destination sync | False / False / False | Identical |
| Throttle | NO_THROTTLE | Identical |
| Grid / N / subblock | 11×8 / 8 / 1×4 | Identical |
| K block | Sliding8, full16 | Identical |
| M block | min(4, physical M tiles), configs1–4 built at setup | Identical |
| Projection storage | Existing BFP8 DRAM weight; FP32 result in caller-requested memory | Identical |

The existing precision metadata should report the effective new compute config.
No additional checkpoint, cache, RoPE, lane, output or activation tensor is
needed. Resident tensor payload delta is zero; only a host compute-config
object is created at setup. This does not predict transient kernel/CB changes.

## Source proof and validation attribution

A prospective exact diff should be limited to the factory keyword/call and the
QKV constructor's compute-config selection. Remove those additions in an AST
comparison and require equality with the archived b585 snapshot. In particular,
`MinimalPrefillQKV.__call__`, the generic minimal projection call, tied-K/V
concatenation, decode wrappers, normalization, cache validators/fill/update,
tail-retention and tight-capacity selection must remain identical. Compare the
resolved production config field-by-field with each passing paired probe.

The unchanged `__call__` still delegates a logical one-row decode input to
`decode_source`; a one-tile prefill matrix has 32 physical rows and uses its
setup-built Mblock1 minimal config. That distinction matters for short/tail
coverage.

Numerical equivalence of prefill is **not** established by the source diff:
lower fidelity changes Q/K/V values and therefore the cache used by later
decode. The full long-context control now passes every required sampled row at the existing .995 threshold; both512-step stress gates also pass. Sliding’s row32 failure demonstrates why aggregate prefill PCC cannot replace this gate. Integrated current-source headline, small/tail/tight
prefill, request-lifecycle and separate Watcher checks bind the selected default
and its setup configs. Exact original v6 reports remain attributed to b585;
probe results at b585 plus a setup override must not be relabeled as production
v7 execution. Unchanged public orchestration/decode/lifecycle coverage can be
retained through the explicit source map, with the new affected-path results
listed separately in a v7 manifest.

Final native profiles must verify HiFi2 on the actual minimal QKV rows, the same
operand dtypes/K blocks, complete operation windows and same-run host/device
scope. V5/v6 CSVs stay archived under their recorded hashes. The checklist's new
fidelity trial closes only once the real-input acceptance and integrated source
binding are complete; current grid trials can already reject11×10 on measured
evidence.

## Completed integrated acceptance scope

The parent completed12 commands for the production source: all eight full-attention public contracts (B32, prefix, reuse, BF16-cache compatibility,262144,262143, headline and Watcher), sliding headline/Watcher, the four-case small-input pytest, and both-kind1025/512 stress. The two new full long results will therefore bind the selected fidelity directly to the integrated source. Sliding’s default remains HiFi4; its unchanged B32/prefix/cache/max and lifecycle coverage stays attributed to v6/v5 through the exact source map. The v7 manifest distinguishes these fresh results and five additional boundary controls from inheritance; it does not claim another full18-command campaign.
