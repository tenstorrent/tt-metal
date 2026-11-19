# Ordered indexer projection: final conversion AutoFix

This records the standalone operator diagnosis and pass, not complete decoder acceptance.
The failed v2 test is retained unchanged, along with its five exact executed source files in
`tests/diagnostic_sources/ordered_projection_device_v2_sources.json`; each preserved source
hash was checked against that run's provenance before editing.

## Causal observation

On real Q32 input starting at105568, all20,480 device FP32 accumulator bit patterns exactly
match the independently computed blocked high-then-low CPU sums. Direct BF16 packing
still differs at two exact midpoints:

| Local row,column | Exact FP32 sum | Direct pack | Literal HF / integer RNE / device typecast |
|---|---:|---:|---:|
|15,342| -0.0518798828125 | -0.052001953125 | -0.0517578125 |
|16,328| -0.13330078125 | -0.1337890625 | -0.1328125 |

The ordinary device FP32-to-BF16 typecast matches both CPU integer ties-to-even and literal
HF at every20,480 output values. Thus the measured discrepancy is the final conversion,
not the accumulation order. The v2 strict BF16 gate correctly failed before trace capture;
it supplies no traced candidate pass. Device wall-time diagnostic62.386ms includes host
audit, dispatch and synchronization and is not a warmed device performance claim.

## Minimal candidate change

`tt/ordered_projection.py` now defaults to the unchanged generic arithmetic with FP32
output, followed by `ttnn.typecast(..., bfloat16, output_tensor=output)`. The caller's
preallocated BF16 output address is preserved. The80KiB padded FP32 intermediate is
per-call and ephemeral; setup retains only immutable BF16 weights. There is no host
conversion inside forward and no persistent runtime state. `direct_bf16_pack=True`
retains the failed behavior for diagnostics; `fp32_output=True` exposes unrounded sums.
These mutually exclusive diagnostics do not alter the strict probe acceptance bar.

The conversion API's preallocated output is supported in
`ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp` and exercised by
`tests/ttnn/unit_tests/operations/eltwise/test_eltwise_mixed_dtype.py`.
Blackhole `ckernel_sfpu_typecast.h` explicitly rounds FP32 bits before BF16 pack.
The repair changes no compute-kernel, reader, writer or block order. After the operator
pass, the parent integrated the ordered projection constructor in QSA; complete QSA
validation remains a separate gate.

## Closed operator validation and remaining scope

`ordered_projection_device_v3.json` closed exit0 in55.556759seconds. All32 logical
counts at10real input starts pass:320literal/ordered CPU cases,32warmed device cases,
and352changed/repeated trace replays are BF16 bit-exact. Output addresses are preserved,
runtime audits pass, no output is read before replay, and program-cache count remains33
before and after all traces. Another prepared count executes eagerly while each trace
is live. Watcher contains860lines with no fatal/invalid/overflow/sanitize/assert/corrupt/hang
matches. The failed v1/v2 runs remain failures; the CPU-only record intentionally keeps
`passed:false` because no device ran.

`ordered_projection_operator_closure_v1.json` verifies all four original process/log
identities and exact source bindings. V1/V2 use preserved historical bytes. V3's current
module differs only in module/class docstrings: the preserved validated source matches
its recorded SHA and executable AST remains identical. Current reader, writer, compute,
probe and runtime-audit helper match the v3 source hashes exactly. The closure makes no
whole-stage acceptance inference. Its reproducible command is:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.close_ordered_projection_evidence
```

Full-context QSA correctness, integrated decoder regressions and final performance remain
separate requirements. General NaN/Inf, denormal and other CPU backend equivalence are
not established by this operator proof.

## Immutable diagnosis identities

- `doc/functional_decoder/ordered_projection_device_v2.json`: `901b87dbddda6ff2090175182016cc72e29077d17e7df89601c8bf0e819b5d68`
- `doc/functional_decoder/provenance/ordered_projection_device_v2.json`: `997245f3bf8e432ae6e3c21553f5081027cc8836a2bb9958b531f24cd95b12ef`
- `tests/diagnostic_sources/ordered_projection_device_v2_sources.json`: `d57afe659fc2e4ac3491991b5081d15f6592b8356527c2989cbaba211acc3922`
