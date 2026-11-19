# Single-token ordered recurrence identity audit

Status: isolated source candidate, **hardware compilation, correctness and performance pending**. No devices were opened or jobs launched in this task. Parent reports the production B32 scan at 167.8 us. This candidate uses the current production scalar-zero reader; it does not mix or modify the queued fast-zero reader experiment.

## Contract and hypothesis

Keep the accepted PCC >= .995 and existing stricter gates. The earlier raw recurrence's long-answer failure means algebraic recurrence equivalence alone is insufficient: preparation packs, operand unpack precision, and ordered dual-product accumulation matter.

Production `tt/kernels/flat_ordered_post_loop/compute.cpp:146–179` prepares `kdec=k*1`, transposes it, materializes `Tinv=I`, then calls the generic scan. `chunk_gdn_math.hpp:1045–1188` forms packed `diff=nkd@S + I@vbeta`, materializes `vnew=I@diff`, then forms output and updated state as two ordered matmul products in the same DST accumulation. For one token, the inverse is exactly the identity; no general multi-token triangular inverse is eliminated.

The smallest candidate aliases `vnew` to the existing packed `diff` and removes only the standalone `I@diff` and the Tinv copy. An independent switch also replaces `transpose(k*1)` with `transpose(k)`. It deliberately retains the identity matmul **inside** the diff accumulation, the diagonal-decay matmul inside the state accumulation, and the intra-attention masking. These remaining operations preserve the passing accumulation order.

## Rounding argument and limits

The helper retains FP32 storage, HiFi4, FP32 destination accumulation, and default unpack mode. `tt_metal/jit_build/genfiles.cpp:1069–1076` selects Tf32 as the conditional destination format for FP32 inputs in this BF16/FP32 kernel; `data_format.cpp:138–155,210–217` applies it absent an explicit unpack-to-destination override. Removing scratch CBs does not change the remaining BF16 format family.

Let R be that operand-unpack quantization. The old inverse step computes a packed identity product from R(diff); its consumers unpack again. For ordinary finite inputs, multiplication by exact one and zero plus idempotent quantization imply R(pack(I@R(diff))) = R(diff). The candidate preserves the packed diff and its WAIT, so consumers still see the same R boundary. It does **not** fuse away the diff pack or move the subtract/cancellation into an eltwise operation.

For K, the old sequence is k -> Tf32 scalar multiply by exact one -> FP32 pack -> transpose -> Tf32 matmul operand. The new sequence transposes k directly. `tt_metal/hw/inc/api/compute/transpose.h:37–75,120–140` selects ordinary Src unpack when the destination format is Tf32; only Float32/Int32/UInt32 select the full-precision unpack-to-destination branch. Thus transpose itself preserves the quantization that the removed multiply performed.

These are source-supported hypotheses, not a hardware proof of bitwise equivalence. Signed zeros, denormals, nonfinite inputs and low-level FPU identity-product behavior could invalidate a blanket equivalence claim. Changing unpack mode or architecture requires renewed validation. The direct probe still requires bitwise state equality against the passing ordered TTNN kernel; output and PyTorch comparisons retain .995. A failure must be investigated at the first differing state instead of relaxing a gate.

## Candidate and ownership

New helper: `tests/multichip_recurrent_identity.py`. New kernels: `tests/kernels/recurrent_identity/`. New process-local wrapper: `tests/multichip_recurrent_identity_candidate.py`.

- `--keep-k-identity`: remove Tinv copy and standalone identity matmul only. CB13 and CB16 removed: 434176 -> 413696 bytes/core.
- Default: additionally remove K scalar multiply and CB18: 397312 bytes/core.
- Compile-time argument 4 controls the K choice; it participates in program identity. Existing epsilon stays at argument 3.
- `identity_scan.hpp:8` statically requires Ct=1. Caller sets `.vnew=diff` and `.ointer=diff`; the diff buffer is pushed once, consumed by both output/state blocks, and popped exactly once at their shared final release. Do not introduce an early pop or a second reserve/push for the alias.
- `compute.cpp:153–172` chooses the transpose input. K remains available until after transpose and retains its original final pop.
- Reader, writer, norm helper, preparation and postnorm bodies are byte-identical copies of current production. Repeated-head ownership is unchanged; compute startup occurs exactly once at kernel entry (`compute.cpp:187`).
- Ordered diff body and both output/state accumulation bodies are byte-identical to the generic scan. Four standalone identity tile matmuls/head are removed from 76 recurrence tile matmul calls, leaving 72. This is an operation count, not a latency prediction. The optional K path additionally removes four scalar-multiply tiles and their pack/CB traffic.
- The wrapper patches only the imported helper binding during this process and restores it afterward. Private B1 and incomplete/subset/mixed pools continue through the production dispatch. No context, padding, dtype or public sequence-length contract changes.

## Validation and queued experiments

Host checks passed: Python AST parsing; both harness `--help` paths; exact copied-body comparison; CB byte counts; eight fresh unique job labels; reference paths exist. `host_validation.json` records these checks and `source_manifest.json` hashes original and candidate files. No C++ device compilation was attempted.

`../recurrent_identity_jobs.json` has eight independent serialized jobs: inverse-only then all-identities, each with the existing direct watcher ownership sweep, B32 and B6 16-step state/benchmark tests, and B32 4095-prefix/1024-answer state/benchmark test. Watcher and timing jobs remain separate. The direct sweep retains batches 1,2,13,14,16,17,26,27,31,32, repeated heads, DRAM/L1 alternating inputs, independent persistent states and bitwise state checks. All model gates and reference files are inherited unchanged. Job labels/output paths are new; the runner captures sources and refuses an existing command record.

Parent may run using the existing serialized device protocol, after its active queue and STOP ownership permit:

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/recurrent_identity_jobs.json
```

Focused commands are in the immutable job JSON. Wrapper switches are `--direct-probe` and `--keep-k-identity`; all other flags pass through to the original harness. The original candidate-only `--no-gate-padding` flag is intentionally absent: current production already implements the accepted behavior and the standard stack harness does not accept that flag.

If inverse-only fails direct equality, locate the first difference between old I@diff and unpacked diff; retain an explicit Tf32 rounding boundary if needed. If inverse-only passes and default fails, the transpose/multiply equivalence is disproved and the K identity should remain while investigating the unpack path. If direct/short tests pass but long-answer fails, no integration is justified: inspect the first recurrent divergence and preserve all original precision gates. Only a passing variant's final-path warmed measurements can establish a gain. Fast-zero can be combined later as a separately measured cumulative candidate.
