# AutoFix — contribution parity

Status: observed precision, native-fill, long-context attention and profiler
collection failures are repaired and verified. All hard stage gates pass;
independent round-2 review returned clean-pass. Checkpoint details are in
work_log.md. Earlier sections are history.

Failing observed contract: length-511 real-weight layer0 output passes overall
PCC 0.995 but minimum per-row contribution PCC is 0.99407857. The stricter
contribution gate was added to expose hidden attention/state defects. Earlier
independently sampled shorter lengths pass. No inference of monotonic drift
or final-token failure is warranted.

Diagnosis: [AUTODEBUG.md](AUTODEBUG.md). Launcher initially failed because
unprivileged user namespaces are disabled. Retried with
`AUTODEBUG_SKIP_CHILD_SANDBOX=1` under the session's authorized unrestricted
execution and inspection-only subagent scope. Logs: `autodebug_launcher.log`
and `autodebug_retry.log`.

Priority/discriminators:

1. Folded `(1+w)` RMSNorm weights rounded to BF16 before multiplication, whereas
   HF multiplies FP32 scales. Confirmed against source and actual weight stats.
   First candidate changes only the four folded norm weight uploads to FP32.
   Rerun identical seeded contract and compare per-row metrics.
2. BF16-versus-FP32 Q/K L2 boundaries. Only test after candidate 1; preserve
   FP32 recurrence. Compare matched raw Q/K and output.
3. Gated RMSNorm gamma and SiLU rounding boundaries, independently controlled.
4. Power-product inverse cancellation. Isolated CPU discriminator subagent
   reconstructs the real failing input and checks inverse residuals; it does
   not use hardware or edit the decoder.

Validation pending: hardware is currently occupied by the serialized
full-context baseline run. Do not interpret source changes as measured repair.

## Completed CPU discriminators

`inverse_diagnostic/REPORT.md`: exact saved failing activation, power-product
inverse versus row substitution, inverse-only contribution minimum PCC
.9999974874. This demotes inverse cancellation for this input; it does not
model TT operand intake or device arithmetic.

`norm_diagnostic/REPORT.md`: all-HF rounding control is bit identical to HF.
Original combined CPU rounding boundaries yield minimum contribution PCC
.9962499942; fixing only folded-scale rounding yields .9993809729. Adding
FP32 convolution accumulation yields .9997875286. These discriminate repair
priority, not device success. Only the folded-scale repair is in the decoder.

## Long-context execution investigation

`triage_long/HOST_DIAGNOSIS.md` records the full-context run and FetchQ
evidence. The raw queue is full (1535/1536 entries nonzero), host cursor and
fence both0x1d084. These snapshots followed full tt-triage. The checkout's
`triage.py` halts Blackhole cores and suppresses resume; consequently capture
interference is an unresolved cause. Do not call this a proven preexisting
queue/runtime defect. No unpatched private-memory halt/resume script was run.

`run_long_context.py --drain-chunks` prepares a diagnostic between-chunk
drain experiment if normal device recovery succeeds after the timeout. It
retains full context and all-token device math, but has not run and is not
a verified repair. The production decoder does not include this workaround.

AutoFix status remains pending device access/validation. No acceptance bar
has been lowered, and no stage completion or physical capacity reduction is
claimed.

## Precision repair verified on device

`gdn_contract_fp32norm.log/json` passed on physical chip0 after normal mesh
recovery. The only numerical implementation change was FP32 storage for
folded norm scales. The identical seeded length511 case improved minimum
per-row contribution PCC from .9940785704 to .9980010967 (worst row350).
Lengths1,31,32,33,127,128,129,511,512,513,1025,17 all passed both whole-output
and per-row contribution PCC>=.995. All7 prefix-resume windows passed, and
all7 dropped-history controls failed the same gate. Mixed rows with starts
32/33 and lengths33/17 passed HF comparison. No threshold changed.

This closes the observed length511 precision failure. Other CPU boundary
differences were not bundled into the fix because this isolated repair closes
the measured contract. Full-attention, trace, and advertised-context
regressions remain required before stage acceptance.

## Runtime creation audit and native fill repair

The native-wrapper audit found `ttnn.full` on the unaligned continuation path
used the host implementation for small int32 ROW_MAJOR tensors
(`operations/creation/creation.cpp:68,181`). Python from/to-torch sentinels
cannot see this C++ upload. Replaced it with `ttnn.moreh_full`, which dispatches
`prim::full` on device. The first1×1 int32 physical fill failed Watcher:
`full_contract_device_fill.log` reports a64-byte NoC write overflowing its
circular buffer in `full/device/kernels/writer_full.cpp`.

Source factory sizes the buffer with the logical page size, while the writer
uses `TensorAccessor::get_aligned_page_size()`. The target-local candidate
fills1×32 int32 (128 bytes), then device-slices to the logical batch and
reshapes the position vector. No shared tt-metal source was changed and no
host numerical/upload fallback remains in that path. Final Watcher and
retained-trace validation are running under `*_padded_fill` artifact names.

The failed process aborted itself; no manual termination was performed.
`recovery_fill_smoke.log` passed normal open/add/read/close in2 seconds before
the retry, with no operator reset.

## Native-fill repair verified

`full_contract_padded_fill.log/json` passes the entire real-weight full-attention
contract under `TT_METAL_WATCHER=10`. `full_trace_reuse_padded_fill.log/json`
passes all six changing requests with one retained trace. The device-fill
repair is now verified on the paths that exposed the original overflow.
`watcher_audit.json` separates the failed tiny-fill experiment (exit134) from
clean runs (exit0), retaining both raw compressed logs and console evidence.

## Profiler collection repair

The first length512 linear-attention profiling command completed its model
checks but failed Tracy postprocessing. The console reports profiler DRAM
buffers full and dropped markers, followed by a missing-device-op assertion.
The harness now offers `--profile-drain`: `ReadDeviceProfiler` runs between
complete passes, outside measured signposts. This changes collection only,
not the decoder. Rerun required before accepting performance evidence.

## Full-context attention accuracy investigation

The clean baseline completed full262143-token prefill in150.317 seconds,
without triage or chunk drains. It passed selected whole-output PCC .9984699,
but minimum contribution PCC was .9692672 and traced decode at context262144
failed at .9563047 (`full_context_262143_retry`). This is a numerical failure,
not a demonstrated capacity limit or dispatch deadlock.

Linear attention passed the same full-context test: prefill .9999624342,
minimum selected contribution .9996092708, traced decode .9999860667
(`gdn_context_262143`). The full-attention HF oracle was independently checked
against actual installed HF, bit identical including context262144
(`long_attention_debug/HF_ORACLE.md`).

An isolated AutoDebug process inspects native SDPA precision. The device-only
tail probe initializes paged state from HF-produced K/V outside forward, then
runs the actual final decoder chunk and traced decode. Key chunk32 reproduces
the failure (prefill .9982503, minimum contribution .9702222, decode .9557780),
which demotes accumulated TT projection/cache-write drift. Chunk256 improves
decode to .9995482 but minimum prefill contribution .9927683 remains below the
stricter .995 check. Chunk512 is being tested. These diagnostic probes are not
substitutes for full public all-token validation after a repair.

## Long-context repair verified and integrated

The complete candidate run, not just the HF-cache probe, passed:
`long_attention_debug/all_tokens_chunk512.json`: prefill .9998875056,
minimum selected contribution .9977730592, traced decode .9995742584.
Source inspection and the matched geometry probes support native reduction
precision sensitivity; they do not separately quantify numerator, denominator,
maximum and correction-factor rounding. No claim of fully FP32 SDPA is made.

The final decoder uses512-key blocks when cache extent is at least512 tokens,
and32-key blocks for smaller extents. Setup appends uniquely owned finite pages
to large page tables so their extent is divisible by512. Native chunked SDPA
also requires its start aligned to that key block; public prefill therefore
uses exact one-token updates until aligned. Logical lengths and starts remain
unrestricted within the advertised context. All changes stay in the autoport.

Final-source checks passed:
- `full_contract_final`: all short lengths, resumes and dropped-history
  controls, with Watcher10 clean.
- `full_trace_reuse_final`: one retained trace across all six requests.
- `full_final_context_4097`: prefill .9999854086, decode .9999824266.
- `full_final_context_262143`: prefill .9998875056, minimum contribution
  .9977730592, decode at262144 .9995742584; eager/replay bit equal.
- `full_final_context_262144`: exact-max prefill .9998768214 and minimum
  contribution .9977730592. Near-end continuation starts262111/262112 pass
  minimum per-row contribution .9991817738, using the actual filled TT cache.

The long harness now asserts the contribution gate and records eager-versus-
traced equality. Full-context execution is validated without a capability
reduction, chunk drains, host model fallback, shared runtime changes or reset.

The profiler drained retry still dropped some within-pass markers despite
successful CSV generation. Final measurement increases the documented native
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` to5000 and retains between-pass drains.
Accept timings only after a clean marker audit of the final profiling logs.
