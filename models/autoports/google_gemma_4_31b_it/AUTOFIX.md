# AutoFix: full-attention long-context decode

Status: repairs verified by final full-context, batch, boundary and watcher runs. Independent stage review tracked in doc/functional_decoder/stage_review.md.

Failing contract: HF-vs-TTNN layer PCC >=0.995 through the advertised262144-token
context, including trace replay. Full prefill passed, but eager/traced decode=.990203.
The independent `AUTODEBUG.md` identifies persistent BF16 online-attention state and
the default K32 / one-core-per-KV-head serial recurrence as the leading cause.
Source inspection confirms those defaults and the BF16 intermediate/statistics CBs.

Discriminating controls in `doc/functional_decoder/long_attention_probe3.log` and
`long_attention_probe.json` supply exact HF K/V and compare TT and HF queries.
The TT query itself has PCC .999995768. With exact HF queries and cache, default
attention PCC=.896732 and final layer=.990482. Changing only exponential mode at
K32/core1 gives attention=.897188 and layer=.990716: insufficient. Changing only
core cap to16 (K32, approximate exp) gives attention=.998532/layer=.999836.
K128/core16/precise exp gives attention=.999972/layer=.999992. TT-query variant gives
attention=.999973/layer=.999993. This implicates the length of the serial BF16
recurrence, rather than query projection, cache production or trace execution.

Candidate order:
1. Explicit decoder-local split/chunk config: verified by isolated attention controls.
2. Promote kernel persistent accumulators: would touch generic kernels and is unnecessary
   if the supported program config satisfies the functional contract.
3. Changing context or acceptance threshold: not a repair and not attempted.

Minimum runtime change: full-attention decode now uses 8x8 grid, K128, core cap16,
precise exponentials; sliding decode is unchanged. BF16 tensors and HiFi4 compute
remain. This bounds batch-1 maximum-context recurrence to128 local chunks instead
of8192. No performance claim is made. An earlier K512 candidate needed3118336 bytes
per-core CB space, exceeding1572864-byte L1; the K128 candidate ran successfully.

Files: `tt/functional_decoder.py`; diagnostic harness `tests/probe_long_attention.py`.
The validation harness also fixes a batch-0-only exact-maximum comparison and adds
an explicit option to reuse unchanged maximum-prefill evidence during decode repair.

Remaining checks: actual TTNN-produced prefix cache, repeated traced decode, changed
inputs, batches, watcher, long-reference control, and independent stage review.

Long-reference control completed (`long_reference_control.json`): SDPA versus
explicit HF-style eager attention PCC .999996841, final-layer .999998689;
SDPA versus explicit FP32 attention PCC .999999762, final-layer .999999702.
This rules out a reference change large enough to explain the failed .9902 gate.
`runtime_identity.json` records the source base, loaded extension/library paths and
hashes, compute policy and the explicit final decode override. No native library
or generic kernel was changed by this stage.

End-to-end full-context validation passed: `full_context_fixed.json/.log`.
TTNN-generated262143-token paged prefill, eager and three identical traced decode
replays at262143, then changed-position replay. Decode PCC=.9999799132;
changed-position PCC=.9999930859. Allocation tracking includes program-cache
allocations; runtime conversion guards pass. Exact262144 prefill evidence remains
valid because prefill arithmetic was unchanged. Batch/watcher/independent review
checks remain stage work, rather than evidence for a kernel-source repair.

## Logical-tail trace reuse repair

A new logical34-token prefill after warming33 (same64-row physical bucket) exposed
late `program_cache: FillPadDeviceOperation` buffers. The allocation tracker refused
replay before corruption (`new_bucket_case.log`). This was a real preparation flaw,
not an allocation to acknowledge or a reason to skip program-cache tracking.

`prefill_forward` now exposes the existing tile storage through metadata-only
sequence views, slices aligned physical chunks, concatenates physical outputs and
returns a logical output view. `prefill_chunk_forward` likewise trims with a view.
The view preserves backing storage and padded shape; no padding values are copied
into valid rows. Causal attention excludes future padding; incomplete tail pages
remain request-private. Inputs/outputs retain their backing allocation through
normal tensor ownership. No allocation exemptions were introduced.

`new_bucket_fixed.json/.log` passes all boundary cases, resume, prefill determinism,
changed inputs and six prefill/decode interleavings, including previously unwarmed
logical34 after prepared33, with one trace ID, zero request-time releases and
program-cache allocations included. Full-context and batch/watcher revalidation is
running because the public prefill padding path changed.

The PCC helper now accumulates in FP64. The old large batch32 FP32 correlation
rounded a prefill result to1.0; final batch evidence is being regenerated with the
stable helper, rather than reporting this as exact equality.


Final revalidation completed: both `*_context_post_view.json` include exact262144
prefill and traced decode at262143 using actual TTNN cache. All ten short batch,
boundary and watcher `*_attention_*_final.json` checks pass with FP64 PCC, and
synthetic pytest passes4cases. `validation_queue.json` records all12commands and
zero exit codes. These supersede the earlier pending-validation notes above.
