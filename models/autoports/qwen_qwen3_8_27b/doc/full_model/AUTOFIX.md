# AutoFix — finite preparation and fixed slots

Diagnosis: `AUTODEBUG.md`. The inherited logical-length prefill path specializes programs on exact tail lengths and page-table suffixes; after decode capture those persistent program buffers are unsafe. Padding before the linear recurrence also advances state unless validity and logical history are explicit.

Selected repair: preserve the original decoder path, adding an optional prepared chunk path. Physical sizes are1 and32..4096 in32-token increments. Active rows in each eager prefill group share that recurrence extent. Device masks zero post-convolution q/k/v and nonlinear decay/beta; runtime row indices select the exact three-token history and final hidden row through the common embedding op. Prepared fill-page windows and tensor-valued chunk starts remove continuation-specific slices. Inactive KV fills are skipped by eager scheduling, and inactive recurrent state is preserved. No weight, activation, KV, CCL, fidelity, residual, or native-RS policy is relaxed.

Prepared P1 retains the sequential FP32 recurrence. Masked decode uses separate all-active and inactive-slot signatures: the former retains the inherited decoder op sequence; the latter adds state selection and uses position−1 to skip KV/SDPA work. Safe separate RoPE indices remain nonnegative. Skipped SDPA output is discarded for inactive users.

The generator allocates persistent bucket metadata and sampler/decode inputs before capture. It prepares representative real layer kinds for all physical buckets and terminal variants, then warms the full decoder stack and sampling before capture. Program-cache allocation tracking remains enabled for acceptance, with no allocation exemptions. Public factory preparation and fixed-slot high-level batching are being validated.

Rejected fixes: merely pad hidden rows (changes recurrent state); warm a few logical lengths (unbounded missing signatures); disable allocation tracking (hides unsafe buffers); retire traces per request (violates lifetime contract); run mixed active rows through an oversized GDN extent (adds padded state arithmetic).

Measured checks so far:

- `probe_masked_trace_01.log`: reduced real layers0/3 plus complete terminal, logical33/physical64, split replay passed with allocation/program-cache tracking.
- `probe_buckets_01.log`, `prefill_bucket_equivalence.json`: lengths1/2/31/32/33/127/128/129/255/256/257/4095/4096/4097. All convolution histories exactly match the inherited path on all four devices; recurrent PCC1.0. Minimum terminal logits PCC0.9999272849774414 at4095; physical4096 selects the existing BFP8-output fused prefillRS branch, whereas logical4095 previously used the separate row projection. Full-model accuracy coverage of this branch remains required.
- `probe_contract_02.log`: reduced same-signature feedback, position, and page-map replay checks pass after prepared-path integration.

Outstanding: full finite-domain closure; B8/16/32 and mixed-slot validation; full-context awkward-tail coverage; full-model readiness/qualitative/performance gates; profiler audit; final independent review. No AutoFix failure or stage pass is declared.


## Isolated follow-up report (historical source audit)

# Generator fixed-slot and ownership contract AutoFix

Source audit on 2026-10-24 found stale host-derived prefill page windows after bound device-map mutation; success-only restoration of caller-owned cache during preparation; stale inactive prefill outputs; untyped all-inactive decode returns; ignored continuation offsets; and silently greedy host compatibility under non-greedy sampler settings.

Changes: `models/autoports/qwen_qwen3_8_27b/tt/generator.py` restores external cache in `finally`, cleans partial traces, validates bound prefill cache, clears prefill outputs, preserves requested inactive output types, rejects non-greedy host compatibility, counts host token feedback, and accepts scalar 256-aligned `start_position`. Device-authoritative RM integer gather with persistent runtime indices fills page windows; no host logits or state fallback is introduced. Slice was rejected because source hashes start/end and would require position-specialized warmup programs. Gather validation accepts RM and UINT32 indices; data and output retain integer dtype. Prompt token indexing remains local while RoPE/page offsets become absolute.

`tests/probe_mixed_slots.py` adds nonzero external-cache restoration, in-place device map mutation and exact KV relocation, runtime continuation offsets, and all-inactive output/cache assertions. Existing mixed-slot state and trace checks remain.

Validation: `python3 -m py_compile` on the two modified Python files and `git diff --check -- <files>` passed. No hardware was executed by this agent. Remaining runtime checks: RM gather integer/out behavior on installed build, warmed program reuse with changed runtime indices, mixed-slot assertions under allocation tracking, and preparation failure injection. Restoration cannot guarantee success after a fatal device failure that prevents synchronization or trace cleanup. No performance claim or commit is made.


## Validation follow-up

Full finite-domain preparation, mixed external-cache restoration/device-map/inactive-row controls, native all-layer262144 capacity, AIME top-k and shared qualitative checks subsequently passed; see work_log.md and current artifacts. The historical report above predates the128-aligned continuation enhancement and EOS repair. Initial stage review identified unused-tail page-window indexing; source repair and targeted near-end/logit validation are now tracked in stage_review_initial.md. No AutoFix failure is declared. Final stage review and performance/profile completion remain pending.
