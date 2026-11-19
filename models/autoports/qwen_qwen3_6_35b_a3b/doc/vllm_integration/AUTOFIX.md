# AutoFix: serving adapter and continuation ownership

Diagnosis: `AUTODEBUG_sampling.md` and `continuation_owner/result.json`.

The first request reached the canonical sampler but rejected disabled logprobs.
The plugin represents disabled counts as -2; the model represents them as zero
with an independent enable flag. The adapter now copies the sampling dataclass
and normalizes only this disabled sentinel. It preserves real requested logprobs,
penalties and all other feature values. Host boundary tests invoke the real
neutral-feature validator; four tests passed. Reduced serving then passed nine
requests with device sampling and no out-of-vocabulary token.

An independent CPU investigation found an earlier inconsistent transition in
the shared plugin: prefill continuations relinquished their existing slots
without moving device state. Alternatives considered were adapter identity
inference (invalid: no request identity at that boundary) and retaining live
owners in the allocator (correct owner). The latter fails three regressions on
the pinned base and passes the complete 2306-test host suite after repair.
`plugin_changes.md` contains the required pre-edit record, test commands/artifacts,
and local plugin commit. Reduced serving completed nine requests afterward;
final full-model serving remains pending.

`readiness_vllm/adapter_commands_probe/result.json` passes through the real
adapter factory, external allocation, warmup, prefill, decode and readback hooks.
It proves unreadable stale token/position inputs are not inspected, queued
outputs remain distinct, positions advance once, page-only updates preserve
tokens/positions, unchanged pages perform no copies, and state remapping retains
dormant recurrence/conv/token/position/RNG data on all four chips. Trace and
program-cache identities remain stable. Its changed-page test targets unused
tails; allocator-driven page growth is separately exercised by serving.

These are verified repairs, not AutoFix exhaustion. No profiler, reset, shared
interface addition, or remote push was used. Full sampling, final qualitative,
all-layer trace reuse, context acceptance and performance remain stage work.


The retained2/32 decode candidate is now verified on reduced layers0/3 by exact fixed32-vs-candidate logits/state trajectories (eight cases,76model replays). The unsafe singleton candidate was rejected before serving. The actual adapter command probe also passes after enabling the documented prepared bucket list. Full40 quality, same-server lifecycle and untracked performance remain required; this reduced evidence does not establish full-model quality or speed. See `work_log.md` and `readiness_vllm/bucket_probe/result.json`.

## Full40 retained-bucket hang investigation

AutoTriage localizes the initial failure to device1 NCRISC reader_combine zero_pages write-ack barrier. Independent audit refutes unsupported AR semaphore-wrap/workspace-lifetime claims; all-reduce is downstream. Source inspection finds correct52-page sender accounting and distinct NoC roles, but the decisive live expected counter was unavailable: standard checker excludes NCRISC, and focused mapped reads then timed out on the first NoC status register. Reports and diagnostic limits are under `AUTOTRIAGE_BUCKET_HANG.md`, `BUCKET_CCL_HYPOTHESIS.md`, and `triage/DIAGNOSTIC_TIMEOUT_AUDIT.md`. No speculative production kernel patch was kept.

After an externally observed host reboot at13:39:46 (before the original job deadline), no owners remained; fresh4chip trivial op passed without a stage-issued reset. A minimal exact native zero_pages discriminator passes5000 retained replays, all-chip output checks, stable2program entries,1capture/1release (`readiness_vllm/zeroing_probe/trace_v2.json`). This narrows the reproduction; it does not certify the full combine or B2model. The real tracker positive control rejects a live postcapture buffer and permits correct clean replay (`logs/trace_positive_control.log`). Full40 same-prompt retained32 vs2 contrasts are next. AutoFix remains active, not exhausted.

## 15:30 UTC recurrent standalone failure; AutoFix active

Same-code full40 controls under original timeout1800 (PID308144, wrapper308143, start15:22:33, deadline15:52:33) complete q01greedy256 and q01sampled256, then stall q02greedy step159, preceding token22238. Host snapshot logs/tt_controls_post_reboot_stack.txt and triage/post_reboot_controls_hang.txt identify the same device1 physical(1,2) NCRISC reader_combine zero_pages DRAM write-ack barrier and captured op320374 as the preboot failure. Other chips' all-reduce waits are downstream. Exact command: TT_METAL_TRACE_ALLOC_TRACKING=0 timeout1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_quality_controls --backend tt --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/qualitative_controls/tt_outputs_post_reboot.json (actual CLI separates timeout/value). Log: logs/tt_controls_post_reboot.log. Standard bounded triage succeeded; no custom MMIO reads were repeated. Process remains untouched until original timeout; no reset yet. Partial RUNNING JSON is not a pass.

Independent source-only AutoFix reports: AUTOTRIAGE_RECURRENT_ZERO.md finds a missing unconditional reader NoC0 write-ACK epilogue in the native fused SwiGLU predecessor; kernel entry rebases counters, so late prior ACKs could overshoot the next exact-equality barrier. Actual overshoot is not yet measured. RECURRENT_HANG_MODEL_AUDIT.md finds no model scratch geometry/lifetime defect; only slot0/model_2 executes, so bucket switching and async are unnecessary. Preparing synthetic multicast-to-native-zero accounting A/B and exact three-request full-model discriminator. No production kernel patch applied yet.

## Reader acknowledgement completion repair (15:58 UTC)

Independent source audits AUTOTRIAGE_RECURRENT_ZERO.md and NOC_COMPLETION_AUDIT.md identify native fused SwiGLU reader returning with nonposted NoC0 multicast ACKs incomplete. Dedicated-mode next-kernel entry rebases expected counters; dispatch completion does not drain reader ACKs. Synthetic ordinary-L1 counters confirm up to248 outstanding after sent flush and245 at its final observation. The sole-barrier variant has0 after drain. Both5000-replay synthetic controls pass with no subsequent overshoot, so the specific full-model hang's attribution remains unproven (readiness_vllm/noc_boundary_probe/synthetic_comparison.json). Native actual fused-op→zero-helper baseline also passes1000allzero+1000one-expert replays with selected BFP8/LoFi, exact replay/eager output, analytic output check and exact zeroing on4chips.

Minimum source-contract repair adds one unconditional noc.async_write_barrier() after the outer expert loop in ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp. It drains metadata in empty paths and final H multicasts in nonempty paths. Existing sent-only overlap points, writer, combine zeroing, ACK equality, sampler and precision are untouched. Repository pre-commit for the native edit passes (logs/native_barrier_precommit.log). The same native discriminator is now rerunning to native_drained.json; original full40 suite and final serving gates still required. No shared plugin or vLLM interface change.

16:07UTC repaired full40 control PASS12outputs3072tokens/3060model_2 replays;3captures and3shutdown releases. Original q02greedy now completes256; q01greedy/sample tokens exactly match both completed pre-repair outputs. All12 outputs read and judged coherent/on-topic, no gibberish/language drift/contamination or mechanical looping. They remain thinking at256cap as HF does; no final-answer completion claim. q03's isolated opening double-word thinking is present with the identical first7tokenIDs in HF greedy and sampled controls; not a generation loop. Detailed verdict readiness_vllm/qualitative_controls/tt_reader_drain_quality_review.json. Original full-model gate now passes after the completion repair; exact ACK overshoot in the original live failure remains unmeasured.
