# Optimized-full-model local B1 evidence

This is the standalone TTNN optimized-full-model stage 07 investigation for
Qwen/Qwen3.8-27B, with the user's explicit above-40 warmed B1 TSU objective.
The source is an isolated checkout at `/data/mvasiljev/qwen38-local-tsu-ttft`,
branch `mvasiljevic/qwen38-local-tsu-ttft`, base
`206dd3a0ec6f6e1494d4408e6590e47d4bc97ae0`.
The validated configuration is **opt-in**: `QWEN38_COMPACT_B1_RECURRENCE=1`.
The default-off policy is retained for integration review; unflagged throughput
is the control. No live vLLM server or SWE task is profiled or accelerated here.

The actual contract is installed `07-optimized-full-model.txt`, with optimize,
multichip, tracing, qualitative-check, target-workload, tt-device-usage and fresh
xhigh independent stage review. Full64 native context 262144, selected precision,
residual/cache formats, on-device split greedy/stochastic-capable sampling,
position/RoPE/page-table state and nonaligned/fixed/inactive request support are
preserved. The measured target remains 4096 input / 128 output / three warmed
repeats. No accuracy threshold, decode window or capacity is reduced.

## Exact source and runtime identities

Raw evidence is local under `/home/mvasiljev/build/qwen38-swe/local-tsu-ttft`
(`/evidence/local-tsu-ttft` in container `qwen38-swe-local`).
`final-source-freeze-v4.json` identifies all 33 runtime/config/kernel files and
four relevant test runners. The native and strict receipts contain the same 34
runtime-plus-runner hashes; quality and profile runners carry their applicable
matching manifests. The selected source is frozen throughout final hardware runs.

The compiled backend is retained `3fbf3dbf1e88112aae1ca39cff9d1bc8a1da66e8`;
`_ttnncpp.so` SHA256 is
`afd47e3a3337851b2938ca27e3438ece90d3eb5aa99d78261ef1d9b45b7019d0`.
Python is 3.12.12. Checkpoint
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
is the verified local 32-file, 55,586,114,863-byte copy. Fresh BF16 HF AIME reference
SHA256 is `e16f1192147d682eaa91d1957e5d5e39a981dcde106b6364b2255f8cc7d8aa1f`.

The NVMe compiler/header copy is byte-verified against retained runtime sources
(`runtime-nvme-byte-proof.json`); the compiled backend is unchanged. Relative SDK
kernel paths can resolve from CWD before TT_METAL_HOME: native/profile/watcher
runs use container `/workspace`, quality uses the isolated source checkout. Both
tracked SDK C++ trees are identical and clean (ttnn tree
`d2f1a480df2967d613a26bba10b3197bfd516237`, tt_metal tree
`c1cc3ba6605a47f5a3b9095b93e68fb40bc42bc6`), recorded in
`kernel-cwd-and-backend-identity.json`. This distinction is not hidden by the
NVMe copy proof. Profiler uses a separate fixed-header root/cache; header SHA256
`b6a21795e2d8f82fb54d20586cf0c65e2ed579378aee8a18bd986b11e106163d`.
Watcher and profiler never run together.

## Native matched before/after

The lease-guarded `run-nvme.sh` runs `tests.benchmark_grouped_model` with tracking0
and program-cache allocation exclusion0, plus:

```text
--checkpoint /evidence/weights-qwen38-27b
--grouped --masked-recurrence --flat-prefill
--batch 1 --context 262144 --length 4096 --outputs 128
--prefill-chunk-size 4096 --cache-state-fields all
--skip-qualitative --repeats 3 --output <receipt>
```

Candidate adds `--compact-b1-recurrence`. All device commands use
`bash /home/mvasiljev/build/qwen38-swe/with-device-lease.sh /root/qwen_local_tsu_ttft ...`.
No lease lock inode is replaced. Other agents pause CPU work through clean timing.

`full64-b1-final-v4-native-{control,candidate}.json` and
`full64-b1-final-v4-native-comparison.json` record median TSU
**36.090929548 → 40.673091044 (+12.696158%)**. Candidate repeats are
40.671142981 / 40.674131702 / 40.673091044, all above 40. TTFT medians are
436.932690907 → 436.612105928 ms; there is no meaningful TTFT improvement.
All 3×128 tokens and mixed repeats match exactly. Each loop has 127 model and
sampler replays, no host refresh/copy/capture or inner readback, one final boundary
readback and nonblocking on-device sampling. Control/candidate UMD closes are
15:49:32.499 / 15:53:56.730 UTC on 2026-10-28. Startup/load/JIT are separate fields.

Earlier native serial-only 37.9550 TSU and combined-v2 40.6769 TSU remain valid
experiment history, not substitutes for the final v4 source measurements.
The unflagged baseline was first 36.0894 TSU and independently reproduces above.
No teacher-forcing or reduced-fixture number is substituted for token-out TSU.

## Numerical, qualitative and trace closure

`full64-b1-final-v4-strict-{control,candidate}.json` uses tracking1/skip0 and the
full native allocation. Five complete 512-entry maps (conv, FP32 recurrent,
decoded paged K/V across all ASICs), all 12 continuation-logit hashes, fixed128
and ragged33 tokens match exactly. The prefix4097 +7-token bridge +129-tail
covers nonzero, nonaligned continuation. The B1 all-inactive API bridge executes
no replay and leaves all state unchanged. Three trace IDs/captures stay stable.
The comparison receipt is `full64-b1-final-v4-strict-comparison.json`.
Both tracked full64 arms prepare all 129 supported physical prefill buckets;
no corruptible-buffer or program-cache exclusion is added by this change.

`full64-b1-final-v4-quality/` refreshes AIME and shared qualitative evidence on
that exact source. AIME prefill top1/top5/top100 is 99/100/100%; teacher forcing
is 98/100/100%. Shared 512-token native-chat explain133 / structured114 / code158
all reach EOS with correct three-sentence English, unfenced 4/3 JSON and an
order-preserving Python deduplication function/example. Prompt tokens, native
chat template, thinking mode and cap match fresh HF controls. The AIME100-token
free-running answer is coherent partial reasoning, as is its HF control; it is
not a completed-answer accuracy claim. The installed scoped stage-07 output
checker exits0 (`full64-b1-final-v4-quality-stage07-check.log`).

Short203/shared-chat timing fields are diagnostics from serial-to-final quality
runs, not unflagged prefill speedup claims. The selected guard excludes every
prefill path (including prepared T1), so unaffected prior prefill evidence is
retained; the applicable full target TTFT and nonaligned correctness are refreshed.

Integrated `watcher-b1-final-v4-{control,candidate}.json` passes with watcher10,
NOINLINE, all features enabled, tracking1/skip0 and no profiler. This explicitly
reduced0/3 safety fixture warms its entire exercised domain [1,32,64,4096]; five
16-entry state maps, 12 logits and tokens are exact. It is not a full64 timing
or general-warmup claim. Full64 all-bucket coverage is the separate tracked pair.
Component all-device operand/head-normalization checks additionally cover the
new NoC readers/writers before and after replay.

## Profiling and accounting

Current real profiles are `profile-b1-final-v4-control/` and
`profile-b1-final-v4/`, each layers0/3 plus full embedding/norm/head/sampling/
feedback, native allocation and physical4096 prefill. They retain actual
`tt-perf-report` tables, raw/window CSVs, source and header provenance. Profiling
uses explicit support count100000/disabled live Tracy push, not the earlier
failed setup. Decode device-op count is 568 → 484 (142 → 121 per ASIC); the
linear segment loses 21 structural ops, with its original FP32 compute retained.

`final-v4-profile-attribution.json` checks unchanged seven-op prefix and 68-op
full-attention/terminal suffix before scaling48 linear +16 full layers on each
ASIC. It corrects two embedding-DRAM conversions that occur only in the first
linear layer, instead of multiplying them48 times. Corrected representative
kernel-work estimates are **26.790020 → 23.694282 ms**, versus E2E
27.707793 → 24.586280 ms. E2E / corrected estimate −1 is 3.43% /3.76%, below the
10–15% investigation trigger. These are profile extrapolations, not directly
measured full64 device time. Raw two-layer envelopes remain separately reported.

The payload roofline estimate is 7.700880 ms: decoded weight tiles
3,829,473,280 B/device, compulsory KV chunk reads37,879,808 B/device and FP32
recurrent read/write75,497,472 B/device at nominal512 GB/s/device. BF4/BF8 tiles
include exponent payload (576/1088 bytes). All127 positions read4352-token SDPA
chunks, not the entire262144 allocation. Conv/norm/activation/cache writes/CCL
and control traffic are omitted from this lower bound. It is not measured DRAM
traffic/utilization. [perf_summary.json](perf_summary.json) reconciles the numbers.

Actual decode rows use BF4/LoFi body and BF8/HiFi2 head. Fused prefill rows show
scalar HiFi4 because the op contains a LoFi matmul and independently HiFi4 ring
reduction; compiled chlkc descriptors and raw multi-kernel attributes prove both.
Tracy's scalar field keeps the last kernel's fidelity. Raw CSV is not rewritten.
Likewise, fused advice's108 cores are88 matmul +20 reduction workers, not108
compute cores. `fused_prefill.py` uses rectangular11×8 compute and RS offset(0,8);
naively increasing compute to110 overlaps the reduction region. Generic>100%
FLOP utilization in those rows is not a physical utilization claim.

Advisor decisions are measured under the selected precision. DRAM-sharded head
capacity failures were adapted to legal16K/R2/K5 and32K/R3/K5 and remained slower.
Local split-topk1 was padded to a legal128-width path, passed real/tie checks but
was slower; Sampling1D is retained. B1 M4096 shared-gather/MLP direct-L1 and110-grid
probes include producer/consumer transfers; legal110/H1 is slower after H13 L1
failure. Mix L1 is likewise adapted to a legal smaller output block and slower.
A full64 direct-L1 trial (`full64-b1-prefill-l1-trial-clean.json`) passes exact
128-token trajectories but TTFT median437.450749 ms versus436.612106 selected,
so no placement change is promoted. These reject particular configurations,
not entire layout/precision families. The detailed experiment log retains exact
configs and failed/adapted receipts.

## Memory, context and anomaly classification

[memory_summary.json](memory_summary.json) and the unchanged-capability addition
to [context_contract.json](../context_contract.json) account for actual nativeB1
owners. Construction normal DRAM is identical8,695,110,656 B/device. Prepared
normal DRAM is8,865,768,448 →8,865,866,752 B (+98,304), solely unmatched runtime/
program ownership; named weights/cache/metadata are identical. Fixed TRACE arena
capacity stays128MiB; used occupancy25,624,576 →24,117,248 B does not return any
ordinary DRAM capacity. These are setup checkpoints, not continuous request peaks
or concurrent-serving capacity. [workload.md](../workload.md) retains that scope.

The initial B1 memory audit's coarse graph type labeled TRACE as DRAM. Numeric
buffer_type4, the three exact TRACE addresses and aligned sizes explain its entire
25,624,576-byte excess; normalDRAM0 independently matches the allocator. The
historical failing receipt is retained; corrected offline replay and the candidate's
independent DRAM/TRACE allocator views resolve the anomaly without ignored bytes
or a trace exemption. The domain parser has dedicated CPU regressions.

The earlier pad32 sampler hang was preserved/triaged before owned recovery; one
tile was published against a two-tile compute wait. All reset/list/mesh exit codes,
triage and the slower legal pad128 probe are retained. No later reset is used.
First profiler setup omitted launcher options and failed postprocessing; its
hardware closed cleanly and its timings are not accepted. Corrected real profiles
succeed. Every prototype failure is separate from validated-source evidence.

## Final closure

The coherent same-K2/C80 tiny16 MLP experiment passes exact all-device component
and complete-window checks, including watcher10. Its clean whole-window medians
are193.797350→194.179593us (linear0) and191.883817→192.193408us (full3), before/after
300 replays/arm. It is not selected. The initial reshape/API and artificial debug
L1 failures were adapted, and a real prototype FPU-vs-selected-SFPU mismatch was
fixed before measuring. This is a configuration rejection, not a global family
claim. Its source/failed/adapted receipts remain local under coherent-tiny-mlp-*
and candidate-staging; selected runtime files never changed.

All required hardware gates are complete. Fresh independent final review and a
local stage-owned commit follow. No push is authorized or performed.
Large CSV, Tracy, tensors, logs, arrays and weights remain local. Only compact
runtime/probe code and documentation enter the source patch.

Independent fresh xhigh review returned **clean-pass** after inspecting the actual
goal, source, outputs and raw evidence. Report: [STAGE_REVIEW.md](STAGE_REVIEW.md),
SHA256 `e57d6b2d9b1876b6762dcedc20611084d882cdf658f296ca81ec2c06096934b6`.
Final hardware closed normally; fresh16:46:59UTC empty-fuser/container-only-sleep/
all-four tt-smi proof is `FINAL_HARDWARE_HEALTH.json`. Devices transferred through
the lease wrapper to the memory investigator at generation7. No hardware is
consumed by this checkout during local checkpoint creation.

Local checkpoint checks: syntax, JSON/category reconciliation and exact staged
37-file freeze checks pass. Black23/isort5/autoflake and clang-format19 checks
pass on new/formatted helpers/probes; existing compact generated legacy style
is retained. The broad repository pre-commit invocation could not reach hook
results because its full-worktree Git diff blocked on unrelated NFS metadata;
its log/owned-process cancellation are preserved and no pre-commit pass is
claimed. An explicit local index contains only the28 reviewed files, and every
validated runtime/test blob is checked against freezev4 before commit. Git
plumbing creates the local checkpoint without rescanning unrelated NFS source.
