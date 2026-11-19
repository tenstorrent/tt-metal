# AutoFix: benchmark execution integration

Failing contract: Stage11 requires complete prefill/decode host wall phases and matching modeled work; Stage10 audit logs only enqueue markers and cannot join asynchronous completions. AutoDebug source inspection confirms the returned DecodeOutput generation remains stable through the plugin event wait and host conversion.

Repair: opt-in CPU-only PhaseRecorder buffers timestamps and immutable request/slot/position records. The adapter records blocking prefill return and existing post-event decode processing. SIGUSR1 exports snapshots only when the collector observes idle state. No synchronization, device read or live profiler was added. The model math, precision, context, traces and sampling route remain the selected implementation. Source commit6ef68e29.

Independent host audit:10 synthetic checks passed, covering two outstanding generations, stale advisory positions, remaps, slot reuse and threaded completion. Direct synchronous adapter calls outside the pinned plugin are outside the recorder scope; failed calls cause unmatched-timeline rejection and require restart. Hardware validation pending.

Other setup integration failures: vLLM client download timed out at30seconds; retry withUV_HTTP_TIMEOUT=300 succeeded. GPQA dataset denied unauthenticated access; accessible common MMLU-Pro/IFEval subsets selected without changing any question after scores. Precommit formatter changes applied and required hooks pass before the first server launch.

## Performance tokenizer transport fault

First live collector smoke rejected a4523-token server prompt where the client requested4096. AutoTokenizer class changed between client Transformers5.23.0 and server5.12.1 despite identical snapshot. Minimal fix: pin the separate performance client to Transformers5.12.1/tokenizers0.22.2. No serving source change. CPU independent audit and all planned prompt length/distinctness checks pass; fresh live smoke pending. Failed artifacts retained under setup/collector-smoke/.

## Canonical capacity flag

Attempt1 failed before warmup/accuracy: checker recognizes --max-num-seqs but readiness emits --max_num_seqs. Running startup/config evidence proves32 slots. Retained exact failed attempt in attempts/attempt1. Added the identical capacity using canonical spelling in additional-server-args, retaining the same effective setting; actual command now includes both accepted aliases with equal values. No identity rewriting or checker weakening. Shell syntax and git diff checks pass. Commit b08fe50aea6f34b9790d8eec5005ed3d7d37df6d. Owned idle server stopped for fresh launch.

## Attempt2 shared-accuracy timeout

Observed31/32 normal responses; final IFEval HTTP request exceeded installed fixed3600s timeout while parent client budget7200s had3583.67s remaining. aiohttp total timeout aborts upstream gather and suppresses completed-task score export. Independent source audit confirmed the cause. Recover complete-task scoring offline using exact saved request instances; leave missing response unscored and preserve all frozen questions. Repair workflow source to propagate remaining parent deadline into child timeout, with CPU regression coverage. Installed package and raw responses remain unchanged. No third inference attempt.

Repair completed in editable skills-v24 commit `f61cb04660b549b370258c9d253a55421f4d4cfa`: remaining parent client seconds are explicitly propagated to the per-request timeout, with positive finite validation and upward rounding for upstream integer conversion. Parent wall timeout remains authoritative. 744 CPU tests passed, 1 skipped; package/version checks passed. Installed package unchanged, hardware rerun unverified.

The corrected tokenizer smoke completed both exact 4096-token requests; the phase collector reconciled every request/chunk/decode position in that smoke and all 32 warmup requests. Phase accounting is hardware-validated for these setup workloads only. All required repeated performance profiles remain missing after accuracy abort. Offline replay recovered complete MMLU-Pro and partial IFEval upstream scores without inference or altered raw responses.

Approved third attempt: the default runner budget guard correctly rejected 28,800 seconds. Add an explicit, provenance-bearing owner override shared by runner/checker while retaining the default guard. The HTTP timeout repair remains in the editable workflow. The outer server timeout becomes configurable (default unchanged) and is set to 43,200 seconds for this approved attempt, covering both clocks. Server support artifacts use an explicit separate directory to preserve prior state/source evidence.

## Runner-side budget-check mismatch

Reproduced the exact cached-checker exit 2 after completion. The original runner-delivered owner approval authorizes 28800 client seconds and 10800 setup seconds. Existing workflow repair f287316 is sufficient: the 0.1.26 stage script exits 0, and substituting only its batch_plan in memory makes all cached evidence gates pass. Focused CPU regression: 29 passed. No further source or hardware change. The installed 0.1.24 script still rejects the authorized budget; this is explicitly refuted in [work_log.md](work_log.md), with transcript and diagnostic evidence.
