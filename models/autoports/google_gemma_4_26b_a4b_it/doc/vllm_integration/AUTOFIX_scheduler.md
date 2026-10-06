# Scheduler call compatibility repair

2026-09-27. Status: **host-verified for the target DP1 scheduler**. No device access or live server calls were performed by this investigation.

## Retained change

[AUTODEBUG_scheduler.md](AUTODEBUG_scheduler.md) traces the failure to installed vLLM0.26 `EngineCore.step`, which unconditionally passes a boolean prefill-throttling argument to `schedule`. The older plugin accepted no argument, so its first request crashed before model execution.

Changed `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py`:

- `TTScheduler.schedule(throttle_prefills=False)` accepts omitted, explicit false, and explicit true calls.
- Omitted/false calls preserve the existing prefill-preference and decode-fallback policy.
- With true in default scheduling mode, a running decode and non-bound prefill capacity select the existing decode-only path, deferring both waiting prefills and running prefill continuations. This evaluates the installed base scheduler's throttle condition before TT prefill filtering hides the decode requests.
- Capacity-bound or decode-free steps keep prefill progress. True combined with an explicitly forced coordinated mode raises `NotImplementedError`; no unverified cross-rank policy is silently applied.

Only `scheduler.py` and `tests/test_scheduler_compatibility.py` are implementation/test changes in this repair. The lane coordinator is unchanged.

## Tests and results

The initial eight-case focused suite failed **7 cases and passed 1** before the repair. The real installed `EngineCore.step` test reproduced the same `TypeError` at `core.py:587` as the saved server traceback. The legacy no-argument case passed.

After the repair, the final suite contains **9 passing tests**:

1. The actual installed `EngineCore.step` reaches a stub executor boundary through the repaired TT scheduler.
2. Omitted/false calls preserve prefill preference and restore hidden queues/running requests.
3. True defers waiting and partial prefills while scheduling decode.
4. No-decode and capacity-bound cases continue prefill.
5. Forced-mode true calls reject unsupported coordinated throttling explicitly.
6. A real installed `TTScheduler -> AsyncScheduler -> Scheduler` lifecycle uses actual `Request`, `SchedulerConfig`, `CacheConfig`, `ParallelConfig`, `KVCacheConfig`, and `KVCacheManager` objects. It schedules a 31-token prefill for A, applies synthetic sampled token42, adds waiting B, schedules only A's one-token decode with throttling, then admits B's 31-token prefill when capacity is marked bound. No base scheduler or allocator method is mocked in this case. Only unused multimodal/structured-output collaborators and minimal model/config metadata are supplied as stand-ins.

Exact command, from `/workspace/tt-metal`:

```bash
python -m pytest vllm/plugins/vllm-tt-plugin/tests/test_scheduler_compatibility.py \
  -q --disable-warnings --tb=short
```

Formatting used `python -m black --target-version py310` on the new test file. `git -C vllm diff --check` passed. No C++ or CMake changed, so no build was required.

## Scope and remaining evidence

A bounded check of existing lane-coordinator tests exposed an independent vLLM0.26 mismatch: `TTLaneCoordinator` cannot instantiate because it lacks the newly abstract `pause_state` and `set_pause_state` members. Temporary lane signature edits and their new tests were removed rather than expanding beyond the DP1 target. Existing lane-coordinator tests therefore remain incompatible with this installed version; this repair does not claim DP lane support.

The real host lifecycle proves scheduling and allocator call compatibility for its small request shape. It does not prove accelerator execution, logits correctness, sampling output, or final serving readiness. The main stage owns reduced-server retry and subsequent failures after scheduling.
