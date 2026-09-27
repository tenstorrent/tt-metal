# Scheduler call compatibility diagnosis

2026-09-27. Source/host investigation only; no device access or server calls.

The reduced-server traceback records installed vLLM0.26 `vllm/v1/engine/core.py:587` calling `self.scheduler.schedule(self._should_throttle_prefills())`, followed by `TypeError: TTScheduler.schedule() takes 1 positional argument but 2 were given`. The default single-engine implementation of `_should_throttle_prefills` returns `False`; this is an unconditional additional argument, not a DP-only request.

The checked-out TT plugin's `TTScheduler.schedule` (`scheduler.py:136`) and `TTLaneCoordinator.schedule` (`lane_scheduler.py:494`) accept only `self`. The vLLM checkout base uses the older no-argument interface, explaining why that plugin signature previously worked. The prior request-validator repair advanced execution far enough to expose this independent failure. It happens before `model_executor.execute_model`, so neither model tracing nor hardware kernels cause it.

The installed base `Scheduler.schedule(throttle_prefills=False)` does more than accept the argument: at lines463–467 it defers prefill when the flag is true, prefill capacity was not bound, and at least one running request is a decode. Deferred prefill includes running partial-prefill chunks and new waiting prefills; decodes can still run. The `False`/omitted paths preserve historical scheduling.

A TT fix must accept the optional argument and preserve TT's requirement that each step contain only prefill or only decode. Simply adding an ignored argument fixes single-engine `False` calls but silently loses requested throttling. Simply forwarding true after `_schedule_prefill_only` hides decode requests also fails: the upstream condition no longer sees the running decodes. Select the existing decode-only path when the true throttle condition applies to TT's default policy. Forced coordinated modes must remain homogeneous across ranks/lanes.

Recommended discriminating host checks: drive the real installed `EngineCore.step` until a stub executor boundary, proving its boolean call reaches the TT scheduler; test omitted and explicit-false legacy behavior; test true with running decode plus waiting/partial prefill; test capacity-bound and no-decode cases; assert hidden queues and running requests are restored. Check coordinated schedule entry accepts the installed signature while preserving one common lane mode. Live retry remains required after host verification.
