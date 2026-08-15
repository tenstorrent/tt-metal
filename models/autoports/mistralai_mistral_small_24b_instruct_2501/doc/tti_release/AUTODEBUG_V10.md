# AutoDebug v10: async page boundary and stale state slots

## Scope and method

This report records the two independent EngineCore failures observed by the
authoritative v10 native TTI release run. The packaged fresh-context AutoDebug
runner could not start because its `bwrap` sandbox was denied the loopback
`RTM_NEWADDR` operation by the host. Diagnosis therefore continued with the
skill's serial fallback: preserve the exact failure evidence, reduce each
failure to a source-level invariant, add a focused regression, and run an exact
hardware control before resuming the native workflow.

No score mask, release waiver, implementation substitution, or reduced serving
context was used.

## Failure 1: missing KV page at an exact boundary

The GPQA run failed when host scheduling state reported 2,429 computed tokens
while the async TT device state had advanced to position 2,432. With a 32-token
page, position 2,432 is the first token of logical page 76, but the host had
allocated only through position 2,431. The TT async pipeline can have the
current submitted device step plus two unresolved `UniProcExecutor` outputs,
so the previous one-token KV lookahead was insufficient.

Nested-vLLM commit `971ee6cfcdd97a36a98e26f96ff7dda08441d219`
reserves three async KV lookahead tokens, while retaining any larger speculative
lookahead. Its regression reproduces host position 2,429/device position 2,432
and proves page 76 is allocated.

The exact P300x2 control used an actual 2,399-token prompt and requested 96
output tokens, crossing position 2,432 twice. It completed 1/1 requests with
zero failures. The resumed native GPQA component then completed all 90/90
samples with no request or engine errors.

## Failure 2: stale device state slots after an idle cleanup

After the native benchmark's single-concurrency point drained, the following
32-concurrency point failed while admitting 31 new prefills:

```text
AssertionError: no free device state slot for 31 prefill(s):
held=[0, 1, ..., 31], capacity=32
```

vLLM can retire the final request in a scheduler-only step containing zero model
work. That step is not delivered to the TT worker, so its host-side request and
slot entries can survive until a later model step. Repeated idle transitions can
therefore leave older off-batch owners in all 32 slots even though only the
immediately preceding batch can still have live device state.

Nested-vLLM commit `aab6d846caf95c5e9cf8038f3338650a9132c383`
snapshots the immediately preceding persistent-batch owners. Only when slot
pressure proves the map impossible does allocation reclaim older off-batch
owners; it preserves recent owners and still raises if the stale set is
insufficient. The regression recreates 32 retained owners followed by 31 new
prefills, proves the recent owner remains in slot 31, and proves the 31 older
owners are reclaimed into unique slots.

The exact production-server lifecycle control ran the workflow's point 1
(concurrency 1, 8 requests) immediately followed by point 2 (concurrency 32,
256 requests). It completed 8/8 and 256/256 with zero failures. The resumed
native benchmark repeated those points successfully and continued through the
long-context sweep without slot or page failures.

## Focused verification

The combined host regression command is:

```text
python -m pytest -q \
  plugins/vllm-tt-plugin/tests/test_state_slots.py \
  plugins/vllm-tt-plugin/tests/test_scheduler_async_lookahead.py \
  plugins/vllm-tt-plugin/tests/test_async_decode_preemption.py
```

It passed 10 tests. Hardware controls and the final native component reports are
listed in `RUN_NOTES.md`; raw per-request completion corpora are intentionally
excluded from the repository artifacts.
