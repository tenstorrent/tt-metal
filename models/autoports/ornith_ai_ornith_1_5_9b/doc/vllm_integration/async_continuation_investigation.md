# Async chunked-prefill continuation investigation

2026-11-13. Read-only AutoFix investigation of `diagnostic6_async/server.log`;
no hardware/server operations or source edits. One AST-isolated host check.

## Diagnosis

**Plugin slot allocation moves a continuing prefill request to another slot
without moving its state.** Ornith correctly rejects the resulting mismatched
prefix. This is independent of the missing original prompt boundary on resumed
requests: the failing scheduler output has `resumed_req_ids=set()` and the
continuing request has zero generated outputs.

Do not relax `Continuation must follow this binding's exact prior end`, enable
`allow_missing_history`, or silently rebuild from zero to hide this mismatch.
The ownership map must describe where the existing recurrent/cache/prefill-tail
state actually resides.

## Exact observed sequence

Let A be request `cmpl-a37069f2519cdeb8-0-a46a7b7d` (2051 prompt tokens), B be
`cmpl-a7d1b10e27321b6a-0-adf9b73a` (1033 tokens), and C be new request
`cmpl-9e92701434dbfa4d-0-99486288` (129 tokens).

| Log lines | Submitted operation | Slot ownership consequence |
|---|---|---|
| 463-466 | A prefill `[0,1024)` then `[1024,2048)`, slot 0 | A owns slot 0. |
| 467-469 | A final `[2048,2051)`, slot 0; B first `[0,896)`, slot 1; host sampling because B is intermediate | B's prefix end 896 and recurrent/tail state belong to slot 1. |
| 470 | One device decode, `reload_inputs=true` | A can decode while B remains an off-batch live prefill owner. |
| 471-472 | Prefill row order C, B; C `[0,129)` assigned slot 1, B `[896,1033)` assigned slot 2 | B has been relocated in host bookkeeping, but no prefill remap accompanies the call. |
| 524-531 | Whole-batch preflight fails in `prefill_runtime.py:427` | Slot 2 does not have B's prior end 896. |

The scheduler dump at line 474 explicitly identifies B's `num_computed_tokens`
as 896 and `num_output_tokens` as zero. C is newly scheduled. This is ordinary
chunk continuation with interleaved decode/new admission, not preemption or a
request asking to resume generated history.

The single decode event does not log its remap. Nevertheless A at slot 0 is
the only completed prompt in this sequence; the allocator reproduction below
matches the observed outgoing `[1,2]` slots exactly. There is no evidence that
sampling mode itself corrupted cache contents.

## Exact source chain

`P` is `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`, inspected at
`042a94863ad27d88207bcf71abdf67122c6f4c27`.

1. `P/src/vllm_tt_plugin/model_runner.py:1295-1336`,
   `_alloc_prefill_state_slots`, excludes **every request in the current
   prefill batch** from `held` (`req_id not in prefilling`, lines 1311-1316).
   It then chooses slots by current packed row and overwrites
   `_req_state_slot[req_id]`; it never first checks whether a prefilling request
   already owns state from an earlier chunk.
2. For map `{A:0, B:1}` and rows `[C,B]`, initial `held={0}`. C's row 0 is held,
   so it takes lowest free slot 1. B's row 1 is now held, so it takes slot 2.
   B's original slot was incorrectly treated as free for the newcomer.
3. `model_runner.py:2765-2774` calls that allocator for `is_prompt`. The
   `_decode_state_slot_remap` path is only used for decode. Prefill does not
   submit a matching state permutation.
4. `model_runner.py:4129-4142` passes the selected slots as `empty_slots`.
   `tt/generator_vllm.py:189-199` passes them unchanged to canonical prefill.
5. `tt/prefill_contract.py:328-367` checks the supplied slot's existing prefix
   history. Its precision/profile rebuild path requires `bound['end']==start`;
   a wrong-slot history cannot qualify.
6. `tt/prefill_runtime.py:425-427` requires the slot's prior end to equal the
   continuation start. This is the correct fail gate. Its `remap_slots`
   implementation at lines 211-233 already supports simultaneous movement of
   eager-prefill history when a real model remap is requested.

Model decode already calls `prefill_runtime.remap_slots` with its accepted slot
remap. Missing a second model-side callback is not the cause: the offending
ownership reassignment happens in the plugin's **prefill allocator**, with no
remap command at all.

The log's `releases:0` counter is not evidence that request release callbacks
never ran: the generator increments that named counter for trace release near
`generator.py:1257`, not in `release_request`. Do not diagnose a release leak
from that counter.

## Host reproduction and smallest repair boundary

Used `python_env/bin/python` to AST-extract the actual
`TTModelRunner._alloc_prefill_state_slots` into a namespace with a fake runner:
capacity 3, live A/B/C, ownership `{A:0,B:1}`. No plugin, TT, or model import.

```text
before {'A': 0, 'B': 1}
prefill rows ['C', 'B']
returned slots [1, 2]
after {'A': 0, 'B': 2, 'C': 1}
```

The minimum repair belongs in the existing plugin allocator, with no new
capability/configuration/interface: reserve all live owned slots before
allocating newcomers, return an existing request's current slot unchanged,
and use free slots only for requests without an ownership claim. In the
reproduction the correct result is `[2,1]`, keeping A at 0 and B at 1. Reserve
continuing owners before processing any row; merely checking B's existing
slot when its turn arrives is too late because C has already claimed it.
Validate uniqueness/capacity and retain the existing fail-closed exhaustion
behavior. Preempted/finished requests already lose their claims through
`_release_dead_state_slots`; fresh re-admission can then allocate normally.

`P/tests/test_state_slots.py` is the natural host regression location. Its
current “re-prefilled live request must keep its own slot” check uses a case
where request slot already equals preferred row, so it misses this collision.
Add a continuing owner whose slot differs from its new packed row and a new
request ahead of it; maintain a fake device owner array and assert map/state
agreement without any gather. Also cover reversed continuation row order,
an off-batch owner, and allocation following real preemption/release.

After that host repair, rerun the exact reduced mixed workload with async and
sync controls, preserving Ornith's continuation gate. Verify B stays in its
original slot through the second chunk and request outputs agree. Existing
successful batches do not establish this interleaving. No repaired device
result is claimed by this investigation.
