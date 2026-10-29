# Resident decode: closing the served step gap (plan row A12)

Status 2026-10-29 18:45: design. The measured gap at 32 users on the 1 M profile: the device step is 48.5 ms (the
harness's traced replay), the adapter's step is 61.2 ms host wall, the vLLM bench's inter-token 91 ms.

## Where the 13 ms per step go today (the version-1 reload contract)

`GLM53FlashForCausalLM.decode_forward` declares `supports_async_decode: False`, so the plugin sends every step with
`reload_inputs=True` and the adapter, each step:

1. stages the wire rows (`_stage_rows`: tokens / positions / page table per logical slot, the slot remap);
2. uploads the device feed (`SlotTable.device_feed` -> `model.ids_dev`, `model.positions`: one packed write under
   `GLM53_DECODE_PACKED_INPUTS=1`);
3. replays the traced step (`model.decode_step`, ~48 ms on device);
4. reads the sampled ids back (one read per row), synchronously, and returns them to the plugin.

Steps 2 and 4 serialize with the device: the upload must land before the replay, the read waits for it. Nothing of
the next step is submitted before the read completes, and vLLM's scheduler runs between (no-async-scheduling).

## The plugin's contract for the alternative (`async_decode.plan_decode_reload`)

With `supports_async_decode: True` the plugin keeps a *decode chain*: a step submits without `reload_inputs` while the
chain is valid, and the explicit commands name every transition:

| command | when | the adapter must |
|---|---|---|
| `reload_inputs` | the chain starts (first step after a prefill), the decode layout changed (rows added / removed / remapped), a verify committed and the next plain step starts a new chain, the host answered a step (the plugin's host-answered reload) | upload tokens / positions for every wire row (today's path) |
| `reload_page_table` | a page table changed | upload the indirection tables |
| `reload_sampling_params` / `reset_sampling_state` | sampling changed | nothing for greedy argmax |
| none | the steady state | replay only: the device feeds itself |

## Model side: the traced step feeds itself

The traced decode step already samples on device (`sample_on_device_mode: all`, the head's argmax -> `ids_dev` read
by the host). For the steady state the captured graph gains two in-graph updates after the head:

* `ids_dev[live] <- argmax` (a copy inside the trace: the next replay's input token is this step's sampled id);
* `positions.pos[live] += 1`, and with it the per-step derived values the KDA / DSA decode read from `positions`
  (`DecodePositions`: the ring row, the pool id, the valid-length bounds). Today the host recomputes them per step;
  resident mode derives them on device from `pos` (a few elementwise ops on `[B]` int32 tensors, part of the trace).

Parked rows (no request) keep decoding their own garbage, as today; their ids are never read.

A reload transition writes the same buffers from the host (today's path) and the chain continues from them, so the
captured graph is the same in both modes: the only difference is whether the host wrote the buffers before the replay.

## Host side

* `decode_forward(reload_inputs=False)`: skip `_stage_rows` / `device_feed`, replay, read the ids. The plugin's
  `reload_inputs=True` keeps today's path exactly.
* The ids readback stays per step for now (the plugin needs them for vLLM's output processor); with vLLM's async
  scheduling the scheduler of step N+1 overlaps the device step N, and the plugin's host-answered reload handles a
  step the host must answer (a stop, a sampling change).
* Speculative decode (the verify / commit graphs) is a chain break by the plugin's rule (verify -> plain reloads);
  the served profiles do not speculate yet (the partitioned cache has no verify path), so nothing changes there.

## Expected gain and the order of work

* The upload and its serialization: ~5-8 ms of the 13 (measured history: packed inputs took the step from 192 to
  174.6 ms when the device step was ~150 ms; the readback is the remainder).
* vLLM's async scheduling (unlocked by `supports_async_decode`): part of the engine's 30 ms per step at 32 requests.
* Burst TTFT (5.2 s mean for 32 arrivals) is the other item: a decode step between prefill chunks needs the traced
  decode to tolerate a mid-prompt slot (the plugin's stable prefill slot plus a model-side guard that skips the
  slot's KDA state write for a row mid-prefill).

Order: (1) the in-graph feedback + the `supports_async_decode` flag behind `GLM53_VLLM_RESIDENT_DECODE=1`, validated
by the served smokes (stages a / b / d: the ids must be identical to today's reload path for the same prompts);
(2) the bench A/B at 32 users; (3) vLLM async scheduling on; (4) the interleave.
