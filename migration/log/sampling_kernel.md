# sampling_kernel.cpp (TIER 1 #11, sender, flag-only loop barrier)

## Block migrated
The single-device loop-barrier readiness broadcast by the final core (orig lines ~210-213, LEGACY
free functions): `noc_semaphore_set(ptr,1)` + `noc_semaphore_set_multicast(...)` +
`noc_async_write_barrier()` -> `loop_pipe.send_signal(1)`. Kept the trailing
`noc_semaphore_set(local_ready_sem_ptr, 0)` (sender clears its OWN cell for the next iteration).
The non-final-core wait branch (else: wait(1)+set(0)) left raw (a plain local-sem doorbell, no mcast).

## Pipe template args / risk notes
`Pipe<>` = `<EXCLUDE_SRC, Flag, ...>`. data_ready = local_ready flag sem; consumed = same id (unused
on this control path). McastRect{loop_mcast_start/end, loop_num_dests}. Constructed inside the
`is_final_core && num_dests>0` constexpr branch. Used global `Noc` (not dataflow_kernel_lib::Noc).

R5 NOC1-SWAP: orig used a local get_safe_multicast_noc_addr that swaps coords on NOC1. The Pipe's
set_multicast uses plain ::get_noc_multicast_addr (NO swap). FENCE: orig barrier, send_signal flush.
Both risks validated away on-device — the 101-core single-device argmax test passed (the broadcast
hit the right cores; flush sufficed). If a future config runs this broadcast on NOC1 with a
non-trivial rectangle, re-verify the coordinate orientation.

## Call-site diff
~4 lines (set/set_multicast/barrier) removed -> 1 `loop_pipe.send_signal(1)` (+ ~6-line construction).

## Validation
test_sampling_argmax_single_device_101_cores[17-0]: SAFE_PYTEST_RESULT: PASS (1 passed in 29.79s),
index=3286.

## Commit
9f93e58ffb0  "apply mcast_pipe to sampling_kernel" (clang-format re-stage)
