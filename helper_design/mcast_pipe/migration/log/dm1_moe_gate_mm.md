# moe_gate_mm dm1.cpp — DEFERRED (coverage-gap)

Kernel: `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/dm1.cpp`
Tier: 2d. Status: deferred (coverage-gap). No code change.

## Why (binding blocker: no runnable test on this chip)
Mapped test `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_mm.py::test_moe_mm`
begins with:
```python
if ttnn.device.is_blackhole() or ttnn.device.is_wormhole_b0():
    pytest.skip(reason="Disabled by issue #44858: stable nightly moe_mm failures ...")
```
This is a single Wormhole b0 chip, so the test unconditionally skips — there is no way to
device-verify a migration of dm1.cpp here. ~1-2 tries rule: deferring on coverage rather than
burning attempts.

## Secondary note (migration is also non-trivial)
The only intra-chip rectangle mcast is the collector core's group-mask broadcast (lines 367–373):
- `noc_async_write_multicast_one_packet(local_group_masks_addr, group_masks_noc_addr, 2048, num_dests=7)`
- `*my_semaphore_ptr = 1; noc_semaphore_set_multicast(semaphore_addr, group_semaphore_noc_addr, num_dests=7, ...)`

`num_dests=7` IS a compile-time literal, so the count itself is expressible by
`SenderPipe<NOC=1, ..., NUM_ACTIVE_RECEIVER_CORES=7, PRE_HANDSHAKE=false>`. BUT the broadcast
semaphore (`partial_semaphore` / `my_semaphore_ptr`) is heavily multiplexed: the same cell is a
monotone reduce counter incremented by the sender→collector phases (`noc_semaphore_inc`, lines
298/325) and wait_min'd at 1 / 7 / 8 across phases, with manual `*ptr = 0` resets. A `SenderPipe`
ctor's `data_ready.set(VALID)` plus the receivers' `receive()` clear-flag would clobber that shared
counter. Migration would need to be scoped to just the doorbell semantics and carefully audited —
but it cannot be validated here regardless.

## Verdict
COVERAGE-GAP. Test skips on this hardware. Deferred. Helper untouched.
