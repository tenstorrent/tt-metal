# Health

[`health.py`](../health.py) turns one router's liveness samples and main pass into facts about whether it is running:
- whether each of its ERISCs is held in reset;
- how its wall clock moves;
- whether its heartbeat moves.

They are facts, not verdicts. A router can look stuck for several reasons, and the reader, such as the viewer, weighs
them together.

## What it takes and returns

`get_health(snapshot, router)` takes the loaded snapshot and a router decoded by [`routers.py`](../routers.py), and
gives a `Health`:

| Field | What it holds |
| --- | --- |
| `t_ns` | Each sample's host time: the liveness rounds, then the main pass |
| `eriscs` | For each of the router's `erisc<n>`, an `EriscReset`: its physical ERISC, its bit in `soft_reset`, and whether that bit was set in each sample |
| `wall_clock` | A `WallClock`: the 64-bit count in each sample, whether it rises, and its rate in MHz |
| `heartbeat` | A `Heartbeat`: the word in each sample, whether each carries the router's magic, and whether the router's words changed. `None` when the manifest's heartbeat area is null |

Each per-sample tuple has one entry per sample, and `None` where capture didn't read the value. A router capture
couldn't open has no samples.

**The samples.** Capture samples every router's heartbeat and core registers in a few liveness rounds, before its main
pass. The main pass then reads the core registers again, first, and the heartbeat area later, with the other areas.
That last sample takes the time of its first read, `started_ns`.

From one of the T3K snapshots, for a router whose loop was running, with four samples (three rounds 1 s apart, then the
main pass):

```python
health.eriscs["erisc0"]       # EriscReset(physical_erisc=0, mask=2048, in_reset=(False, False, False, False))
health.wall_clock.rises       # True
round(health.wall_clock.mhz)  # 486
health.heartbeat.changed      # True
```

## The names it reads

The meaning of these names is in decode's code (case 2 of "What decode reads, and what it knows" in the
[README](../README.md)). A renamed one fails with `ManifestError`:

- `soft_reset`, `wall_clock_low` and `wall_clock_high` in `archs.<arch>.core_registers`, and `soft_reset`'s
  `reset_masks`;
- the `heartbeat` area's `magic` and `magic_mask`;
- each router ERISC's `physical_erisc`.

## Reset, per ERISC

`soft_reset` has one bit per RISC on the core, and a set bit holds that RISC in reset. Each ERISC's bit is
`reset_masks[physical_erisc]`.

- **Why the physical ERISC.** A router's `erisc<n>` is the kernel's index, and which ERISC it runs on depends on the
  mode. In Blackhole's single-ERISC mode, base firmware keeps ERISC0, so the one processor runs on ERISC1, and
  `erisc0` reads ERISC1's bit.
- **Other bits are set on a healthy core.** Both T3K and Blackhole cores read `0x47000`. Those are the bits for TRISCs
  and NCRISC, which an Ethernet core doesn't have. So "any bit set" doesn't mean "in reset", which is why each ERISC
  reads only its own bit.
- **Outside a run, Blackhole's ERISC1 is in reset.** tt-metal holds it there before a run and after teardown
  (`assert_active_ethernet_cores_to_reset`), so ERISC1's bit set after a close is expected.
- **What it tells a reader.** A core held in reset looks like a hung router, since its heartbeat and counters freeze.
  But nothing is stuck: something stopped it.

## Wall clock

The wall clock is a free-running 64-bit cycle count, read as two 32-bit words. Reading the low word latches the high
one, so capture reads the low word first, as firmware does (`get_timestamp`, `eth_read_wall_clock`). Each sample's
count is `high << 32 | low`.

- **`rises`** says whether each count is above the one before. A clock that doesn't rise means one of two things. The
  core isn't clocked, or a read paired a low word with a stale high one, which can only show when the low word wraps
  (about every 5.4 s at 800 MHz).
- **`mhz`** is the change in count from the first sample to the last, over the change in host time.
  - **Not a constant.** A Blackhole run measured 1350 MHz while the fabric was held and 800 MHz after it closed, and
    the T3K snapshots measure 486 MHz. The manifest doesn't record a rate, so decode measures it.
  - **Host times compare only within one host,** since they come from `time.monotonic_ns` on the capture host.

## Heartbeat

Every `period_iters` main-loop iterations, the router writes its `magic` with a 16-bit count to the heartbeat word
(`FABRIC_KERNEL_HEARTBEAT_MAGIC`, in its main loop). A word whose top half is the magic (`(word & magic_mask) ==
magic`) is the router's.

- **The word is shared.**
  - **Wormhole.** Base firmware writes its own `0xABCD____` there whenever the router yields to it. An idle router
    yields after 10,000 idle iterations, by default (`default_firmware_context_switch_interval`, `WAIT_FOR_IDLE`).
  - **Blackhole.** The word is base firmware's own heartbeat, `eth_status_t::heartbeat[0]`, which both write (see the
    comment in `aerisc_context_switch`). After a close it kept changing without the router's magic.
- **`router_magic`** says, for each sample, whether the word was the router's.
- **`changed`** says whether the router's words differ: `None` with fewer than two of them.
  - In the T3K snapshot, 20 routers had the router's magic in all four samples and changing, and 17 had none. So no
    router words doesn't mean a hung router: base firmware may have had the word each time capture looked.
  - Two equal router words can happen by chance. One router didn't change in one round on a Blackhole run.
- **Base firmware's words are just "not the router's".** Its magic isn't in the manifest: UMD has it, but not in a
  header tt-metal can include.

## What it doesn't cover

- **Telemetry's per-ERISC heartbeat timestamps** (`tx_heartbeat`, `rx_heartbeat` in `FabricTelemetry`). They decode as
  values with the telemetry area, but health doesn't use them, since they are likely written only with telemetry on.
- **Blackhole's link status words** (`base_fw_*` areas). They decode as plain `u32` values.
- **Comparing snapshots.** Health reads one snapshot's samples. Whether a router moved between two snapshots is left to
  the reader.

## Files cited

| File | Symbols |
| --- | --- |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `FABRIC_KERNEL_HEARTBEAT_MAGIC`, `FABRIC_KERNEL_HEARTBEAT_MAGIC_MASK`, `FABRIC_KERNEL_HEARTBEAT_PERIOD_ITERS` |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | the main loop's heartbeat write |
| `tt_metal/fabric/erisc_datamover_builder.hpp` | `default_firmware_context_switch_interval`, `WAIT_FOR_IDLE` |
| `tt_metal/hw/inc/internal/tt-1xx/risc_common.h` | `get_timestamp` |
| `tt_metal/hw/inc/internal/ethernet/tt_eth_api.h` | `eth_read_wall_clock` |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h` | `eth_status_t`, and the comment in `aerisc_context_switch` |
| `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp` | `RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset` |
