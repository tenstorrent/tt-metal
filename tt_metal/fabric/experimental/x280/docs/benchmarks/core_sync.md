# core_sync: the cross-core barrier and its release

`core_sync` is the star-topology barrier the other families use to start their participants
together, and the benchmark of its release. Workers on any core family (Tensix, idle or active
ERISC, DRISC, X280) announce arrival to a director, by an atomic counter increment or a ready
token in their own slot, then poll their own release slot; the director waits for every arrival
(or the host's STOP), writes the release token to every worker and fences. The barrier is
compile-time configured and header-only (`kernels/shared/core_sync_barrier.hpp`, with the
transports in `core_sync_transport_{metal,x280}.hpp`); its storage is caller-owned and is
reinitialized for every invocation. Release destinations come from the host release selector,
which multicasts to a Tensix selection that fills one logical rectangle and unicasts to every
other participant ([harness.md](../harness.md), "Core-sync and packet injection share one host
release selector"). Results are in [results.md](../results.md#core_sync).

## Executables

| Executable | Cases |
|---|---|
| `x280_core_sync_director` | an X280 hart (`firmware-core-sync-director.bin`) directs a barrier over a Tensix grid running `tests/device/kernels/core_sync_probe.cpp`: multicast completion, abandonment on STOP, mixed counter and word arrivals into a Tensix director, singleton and ordered multi-group release (`firmware-core-sync-release-probe.bin`), a renumbered (virtual != NOC0) column, a required multicast that must release exactly its selection, and active-ERISC word workers on both NoCs |
| `x280_core_sync_characterization` | records the raw exit timestamps of a 4 x 4 Tensix grid released by a null run (barrier compiled out), a unicast and a multicast release, with no timing threshold. Built only when named |
| `x280_bench_device_tests` (`CoreSyncHwTest`) | a two-core Tensix barrier, a withheld arrival expiring the director's wait, and idle-ERISC and DRISC workers joining a Tensix release |

The host-only cases of `test_core_sync_hw.cpp` (memory bindings, probe arguments) run in the
`x280_core_sync_offline` CTest, and `x280_core_sync_director_refusal` checks at compile time
that a DRISC core is refused as director. `x280_core_sync_director` and
`x280_bench_device_tests` launch on idle-ERISC and DRISC cores and link slow dispatch.

## Running it

```bash
env -u TT_VISIBLE_DEVICES tt-smi -r 0
TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" build/test/tt_metal/x280/x280_core_sync_director
ninja -C build x280_core_sync_characterization
TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" build/test/tt_metal/x280/x280_core_sync_characterization
```

`X280_CORE_SYNC_DIRECTOR_FW_IMAGE` and `X280_CORE_SYNC_RELEASE_PROBE_FW_IMAGE` name other images.
The renumbered-column case skips on a device whose virtual and NOC0 columns all coincide, and the
active-ERISC case on a device running a single ERISC per Ethernet core.
