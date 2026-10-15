# transport_bandwidth: sustained rate of one transport leg

`transport_bandwidth` measures the achieved rate of one transport leg between the X280 and a
Tensix core: a timed loop that issues the same transfer repeatedly, with no payload inspection,
no application credit and no echo in the timed path. Its design (protocol, validity and byte
accounting, measurement boundary, the frontend/backend contract, the timed loop and the member
table) is [design.md § Transport bandwidth](design.md#transport-bandwidth); its results are in
[results.md](../results.md).

## Members

| Executable | Member | Points |
|---|---|---|
| `x280_transport_bandwidth_tensix_to_x280` | a Tensix data-movement kernel issues `noc_async_write` into X280 LIM; the X280 is a passive target and runs no image | 64, 128, 256, 320, 384, 512, 1024, 2048, 4096, 8192 and 16384 B, each at the DRAINED and INJECTION boundaries |
| `x280_transport_bandwidth_x280_to_tensix` | X280 harts 0..H-1 each issue `SIZE/W` posted stores of `W` = 4 (`sw`), 8 (`sd`) or 64 B (`vse32.v`) from a register-held 64 B pattern through their own paired-range NoC TLB window into their own 64 KiB range of a host-allocated Tensix L1 buffer; the Tensix is passive. The harts start their timed brackets together and split the run's `N` issues evenly. Image `firmware-transport-bandwidth-x280-to-tensix.bin` | H = 1: 64, 256, 4096 and 65536 B at each store width; H = 2 and 4: 64 and 65536 B at each store width; each at the RETIRED and INJECTION boundaries |

Each size is one NoC packet. RETIRED ends on one read through the window after the stores, which
under the posted ordering mode does not wait for them, so it is an issue rate
([measurement.md](../measurement.md#quantities-that-share-a-label)).

An X280 -> Tensix record carries every hart's raw `mtime` endpoints and `rdcycle` delta
(`hart_brackets`); its batch is hart 0's. The tabulator derives the run's span, from the first
hart's start to the last hart's end, and the run's GB/s over it; every hart reads the same tile
`mtime`.

## Running it

Both executables open at 14 LIM ways like every benchmark, and the harness boots resident idle
on a reset-held chip ([harness.md](../harness.md#running-benchmarks)).

```bash
env -u TT_VISIBLE_DEVICES tt-smi -r 0
TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" build/test/tt_metal/x280/x280_transport_bandwidth_tensix_to_x280
TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" build/test/tt_metal/x280/x280_transport_bandwidth_x280_to_tensix
```

`TRANSPORT_BW_N` sets the timed issue count (default 16384); X280 -> Tensix splits it over the
harts, so it must be a multiple of H. `TRANSPORT_BW_W` sets the warm-up
count; unset, the Tensix -> X280 member calibrates it from a `W=128` pilot to 25 ms of device
work and the X280 -> Tensix member uses 128. `X280_TRANSPORT_BW_X280_TO_TENSIX_FW_IMAGE` names another X280 image. Records go to the
characterization output directory ([README](../../README.md#2-build-boot-and-run-a-benchmark)).
