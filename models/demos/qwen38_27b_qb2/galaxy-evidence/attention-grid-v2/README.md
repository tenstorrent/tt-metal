# Attention grid correction and persistent v2 sweeps

The v1 placement diagnostic required 11x10 workers, but the allocated Blackhole
Galaxy reports 12x10. It failed before attention execution and closed the device
cleanly. The dependent reader queue stopped without opening hardware. Original
failure receipts are preserved in the subdirectories here.

The correction uses the runtime grid, records it, and tests both 11x10 and 12x10
assignments. CPU validation passed 254 tests and 40 subtests; both opt-in hardware
tests collected. The new source snapshots leave v1 attempts immutable.

The launch JSON files describe the persistent v2 placement and reader services.
At the launch verification, placement had passed the old guard and produced
numerically passing 32K/B16 timings; the reader controller was waiting for its
clean terminal receipt. These are diagnostic runs, not model promotions.

The sweeps retain existing precision. Placement compares native, row-major 64,
FlashMLA reference 64, row-major 80 and outer-column 80, followed by native again.
The reader sweep compares native intermediate KV barriers with thresholds 4, 8
and 16, then repeats native. KV remains interleaved. Both use six pairs:
32K/B16, 128K/B8, 262016/B4, 32K/B32, 128K/B16 and 262016/B8.

See the parent GDN integration report for bandwidth references, correctness
gates, device lock, timeout bounds and separate full-model qualification.
