# Blackhole classic profiler clock repair

The grouped Qwen reduced profile aborted while pairing `TRISC-KERNEL` markers:
`End marker found without corresponding start`. Its failing low clock words were8/10.
AutoDebug found that the classic profiler reads live HIGH, writes packed HIGH to L1,
then reads LOW. A natural LOW rollover can produce a timestamp almost2^32 cycles
behind adjacent markers; timestamp sorting can then expose an unmatched endpoint.
The original raw endpoints were unavailable, so attribution of that particular
Qwen failure remains probable. A focused device regression reproduces the defect.

`tt_metal/hw/inc/api/debug/kernel_profiler.hpp` now reads live HIGH/LOW/HIGH and
retries when the full HIGH words differ, before masking or writing a timestamp.
This Blackhole-only helper covers ordinary markers, push/flush start/end, and
accumulated timing. Wormhole/Quasar paths, streaming profiler, timestamp encoding,
and host validation are unchanged. Using live HIGH also avoids shared latched-HIGH
interference between RISCs. The extra load/branch affects profiling only; its
performance overhead has not been measured.

The new `tests/tt_metal/tools/profiler/test_clock_rollover.cpp` and four kernel
files exercise the real production marker writer on both DM RISCs and all three
TRISCs. They bracket samples with coherent reference timestamps around natural
LOW rollover and save raw snapshots before normal profiler parsing. No timestamps
are injected or repaired, and no parser assertion is relaxed. Raw artifacts are
outside Git; only code and this compact report are published.

## Validation

Host compile/link against local backend3fbf3dbf1e88112aae1ca39cff9d1bc8a1da66e8
passed with clang20. It warns about a deprecated single-device accessor and the
host search path for transitive libnsl; hardware execution used Ubuntu24 with
libnsl installed. Device0 is Blackhole p300c. Original and fixed isolated source
roots share that compiled backend; only the classic profiler header differs.
Original header SHA256: be5e6ddea32ef81e9053901b8ebaea4bf6f1b44070ad424a830e70b157c91f31.
Fixed header SHA256: b6a21795e2d8f82fb54d20586cf0c65e2ed579378aee8a18bd986b11e106163d.

Run with `TT_METAL_DEVICE_PROFILER=1`, matching `TT_METAL_RUNTIME_ROOT`, and:

```sh
test_clock_rollover --device 0 --cores 8 --rounds 1 --output-dir /evidence/clock-original/raw --expect-tear
test_clock_rollover --device 0 --cores 8 --rounds 1 --output-dir /evidence/clock-fixed/raw
test_clock_rollover --device 0 --cores 32 --rounds 3 --output-dir /evidence/clock-fixed-stress/raw
```

| Reader | Lane boundary checks | Retained samples | Inconsistent timestamps |
|---|---:|---:|---:|
| Original |40|3,761|7|
| Fixed |40|3,786|0|
| Fixed stress |480|45,311|0|

These counts cover five RISCs per core around shared physical rollover events;
they are not counts of independent elapsed clock periods. All runs passed normal
profiler parsing and closed devices. A separate read-only review decoded every
raw snapshot and confirmed the counts, source hashes, and selected TRISC1 assembly:
original HIGH/store/LOW versus fixed HIGH/LOW/HIGH/retry before stores. Push/flush/sum
branches are source-audited, not separately forced by this regression.

Local logs and raw captures are under `/home/mvasiljev/build/qwen38-swe/clock-rollover-*`.
The grouped reduced workload is being repeated with the fixed reader and intact
assertions; no accepted grouped device timing exists yet. An initial retry was
interrupted during uncaptured startup after compile paths revealed that this
backend needs `TT_METAL_RUNTIME_ROOT` as well as `TT_METAL_HOME`; it closed normally.
The subsequent run uses the explicit runtime root and a fresh cache, with compile
include paths checked against the patched overlay.
