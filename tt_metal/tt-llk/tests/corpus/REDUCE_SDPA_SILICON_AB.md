# Reduce-SDPA generated-SFPI silicon A/B

## Result

Blackhole P100A silicon rejects the current generated implementation on
performance, while accepting it on correctness.

| implementation | fresh-process body samples | median | delta |
| --- | --- | ---: | ---: |
| handwritten replay | 839, 839, 839 | 839 | baseline |
| generated SFPI | 914, 914, 914 | 914 | +75 cycles (+8.94%) |

The profiler marker is `REDUCE_SDPA_BODY` on PACK/TRISC2. It excludes SFPU
initialization, prologue, epilogue, packing, and host time. The whole profiled
kernel is also slower (5511 versus 5357 cycles), but that broader number is not
the acceptance metric. Both selectors pass the full 512x64 four-subblock golden
test on the same device.

Evidence is archived at
`/localdev/nkapre/reduce-sdpa-bh-silicon-20260815`. Every sample has a fresh
process/build directory, raw and post-processed CSV, linked pack ELF, complete
objdump, generated `build.h`, log, and SHA256 manifest. The compiler binary hash
is `f584dce043a26fef1ed8d2d11ef9fb6b4903a61c42b2e9af0c9b0701cca5f360`.

Full ELF hashes differ between fresh builds because profiler/debug metadata
contains build-specific values. Extracted `.text` is invariant within each
selector:

- handwritten replay: `d56d5f623e32eef3231c683c2046722cf1690a4f4215c28420f46fa799cb26da`
- generated SFPI: `83984080286a71dae8c4928d37f379e9e20cafe12993dc24478676c5cb38fdbd`

## Disassembly diagnosis

SFPI instruction selection is already correct. The generated body lowers each
source `max` to the same alternating `SFPLOAD; SFPSWAP` pair as the handwritten
body. Functional CRAQ execution confirms exactly 2246 retired SFPU instructions
and 4391 total modeled instructions for each selector.

The difference is instruction delivery. In the linked `run_kernel` body, the
handwritten form contains one eight-entry no-execute replay recording followed
by 64 `TTREPLAY` commands. The generated form contains no replay; its loop issues
the eight `SFPLOAD; SFPSWAP` instructions directly for every group. The static
mnemonic counts make the transformation visible:

| mnemonic | handwritten | generated |
| --- | ---: | ---: |
| `TTREPLAY` | 65 (one record, 64 execute) | 0 |
| `SFPLOAD` | 17 | 71 |
| `SFPSWAP` | 12 | 72 |

This is why local SFPU scheduling alone cannot recover the gap: the generated
pair ordering already matches the replay payload. Replay removes repeated TRISC
delivery pressure by asking Tensix to reissue an already-recorded fixed binary
sequence.

## General compiler opportunity

Add late, post-register-allocation Tensix replay formation. Fingerprint repeated
fixed-encoding Tensix subsequences, record one copy with no execution in a safe
preheader, and replace eligible occurrences with replay commands. This is not a
Reduce-SDPA pattern: it applies to any loop containing an identical, replay-safe
SFPU sequence.

Legality must require a fixed instruction encoding after allocation, replay
capacity, dominance of the recording, no control flow or host-visible side
effects inside the payload, no crossing counter/config/profiler barriers, and
unchanged code for ineligible regions. The first regression should recognize
this eight-instruction load/max payload and also include adversarial near-matches
that differ by LREG, offset, modifier, or intervening counter update.
