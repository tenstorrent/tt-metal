# Reduce-SDPA generated-SFPI silicon A/B

## Result

Blackhole P100A silicon accepts the final generated implementation on both
correctness and performance after generic compiler replay hoisting.

| implementation | fresh-process body samples | median | delta |
| --- | --- | ---: | ---: |
| handwritten replay | 839, 839, 839 | 839 | baseline |
| generated SFPI | 914, 914, 914 | 914 | +75 cycles (+8.94%) |
| generated, compiler-owned load | 855.5, 855.5, 855.5 | 855.5 | +16.5 cycles (+1.97%) |
| generated, typed boundaries + D1 replay hoist | 834, 834, 834 | 834 | -6 cycles (-0.714%) |

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

The first generated fixture used raw `TTI_SFPLOAD`. Although the linked ELF
contains the expected load words, GCC sees each as opaque volatile `.ttinsn`
inline asm, which terminates replay discovery before every compiler-owned
`SFPSWAP`. Expressing the same load through the SFPI destination-register API
makes the full eight-word sequence visible to the existing generic post-RA pass.
The compiler then emits two 8-slot captures and fourteen playbacks, passes the
same physical golden, and measures `855.5` cycles in each of three fresh device
processes. This recovers 58.5 of the original 75-cycle deficit without changing
the compiler or production LLK.

The complete follow-up archive is
`/localdev/nkapre/reduce-sdpa-compiler-load-bh-silicon-20260815`: both-selector
correctness, three fresh processes per selector, 794-byte raw/post profiler rows,
linked ELF/objdump/build headers, provenance, and SHA256 manifest. The earlier
compiler-visibility proof is
`/localdev/nkapre/reduce-sdpa-compiler-load-build-v2/pack.text.objdump`.

The leading explanation for the remaining 16.5-cycle gap is capture placement:
the linked generated body has two static capture sites and fourteen static
playbacks, but each capture site is inside a dynamic loop and is revisited on
the backedge. The handwritten LLK records once outside `REDUCE_SDPA_BODY`.
Matched capture-hoisting A/B must establish full causality; the next compiler
experiment is therefore safe capture hoisting, not opaque-asm decoding or
kernel-specific instruction selection.

That experiment is now complete.  SFPI-GCC `5a849606f` recognizes only
fixed-encoding, compiler-visible replay-safe payloads in a single-block loop,
records them without execution in a dedicated preheader, and replaces the loop
copies with playback.  It rejects calls, opaque assembly, explicit replay
owners, abnormal entries, MEM/GPR-dependent encodings, and unsupported CFGs.
The typed fixture also represents TTINCRWC as a compiler barrier and the
architectural L8 dummy load as a no-result typed builtin.  Final linked code has
two static eight-slot preheader captures and eight playbacks per loop; dynamic
capture executions at block height four fall from eight to two.

The final paired archive is
`/localdev/nkapre/reduce-sdpa-d1-bh-silicon-20260815`.  Both selectors pass the
physical golden; three fresh processes produce `840,840,840` handwritten and
`834,834,834` generated `REDUCE_SDPA_BODY` cycles.  Raw/post profiler rows,
ELFs, objdumps, build headers, logs, provenance, and SHA256SUMS are retained.
The compiler's replay suite is 52/52, full RVTT is 713 pass with the same 15
baseline failures and two expected failures, and ineligible on/off output is
byte-identical.  CRAQ passes functionally but is not used as the performance
authority.

The optimization remains default-off.  Reproduce the D1 binary through the
checked-in LLK harness, without a driver/specs override, by exporting:

```bash
TT_LLK_EXTRA_COMPILER_OPTIONS=-mtt-tensix-optimize-replay-hoist
```

The same option applies to both selectors in a paired process; the handwritten
raw replay path is ineligible, while the compiler-visible generated loop is
transformed.  An empty or unset variable preserves the ordinary harness command
byte-for-byte.

## General compiler opportunity

The first conservative late, post-register-allocation Tensix replay-hoist phase
is implemented behind an opt-in flag. Keep the payload
compiler-visible, fingerprint repeated fixed-encoding Tensix subsequences,
record one copy with no execution in a safe preheader, and replace eligible
occurrences with replay commands. This is not a Reduce-SDPA pattern: it applies
to any loop containing an identical, replay-safe SFPU sequence.

Legality must require a fixed instruction encoding after allocation, replay
capacity, dominance of the recording, no control flow or host-visible side
effects inside the payload, no crossing counter/config/profiler barriers, and
unchanged code for ineligible regions. The first regression should recognize
this eight-instruction load/max payload and also include adversarial near-matches
that differ by LREG, offset, modifier, or intervening counter update.
