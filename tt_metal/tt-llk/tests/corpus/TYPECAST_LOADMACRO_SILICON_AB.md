# UInt16-to-Float16_b compiler SFPLOADMACRO A/B

## Result

The generic compiler formation is correct and replaces the fresh typed
load/cast/round/store sequence with the same alternating-L0/L1 macro launch
shape used by production, but it is not yet competitive on Blackhole silicon.
The scoped metric is `TYPECAST_BODY mean(MATH_ISOLATE)` cycles for one tile.
Every value is from a fresh, separately serialized device process.

| selector | sample 1 | sample 2 | sample 3 | median | versus production |
|---|---:|---:|---:|---:|---:|
| production handwritten | 267 | 267 | 267 | 267 | baseline |
| fresh typed C++ + compiler macro | 313 | 313 | 313 | 313 | +17.2285% |

Both selectors passed the physical-Blackhole correctness test before profiling.
The fresh UInt16-to-Float16_b lane is gated by the checked-in element contract
and PCC greater than 0.99.  The same selector also passed CRAQ correctness.

## Compiler discriminator

The accepted generated face function contains one descriptor setup, eight
alternating `SFPLOADMACRO` launches, and a three-NOP drain.  It contains no
`TTREPLAY` and no explicit load/cast/stochastic-round/store row body.  GCC tests
cover default-off behavior, Blackhole and Wormhole emission, and Quasar refusal;
the focused CRAQ differential passed 100/100 trials on both Blackhole and
Wormhole.

The measured residual is setup placement, not missing inner-body formation.
The RC wrapper invokes the generated face function four times per tile, so its
compiler-owned descriptor setup and call execute four times inside the scoped
zone.  Production programs the same descriptor once during operation init,
outside `TYPECAST_BODY`.

A bounded source-only recovery attempted to expose all four faces as one typed
32-row region.  GCC produced exactly one descriptor and 32 alternating macro
launches, but removing the architectural face-counter transitions was not
correct in CRAQ.  The attempt was rejected and is not present in the accepted
source.  Correctly sharing one descriptor across the four face regions requires
descriptor lifetime/placement across typed RWC boundaries; offset substitution
is not a sound replacement.

## Revisions and evidence

- TT-Metal: `a5c01955`
- SFPI-GCC: `e4b974208`
- CRAQ: `be8e859`
- archive: `/localdev/nkapre/typecast-macro-bh-silicon-20260815`
- archive SHA-256 manifest: `8f753f76704f6313138e457173b343261d35384561ad0caf64da3bf53183ef77`
- production ELF SHA-256: `69e81c3cbb6ecd9c3e515e3292aebc3ed81bbf1225084c62be68163b377e8003`
- generated ELF SHA-256: `30fbe1c79be7a10515953709192a984d4e6312056bfbbae28c40cec48d11cd8c`

