# Reciprocal fresh semantic-C++ silicon A/B

## Result

The fresh typed accurate-BF16 Reciprocal body is a stable scoped Blackhole
silicon win. Each value below came from a separate serialized profiler process.

| Selector | `RECIPROCAL_BODY` cycles/tile |
|---|---:|
| Production | 467 / 467 / 467 |
| Fresh semantic C++ | 459 / 459 / 459 |

The semantic body is 8 cycles, or 1.713%, faster. This is a device-body zone,
not a whole-kernel throughput claim.

## Correctness contract

Both production and semantic selectors passed the paired device correctness
run before profiling. The dedicated suite also passed Blackhole CRAQ for
accurate and approximate BF16/FP32 plus registered-domain edge inputs, and
compiled all ten selector cases for both Blackhole and Wormhole.

Float outputs must pass both the existing per-format element tolerance and
PCC gate; the measured accurate `Float16_b` lane uses `rtol=0.05`, `atol=0.05`,
and PCC greater than 0.99. The edge suite includes zero and the registered
finite Reciprocal boundaries. It deliberately does not force IEEE specials:
Reciprocal is not registered in `SPECIALS_READY_OPS`, and the production path
itself does not preserve the forced-NaN golden.

## Semantic discriminator and compiler mechanism

The test-only body names typed destination values, `approx_recip`, arithmetic,
and `sfpi::min`. It contains no fixed LREG, raw TTI, replay range,
SFPLOADMACRO template, or hand-interleaved schedule. The canonical BH accurate
`sfpu_reciprocal<false>` typed expansion currently triggers an
`rvtt_expand` SSA verification ICE (definition follows use), so the accepted
form expresses a branch-free cubic Newton correction in value space.

The compiler forms a ten-instruction replay capture around the semantic load,
reciprocal seed, correction, min/swap, store, and increment, then issues seven
replays. Production's accurate BF16 correction remains a larger handwritten,
unrolled sequence without replay. The win therefore exercises a generic,
operation-independent compiler mechanism: typed loop replay formation.

## Evidence

The immutable local archive is
`/localdev/nkapre/reciprocal-semantic-bh-silicon`. It contains the paired
correctness log, six profiler-process logs, representative ELFs and
disassemblies, and provenance. The SHA-256 of its aggregate `SHA256SUMS` file
is `61844ea997e5b118d7b195ed2a393d43c3a2025fb67d20f8a77b1b3b1483c47b`.

The Blackhole CRAQ archive is
`/localdev/nkapre/reciprocal-semantic-craq-bh-v2`; its `llk_sim.tsv` SHA-256 is
`1da299225765b1ef8cab23b999eb0eaf4c0051ead554a7bc61e2dc396e1218db`.
The CRAQ modeled-cycle sign is not used as a silicon claim.
