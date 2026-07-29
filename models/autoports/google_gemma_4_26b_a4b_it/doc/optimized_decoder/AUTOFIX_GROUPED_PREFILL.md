# AutoFix: grouped expert-prefill boundary failure

Date: 2026-07-29 UTC

## Failure

The clean optimized default selected `prefill_expert_chunk_size=128`. Both
logical-boundary suites failed once physical padding produced more than one
32-token sparse group:

- sliding attention minimum PCC: `0.888242`;
- full attention minimum PCC: `0.900904`;
- unchanged functional bar: `0.995`.

The failure was deterministic, affected both meaningful layer kinds, and
started at logical length 33 / physical length 64. It was therefore treated
as a failed stage gate, not waived as numerical noise.

## Focused hypotheses

| Hypothesis | Isolated evidence | Verdict |
| --- | --- | --- |
| fast block geometry caused the error | chunk 128 also failed with legacy `per_core_n=11`, gate/down block 1 | refuted |
| tail program caused short failures | physical 64/96 used tail geometry and failed; chunk 32 never needs it and passes | refuted |
| BFP8/LoFi was intrinsically below the bar | the identical dtype/fidelity and fast geometry pass when split into one-tile calls | refuted as root cause |
| grouped sparse invocation crossed an unproven contract | same-source A/B changed only chunk 128 to 32 and moved all 20 cases above the bar | verified decoder-level boundary |

The fresh read-only AutoDebug investigation in `AUTODEBUG.md` independently
reached the same conclusion. It ranks grouped down work splitting and grouped
gate/up enumeration/layout as the two lower-level causes still worth isolating.

## Proven fix

The smallest repair changed only the public defaults in
`OptimizedDecoder.__init__` and `OptimizedDecoder.from_state_dict` from 128 to
`TILE_SIZE` (32), plus the defaults test. The fast proven geometry remains:

- `prefill_expert_per_core_n=2`;
- gate/up `in0_block_w=44`;
- down `in0_block_w=11`;
- BFP8 weights and LoFi compute.

The environment override remains available for explicit grouped-sparse
diagnostics, but values above 32 are not the runtime default.

## Post-fix gates

- all 20 logical boundary cases pass; minima `0.995340` sliding and
  `0.998048` full;
- clean default suite: `16 passed, 10 skipped` (only opt-in capacity/perf
  groups skipped);
- physical non-aligned prefill at 262143 and traced decode at current position
  262143 pass for both layer kinds;
- warmed prefill improves from 680.955 to 120.697 ms at batch 1 and from
  21780.254 to 3856.548 ms at batch 32 for sliding attention;
- watcher stress selection: `7 passed`, zero watcher errors/asserts;
- post-run `tt-smi -ls --local` sees all four P300C devices.

Final decoder/test SHA256 values are respectively
`803f0e19451926ce7f5529a05498aeadee5cc186c4e0cb408d53e0de8cef9e7e`
and `829da22cc60600f8bbe2a17064c2b96cb19e74342b4e5e15b8d45dd4184d3b41`.

## Disposition

The decoder fix is complete and proven. A future lower-level TTNN investigation
may compare grouped versus independent sparse gate/up and down calls at groups
1/2/3/4. Until that primitive has its own regression test, grouped chunks
above 32 remain diagnostic-only.
