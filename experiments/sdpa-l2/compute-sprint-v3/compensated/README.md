# Group-two protected numerator prototype

Final results and the selected `group2_valid/` revision are in [REPORT.md](REPORT.md).
All 60 final records pass. The result is a long-context research option, not
an unconditional replacement or a production promotion. The sections below
retain the design history and early-stage evidence.

Private numerical-relaxation research; v1/v2 and production sources remain
unchanged. Q256/K512/D128, all CB formats/capacities, input slots and original
readers/writers are preserved. Fidelity, preparation, exp, denominator
compensation, reciprocal and final normalization arithmetic are unchanged.

## Representation and recurrence

CB8 is the protected numerator,64 BF16 tiles, with per-row hi4/lo4 planes.
CB9 retains the same64-tile allocation, now per-row current-PV4/local4.
CB8 is zeroed and fronted once per Q job. The existing first-K CB9 low-plane
clear initializes local to zero. K0 copies each BF16 PV chunk into rootHi;
rootLo stays zero. Maxima and denominator still use the original ping-pong.

For later K chunks, fixed group boundaries are even zero-based K indices;
the final K chunk always flushes. A changed/nonfinite maximum in any of the
64 rows also forces that row group's early flush. The guard is the exact
finite-bit comparison from the frozen v2 early-guard implementation.

- Unchanged maximum and odd nonfinal chunk: local is known empty after the
  preceding boundary/bootstrap, so copy the current BF16 PV into local.
- Unchanged maximum and boundary/final chunk: form FP32
  `(rootHi + rootLo) + (local + chunk)`, split into nearest-BF16 hi/lo,
  and clear local. The local-plus-chunk sum is not spilled to BF16.
- Changed maximum: form FP32
  `((rootHi + rootLo) + local) * canonicalCorrection + chunk`, split into
  BF16 hi/lo, and clear local. This rescales every live contribution before
  adding the new PV. It is an explicitly allowed arithmetic reassociation.

There is no dropped update or unprotected long running sum. Group size2 is
fixed; the existing state probe found group4 failures, so it is not enabled.

## Ownership and synchronization

Protected CB8 remains logically fronted while its owned tiles are updated
in place. Scratch CB9 is published rowwise and popped as a full64-tile
allocation at each nonfinal K end, wrapping its read/write bases while its
local low plane survives. PV writes only its high plane. Every fold clears
local, including forced odd-chunk folds.

The original correction-CB publication still fences preceding PV writes.
K0 bootstrap has an explicit PACK_DONE publication fence. The later CB9
row publication drains root/local writes before normalization or subsequent
K work. On final K, normalization reads/pops fixed CB8, not ping-pong parity;
CB9 is separately balanced. Root reads are relative to its advancing read
pointer during normalization, while root writes use global row indices and
its unchanged wrapped write pointer. No pack/SFPU completion fence is removed.

The compute integration received independent source review before the first
bounded JIT/smoke. Initial device checks are recorded below; completed final
qualification is summarized in REPORT.md.

## Isolated layout revisions

`group2/` is the first frozen prototype described above. `group2_direct/`
writes odd-K PV directly into the local low plane. Odd identity/nonfinal
steps need no copy; odd flushes explicitly use old-local zero and the low
plane as the new chunk. Even steps retain the original grouped recurrence.
Both materialized-PV sites (partial first row and later full rows) use this
offset; all readers and input buffers are unchanged.

`group2_noclear/` additionally omits local-zero stores/packs after even
folds. Local is then *logically* empty, but its physical low slots may be
stale. The next odd PV overwrites them before any read; the odd helper
explicitly ignores old local. Odd changed/final folds still clear local,
because a following even step must consume zero after an early flush.
Final normalization never reads scratch local, and each new Q job rezeros
it. These changes received independent lifetime/source review.

`group2_replay/` uses an18-instruction raw replay for the identity-even fold.
It retains the grouped association and frozen BF16 round/store macros.
The independent review checked macro latencies, offset increments, final
drain, and reinstatement of the full denominator replay after the overlap
of replay slots0–17 and15–30. It passed the initial five-case E suite.

## Initial device evidence

E first-K, final-odd, retained-local uniform, changed-max, and alternating
identity/changed-max tests pass (Q512 or1024, K512/1024/1536/4096).
The direct and no-clear five-case suites passed all numerical and raw trace
gates; these short cases also happened to match the baseline output bits.
G original group2 K3 resident passed. This is not yet long-context E/G
qualification. The independent B reviewer has additional evidence in
`../review/`.

Initial alternating paired resident screens use Q256/K512/D128, four
repeated Q jobs and64 K iterations, preprocessing excluded:

| Candidate | Fresh v2 earlyguard ms | Candidate ms | Time change |
|---|---:|---:|---:|
| Original group2 |8.771333|9.122709|+4.01%|
| Direct PV→local |8.773572|8.920255|+1.67%|
| No even local clear |8.781272|8.763812|−0.20%|

Sources are `e-group2-resident-screen-v1.json`, `e-direct-screen-v1.json`,
and `e-noclear-screen-v1.json`. The last small difference is not a robust
speedup claim; sustained long-context controls are still required.

Sustained E q8/k512 resident controls (seven warmups, five alternating
pairs) amortize the new per-query root initialization:

| Candidate | Fresh v2 earlyguard ms | Candidate ms | Candidate TFLOP/core |
|---|---:|---:|---:|
| No even local clear |139.550986|138.909453|1.978828|
|18-op identity fold |139.570555|131.319082|2.093206|

The replay version reduces time5.912% in this resident experiment. Sources:
`e-noclear-resident-long-v1.json`, `e-replay-resident-long-v1.json`.
Both have exact eager/two-trace equality and accepted numerical metrics;
resident inputs favor identity corrections, so this is not a distinct-input
or end-to-end speedup claim. Final validity-path E/G results are in REPORT.md;
the independent B transfer is documented in ../review/.

`group2-codegen-v1.json` records selected generated-ELF hashes and instruction
windows. The changed-max fold emits an actual `sfpmad` immediately followed
by BF16 `sfpstochrnd`, then residual subtraction and another BF16 round;
it is not a separate multiply/add substitution. This is a selected codegen
check, not a complete ISA simulator or compiler provenance closure.

## Validation contract

`benchmark.py` defaults to frozen v2 early-guard as the baseline;
`--baseline v1` supplies fresh original-winner controls. Candidate versus
baseline bits may differ. Eager versus two mandatory real trace replays must
match raw bytes. Preparation, input immutability and source hashes remain
strict. `../numerics.py` evaluates the original BF16-input FP64 reference:
candidate L2 must be at most1.05×baseline L2+0.0001 percentage points per case.
Numerical failures are written as data after a clean close, not asserted as
device failures. Row tails, PCC and explicit zero-reference absolute metrics
are recorded and require review.

The standard-library `state_probe.py` uses explicit BF16 nearest/ties-away and
simplified final normalization; it is not an attention or device oracle.
At512 chunks group2 passed all7 state cases; group4 failed4/7. See
`state-probe-512-v1.json`. Model identity is based on correction==1, unlike
the actual finite-max-bit guard; model branch frequencies are not hardware
predictions.
