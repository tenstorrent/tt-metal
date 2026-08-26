# OWNER RATIFICATION — threshold/hardshrink store-sink license (2026-08-26)

Verbatim owner order (nkapre, 2026-08-26, in response to lane HK's two
named owner questions):

> yes do that ++ open it also swarm update artifact.. the last table
> should have clear explanatins of how we choose/enable/disable
> compile rpasses per activation.. that is important

## Scope as ratified

Lane HK's certified-floor close named two owner questions:

1. **Licensed sink** ("yes do that"): the store-fold S2 sink for the
   predicated value-merge shape is LICENSED for threshold-fresh /
   hardshrink-fresh class shapes.  Basis accepted from HK's evidence:
   the sunk (store-under-predicate) form is CLOSER to the torch golden
   than the current unconditional write-back — the sem all-lanes store
   canonicalizes Dst (BF16 denormal-flush, 254/2^16 exhaustive
   witnesses in tt/proofs/store-sink-roundtrip) where the golden keeps
   the bits.  This is a value-changing license and follows the EJ
   licensed-knob discipline in full: license tokens ride entry flags,
   cells marked LICENSED, never merged with unlicensed cells,
   paired-CRAQ inequality recorded LICENSED-EXPECTED, correctness
   authority = device-golden at the row's documented tolerance, and
   the license admits only where accuracy vs golden is
   hand-matched-or-better (here: strictly better on the flush
   witnesses).

2. **Fresh-source door OPENED** ("++ open it"): restating
   store-under-predicate in a fresh semantic body is permitted by the
   owner.  Execution preference remains the COMPILER route (the
   licensed sink, item 1) — a source restatement is transcription and
   is the fallback only if the licensed sink fails its gates.

3. Dashboard order (same message): the artifact's final table must
   explain how compile passes are chosen/enabled/disabled per
   activation class (ON set vs Init(0) booking knobs vs drop-one vs
   licensed; promotion protocol).  Executed in the dashboard
   generator, not in this repo.

Recorded by the orchestrator session (Claude, operated by
nkapre@tenstorrent.com).  Implementation lane: HL.
