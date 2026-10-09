# Higher-precision LM-head Galaxy check

The BFP8/HiFi2-head configuration completed G0 on all eight TP4 replicas on
October 9, 2026 at approximately 07:50:53 UTC. The JUnit receipt records one
test, zero failures/errors/skips, and 2,660.287 seconds. Every replica passed
the concurrent-versus-isolated timing gate; the largest ratio was 1.000006.

This uses native GDN recurrence, accurate full-tile decode attention, BFP8 KV,
FP32 recurrent state and the original BFP4/LoFi decoder weights. Only the LM
head differs from the previous native control. The committed model source is
`d3e8d6021f7aadcb28ba3903d79bd04b288a2819`.

The short-public-prompt, one-user-per-replica measurement requests 128 output
tokens, repeats five times, and excludes prefill/readback. It intentionally
continues past EOS for timing; its saved text is not an API-completion test.
The same-shape native head gave 286.27 aggregate output tok/s; this head gave
281.33. Per-replica TPOT increased 1.61-1.76%. This does not establish the cost
at 32K context or higher batch, nor any accuracy improvement.

- [full-model.json](full-model.json): exact policy, source hashes, chip groups,
  outputs and measured comparisons.
- `hardware.xml.gz`: unmodified JUnit bytes, compressed for publication.
- [manifest.json](manifest.json), [runtime-model-spec.json](runtime-model-spec.json),
  [values.yaml](values.yaml): a newly prepared experimental bundle using the
  actual passing receipt, exact committed source, and corrected health probes.

The native-policy image built earlier remains unchanged. A separate v6 image
build uses this head bundle; its result and all container checks remain pending.
The head controller has advanced to full GPQA. All six checks in
[api.json](api.json) passed: health, model listing, nonstreaming chat, streaming
greedy equivalence, multi-turn chat and concurrent greedy repeatability at
concurrency 128. At 08:09:23 UTC, GPQA had completed 116/198 questions with
110 correct and zero truncations. This is a progress snapshot, not a final
score or accuracy qualification. G0 and API checks alone do not qualify a release.
