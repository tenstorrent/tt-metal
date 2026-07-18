# Second independent stage review

Verdict: `more-work-needed`.

The fresh reviewer accepted every prior remediation and found one remaining
evidence defect: the final shard-advisor capture constructed cosine and sine as
`[1,batch,1,head_dim]`, while production `_decode_rotary_positions` returns
`[1,1,batch,head_dim]`. The malformed capture caused the reported padded RoPE
shape and invalidated the claim that the two rotary operations were unfixable.

The capture now uses the production shape. The corrected report is retained in
`shard_advise/final_graph_corrected/`: 25 operations, 22 final choices, two
spills, and only `nlp_concat_heads_decode` unfixable because its input must be
sharded. The corrected feasible RoPE transpose/block/height layouts were added
to the opt-in `final_shard_advisor` family and tested on the physical TP=4
mesh. Real PCC passes; warmed prefill/decode are 2.892734/0.451892 ms; and the
coherent 96-core two-layer boundary improves from 0.888366 to 0.819200 ms at
PCC 1.0 with zero inter-layer collectives. The family remains rejected because
its decode is 15.45% slower than the final 0.391425 ms default.

Because applying the corrected feasible family changed the implementation
hash, all final-default PCC, performance, batch-1, two-layer, full-context,
broad-suite, strict-fallback, watcher, profiler, tt-perf-report, and health
artifacts were regenerated from the new source hash before final rereview.
