# Decode buckets in one serving process

Experimental opt-in: set `QWEN_DECODE_BUCKETS=1` and start the server with
`max_num_seqs=16`. Supported serving capacities are 1, 8, and 16; capacity
above 16 is rejected before model loading when the option is enabled.

| Active requests | Logical decode batch |
| --- | --- |
| 1 | 1 |
| 2–8 | 8 |
| 9–16 | 16 |

The server stays running across occupancy changes. The generator's existing
active-slot handling invalidates and captures traces when the active set
changes. This first implementation retains one current trace, not a cache of
all three traces. The option applies to decode; prefill retains its existing
occupancy-dependent implementation.

Tokens, positions, and page-table rows are packed on device. Unused bucket
positions are -1. Full-attention KV pages remain shared with their original
page IDs. Linear-attention state is gathered once before trace capture and
remains resident at the selected bucket shape during steady decode. It is
published before prefill or reset, then refreshed in place before decode;
the same captured graph remains reusable across same-shaped requests.
The owned slot-zero B1 prefill trace and fresh-request reset operate directly
on resident B1 state, avoiding a round trip through the 16-slot state buffer.
Slot remapping, cache rebinding, or another active set releases traces and
publishes state before discarding the resident buffers, preserving inactive slots. Capture
warmup backs up and restores the resident state, not the stale scheduler copy.
Logits return to scheduler order before
the existing sampler, so request seeds and asynchronous token feedback do not
change row ownership. Hardware tile padding still applies.

The direct model API retains a per-token gather/scatter fallback if the caller
does not prepare a resident bucket. The serving generator prepares one outside
capture. These state transfers must not appear in the steady-state trace.

Host tests cover every occupancy, non-contiguous/reversed slot mappings,
inactive-state preservation, repeated bucket transitions, resident warmup
backups, prefill synchronization, and discarding replaced request state.
After a local device reset, the reused CI image passed both two-layer and
full-model traced comparisons at occupancies 1, 5, 8, 9, 16, then 1 again.
Four greedy decode tokens per request matched the fixed-capacity control.
The initial per-token-copy implementation measured roughly 76 versus 36 ms/token at occupancy
1, 76 versus 52 ms at occupancy 8, and 77 versus 83 ms at occupancy 9.
These are local synchronous adapter measurements, not serving benchmarks;
prefill is reset between comparisons and continuous request turnover is not
covered by that initial test.

The resident-state implementation subsequently matched the per-token-copy
bucket implementation for all 128 generated tokens at occupancies
1, 5, 8, 9, 16, then 1. Native serving also completed 1→5→16→1 request
turnover with unchanged B1 output after returning to B1. There are 90 passing
model host tests. Cross-shape B1-versus-B16 generated text is not required to
be identical: longer greedy streams can diverge due to batch-dependent rounding;
the state optimization is checked against the same-shaped control.

Matched native `vllm bench serve` controls use the reused CI image, full model,
context 262144, shared KV pool 1050592, greedy 128/128/c1, one warmup,
eight measured requests, and identical settings except serving capacity.
Fixed capacity 1 measured 24.68798 ms mean TPOT (40.50555 tokens/s/user)
and 70.31799 ms mean TTFT. Final resident B1 inside capacity 16 measured
24.72533 ms (40.44436 tokens/s/user) and 68.71432 ms mean TTFT.
This restores decode throughput to within 0.16% of the matched fixed-B1
control, without a warmed TTFT regression. Aggregate output throughput includes
TTFT and is a different metric. An earlier run of the same final code had one
decode-latency outlier and averaged 38.83566 tokens/s/user; the detailed repeat
above had 24.75 ms P99 TPOT. This is local evidence, not a CI performance verdict.
The native protocol checks are in `tests/benchmark_bucket_server.py`;
the same-shape device comparison uses `tests/run_decode_buckets.py
--layers all --steps 128 --control-buckets`.

Remaining release qualification: longer full-model logit/accuracy comparisons,
broader queued asynchronous replay and prefill interleaving, full
C1/C8/C16 context sweeps, and the agentic evaluation suite.
Do not promote this flag to release defaults based on host tests alone.
