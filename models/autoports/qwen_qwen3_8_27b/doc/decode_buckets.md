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
page IDs. Linear-attention state is gathered before decode and scattered back
afterwards, preserving inactive slots. Logits return to scheduler order before
the existing sampler, so request seeds and asynchronous token feedback do not
change row ownership. Hardware tile padding still applies.

State gather/scatter runs inside the trace on every token. This has overhead
and must be measured against fixed B1/B8/B16 before promotion. Avoiding those
copies would require persistent bucket state with explicit synchronization at
prefill, slot remapping, capture warmup, and occupancy transitions.

Host tests cover every occupancy, non-contiguous/reversed slot mappings,
inactive-state preservation, and repeated bucket transitions. They do not
establish TTNN device correctness, trace replay correctness, or performance.
After a local device reset, the reused CI image passed both two-layer and
full-model traced comparisons at occupancies 1, 5, 8, 9, 16, then 1 again.
Four greedy decode tokens per request matched the fixed-capacity control.
The short full-model test measured roughly 76 versus 36 ms/token at occupancy
1, 76 versus 52 ms at occupancy 8, and 77 versus 83 ms at occupancy 9.
These are local synchronous adapter measurements, not serving benchmarks;
prefill is reset between comparisons and continuous request turnover is not
covered. All 82 model host tests passed against that image.

Remaining device qualification: longer full-model token/logit comparisons,
queued asynchronous replay, active-slot turnover and prefill
interleaving, then C1/C8/C16 serving latency and the agentic evaluation suite.
Do not promote this flag to release defaults based on host tests alone.
