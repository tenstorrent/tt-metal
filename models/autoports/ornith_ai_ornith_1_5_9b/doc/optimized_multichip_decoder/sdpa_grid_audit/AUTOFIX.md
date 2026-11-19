# SDPA compute-grid repair

The failed width10/11 configurations kept Q on an8-wide shard prefix while SDPA derived reducer/Q-source coordinates from compute width. Batch8 was the first address mismatch. Parent verified the source chain in `sdpa_decode_program_factory.cpp:342` and the existing wide-grid regression in `test_sdpa_decode.py:326`. See `AUTODEBUG.md` for the complete inspection.

The candidate now passes an ordered `CoreRangeSet` whose first B cores are the actual Q shards, followed by every remaining compute core. K/V-cache update layouts, QKV creation, output layout and public logical batch are unchanged. An ordered list is required; merging or sorting ranges is invalid. The first repair covered B32. The source-review agent independently verified that an ordered Q-prefix plus disjoint remainder generalizes to B2..32, retaining all compute cores. Hardware boundary gates for the generalization remain pending.

## Completed hardware discrimination

Commands are in `sdpa_ordered_jobs.json`; every result has an immutable command/source snapshot and paired log. All rows use TP4,4Kprefix,128 evolving traced steps and final convolution/recurrent/KV checks. The previous width8/chunk256 control was1.271819ms. These are candidate results, not production defaults.

|Grid/chunk|Packed two-layer decode ms|Minimum PCC|
|---|---:|---:|
|sdpa_ordered_g10x10_k128_4k|1.264114|0.998714328|
|sdpa_ordered_g10x10_k256_4k|1.267772|0.998682380|
|sdpa_ordered_g10x10_k512_4k|1.274274|0.998716056|
|sdpa_ordered_g11x10_k128_4k|1.258012|0.998714328|
|sdpa_ordered_g11x10_k256_4k|1.260805|0.998682380|
|sdpa_ordered_g11x10_k512_4k|1.268458|0.998716056|

All six previously failing wider-grid families pass after the mapping repair. No arithmetic gate, cache mask, or allocation tracking exemption was added. The best4Kcandidate is11x10/chunk128, about1.1% faster than width8/chunk256. Before selection, compare short/boundary batches and64Kcontexts with chunks128/256/512 (`sdpa_ordered_contract_jobs.json`). Production remains8x8/chunk256 until those comparisons finish.
