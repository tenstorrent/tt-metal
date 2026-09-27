# Native sharded decode RoPE candidate

Status: source-derived, unapplied and unmeasured. Runtime base hash and candidate
patch are in `sharded_rope_provenance.json` and `sharded_rope_candidate.patch`.

The current FusedAttention decoder embeds one cosine/sine row, slices and repeats
it separately across Q4 and K2(sliding)/K1(full) head rows, then casts to FP32.
The native interleaved RoPE path treats heads as sequence positions. Native
`rotary_embedding_hf(..., is_decode_mode=True)` instead broadcasts cosine/sine
row0 across head rows in a height-sharded tensor. Source validation explicitly
accepts input[1,batch,heads,D] and tables[1,batch,1,D]. D256/512 are supported.

The candidate maps Q or K to one L1 core with shard[32,D], maps FP32 cosine/sine
to the same grid, runs sharded decode RoPE, and restores the original input
memory config. It removes table slicing/repetition and their layout conversions;
it adds input/table sharding and output interleaving. FP32 values and the exact
existing compute config are preserved. Prefill is untouched. Setup/measurement
must enable the candidate only for TP4, leaving the optimized TP1 baseline intact.
No persistent tables or new full-context copies are needed. Batches are handled
by the decoder's existing slot loop; local heads fit one32-row tile.

This is not accepted based on fewer operations. Measure real4096/128 paired
PCC, current positions/cache, exact replay and warmed latency for both layer
kinds after the indexed expert failure is repaired. If sharding overhead loses,
keep that full-layer result; if native decode numerics fail, localize rotary
outputs before rejecting the family. The native factory uses sharded output
buffers, so retain sharded output then explicitly interleave rather than asking
it to write directly into a DRAM interleaved destination.
