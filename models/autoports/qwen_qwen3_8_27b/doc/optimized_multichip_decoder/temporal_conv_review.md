# Batched temporal-window convolution review

Inspection-only continuation, 2026-10-24. No hardware execution or implementation edits.

The proposed decode adaptation is legal under the current operation contract and mathematically equivalent for retained outputs. It deserves a whole-layer candidate measurement; the operation's B1 input restriction does not reject this representation.

For each user b, construct four consecutive raw BF16 rows `[history[b,0], history[b,1], history[b,2], projected[b,0,:2560]]`. Flatten the user/window axes to `[1,4B,2560]`, pad the sequence internally to `ceil(4B/32)*32`, and retain only convolution output rows `4b+3`. The four-tap reader uses rows `r-3,r-2,r-1,r` in this order, so retained row `4b+3` consumes exactly that user's four rows. The discarded first three outputs of each window may contain preceding-user data; they have no path into the retained output because this op is feed-forward convolution, not recurrence. Sequence tile boundaries are also safe because 32 is divisible by4, and each retained row's complete four-row window lies within a tile.

Source evidence:

- `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp`: rank3/B1, BF16 interleaved input, positive tile-aligned sequence, `[1,3,Q+K+V]` row-major history, legal tiled taps, and channel-chunk divisibility are the relevant checks. `[1,T,2560]`, widths512/512/1536 and channel_chunk_size128 satisfy them.
- `.../device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp`: each output row uses four shifted input rows; history or predecessor supplies only rows before a sequence segment. The compute kernel performs four tap accumulations then pointwise SiLU; discarded outputs do not feed later outputs.
- `ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp`: with `sequence_parallel_axis=0` on mesh `[1,4]`, partitions=1 and no local split occurs. Keep `actual_start=self.conv_start` containing0. Do not use the tensor-parallel axis1 for this artificial local sequence.
- Ordinary `qkv_causal_conv1d_silu` leaves history unchanged. Supply a valid `[1,3,2560]` history, such as `state.conv[:1]`, and alias it as predecessor_carry. Its values do not affect the retained rows. The exchange/carry-returning variant is unnecessary and its synthetic final carry would not represent all users.
- `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp:203`: strided slicing requires ROW_MAJOR input.

Implementation constraints for the candidate:

1. Build and flatten windows in ROW_MAJOR before tilizing. A tiled `[B,4,C]` has28 padded rows per user; reshaping must preserve logical rows, not concatenate that padding.
2. Untilize Q/K/V outputs before extracting rows `3:4B:4`, or reshape row-major output to `[B,4,C]` and select `[:,3:4,:]`. Retile only when the subsequent consumers require it. Direct stride4 tiled slicing is not legal in the primitive.
3. Keep original `projected` for Z, decay and beta; only its first2560 raw channels participate in the packed convolution. Preserve the existing per-user history update, or an exactly equivalent update from each raw window's final three rows. Do not use convolved/SiLU outputs as history.
4. Initially scope the candidate to decode `length==1` and B>1 so prefill and B1 retain their existing paths. Precompile each physical batch/window shape before capture and include the new packing/extraction operations in tracking validation.

Materiality: the current batch loop performs one padded32-row convolution per user. With channel_chunk_size128, there are20 channel blocks. B32 therefore executes640 work items across32 launches; packing uses128 rows,80 work items and one launch. B8 reduces160 to20 work items, B16 reduces320 to40. The baseline B1 profile records roughly12us for one convolution on20 cores, so a material reduction is plausible. This is not a measured latency prediction; all conversion, extraction and state-update costs must remain inside the candidate timing.

A host-only indexing check used actual2560 channels, distinct integer histories/current rows/taps, batches1/3/8/9/16/31/32 and three advancing history updates. All768000 retained-channel comparisons were exact, including padded tails. This confirms the indexing algebra only, not device numerical or performance acceptance.

Focused follow-up: compare real-weight per-user-loop control and packed candidate at B8/B16/B32, plus an awkward batch such as3 or9; inspect every user's Q/K/V or final output, conv/recurrent state, and multiple advancing decode steps. Check warmed traced whole-layer timing and representative allocation-tracked replay. B1 and prefill remain regression controls if their source branches are unchanged. Final stage verdict remains pending acceptance and profile evidence.
