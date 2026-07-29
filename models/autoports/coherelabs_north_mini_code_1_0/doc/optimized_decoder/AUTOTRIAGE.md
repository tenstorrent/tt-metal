# AUTOTRIAGE

## Diagnosis

- The first serving-batch prefill failure was an invalid special-case fused-QKV matmul geometry for logical `[batch=32, seq=1]`; watcher caught an out-of-bounds BRISC runtime argument in `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`. The attempted per-user workaround then exposed a `TilizeWithValPaddingDeviceOperation` producer/consumer stall. The model-side fix is to pack all logical users into one token axis for QKV/O matmuls and restore independent user axes only for RoPE, cache fill, and SDPA.

## Triage Evidence

- `doc/optimized_decoder/triage/user_chunk_hang.txt` identifies operation 1 as a running `TilizeWithValPaddingDeviceOperation` on logical shape `[1, 1, 1, 2048]`, BF16 DRAM interleaved, using 52 cores on device 3.
- BRISC writers are blocked in `cb_wait_front()` in `writer_unary_interleaved_start_id_wh.cpp`; tilize math/pack RISCs remain in hardware startup/synchronization, and NCRISC readers wait for BRISC notification. This directly proves that the output consumer is waiting for tiles the tilize pipeline never produces.
- The broad Ethernet/NoC counter warnings occur while that operation is still running. They are downstream observations from a non-idle stalled device, not evidence of an independent fabric root cause.
- `doc/optimized_decoder/triage/token_pack_hang.txt` showed the same stop site only after the earlier fatal/stalled runs without a hard device reset. After terminating the owner and issuing `tt-smi -r`, the token-packed implementation passed normally and under watcher at PCC `0.9986950605380849`; the watcher log contains no assert, error, hang, NoC sanitizer, or tripped-watcher signature. This establishes the second token-pack capture as stale device state rather than a token-pack defect.

## Source Evidence

- The rejected special fused path treated the physical `batch * padded_seq = 1024` rows as a conventional large-M fused-batch matmul. Its custom core geometry produced runtime arguments that passed without watcher but violated the Blackhole reader kernel's bounds under watcher.
- The rejected per-user path repeatedly converted a logical `[1, 1, 1, 2048]` row-major tensor to tile layout. Triage shows the corresponding 52-core tilize operation's writers waiting for output CB data while its compute pipeline never advances past startup, so the conversion cannot satisfy its producer/consumer contract for this sub-tile shape on the observed runtime.
- `OptimizedDecoder._attention_prefill_user_chunks` now performs one row-major reshape from `[1, batch, seq, hidden]` to `[1, 1, batch * seq, hidden]`, tilizes a multi-row token matrix, executes QKV once with `batch_size=1`, restores `[batch, heads, seq, head_dim]` for per-user attention/cache semantics, then repacks attended rows for one O projection. No per-user one-row tilize remains.
- The packed token count is exactly `batch * seq`; reshapes and permutations preserve that count. Cache fill still slices each user independently, and SDPA retains a distinct batch dimension, so packing does not mix user attention histories.

## Downstream Effects

- The special fused geometry caused the watcher BRISC assert directly.
- In the per-user workaround, blocked BRISC writers, compute startup waits, NCRISC notification waits, host synchronization, and non-idle NoC/ethernet counters all fan out from the stalled tilize operation.
- A hard reset was required after the fatal/stalled experiments. Treating the immediate post-failure token-pack stall as a second source failure would have misattributed stale device state.

## Proposed Fix

- Keep the token-packed non-aligned multi-batch prefill path and remove the invalid special matmul geometry and one-user-at-a-time tilize workaround.
- Validate it from a reset device under normal execution and watcher, then include multi-batch non-aligned correctness coverage in the full optimized suite.
- Preserve the original aligned prefill and traced decode paths; the packing conversions are confined to the non-aligned multi-batch compatibility path.

## Uncertainty

- The precise low-level defect inside the Blackhole 52-core one-row tilize kernel was not patched because that is outside this stage's model-owned scope. The model-side route avoids the proven stop shape.
- The model-side route passed after reset at batch 32/sequence 1 (PCC 0.998695) and batch 2/sequence 33 (PCC 0.998729), both normally and under watcher. Broader lengths retain the same token-count-preserving reshape contract, but exhaustive context-length enumeration is outside this stage.
