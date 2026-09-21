# Extra SALAD fence audit (isolated prototype)

Only the three `PACK_DONE` post / wait-on-zero / get instructions immediately below `SDPA_STREAMING_NUMERATOR_COMPENSATION` in `salad_correct_fused` are removed. No other synchronization or pack-L1 toggle changes. This is restricted to the sprint's noncausal Q256/K512/D128 full-compensated E/G path, not a claim for all header instantiations.

## Publication is a real pack-completion boundary

`tt_metal/hw/inc/api/dataflow/circular_buffer.h:43` dispatches `push_back` to `llk_push_tiles`. `tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h:47` increments the local received count, then queues `TTI_STALLWAIT(STALL_THCON, PACK)` before `TT_STOREREG` publishes the count. The comment explicitly states that publication occurs only after packing finishes. `llk_io_unpack.h:21` polls that published counter. This is not merely a software-side unconditional push.

## Required data precedes the correction publication

In frozen FAST `compute_streaming.hpp`:

- `salad_correct_fused` first waits on old output/sum and the correction CB, before the removed fence (`1207–1214`).
- The first PV row is produced by the phase-two drain before the remaining-row loop (`2331` onward).
- For each later PV row `r+1`, the loop computes and publishes correction `r` first (`2607–2613`), then produces PV `r+1`, then corrects row `r`. The PV row `r` being read was produced in the preceding iteration, so it precedes the correction publication. The removed fence also waits for the unrelated newly packed PV `r+1`.
- On the final loop iteration, the final row's correction is deliberately computed/published after its PV row (`2664–2671`) before both final SALAD calls.
- First K iteration does not call SALAD. Its output publication remains untouched. Intermediate and last K iterations use the ordering above; final normalization happens after each corrected row and its original waits remain unchanged.
- Q boundaries run the original CB pop/push and initialization paths. Multiple Q jobs and odd K-chunk counts are explicit tests, not inferred from a single resident Q block.

All current-chunk denominator P-to-L1 accumulation was completed during the earlier score-exp phase, before these correction publications. Both old recurrence buffers retain their existing waits. The three removed semaphore operations form a balanced local triplet; no unmatched token is introduced.

This source argument motivates testing; it is not by itself device qualification or a speedup claim. Small distinct two/three-K-chunk tests and multi-Q changing-max tests precede timing. A candidate that changes any output byte or hangs must be rejected or investigated, with recovery controlled by the parent agent.
