# Advisor contribution at decode batch 32: measured zero

Full-model decoder-layer estimate: **5,672.576 us before and 5,672.576 us after, ±2.336 us**. The frozen incumbent remains shipped, so `$shard-advise` contributes a measured **0.000 us/model** at this stage.

The control is one dense layer measured by the required template in a fresh process: median 0.373080 ms from five blocks of 50 trace replays after 10 warmups; repeat spread 0.146 us. The model config has 16 dense layers and no other layer kind. Capture batch, requested batch, and measured batch are all 32. Capture occurred after the incumbent freeze at advisor pin `618cd4e75d` with the executed BFP8 weight policy.

Reconciliation accounts for 354.536 us (100%): 258.260 us agrees with shipped, 40.420 us is in differing L1 chains, 25.497 us is advisor-DRAM-resident, 26.217 us is untraced cache update work, and 4.142 us is conversion boundary time. The advisor drops 2.822 us/layer of shipped conversions; 1.320 us/layer is a real boundary the advisor agrees with and was reported but not screened. Layer handoff begins and ends in L1, so no out-of-scope handoff cost was booked.

Screening followed reconciliation ranking. The top 1.491 us SDPA→concat boundary candidate was attempted despite the advisor's own unfixable concat result: paged GQA SDPA explicitly rejected sharded output (`Sharded output not supported for GQA`), while `nlp_concat_heads_decode` requires that unavailable sharded output. The 0.665 us residual-chain candidate was extended after the first neighbor constraint, then tested on an exactly-dividing 64-core grid above the advisor's 22/55-core choices. It ran at 0.420341 ms versus 0.373080 ms, 12.7% slower. The 32-core row-wise isolate failed because sharded RMSNorm forbids non-rectangular grids.

The DRAM-resident disagreements were also screened. Moving rotary K to DRAM failed because `paged_update_cache` requires a sharded input. DRAM concat output measured 0.373237 ms with repeats `[0.379504, 0.373371, 0.373237, 0.373003, 0.373117]`; these overlap the incumbent repeats and the median is slower. The remaining chains have zero advisor-attributable boundary value and were marked below threshold.

No placement change shipped, so the existing real-weight incumbent oracle remains authoritative. Every losing knob remains default-off in `MD_LLAMA32_ADVISOR_CHAIN` and is listed in `final.json`.
