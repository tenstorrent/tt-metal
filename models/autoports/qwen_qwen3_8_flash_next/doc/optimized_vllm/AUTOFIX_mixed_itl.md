# AutoFix: mixed-traffic client ITL

Status: diagnosis independently source-verified; experiments in progress, no passing claim.

The contract is worst client-observed ITL <=1000 ms for a background decoder while
4096- and 32768-token prompts prefill, C262144, resident2, real TT-plugin serving.
The complete alignment8/interleave1/1 runs failed: budget33 maxima2557.9/2599.3 ms;
budget17 maxima1587.8/1620.5 ms. Both completed without request errors. The later budget9 control and interleave1/2 experiment are recorded chronologically below.

[AUTODEBUG_mixed_itl.md](AUTODEBUG_mixed_itl.md) identifies a final-prefill publication
barrier. Independent source review confirmed: vLLM core.py697-701 submits the next
worker call before popping/publishing an older result at740-757; the uniprocess
executor108-117 executes inline despite non_block; plugin model_runner3245-3255
drains previous decode before prefill; generator987-995 returns intermediate
prefills immediately but final prefill calls blocking _read_tokens640-643.
All four completed worst gaps match the penultimate-to-final prefill_result
interval within0.8 ms. This is strong timing/source evidence; raw client SSE
adjacency was not retained, so exact pair attribution remains an inference.

Priority experiments:
1. Finish unchanged budget9 control, preserving both foreground windows and cleanup.
2. Existing interleave1/2 at budget17, alignment8, async on: P,D1,D2,P permits
   publication of D1 after D2 submission before final P. Measure official client
   worst ITL, foreground TTFT and background coverage. No plugin source/interface change.
3. Interleave1/3 if needed; D2-D3 may exercise steady overlap. This changes offered
   decode service share and may lengthen TTFT; report the tradeoff.
4. Smaller physical chunks (budget5/alignment4 or3/alignment2) only if necessary;
   preserve full context and logical tails. Do not substitute budget8 for9 and
   claim smaller work: both select physical8 at alignment8.

A kept config requires a consistent three-budget sweep, largest passing value,
then matched primary/CI before/after at that same config, final mixed-shape trace
retention/allocation safety, stale inputs and async history. Never remove final
prefill's required token read, return stale data, relax the ITL bound, suppress
valid tails, or add a deferred-prefill plugin interface without authorization.

No runtime source changes have been made for this diagnosis. No profiling,
reset, or unrelated process termination was used. The current server continues
unchanged; live process ownership remains with the serial sweep harness.

08:08 UTC: budget9 completed,4K/32K worst ITL1064.0/1098.8 ms; no errors. All three1/1 arms fail. Each drained0/0, no owned processes remained, and fuser found no handles. Complete manifest/raw results archived under chunk_interleave1/. Changed only existing decode_interleave_decode_steps from1 to2 in run_server.sh and benchmark.sh; aligned sweep/comparison metadata. Next consistent sweep order17,33,9, same alignment8/context262144/resident2/async/precision. Model, native runtime, plugin and vLLM source unchanged.

08:50 UTC: interleave1/2 budget17 completes with4K TTFT243952.3ms/worst ITL1184.1ms and32K TTFT1987248.8ms/worst877.5ms. The32K bound passes, but4K fails, so17 is not eligible. All requests and owned cleanup pass. Continue unchanged33/9 arms. Source/log review rules out premature foreground admission: background is128 tokens, with27 completed decode validations and about2.81s steady decode before first4K submission. Extending warmup is not justified as a repair. Client raw timestamps remain unavailable; a localized event check is in progress.

08:54 UTC independent completed-log/source analysis: interleave1/2 removes the
previous two-prefill publication gap. Both foreground windows have zero
captures, only physical16-token prefill replays, no host sampling and no OOV.
The32K final D2-submit to final-prefill-return interval877.506ms matches the
client877.5ms maximum. For4K the largest validation gap854.638ms and reconstructed
publication opportunity852.369ms do not explain the client1184.1ms maximum.
These are source-derived opportunities, not client receipt timestamps. Content
filtering, delivery/coalescing, client scheduling or a crossing-window gap remain
unproven possibilities. The4K run also has16 page-only refreshes, while32K has
none due to boundary phase; this is coverage, not causal proof. Keep17 rejected.
If9 fails, retain raw background SSE receipt times/content-empty flags and the
maximizing pair in a targeted repeat with the official metric unchanged before
considering interleave1/3. If9 passes both windows, selecting9 is justified.

09:22 UTC: interleave1/2 budget33 fails both windows:4K1339.8ms and32K1364.7ms worst ITL; TTFT183815.3/1489260.9ms. No errors and clean owned shutdown. It remains ineligible. Budget9 is the final unchanged arm.

10:20 UTC: budget9 passes both client windows:4K592.1ms and32K615.7ms worst
ITL, with foreground TTFT355182.9/2918032.7ms. No request errors. The consistent
17/33/9 sweep selects9, the largest tested passing value. All six windows have
an active decoder at every foreground prefill, zero captures/retirements,
zero host routes and OOV. Model/adapter/common-sampler/server-script hashes
remain unchanged. Process/device handles are gone, but complete orderly trace
teardown is separately failed under timeout0; see chunk_v4_cleanup.json and the
proposed shared profiler-call removal. This closes the mixed-traffic budget
repair only, not final stage acceptance. No raw-timestamp/interleave1/3 follow-up
is needed for the selected9 result;17's failure remains rejected evidence.
