# Shared MLP owners: measured native result

The optional shared-owner path removes duplicate packed gate/up weights. It
remains disabled by default: its measured memory saving comes with a 3.67%
decode throughput regression. This is a native result, not a serving capacity
or prefix-retention result.

Both arms used source `91d473c322c911f5ac5df6f42ae97c586a5a09e0`, all64 layers,
B32, native262144 context/table geometry, grouped/masked/flat enabled, BF4/LoFi
MLP weights and K blocking2. The separate gate/up owners retain BF16 rounding
before SiLU. The external native pool was bounded to8192tokens/row; it does not
represent32 full native-context histories. The retained local backend differs
from the pinned CI image. Raw evidence is outside Git under
`/home/mvasiljev/build/qwen38-swe/memory-prefix`.

| Actual bytes per device | Packed control | Shared owners |
|---|---:|---:|
| After construction |9,980,379,648|8,376,058,368|
| After preparation |10,120,532,480|8,528,105,984|
| Observed allocation-event peak |11,652,462,080|10,048,108,032|
| DRAM freed by releasing dynamic traces |80,609,280|92,536,832|

Physical-owner census and allocation events reconcile exactly at checkpoints.
Removed packed owners account for1,604,321,280bytes/device; the measured
prepared saving is1,592,426,496bytes/device (1.483GiB), after11,894,784bytes of
additional runtime/trace allocation. The event peak covers construction,
preparation and the tested requests; it is not a whole-workload serving peak.
Measured trace release is separate from the serving128MiB configured arena.

Tracked full64 cases257/513 exercise ragged rows and the512+1 continuation.
All128 logical state fields across all4 devices, logits and generated token
arrays match exactly. Recurrence/conv checks preserve FP32/BF16 respectively;
BF8 K/V are compared after logical conversion, not as raw packed buffers.
Every request allocation reconciles, teardown completes, and both devices
close cleanly. Earlier reduced checks additionally cover1/33/255/256/511/512.

The separate clean run used4096 input tokens,128 output tokens, three repeats,
tracking/profiler/watcher disabled, no steady-state host refresh or recapture.
Median TTFT was15,594.344 versus15,595.041ms. Median per-user decode throughput
was12.76696 versus12.29814tokens/s. All32 output trajectories match exactly.
Three native-chat qualitative prompts used the same512-token cap and normal
EOS policy as pinned HF controls; both TT arms produced identical complete
outputs. Manual review found the requested English sky explanation, correct
apple-count JSON, and ordered-deduplication Python. This is a small qualitative
check, not a scored accuracy qualification.

Canonical artifacts: `full64-{control,shared,comparison}.json`,
`full64-clean-{control,shared,comparison}.json`, and `FULL64_PAIR_HANDOFF.json`.
The [compact physical-owner ledger](SHARED_MLP_FULL64.json) contains both
checkpoint inventories, explicit unnamed residuals and raw artifact hashes.
The earlier `SHARED_MLP_REDUCED.json` records the initial reduced checkpoint;
its full64-pending status is historical.

Next experiments preserve K2, precision and activation rounding while testing
reader/core layouts. Two readers require local N4352 padding to4608; forecast
extra separate-owner bytes are94,371,840/device at64layers, giving a forecast
constructor saving1,509,949,440bytes. Cores16 versus80 changes activation
sharding. All three R2/C80, R1/C16 and R2/C16 variants now pass reduced257/513
exact state/logit/token and physical allocation checks at source809ffc6b3d1.
Clean reduced4096/128/three-repeat timing still loses2.26% for R1/C16 and2.54%
for R2/C16; outputs remain exact. These variants were not expanded to full64.
A separate reduced R1/C80 watcher10/NOINLINE run passed with all features
enabled and no profiler. See [compact geometry evidence](SHARED_MLP_GEOMETRY.json).
Independent review rederived the measured full64 evidence and retained a
more-work-needed/default-off verdict. Independent final review, actual CI B32 ownership/retention and
matched multi-turn cached-prompt measurements remain open. No current B32
serving hit-rate or higher-concurrency acceptance follows from these results.
