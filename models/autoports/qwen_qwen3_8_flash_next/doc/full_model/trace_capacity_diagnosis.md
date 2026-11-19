# Retained prefill trace capacity failure

`batch32_hf_v1.json` and its log preserve the failed all-48-layer run with
allocation tracking and program-cache checks enabled. Preparation completed in
350.905 seconds. The first public mixed prefill (32 prompts of lengths203..234)
failed while ending a prefill trace, before decode accuracy could be measured.
The device closed cleanly; no reset or precision change followed.

Prepared free DRAM was521,230,528 bytes per bank, or4,169,844,224 bytes per rank
across eight banks. At failure, allocated DRAM was4,251,289,792 bytes per bank,
free space21,051,584 and largest contiguous space17,688,000. The next trace
required143,278,080 bytes per rank (17,915,904 per bank). Allocated growth since
preparation was4,001,431,552 bytes per rank.

The original generator keys prefill traces by `(count, slot)` and retains them
until close. This call requires32 `(32, slot)` traces and31 distinct tail/slot
traces. Retention, rather than weights or the logical context alone, exhausts
capacity. A trace-capacity failure from this design is not evidence for reducing
the advertised262144-token context or batch capability.

The selected tracing skill requires lifetime retention and prohibits ordinary
request-time eviction or recapture. Consequently, the initially considered LRU
approach was rejected before implementation. The repair under investigation uses
six physical chunk sizes (32,16,8,4,2,1) and a common prefill workspace across
slots, with device-to-device state transfers at prefill request boundaries.
Batch1 can use its own cache directly. It must preserve all logits, absolute
positions, recurrent/PLE state, shared page ownership, and split-sampling input
lifetimes. No eager or host-compute fallback is proposed.

This is a diagnosis, not passing evidence. Reduced tracked mixed-slot and
nonaligned continuation tests, before/after timing, full-model accuracy, and the
B32 HF/lifecycle test remain required. Decode and sampling signatures also need
their retained memory accounted for; solving prefill alone does not prove that
every subsequent schedule fits.
