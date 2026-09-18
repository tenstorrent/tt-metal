# Real prefill cancellation and native restart

**PASS — 18 September 2026.** One real prefill request was cancelled after its
first 1K chunk, then a distinct 32-token request completed native KV transfer
using the retained owner/cache allocations. Both endpoints used full 32-chip
Galaxies, SP=4/TP=8, a 2K two-slot BFP8 cache, and all 32 model layers.

| Check | Accepted result |
|---|---|
| Real scenario work | 2 full32 H2D/model calls and 64 real layer acknowledgements; no synthetic acknowledgements |
| Source warmup | 2 additional full32 calls, outside the scenario acknowledgement counts |
| Delayed destination | All 16,384 intended first-chunk pages remained unchanged before destination expectation was armed |
| Cancel terminal | Source OK (0), passive INTERNAL (6), both reporting 0 completed tokens |
| Restart transfer | 512 packed pages / 2,228,224 bytes matched exactly from source slot 0 to destination slot 1 |
| Restart completion | Both endpoints OK (0), 32 completed tokens |
| Ownership and shutdown | Both epoch stop barriers passed; retained allocations released after native I/O stopped; all 32 chips closed cleanly on each endpoint |
| Integrity | All 391 source pins unchanged; controller, both dispatches and final verifier exited 0 |

## What the sequence establishes

Epoch A registers the 2K request before the real H2D/model call. The source
prepares only [0,1024), completes that chunk, synchronizes its writes, saves the
selected source bytes, and publishes 32 real layer acknowledgements. While the
passive holds its expectation, the source audit remains register-only and all
16,384 destination pages are unchanged. The named handshake then arms the
destination. Cancellation follows the observed peer-ready and 32 first-chunk
layer commands, before a second chunk or seal.

Both epoch-A native managers stop before destination sentinels are rewritten or
epoch-B managers start. The owners, caches and source runtime remain allocated.
The restart uses a distinct 32-token prompt and fresh manager/client identities.
Its 512-page source oracle is captured after that request's synchronized writes
and before its first acknowledgement. Every selected configuration/layer group
changes from its pre-request snapshot; this detects a wholly skipped group,
without claiming every value was recomputed. The destination then matches those
current source bytes exactly across all 16 K/V configurations and 32 layers.

Both epoch-B managers stop before either owner releases the cache. Selected
source bytes remain unchanged through native shutdown, destination restart
bytes remain exact, and both cleanup reports are empty. The final 32-chip closes
were 12:34:52.480 UTC (passive) and 12:35:49.543 UTC (source). The handback recorded
empty job steps and available physical locks.

## Scope and provenance

This does not prove bytes were in flight when cancellation arrived, device-kernel
interruption, or abrupt physical peer-loss recovery. It adds no numerical,
PCC/golden, throughput or latency claim, and runs no decoder or SC4 work. Coverage
is limited to the 2K allocation and selected ranges above.

The accepted device result belongs to frozen cancellation harness003. The
portable fixture is a separate host-validated configuration/path successor; its
publication is not another device execution. Earlier failed attempts remain
preserved separately.

The [machine-readable evidence summary](migration-prefill-cancel-restart.json)
records the exact source, bridge/binary, plan and receipt hashes. Root acceptance:
52718f795a2229bfe42c7c93fb51401b69356db2ae52dad0ed58e756df417ed6.
The original raw evidence stays in the task evidence store; this public summary
contains no private deployment paths, raw logs, weights or binaries.
