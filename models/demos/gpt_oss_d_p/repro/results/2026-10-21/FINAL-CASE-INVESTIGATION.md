# Final 64K-input / 1K-chunk migration stall

## Preserved observation

On October 21 at 04:51:16 UTC, the fourth request (third measured request) of the final sweep case stopped completing migrations. The warmup and first two measured requests had completed. Prefill itself completed all 65,472 uncached prompt tokens by 04:51:32.454 UTC, but no response token arrived. The 300-second client bound failed the case; the serial coordinator exited without launching anything else.

The case is preserved under `evidence/sweep/failed/pd-64k-c1k-attempt1`. The four worker/manager services were stopped with verified launch records (no forced kills). Their files are preserved under `runtime/sweep-pd-c1024/before-admission-fix/services`. Both loaded model owners were retained through the repair and retry.

## Diagnosis reproduced in a host regression

`ControlPlane::handleRemoteSlabResponse` releases the temporary send reservation when a receiver grant arrives, then calls `submitMigration` to dispatch local children. The original `submitRequest` applies the fresh-admission FIFO rule even to such a child, which already holds remote receive capacity. A queued newer parent therefore consumes the freed send slab and waits for receive capacity held by the older child. The older child is behind that parent in the admission queue. This creates a circular wait; the reservation reclaimer and command timeouts subsequently report failure.

The first affected pair was migration IDs `281474976736015` and `281474976736016`, followed by IDs ending 017 and 018. Logs show send slots released at 04:51:16.397, newer parents taking them before the local children can run, then reclamation after roughly three seconds. Both send and receive pools have two slots (four slabs split by direction).

## Validation and outcome

A host-only regression, `ControlPlaneDataPlane.PreparedLocalChildRunsBeforeFreshAdmission`, gives the sender one send slab and withholds the second receiver grant while the first migration holds the receiver's sole slab. The first already-prepared child must reach the data plane before the second parent can obtain receive capacity. Observe failure before applying the narrow FIFO exemption, then run the host control-plane suite and a fresh warmup plus three measured requests for only the failed case. Preserve all earlier measurements and identify the retry's source revision separately.

The old-code regression failed with zero data-plane submissions instead of one. With the FIFO guard restricted to fresh admissions (`!plannedSlabs.has_value()`), all **80 host control-plane tests passed**, including the new regression, FIFO ordering and cancellation tests. The repaired native binary SHA-256 is `06ce4084162e7ffa4d3a63c0977b7d681781e72244fb86c5acbc006d0fd78657`. The silicon retry completed at **05:04:41 UTC**: a fresh warmup and all three measured requests passed. Mean TTFT was **24.256486 s**, prefill **24.113500 s**, native decode **257.569 t/s/user**; each measured request completed all **2,304 migrations** with **2,563,153,920 wire bytes**. The all-58-case saved-evidence audit then passed with zero errors.

The repair is commit `5ca79e3076f193c092962d34a3b69731e39aff91`. Only this final P/D case was remeasured with it. Earlier accepted measurements and their source revisions are preserved. This validates the observed failure and retry, not a long-running stability guarantee.

No golden, PCC, hardware reset or model restart is involved. Both loaded model owners were retained.
