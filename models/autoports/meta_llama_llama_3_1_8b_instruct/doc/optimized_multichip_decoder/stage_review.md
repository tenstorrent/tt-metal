# Independent stage review

Final verdict: **clean-pass**.

The final fresh reviewer inspected the original optimized-multichip-decoder
contract, implementation, exact QKV32/O32 default, candidate provenance,
context contract, PCC/performance logs, fallback/watcher gates, and
replay-only profiler artifacts.  It returned no required work, other
concerns, or hard-check gaps.

Review repairs completed before the clean pass:

1. Moved the minimal-all-reduce workspace from per-layer ownership to one
   mesh/stack-shared pool and proved 32-instance sharing plus a two-decoder
   hardware chain.
2. Adapted packed gate/up to the advisor/minimal family, swept projection
   geometry per role, and captured warmed replay-only profiling.
3. Reran packed and material near-winning role families cumulatively under
   QKV32, promoted O block32, and refreshed exact-final performance, PCC,
   watcher, and profiler evidence.
4. Reconciled exact-final PCC and profiler signpost accounting, corrected the
   bounded geometry explanation, and classified every historical candidate
   and profiler artifact so only QKV32/O32 evidence is authoritative.

Controlled anomalies remain documented in `work_log.md`: ACTIVE_ETH watcher
image capacity, the rejected 100-replay profiler marker overflow, and noisy
process-level prefill E2E.  None affects the selected runtime path.

The final rereview made no file or hardware changes.  Its only residual note
is that the historical QKV16/O8 XML internally says `variant=default`, as was
true when captured; chronology, PCC, latency, and its corrected filename make
the historical configuration unambiguous.
