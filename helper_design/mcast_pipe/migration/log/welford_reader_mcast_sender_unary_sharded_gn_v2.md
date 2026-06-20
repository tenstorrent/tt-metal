# welford_reader_mcast_sender_unary_sharded_gn_v2.cpp — mcast_pipe migration (Tier 2a)

**Status:** DEFERRED (helper design gap) | **Validation:** n/a (not attempted past static analysis)

## Why deferred — runtime per-rect num_dests (helper DESIGN gap)
Identical blocker to its non-welford twin `reader_mcast_sender_unary_sharded_gn_v2.cpp` (see that log).
This welford multi-rect sender also broadcasts to up to 3 NoC rectangles per group, and each
rectangle's recipient count is a **runtime arg**:

```
const uint32_t num_mcast_cores_mid_group   = get_arg_val<uint32_t>(6);
        num_mcast_cores_first_group        = get_arg_val<uint32_t>(11);  // runtime
        num_mcast_cores_last_group         = get_arg_val<uint32_t>(16);  // runtime
```

The v7 `SenderPipe` requires `NUM_ACTIVE_RECEIVER_CORES` as a **compile-time template parameter**;
a runtime per-rect count cannot be expressed. Same host source
(`groupnorm_sharded_program_factory.cpp`) feeds both senders.

Closing it needs a SenderPipe runtime-`num_dests` mode — a helper DESIGN change, out of scope per
SUBAGENT_CONVENTIONS (do NOT modify the helper).

## Action
No kernel changes committed. Ledger set to `deferred` with this reason. (The welford gn_v2 RECEIVER
is already migrated@v7 — only the multi-rect SENDER side hits the count blocker.)
