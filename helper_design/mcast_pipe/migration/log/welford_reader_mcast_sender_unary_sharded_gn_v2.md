# welford_reader_mcast_sender_unary_sharded_gn_v2.cpp — mcast_pipe migration (Tier 2a)

**Historical v9 checkpoint:** MIGRATED | **Validation:** Welford inventory 108 passed,
2 expected skips; fixed/default-routing nodes 19 passed, 6 expected skips.

The raw acknowledgement gate executes before remote L1 source reads. The
persistent no-handshake middle `SenderPipe` owns the linked send, while
optional first/last pipes send the same payload. Mapped inventories validate
their host-generated receiver partition. The deferred v7 discussion below is
historical.

The legacy sender's v9 A/B isolation produced PCC `0.983672` when the shared
gate was moved after remote L1 reads and recovered when baseline ordering was
restored. The Welford sender preserves that same proven pre-read gate.

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

Closing it needs a SenderPipe runtime-`num_dests` mode — a helper DESIGN change, which was out of
scope for that historical frozen-helper rollout.

## Action
No kernel changes committed. Ledger set to `deferred` with this reason. (The welford gn_v2 RECEIVER
is already migrated@v7 — only the multi-rect SENDER side hits the count blocker.)
