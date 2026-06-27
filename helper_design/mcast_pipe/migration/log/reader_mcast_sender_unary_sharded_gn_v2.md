# reader_mcast_sender_unary_sharded_gn_v2.cpp — mcast_pipe migration (Tier 2a)

**Status:** DEFERRED (helper design gap) | **Validation:** n/a (reverted)

## Why deferred — runtime per-rect num_dests (helper DESIGN gap)
This C2 multi-rect groupnorm sender broadcasts the global reduce result + flag to up to 3 NoC
rectangles (mid / first / last group) per exchange. The per-rect-loop call-site residue (one `send()`
per rect, each with its own rect + count) is the sanctioned approach — BUT the recipient COUNT for each
rectangle is a **runtime arg**, not compile-time:

```
const uint32_t num_mcast_cores_mid_group   = get_arg_val<uint32_t>(6);
        num_mcast_cores_first_group        = get_arg_val<uint32_t>(11);   // runtime
        num_mcast_cores_last_group         = get_arg_val<uint32_t>(16);   // runtime
```

Host (`groupnorm_sharded_program_factory.cpp` ~L1144-1170) computes each mcast group's size
(`mcast_group_mid.size()`, `mcast_group_first.size()`, ...) per sender core and per group-split and
pushes it as a **runtime argument** — it varies per core under one compiled binary.

The v7 `SenderPipe<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES, ...>` takes the recipient
count as a **compile-time template parameter** (`NUM_ACTIVE_RECEIVER_CORES`). There is no runtime
`num_dests` ctor/`send()` parameter — the count feeds `mcast_dests`, the degenerate `== 0` branch, and
the loopback `+1`, all keyed off the template. A runtime per-rect count cannot be expressed.

This is the same class as the conv 1D-HS sender deferral already in the ledger ("single-count
SenderPipe cannot represent... num_dests off-by-one"), but more fundamental: the count itself is
runtime. Closing it needs a SenderPipe with a runtime `num_dests` (a new mode/overload) — a helper
DESIGN change, out of scope for this rollout per SUBAGENT_CONVENTIONS (do NOT modify the helper).

## Action
Reverted the in-progress migration (`git restore`); tree left green / on raw primitives. No commit of
kernel changes. Ledger set to `deferred` with this reason.
