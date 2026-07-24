# SDPA KV-reuse multicast — intermittent handshake hang (Blackhole)

The generated flash-attention SDPA op has an **opt-in** KV-reuse multicast path
(`Mcast1D` PerRow, via `ttnn/cpp/ttnn/kernel_lib/mcast_pipe`). On Blackhole it
**intermittently hangs**. Two distinct bugs were found — this branch isolates the
second one.

## Bug 1 — DETERMINISTIC fan-out overcount (already fixed on this branch)
`McastRect::area()` counted Blackhole's non-worker NoC columns 8,9 inside the
mcast bounding box, so `num_dests` was too high and the non-posted multicast
waited forever for acks from cores that don't exist. **Fixed at source** by
@sjovic's `area()` correction (`llk_helper_library` `4268da8b519`), which is
included in the **base commit** here. This bug hangs *every* time an ≥8-wide
row is used, so it is not the intermittent one.

## Bug 2 — INTERMITTENT consumer-ready lap-race (this branch's subject)
`SenderPipe::send` (PRE_HANDSHAKE) does:
```cpp
consumer_ready_.wait(ack_count_);   // wait for all receivers' "ready"
consumer_ready_.set(0);             // absolute reset — NOT atomic with the wait
```
Receivers increment `consumer_ready` **monotonically** (`up()`) and never reset it.
A receiver that laps the sender — posting its **next** send's ack in the window
between the sender's `wait()` returning and its `set(0)` — has that ack **clobbered**
by the `set(0)`, so the sender's next `wait(ack_count_)` is permanently one short
→ **deadlock**. Timing-dependent ⇒ rare / non-deterministic.

## Branches
- **outer** (`tt-metal`): `dnijemcevic/sdpa-mcast-hang-repro`
- **inner** (`tt_ops_code_gen`, eval + golden tests): `dnijemcevic/sdpa-mcast-hang-repro`
  (commit `a5571e1`). Layer into `tt_metal/third_party/tt_ops_code_gen` and create
  the `.claude` → submodule symlink for the golden suite. Not required for the
  self-contained repro below.

## Commits on the outer branch
- **base** — `area()` fixed, lap-race PRESENT → hangs intermittently.
- **tip (proposed fix)** — reset-free `consumer_ready` (`wait_min` on a running
  cumulative count; adds `consumer_acks_seen_`). `git diff base..tip` = the whole fix.

## Repro (self-contained — needs only this outer clone)
The mcast path is opt-in (`TTNN_SDPA_KV_MCAST=1`) and eligible only when
`b*H_q == grid_rows` with no mask. The 9k perf shape `(1,10,9472,128)` is eligible
on an 11×10 Blackhole (`b*H_q = 10 = grid_rows`). This clone's `build_Release`
already has tracy/profiler on.

```bash
# 1. reproduce the hang (may take many iterations — it is rare)
git checkout <base>
export TTNN_SDPA_KV_MCAST=1
scripts/tt-probe.sh --dev scaled_dot_product_attention < repro_mcast_hang.py
#    a hang → dispatch timeout → triage report at generated/tt-triage/triage.txt
#    (look for cb_wait_front / noc_semaphore_wait in SenderPipe::send / ReceiverPipe::receive)

# 2. verify the fix
git checkout <tip>          # or apply only the mcast_pipe.inl hunk
scripts/tt-probe.sh --dev scaled_dot_product_attention < repro_mcast_hang.py
#    runs clean, PCC ~0.9996
```

`STRESS_ITERS` (env, default 40) controls the loop count.

## Key files
- `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.inl` — `SenderPipe::send` (the handshake; the fix)
- `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` — `McastRect::area()` (bug-1 fix) + `consumer_acks_seen_`
- `.../scaled_dot_product_attention/kernels/scaled_dot_product_attention_reader.cpp` — `USE_MCAST` sender/receiver lockstep loop
