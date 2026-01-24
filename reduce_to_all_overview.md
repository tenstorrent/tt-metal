## Overview

The `reduce_to_all` operation implements **SDPA (Scaled Dot-Product Attention) tree reduction** across 4 devices in a ring topology. Each device starts with partial SDPA results (m, s, l tensors) and after the operation, all devices have the fully reduced global result.

### Ring Topology

```
    ┌─────────────────────────────────────────┐
    │                                         │
    ▼                                         │
  ┌────┐  forward   ┌────┐  forward   ┌────┐  forward   ┌────┐
  │ D0 │ ─────────► │ D1 │ ─────────► │ D2 │ ─────────► │ D3 │
  └────┘            └────┘            └────┘            └────┘
    ▲                                         │
    │                                         │
    └──────────────── backward ───────────────┘
```

### SDPA Reduction Math

Each device has partial results:
- `m` = max of attention scores
- `s` = weighted sum (softmax numerator)
- `l` = sum of exponentials (softmax denominator)

To merge two partial results (subscripts 1 and 2):
```
m_new = max(m1, m2)
P1 = exp(m1 - m_new)    // rescaling factor for result 1
P2 = exp(m2 - m_new)    // rescaling factor for result 2
s_new = s1 * P1 + s2 * P2
l_new = l1 * P1 + l2 * P2
```

---

## Algorithm: 2-Round Neighbor Exchange

The reduction happens in 2 rounds, each round merging pairs of devices:

### Round 1: Adjacent Pairs Exchange (Forward Mux)

```
    D0 ◄────────────────► D1        D2 ◄────────────────► D3
       (forward mux)                   (forward mux)

    After Round 1:
    D0 has: reduce(D0, D1)          D2 has: reduce(D2, D3)
    D1 has: reduce(D0, D1)          D3 has: reduce(D2, D3)
```

### Round 2: Cross Pairs Exchange (Backward Mux)

```
    D0 ◄──────────────────────────────────────────────────► D3
                        (backward mux)

    D1 ◄──────────────────────────────────────────────────► D2
                        (backward mux)

    After Round 2:
    ALL devices have: reduce(D0, D1, D2, D3)
```

---

### Data Flow Diagram (Zero-Copy)


ROUND 1:
═══════
```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     TENSIX CORE (Shard Core)                     │
  │                                                                  │
  │          ┌─────────────────────────────────────────────────┐     │
  │          │                 SHARED L1                       │     │
  │          │                                                 │     │
  │          │  ┌──────────────────────────────────────────┐   │     │
  │          │  │  Input Tensor (sharded) @ 0x1000         │   │     │
  │          │  │  ┌──────┐┌──────┐┌──────┐                │   │     │
  │          │  │  │ L    ││ S    ││ M    │                │   │     │
  │          │  └──┴──┬───┴┴──┬───┴┴──┬───┴────────────────┘   │     │
  │          │        │ CB    │ CB    │ CB ALIASING            │     │
  │          │        ▼ alias ▼ alias ▼ alias                  │     │
  │          │  cb_local_l  cb_local_s  cb_local_m             │     │
  │          │                                                 │     │
  │          │  ┌──────────────────────────────────────────┐   │     │
  │          │  │  MeshBuffer @ 0x2000 (SAME on all devs)  │   │     │
  │          │  │  ┌──────┐┌──────┐┌──────┐                │   │     │
  │          │  │  │ L    ││ S    ││ M    │ ◄──────────────┼───┼──── neighbor
  │          │  └──┴──┬───┴┴──┬───┴┴──┬───┴────────────────┘   │     writes here
  │          │        ▼       ▼       ▼ CB ALIASING            │     │
  │          │  cb_nbr_l   cb_nbr_s   cb_nbr_m                 │     │
  │          └─────────────────────────────────────────────────┘     │
  │                                                                  │
  │  ┌──────────────────────────┐    ┌──────────────────────────┐    │
  │  │       READER (NCRISC)    │    │       WRITER (BRISC)     │    │
  │  │       (NO mux conn)      │    │       (mux connection)   │    │
  │  │                          │    │                          │    │
  │  │  1. cb_push_back(local)  │    │  1. FWD mux setup        │    │
  │  │     (NO memcpy needed!)  │    │  2. FWD mux connect      │    │
  │  │                          │    │  3. Send fused pkt ──────┼────┼──► neighbor
  │  │  2. sem_wait(neighbor)   │    │     (R1 data)            │    │
  │  │                          │    │                          │    │
  │  │  3. cb_push_back(nbr)    │    │  4. BWD mux setup        │    │
  │  │     (NO memcpy needed!)  │    │     (hidden by compute)  │    │
  │  └────────────┬─────────────┘    └──────────────────────────┘    │
  │               ▼                                                  │
  │  cb_local ──────┐                                                │
  │                 ▼                                                │
  │           ┌──────────┐                                           │
  │           │ COMPUTE  │──► cb_r1_result                           │
  │           │  (TRISC) │                                           │
  │           └──────────┘                                           │
  │                 ▲                                                │
  │  cb_r1_nbr ─────┘  ← cb_pop_front() frees R1 neighbor space      │
  └──────────────────────────────────────────────────────────────────┘
```

ROUND 2 (uses SEPARATE MeshBuffer from R1 - different sender!):
═══════════════════════════════════════════════════════════════
```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     TENSIX CORE (Shard Core)                     │
  │                                                                  │
  │          ┌─────────────────────────────────────────────────┐     │
  │          │                 SHARED L1                       │     │
  │          │                                                 │     │
  │          │  cb_r1_result (already populated by R1 compute) │     │
  │          │  Used as local input for R2                     │     │
  │          │                                                 │     │
  │          │  ┌──────────────────────────────────────────┐   │     │
  │          │  │  R2 MeshBuffer @ 0x3000 (SEPARATE!)      │   │     │
  │          │  │  ┌──────┐┌──────┐┌──────┐                │   │     │
  │          │  │  │ L    ││ S    ││ M    │ ◄──────────────┼───┼──── R2 neighbor
  │          │  └──┴──┬───┴┴──┬───┴┴──┬───┴────────────────┘   │     (DIFFERENT
  │          │        ▼       ▼       ▼ SEPARATE CBs           │     device!)
  │          │  cb_r2_nbr_l  cb_r2_nbr_s  cb_r2_nbr_m          │     │
  │          └─────────────────────────────────────────────────┘     │
  │                                                                  │
  │  ┌──────────────────────────┐    ┌──────────────────────────┐    │
  │  │       READER (NCRISC)    │    │       WRITER (BRISC)     │    │
  │  │       (NO mux conn)      │    │       (mux connection)   │    │
  │  │                          │    │                          │    │
  │  │  1. cb_push_back(r1_res) │    │  1. BWD mux connect      │    │
  │  │     (already in CB)      │    │     (setup was in R1!)   │    │
  │  │                          │    │  2. cb_wait(r1_result)   │    │
  │  │  2. sem_wait(r2_sem)     │    │  3. Send fused pkt ──────┼────┼──► R2 neighbor
  │  │     (SEPARATE semaphore!)│    │     (to R2 dest addr)    │    │
  │  │                          │    │                          │    │
  │  │  3. cb_push_back(r2_nbr) │    │  4. Disconnect muxes     │    │
  │  │     (NO memcpy needed!)  │    │     (overlaps w/ compute)│    │
  │  └────────────┬─────────────┘    └────────────┬─────────────┘    │
  │               ▼                               │                  │
  │  cb_r1_result ──┐                             │                  │
  │                 ▼                             │                  │
  │           ┌──────────┐                        │                  │
  │           │ COMPUTE  │──► cb_final_result ────┤                  │
  │           │  (TRISC) │                        │                  │
  │           └──────────┘                        ▼                  │
  │                 ▲                      Write to output tensor    │
  │  cb_r2_nbr ─────┘                                                │
  └──────────────────────────────────────────────────────────────────┘
```

### Fused Packet Flow (Cross-Device, Zero-Copy)

```
  DEVICE A (Sender)                              DEVICE B (Receiver)
  ═══════════════                                ═══════════════════

  ┌─────────────────┐                            ┌─────────────────┐
  │ Writer (BRISC)  │                            │ Reader (NCRISC) │
  │                 │                            │                 │
  │ 1. Pack [L|S|M] │                            │ 1. sem_wait()   │
  │    in L1 buffer │                            │    (blocking)   │
  │                 │                            │                 │
  │ 2. Build fused  │                            │        │        │
  │    packet:      │                            │        ▼        │
  │    - dst = 0x2000 (MeshBuffer addr)          │ ┌─────────────┐ │
  │    - sem_inc    │                            │ │ Semaphore=0 │ │
  │                 │                            │ │             │ │
  │ 3. mux_send()───┼────────────────────────────┼─│ MeshBuffer  │ │
  │                 │     FABRIC                 │ │ @ 0x2000    │ │
  └─────────────────┘                            │ └─────────────┘ │
                                                 │                 │
       NOTE: dst_addr = neighbor_recv_buffer->address()            │
             This is the SAME address on ALL devices!              │
                                                 │                 │
                          Fused packet arrives:  │                 │
                          ┌───────────────────┐  │                 │
                          │ 1. Write data     │──┼─► MeshBuffer    │
                          │    to 0x2000      │  │   [L|S|M]       │
                          │                   │  │       │         │
                          │ 2. atomic_inc     │──┼─► Semaphore=1   │
                          │    semaphore      │  │       │         │
                          └───────────────────┘  │       ▼         │
                                                 │ sem_wait wakes! │
                                                 │                 │
                                                 │ 2. cb_push_back │
                                                 │    (NO memcpy!) │
                                                 │    Data already │
                                                 │    in CB via    │
                                                 │    MeshBuffer!  │
                                                 └─────────────────┘
```

---


Current estimates:
1. Compute: sdpa per round: ~2000 ns; Final normalization: ~800ns; total: 2000*2 + 800 = ~5000ns
2. Fabric latency: mux -> router -> remote router -> dest core : ~2000ns [Packets are in mux channels, dst chip is 1 hop away, 1D ring, ~4k packet]
3. Total traffic: Each worker sends 1 packet in each direction -> 4 workers per link -> 4 packets per link per device
