# Ring-attention trace-replay deadlock — root cause

Traced chunked prefill (gemma4, 4x8 Blackhole galaxy, CP=4 along mesh axis 0) deadlocks
at deep ring depth: chunks 54/59/61 of 64 in the real test, replays 41–121 in the stress
reproducer. Probabilistic. Recovery needs a board reset.

## Root cause (one sentence)

The neighbor-halo exchange and the fused all-gather are **separate programs that share both
a physical core and a GlobalSemaphore**, and the halo's completion does
`noc_semaphore_set(sem, 0)` — a blanket clear that destroys all-gather increments which
have arrived on the same counter but not yet been consumed.

## The two colliding users of `semaphore[0]` on core (11,0)

Gemma4's 60 layers dispatch **two different program families**, because 50 layers are
sliding-window (compact halo path) and 10 are full-attention (dense gather path). Proven
from the inspector's `program_id`s:

```
neighbor_halo_reader/writer   programs 284, 286, 288, ...   (32)
all_gather_reader/writer      programs 574, 576, 578, ...   (32)
intersection: EMPTY                       -> separate programs
ring_joint_reader/sdpa/writer programs: 64 -> present in BOTH families
```

Both families allocate CCL workers from the same `ccl_core_grid_offset`:

| helper | call | cores (num_links=2) |
|---|---|---|
| neighbor halo | `choose_worker_cores(num_links, 1, ...)` | (11,0), (11,1) |
| all-gather | `choose_worker_cores(num_links, 2, ...)` | (11,0), (11,1), (11,2), (11,3) |

and both take `semaphore[0]`:

| user | semaphore | source |
|---|---|---|
| halo reader **and** writer | `semaphores.front()` | ring_attention_all_gather_..._program_factory.cpp:247, 253 |
| all-gather **backward** reader | `semaphore.at(0)` | same file:855 |
| all-gather **backward** writer | `semaphore.at(0)` | same file:932 |
| all-gather forward reader/writer | `semaphore.at(1)` | same file:828, 892 |

Gemma4 hands the *same* two-semaphore pair (`ring_attention_ccl_semaphore_handles`) to
every layer, so **core (11,0) + semaphore[0]** is used by the halo protocol on sliding
layers and by the all-gather backward protocol on full-attention layers, alternating,
back-to-back, with no host involvement inside a replay.

A `GlobalSemaphore` has the same L1 address on every core but a **separate counter per
core**, so sharing the address across *different* cores is harmless. (11,0) is the one
core where the two protocols genuinely alias.

## Why the collision is destructive

The two protocols have incompatible completion semantics:

```cpp
// neighbor halo reader — expects exactly one increment
noc_semaphore_wait_min(incoming_ready_sem, 1);
noc_semaphore_set(incoming_ready_sem, 0);        // <-- blanket clear

// all-gather backward reader — expects a RUNNING COUNT of slices_expected
while (slices_received < slices_expected) {
    noc_semaphore_wait_min(out_ready_sem, slices_received + 1);
    slices_received++;
}
noc_semaphore_set(out_ready_sem, 0);
```

The halo waits for `>= 1` and then zeroes the counter, discarding anything above its own
single increment. Increments are delivered asynchronously over the fabric — the writer's
`send_payload_flush_blocking_from_address` only flushes into the local EDM, so a sender's
kernel completes while its increment is still traversing ethernet. When an all-gather
increment lands on (11,0) while that core has already moved on to a sliding layer's halo
program, the halo's `set(0)` annihilates it. The all-gather reader that was counting on it
is then permanently short and waits forever.

The reverse also corrupts: a late halo `+1` can satisfy one iteration of an all-gather
reader's `wait_min`, which then exits early and leaves a real increment unconsumed.

Only the all-gather side can *hang*, because it waits for a specific count N; the halo only
waits for `>= 1`. That matches the evidence exactly.

## Evidence

`tt-triage --run=dump_callstacks` on a live hang (inspector enabled, attached before kill):

| kernel | stuck at |
|---|---|
| `ring_attention_all_gather_reader` | `noc_semaphore_wait_min` — reader.cpp:201 |
| `ring_attention_all_gather_writer` | `cb_wait_front` (waiting on its own local reader) |
| `ring_joint_reader` / `ring_joint_sdpa` | `RingSDPAOpReceiver::get_next_ring_id_and_consume_one_signal` |

Every stuck all-gather reader is on core **(11,0) or (11,2)** — never (11,1)/(11,3). And the
stuck set is exactly the devices whose (11,0) currently hosts the all-gather reader:

```
(11,0) brisc = neighbor_halo_reader  -> devices 0-3, 24-27      (not stuck)
(11,0) brisc = all_gather_reader     -> devices 4-23, 28-31     (STUCK)
```

Semaphore values read from L1 with ttexalens while deadlocked (`sem0` is nonzero only on
(11,0)/(11,2), `sem1` only on (11,1)/(11,3), confirming ownership):

```
sem0 (backward, cores 11,0 / 11,2)      sem1 (forward, cores 11,1 / 11,3)
 r0 (expects 0):  0 0 0 0  0 0 0 0       r0 (expects 3):  3 3 3 3  0 0 0 0
 r1 (expects 1):  2 2 2 2  1 1 1 1       r1 (expects 2):  0 0 0 0  0 0 0 0
 r2 (expects 2):  2 2 2 2  1 1 1 1       r2 (expects 1):  0 0 0 0  0 0 0 0
 r3 (expects 3):  0 0 0 0  0 0 0 0       r3 (expects 0):  3 3 3 3  0 0 0 0
```

Two independent captures produced identical tables. Note the impossibilities under a clean
protocol, which is what proves cross-protocol interference rather than a plain lost packet:

- **r3 forward = 3 while its forward reader expects 0.** With `slices_expected == 0` the
  loop never runs and the reader immediately does `set(0)` — so those 3 arrived after it
  had already cleared, and nothing in the all-gather protocol will ever consume them.
- **r1 backward = 2 while it expects 1**, and it is nevertheless blocked at line 201.
  A reader expecting 1 cannot be blocked with the counter at 2. The counter it is waiting
  on is not the counter its own protocol has been filling.

Linear-topology counts are from source, not assumed:
`num_targets_forward = ring_size - ring_index - 1`, `num_targets_backward = ring_index`
(`LineTopology::get_distance_to_end_of_line`, ccl_common.cpp:313).

## Experiments (each eliminates a hypothesis)

| experiment | result | conclusion |
|---|---|---|
| fixed depth, 120 replays | survived | not replay-count accumulation |
| shallow depth, metadata changing every replay | survived 120 | not stale-metadata reads |
| deep-only (chunks 40–62), 400 replays | survived | depth alone insufficient |
| full sweep 1–63 | hangs ~41–121 | needs layer-family alternation across depth |
| + 200 ms / 25 ms settle | survived | drain before the next replay avoids it |
| + 5 ms settle | hangs | drain must exceed the in-flight window |
| `ttnn.synchronize_device` after staging | **no effect** | barrier does not cover in-flight inter-chip traffic |
| host reset of the ring semaphores | unreliable | cannot beat an arrival landing after it |
| `num_links=1` | still hangs | not the parallel link workers racing |
| `readback_all` | never hangs | its per-chunk gather + host read supplied the drain incidentally |

## The fix

**1. Give the neighbor-halo exchange its own GlobalSemaphore.** It must not reuse
`semaphores.front()`, which the all-gather backward direction owns on the same core.
`ring_attention_neighbor_halo_exchange_helper` should index a dedicated entry (e.g.
`semaphores.at(2)`); `ring_joint_sdpa_program_factory` passes a 3-element vector; gemma4's
`CCLManager` creates three global semaphores instead of two. This removes the aliasing that
causes the observed hang.

**2. Make the halo consume rather than clear.** Replace

```cpp
noc_semaphore_set(incoming_ready_sem, 0);
```

with an atomic decrement of exactly the one increment it waited for. `set(0)` asserts
ownership of the whole counter, which is only ever safe if nothing else can increment it —
an invariant nothing enforces. Decrementing keeps the "never destroy what isn't yours"
property even if a semaphore is shared again later.

Either change alone stops the observed deadlock; both together make the invariant
structural. Note the all-gather's own `set(0)` (reader.cpp:281) is safe in isolation — it
waits for every increment it expects before exiting, so there is nothing in flight for it
to destroy — but it should get the same treatment for the same reason.

An alternative to (1) is to allocate the halo's worker cores disjoint from the
all-gather's, since GlobalSemaphore counters are per-core. That also works but costs extra
cores and leaves the fragile `set(0)` invariant in place.

## Status: FIXED

Both changes are implemented. Verified with the workaround disabled
(`GEMMA4_TRACE_SETTLE_MS=0`, the setting that previously hung within ~50 replays):

- stress reproducer, advancing depth, **400 replays, no settle** — survived (previously
  hung at replay 41–121)
- 256k readback_final, no settle, **three consecutive runs** — 23.7 s each
  (11058 / 11059 / 11050 tok/s), faster than the 24.7 s the 25 ms settle cost
- 32k traced PCC unchanged: 0.94515 0.98890 0.98914 0.99010 0.98996 0.98885 0.98996 0.98920

`GEMMA4_TRACE_SETTLE_MS` now defaults to 0 and remains only as a bisect knob.

## Reproducer

```bash
GEMMA4_STRESS_REPLAYS=400 pytest \
  "models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_trace_replay_stress" \
  -k advancing_depth -s -v
```

`GEMMA4_STRESS_MINCHUNK` / `MAXCHUNK` bound the depth sweep, `GEMMA4_STRESS_DELAY_MS` adds a
settle, `GEMMA4_STRESS_SYNC_STAGE=1` adds a device barrier, `GEMMA4_NUM_LINKS` overrides the
link count. To capture state, run with `TT_METAL_INSPECTOR=1` and, **while it is still
hung**, attach:

```bash
./tools/tt-triage.py --run=dump_callstacks --llm-output --llm-output-path=/tmp/triage.txt
python3 read_sems.py <pytest-log>     # semaphore values per device/core
```

Do not kill the process first — the state dies with it. Triage reads inspector logs from
`<repo>/generated/inspector`, not the `/tmp/tt-metal/inspector` default.
