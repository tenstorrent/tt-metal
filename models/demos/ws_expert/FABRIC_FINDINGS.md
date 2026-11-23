# Fabric bandwidth: findings

What moving weights (weight streaming) and token rows (EP-4 dispatch / combine) over the Blackhole fabric needs, from
microbenchmarks on a LoudBox flashed to the Galaxy's 12 x 10 grid, at 200G and 400G. The run-by-run history, exact
commands and every number are in WORKLOG.md ("The LoudBox at 200G" onwards); this file keeps what generalizes.

Tests: `tests/test_fx_bench.py` (one-hop flows, kernels `fx_send.cpp` / `fx_recv.cpp`), `tests/test_fr_ring.py` (an
8-chip forwarding ring in the weight-streaming shape, kernels `fr_lane.cpp`, `fr2_send.cpp` / `fr2_proc.cpp`),
`tests/test_hbag_ring8.py` (high_bw_all_gather as the reference CCL).

## 1. One fabric connection fills a link, if its packets fill the router's slot

A packet costs the router roughly a full slot's time whatever its size, so throughput depends on packet size *relative
to the router's configured max payload*, not on packet size alone. With the payload set to the packet size, one worker
connection from L1 reaches 46-48 GB/s on a 400G link (92-96%) at 4352, 8704 or 15232 B; a 4352 B packet under the
15232 B default we use gets 18. The send path (stateful or not, posted, an increment on every packet, BRISC or
NCRISC) does not matter. The old belief that a connection tops out near 25 GB/s came from a benchmark with exactly
this mismatch. Our combine and dispatch send one 7.7 KB row per packet under a 15232 B payload: much of their
per-connection ceiling is this.

## 2. The "second flow on a chip" penalty is two streams sharing a NoC link

Two senders on one chip (any links, any neighbours), or two routers receiving on one chip, dropped to ~40 GB/s each;
a sender plus a different receiving router, or flows on different chips, did not. Each such pair adds up to ~81 GB/s,
one NoC link. NoC traces of the senders show why: worker-to-router writes travel along the ethernet row on NOC1
(y- then x-) or along a Tensix row on NOC0 (x+ then y+), and the routers' landing writes (always NOC1) travel along
the destination core's row, so same-kind streams overlap on a link. It is neither a fabric nor a per-chip limit.

The fix is placement by physical path, not splitting NoCs: a sender directly under its own ethernet core on NOC1 (one
hop), a receiver at the bottom physical row of its ethernet core's column (the landing path 1 -> 0 -> 11, no row
travel). An ethernet core over a column without a Tensix core (harvested, or the dispatch column) gets a NOC0 sender
from the bottom row, because every NOC1 path up the neighbouring column shares that column's last link. With every
link of the mesh busy in both directions this takes the flows from 15-48 GB/s to 44-47, and a 6-link chip from
~110-190 to ~280 GB/s each way.

Two traps on the way: placement must use physical columns (logical column c is the c-th enabled one; every bh-lb-09
chip has x 10 harvested), and the fabric's ethernet channel ids count only unharvested ports (a P150 fuses channels
5 and 8), so `ws_lane5.eth_x` and the lane placement built on it are wrong for channels >= 5 on the LoudBox. Check
any such mapping against a NoC trace.

## 3. Forwarding through L1 at link rate: what it takes

An 8-chip ring of one-way lanes (each chip injects its own packets, every packet travels 7 hops, packets land in an
L1 ring on the next chip) went from 17 to 38.5 GB/s per link direction with both links busy, above
high_bw_all_gather on the same ring (37.7). In order of effect:
- A credit window that covers the round trip: 8 slots ran at 17 GB/s (the credit round trip is a few us), 32 slots
  at 38; more did not help.
- Free a slot when the NoC's sent-write counter says its packet left L1, not with a blocking flush.
- Full-slot packets, the connection's stateful send path, and credits carried in the opposite lane's packets (an
  explicit credit only when that lane is idle).
- Short passes: handling 2-4 arrivals between credit updates is best; 32 collapsed to 6 GB/s, because the credits a
  pass carries are as old as the pass.
- Split the work over the two RISCs of a lane's core: one examines landed packets and does every write to other
  cores, the other only sends. This was the step from 32 to 38.5 with two links; it also keeps the sender's stateful
  command buffers armed (the stateful path keeps state in the command buffer other writes use).

Going through L1 is not faster by itself. high_bw_all_gather lands in DRAM, which is effectively unbounded, so it
needs no slot credits, sends one increment per shard and has a reading RISC and a sending RISC per lane; an L1 ring
must earn the same with a large enough window and the RISC split. DRAM bandwidth was never its limit.

## 4. At 200G a good design runs at link rate; at 400G other limits appear

high_bw_all_gather reaches 23.9 of 25 GB/s per link direction at 200G (96%) and 37.7 of 50 at 400G (75%). The ring
benchmark at 400G stops at ~38-39: with 8704 B packets the sending RISC is busy every cycle (~300 cycles a packet
against ~245 for a bare send loop), with 15232 B packets it waits on router space. Both are above what 200G asks.

## 5. What this means for weight streaming and the Galaxy

- The mux exists because a router accepts one worker connection per direction and a group's 25 cores all send. A
  forwarder core per lane that owns the connection, lands packets in its L1 ring, forwards them, and delivers each
  slice to its group core on NOC1 removes the mux, its ethernet-row paths and the group cores' forwarding work,
  with the same core count. Not built yet; the next step is delivery into group cores in the ring benchmark.
- A Galaxy chip runs 8 links: 8 streams out and 8 in at ~25 GB/s each. One NoC link fits about three, so the budget
  is there, but only with forwarders and landing cores placed by physical path from each chip's real Tensix and
  ethernet harvesting.
- The first Galaxy experiment should be every link busy in both directions (`test_fx_all_links` with the physical
  placement) on 8 x 4, before any lane work.

## 6. Method

- NoC traces: device profiler NoC events. Fabric runs need the scratch patch in `profiler.cpp` (the trace writer
  aborts on fabric coordinates). The connection's stateful send path programs command buffers directly and is not
  recorded: trace with the plain path (`FX_MODE=0`), which runs at the same speed. The trace gives physical (NOC0)
  coordinates; tt-npe was used only to collect, the routes were counted by hand.
- Measure a flow at its receiver (first to last arrival), and keep sends and arrivals counted exactly; every ring
  result above is from runs with every packet accounted for.

## Open

- The 8704 B sender loop's extra ~60 cycles a packet (bookkeeping, credit field, queue).
- 15232 B both ways on one link: 42 GB/s alone, ~39 forwarding; whether the receiving core's own sends slow its
  router's landing writes.
- Every link busy at 200G on this box, and on the Galaxy.
- `ws_lane5.eth_x` / `L5Cfg.place`: map channel ids through each chip's ethernet harvest.
