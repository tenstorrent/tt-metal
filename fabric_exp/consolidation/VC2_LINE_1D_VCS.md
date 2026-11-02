# Several line VCs on 1D fabrics: resource census and plan (2026-11-01; offline, no code)

The idea (user; main tree VC2_LINE_IDEA.md, "Extension" and "Addition"): on a 1D line or ring, up to 3-4 line VCs, each
with its own worker channel, so one router takes N worker connections natively (no mux core) for ops with worker fan-in;
the VCs are not special, the host says how many there are and what each one is, and the router follows the host's
per-VC facts (V4_RECEIVER_OWNERSHIP.md, V4_SENDER_OWNERSHIP.md). On 1D VC1 need not be the inter-mesh VC.

A line VC here = one worker sender channel, one forwarding sender channel (fed by the router facing the other way) and
one receiver channel that forwards straight on, as VC0 has on 1D today. Sources: this worktree (`tt-metal-vc2line`,
VC2 line mode as built; file:line below), the ownership docs, the main tree's RESULTS.md and V4_RECEIVER_SHAPE.md.
Labels: code (read from code), meas (measured: job or offline build), est (estimate).

## 0. Answer

- **N = 2**: no resource is tight: slots 41 per VC, 8 transaction IDs per receiver channel, about 19.5 KB of code on
  two ERISCs (est), the 1-hop windows still reach the wire (est).
- **N = 3**: fits, at a cost: 27 slots per VC (a lone VC's worker window about 9: at the Galaxy wire at 4 KiB 1-hop,
  est), 4 transaction IDs per receiver channel (conservative completions), about 24 KB of code on two ERISCs (est), three
  idle line VCs' per-pass work on E0 (est, below; to measure). The receiver pass and the host's VC arrays are at their
  sizes (MAX_NUM_VCS = 3, three receiver channels).
- **N = 4: what binds first is code size on two ERISCs** (est 28.5 KB on a line, 30 KB on a ring, against 29,216 B left
  in the 29 KiB buffer after about 480 B of runtime args), then **structure** (MAX_NUM_VCS = 3 and the receiver's
  three named channels: a fourth VC is a code change, not a resource), then the windows (20 slots per VC: a lone VC
  window-bound at about 1.5x the wire per packet at 4 KiB 1-hop, est). Transaction IDs (4 per channel), stream registers
  (16 of 30), the flat sender table (8 of 10) and the credit blocks still fit.
- So **N = 3 is the maximum inside today's buffer and structure**; N = 4 needs one ERISC (about 21.5 KB, fits) or a code
  diet, and lines only (rings add first-level-ack code).
- **Measured on the Galaxy (§1.7):** an idle line VC costs a busy VC0 flow +17-20 cycles per packet at small sizes on two
  ERISCs (+53 on one), nothing at 4352 B 1-hop; so with N = 3, a lone flow pays ~+35-40 (~30%) at small sizes (est, if
  additive). Two or three busy VCs share one router's small-packet capacity (+9% total over one VC at today's windows,
  +27% with doubled windows); at 4352 B the wire caps the total. Line VCs buy native fan-in (no mux core, no extra hop),
  not router capacity.

## 1. Resource census (1D line or ring; per router)

| resource | 1D today (VC0) | per added line VC | N = 2 | N = 3 | N = 4 | limit |
| --- | --- | --- | --- | --- | --- | --- |
| channel slots (4,448 B each; 29 KiB buffer: 82) | W 25, F 25, R 32 (meas, slot log) | W + F + R | 2 x 41 (e.g. 12/13/16) | 3 x 27 (e.g. 9/10/8) | 4 x 20 (e.g. 6/6/8) | 82 |
| transaction IDs per receiver channel (E1, one NoC) | 16 (8 with deadlock avoidance) | a range of its own | 8 | 4 | 4 | 16, power of 2 |
| stream registers per Ethernet core (both ERISCs) | 4 (+ fixed ones) | 4 | 8 | 12 | 16 | ids 0-29 |
| flat sender-table positions | 2 | 2 | 4 | 6 | 8 | 10 |
| credit counter blocks (L1, 4 x 4 B per sender channel) | 4 blocks x 10 channels | 0 (sized by the table) | same | same | same | 192 B fixed |
| code, two ERISCs (E0 + E1) | 14,956 B line, 15,016 ring (meas) | ~4.5 KB line, ~5.0 KB ring (est) | ~19.5 KB | ~24 KB | ~28.5 / ~30 KB | 29,216 B |
| code, one ERISC | 9,552 B line, 9,656 ring (meas) | ~4.0 KB (est) | ~13.5 KB | ~17.5 KB | ~21.5 KB | 29,216 B |
| E0 per-pass reads (grouped) | 3 (2 stream + 1 L1) | +3 | 6 | 9 | 12 | 4 overlap fully |

### 1.1 Slots

- 82 slots at 4,352 B payload (4,448 B channel buffers) with the 29 KiB kernel buffer (meas: VC2_LINE_BRINGUP.md phase
  3, 1D 83 -> 82). Today's 1D split (code: fabric_static_sized_channels_allocator.cpp:796-831, the study router only):
  the receiver keeps its tier depth 32 (a power of 2: the receiver ring's index arithmetic is cheapest so), the two
  sender channels share the rest (25/25).
- N line VCs split 82 N ways. A receiver at a power of 2 (8 or 16) leaves the senders the rest. Rings need 2 free far
  slots for an injection (bubble flow control, fabric_builder_config.hpp:108-118), so a receiver of 4 or more.
- The window model, from measured rows: a 1-hop flow's cycles per packet ~ max(wire, RTT / W). Galaxy, 4 KiB, VC2 1-hop:
  4 slots 507, 8 slots 294 (job 167468), so RTT about 2,000-2,350 cycles; the wire ~248. A lone VC needs W >= ~9 for
  the wire at 4 KiB (est); P150 agrees in shape (VC2 2 slots 858, 8 slots 240 against VC0's 135; main
  V4_RECEIVER_SHAPE §35). Multi-hop flows are bound by the forwarding windows the same way (main RESULTS §98: 16
  forwarding slots bind 3-hop on Linear).
- Per N (est): N = 2 keeps 12-16 slot windows (a lone VC at the wire for 1 hop); N = 3 about 9 (at the wire for 1 hop,
  short of it for multi-hop); N = 4 about 6 (~370 cycles per packet 1-hop at 4 KiB, ~1.5x the wire). When the workers
  share the link the windows add up, so the shrink matters only for a lone busy VC.

### 1.2 Transaction IDs

- Code: E1 tags each receiver channel's NoC writes (local writes and forwards on one NoC, router_completions.hpp:89-92)
  with IDs from its own range, RX_CH_TRID_STARTS = channel x NUM_TRANSACTION_IDS (compile_time_arg_tmp.hpp:26-34;
  ct_args:751-770, "TODO: pass from host"); the NoC has 16 IDs (4 bits); a channel's count must be a power of 2
  (GroupIds, router_completions.hpp:49-53). Today with VC1 or VC2: 8 each for up to 2 receiver channels under deadlock
  avoidance, 4 for 3 (ct_args:763; my phase-1 fix of related problem 2).
- N = 2: 8; N = 3 and 4: 4. A completion group (1, 2 or 4 slots; 1 under first-level acks, router_receiver.hpp:111-115)
  takes one ID; fewer IDs than groups only makes the completion proof conservative (a retired ID covers its aliases).
  With 8-slot receivers in groups of 2-4 four IDs suffice; on rings (groups of 1) 4 IDs for 8 slots alias 2:1 (est: a
  later credit, measurable).

### 1.3 Stream registers

- 32 per Ethernet core (both ERISCs share them), ids 0-29 assigned by role, 30-31 pinned
  (fabric_stream_assignment.hpp:28-58). The study router's credits are L1 counters on one or two ERISCs
  (fabric_builder_context.cpp:56-67), so per line VC: 2 sender free-slots streams (worker, forwarding), 1 receiver
  packets-sent stream, 1 downstream free-slots copy (1D: one downstream) = 4. N = 4: 16. Not binding.
- VC-numbered today: VC2's worker free-slots stream is pinned to id 30 and its receiver's to 31 (dual-use with the tensix
  relay and the teardown scratch, fabric_stream_assignment.hpp:40-58); the declared name sets are per VC
  (`num_downstream_free_slots_names_per_vc` for VC0 and VC1, one VC2 name, :62-79). Line VCs need generic per-VC names.

### 1.4 Flat sender table and credit blocks

- 10 positions (device MAX_NUM_SENDER_CHANNELS; host builder_config::num_max_sender_channels, fabric_builder_config.hpp
  :60-77), per-VC bases VC<n>_FABRIC_POSITION_START (ct_args:107-140). 2 per line VC: N <= 5 by the table.
- Credits: four L1 blocks of 4 B per sender channel, sized by the channel maximum (erisc_datamover_builder.cpp:325-351,
  ~192 B); a packet's source channel is a byte (SlotRecords, router_completions.hpp:55-63). Not binding.

### 1.5 Code size

- Measured (offline build p3a, -O3, size_report.py; E0 + E1): 1D 14,956 (one ERISC 9,552); ring 15,016 (9,656); 2D
  16,648 (11,372); 2D + VC2 neighbour 19,512 (14,168); 2D + VC2 line 21,116 (15,360); torus + VC2 line 22,132 (torus
  17,160); 2D + VC1 + VC2 line 28,412.
- A line VC on 2D adds 4,468 B on two ERISCs (E0 +2,168, E1 +2,300) and 3,988 on one; on a torus (first-level acks,
  bubble flow control) 4,972. Taken as the per-VC increment on 1D (est; 1D's receiver decode is simpler, so this is an
  upper estimate; no 1D multi-VC shape builds without code, so nothing closer can be measured offline).
- The kernel buffer is 29 KiB (29,696 B) and holds the runtime args too (~480 B): ~29,216 B for E0 + E1.

### 1.6 Per-pass cost (E0, E1)

- E0 today on 1D: each pass reads, in one group, the worker channel's free-slots stream and connection semaphore (L1)
  and the forwarding channel's free-slots stream, then visits both channels. A line VC adds the same three reads and two
  visits. Main RESULTS §17 (meas, erisc_load_ubench): stream reads issued together overlap up to 4 in flight (four cost
  4 cycles); a group with an L1 read overlaps only if used 5-7 instructions later; an empty channel visit is ~13
  instructions. So N = 2 puts 6 reads in the group (past 4), N = 4 twelve.
- Measured cost of one idle VC: on P150 2D an idle VC2 (neighbour: one worker channel, one receiver) costs VC0 7.7
  cycles per packet at 4352 B, E0's (main V4_RECEIVER_SHAPE §29-§30, the lead after the one-visit change). A line VC has
  a forwarding channel too: est ~10-15 cycles per pass on E0 per idle line VC. P150 1D 1-hop has ~14 cycles of E0 slack
  at the wire (main): one idle line VC fits, two or more show at 4352 B on P150 (est). E1's idle step did not bind on
  P150 (§29 probe p3).
- Busy VCs: each sent packet costs E0 its send (~45 instructions plus the TX-queue latch, §17), whatever VC it is on;
  the link's total over N busy VCs is capped by E0's per-packet work plus the per-pass visits at small sizes, by the wire
  at large ones (to measure: §1.7).

### 1.7 Measured on the Galaxy (VC2 line mode as built; VC2_LINE_BRINGUP.md, "Perf: idle and full VCs")

- Idle VC cost (job 167481, one 8x4 mesh, VC0 1-hop, windows pinned equal): an idle VC2 in line mode costs VC0 +17 to
  +20 cycles per packet at 256 B and 1 KiB on two ERISCs (neighbour mode +10), +53 on one ERISC (neighbour +38), nothing
  at 4352 B (the wire, 248-252, hides it). Multi-hop and multicast rows pay +23 to +31 at small sizes and +46 to +65 at
  4352 B (two ERISCs, line). With VC1 compiled in as well (job 167482, the 2x2 split) the line VC costs +34 to +36.
- So an idle line VC costs about 15% of a two-ERISC router's small-packet 1-hop rate (125 -> 145), and 25-30% on one
  ERISC. For N line VCs with one busy the other N - 1 add (est, if additive): N = 2 +17-20, N = 3 +34-40, N = 4 +51-60
  cycles per packet at small sizes on two ERISCs; at 4352 B 1-hop nothing until a router's per-pass work passes the wire
  (on the Galaxy: VC0 1-hop 125 against 248, so roughly 6 idle line VCs, est). On P150 (wire ~126) the slack is ~14
  cycles: one idle line VC already shows at 4352 B there (est).
- Several busy VCs on one link (job 167482): at 4352 B the total reaches the wire with any VC whose window allows it; at
  256 B and 1 KiB two or three VCs together move ~8.3 packets per us against ~7.6 for one (+9%), ~9.7 with doubled
  windows (+27%), far below the wire. Native fan-in through line VCs therefore adds little small-packet capacity per
  router; it removes the mux core (a Tensix per link) and its extra hop, which is the point of the idea.
- Windows (A/B, job 167482): a 4-slot VC is window-bound alone (VC2: 448-507 cycles per packet against 254-294 at 8
  slots; VC1 4/4: 467-557 against 332-338 at 8/8). At N = 4 (6-slot workers) a lone VC is window-bound (est from the
  same model, ~370 at 4352 B 1-hop).

## 2. The alternative: several worker channels on one VC

Counted the same way (k worker channels on one VC, merging into the VC's far receiver ring the way forwarding channels
do today, e.g. 2D VC0's three forwarding channels into one receiver):

| resource | per added worker | k = 4 workers on VC0 (1D) |
| --- | --- | --- |
| slots | its own W only (no forwarding channel, no receiver) | 4W + F + R <= 82, e.g. 12/16/16: larger windows than 4 line VCs |
| transaction IDs | none (one receiver) | 16 (8 under deadlock avoidance) for the one receiver |
| streams | 1 (its free slots) | 5 + 2 |
| flat positions | 1 | 5 of 10 |
| code | E0 only, one more worker channel (~1 KB, est from 2D's ~740 B per forwarding channel plus connection handling) | ~18 KB two ERISCs (est) |
| per pass | one worker visit (free slots + connection semaphore in the group), ~8 cycles idle (est, like VC2 neighbour's worker) | 4 workers: 6 reads in the group |

The device side is close: a packet's source channel is recorded per slot (SlotRecords) and credits go back per source
channel (counter blocks indexed by channel), as for forwarding channels (code: router_completions.hpp:55-63,
erisc_datamover_builder.cpp:325-351). The host's worker role per channel (SENDER_CH_<i>_IS_WORKER_CHANNEL, the sender
ownership pass) makes "several worker channels on a VC" expressible.

What it gives up:
- isolation: the workers share one far receiver ring and the VC's downstream windows, so one flow whose packets wait
  (a slow destination write, a full downstream) holds slots every other worker's packets need (head-of-line blocking);
  separate VCs keep separate rings and credits;
- the one-worker-per-VC rule (user's rule: mix flows on a link through forwarded traffic, not a second worker): the
  router's per-VC worker facts (one worker credit buffer per VC, V4_SENDER_OWNERSHIP §2 #7) and the worker API (one
  channel per VC) assume it;
- on rings, bubble flow control is a per-injection-channel rule (2 free far slots before injecting); with several
  injectors on one VC it holds only because E0 serializes their checks and sends (est; needs its proof restated).
What it keeps: fewer receivers, transaction-ID ranges and forwarding channels, so bigger windows per flow at N = 3-4
and less E1 code.

## 3. What must change

### Host

- A per-VC role. Today the host decides VCs by number (code): `compute_intermesh_vc_config` sets VC1 for multi-mesh and
  VC2 by option, and VC2 line mode only on 2D (fabric_builder_context.cpp:356-421); on 1D no VC1 or VC2 channel is ever
  created (router_wiring_rules.cpp:275-280: "No VC1 or VC2 channels are ever created"). Needed: a fabric
  setting "number of line VCs" (and per VC its role: inter-mesh or line), with VC1 a line VC on 1D fabrics.
- Per-VC shape: `RouterVcShape` counts per VC from the role (worker + forwarding + receiver per line VC on 1D), not from
  `requires_vc1/vc2` (router_wiring_rules.cpp:243-323); `MAX_NUM_VCS = 3` (fabric_builder_config.hpp:46) and the per-VC
  arrays sized by it for N = 4.
- Allocation per VC count: the slot rows are per VC combination today (fabric_static_sized_channels_allocator.cpp:
  876-960, VC0+VC1, VC0+VC2, VC0+VC1+VC2, line rows); a row per number of line VCs on 1D (receiver a power of 2).
  Streams: generic per-VC names instead of VC2's pinned ids 30/31 (fabric_stream_assignment.hpp:40-79); the flat bases
  per VC exist (`sender_flat_base(vc)`, erisc_datamover_builder.cpp:1334-1335).
- The worker connection API per VC: VC2's private API finds "the VC2 sender at the last flat index"
  (fabric_vc2_connection.cpp:159-191, fabric_vc2_connection.hpp:17-28); generalize to (VC, link) -> that VC's worker
  channel from the host's worker-role arg, for VC0..N-1.
- Per-VC named args already exist for three VCs (erisc_datamover_builder.cpp:1313-1389: PACKED_DOWNSTREAM_VC<n>_...,
  VC<n>_USES_COUNTER_CREDITS, NUM_DOWNSTREAM_SENDERS_VC<n>, ENABLE_FIRST_LEVEL_ACK_VC<n>, VC<n>_DOWNSTREAM_EDM_SIZE,
  ACTUAL_VC<n>_SENDER_CHANNELS; OPEN_DOWNSTREAM_VC<n>_CONNECTIONS at :1453-1457); a fourth VC needs a fourth of each, and per-VC
  first-level acks (VC1's is fixed at 0 today, "must agree across every VC1 link"; a 1D VC1 ring needs its own).
- Bubble flow control by VC role, not number: `bubble_flow_control_enabled_on_vc(vc, vc2_line)` (fabric_builder_config.hpp
  :118) is "VC0, or VC2 in line mode".
- Combination validation moves here from the kernel's gate (below): VC counts, roles, topology, buffer fit.

### Router

- Crossover as a per-VC fact: `downstream_vc` is 1 under FABRIC_2D_VC0_CROSSOVER_TO_VC1, else the channel's own VC
  (router_receiver.hpp:104-107; host define compute_mesh_router_builder.cpp:933-938); the receiver ownership pass makes
  it `forwards_on_vc` (V4_RECEIVER_OWNERSHIP §2 #5, decision 4 there: VC2 line's phase 3 makes every channel cross over).
  Generalize to a host-given "forwards on VC" per receiver channel.
- Line routing by VC role: `vc_routes_as_line(vc) = vc == 2 && vc2_line_mode` (router_common.hpp:150) and its uses
  (downstream streams :166-197, command buffers router_downstream.hpp:47, presets :99-111, live downstreams :66-71); on
  1D every VC is a line VC and the 1D branches assume VC0's streams (router_common.hpp:174 `sender_channel_1_free_slots_
  stream_id`, :197 `vc0_downstream_credit_streams[0]`).
- The supported-combination gate moves to the host: `vc2_config_supported` (is_2d, VC2 sender count by mode, not Z,
  first-level acks only in line mode) and `brought_up` (receiver count = 1 + VC1 + VC2; `!has_vc1 || is_2d`;
  `!ENABLE_FIRST_LEVEL_ACK_VC1`) (fabric_erisc_router_v2.cpp:95-119); the kernel keeps consistency static_asserts.
- Remaining VC-number assumptions:
  - channel counts per VC: ACTUAL_VC0/1/2_SENDER_CHANNELS and the VC<n>_LOCAL_CHANNEL_START chain (ct_args:93-140,
    `vc_of_sender_channel` :392-410, `vc_of_receiver_channel` :414-427, `VC2_RECEIVER_CHANNEL` :188);
  - the sender's workers: `worker_channels = (vc0_channels & 1) | vc2_worker`, `Present<has_vc2_worker, Worker, 2>`,
    `visit_worker<VC2_LOCAL_CHANNEL_START>` (router_sender.hpp:79-121, 170, 620-729) -> per VC from
    SENDER_CH_<i>_IS_WORKER_CHANNEL (sender ownership X1);
  - downstream traits by VC ternaries (router_downstream.hpp:39-43) -> per-VC arrays (receiver ownership H1 #16);
  - the receiver pass names channels 0..2 as scalars (receiver ownership §3 #7: static_assert MAX_NUM_RECEIVER_CHANNELS
    == 3): a fourth receiver channel is a structural change;
  - transaction-ID ranges `ch x NUM_TRANSACTION_IDS` with the rule at ct_args:757-764 -> from the number of receiver
    channels (16 / N rounded down to a power of 2);
  - telemetry: VC0-specific reads (`rx_register_empty()`: VC0's packets-received register, router_telemetry.hpp:126);
  - slot rows: per VC combination (host, above).
- Command buffers: no new ones. On 1D every VC forwards to the same next router, so all line VCs can share VC0's preset
  destination with each forward setting its own ring slot and credit register: VC2 line's shared-preset rule
  (router_downstream.hpp:91-120), which cmd-buf-2 fix (c) generalizes to every VC.

## 4. Bring-up and measurement plan

1. Host and router changes above on top of the ownership variant (after the rebase), with VC2 line mode as the first
   user of the per-VC facts; neighbour and line images byte-identical where the facts are equal (offline, as phases 1-3).
2. **2 line VCs on P150 1D** (Linear, then Ring), then a **Galaxy ring** (one row or column as a 1D ring), data-checked:
   each VC alone, both, every length; rings with bubble flow control and first-level acks per VC; the deadlock-stability
   suite. Then **3 line VCs**, the same.
3. Rows (benchmark, no sync, per-sender lines; 256 B, 1 KiB, 4352 B):
   - **headline**: N workers on one chip, each to the same neighbour, one per VC (N = 1, 2, 3), against today's path for
     the same fan-in, N workers through the test infra's mux on VC0 (its tensix mux core); per worker and aggregate;
   - **idle-VC cost**: VC0 alone with 0, 1, 2 idle line VCs compiled in (1-hop, 3-hop, ring), windows pinned equal
     (as in "Perf: idle and full VCs");
   - **lone busy VC**: one VC's flow with the others idle, at each N's slot row (the window cost of splitting slots);
   - window A/Bs (pinned rows) for any bound claimed.
4. The alternative (§2) only if the user wants it compared: one VC with k worker channels, the same rows.

## 5. Open questions for the user

1. The fabric setting: a number of line VCs (roles implied by topology) or a per-VC role list (inter-mesh, line,
   neighbour) the host validates?
2. N = 4: worth a code diet or one-ERISC only, or stop at 3?
3. On 1D, VC1 as a line VC: always (no inter-mesh on 1D), or only when the user asks for more than 2?
4. Slot policy: equal split per VC, or a weighted row (the busiest VC's worker larger)? Receivers at 8 (power of 2) for
   N = 3-4?
5. The worker API: one call per VC (generalizing VC2's private API), or a VC argument on the public one?
6. The alternative of §2: rule it out (one worker per VC), or measure it beside the line VCs?
7. Rings with 3-4 VCs: first-level acks on every VC (code size, §1.5), or lines only at first?
