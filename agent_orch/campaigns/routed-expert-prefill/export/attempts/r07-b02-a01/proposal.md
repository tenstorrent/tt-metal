# r07-b02-a01: wrap-free gate/up weight multicasts: a top-row sender's gate/up multicast runs on its reader's NoC 0 (+y) instead of the writer's NoC 1 (-y), with the reader's DRAM read issued before the receivers' acks and the next block's read issued under the multicast drain

## Motivation
- r06-b04-a01 (best, 1.6931) removed the wrapped DOWN multicasts and measured DN -7..-14% per mean core on
  8 of 9 shapes (+1.6% geomean). Its reflection, next #1: do the same for gate/up, the other half of the
  r05-b04-a01 wrap diagnosis.
- Today every gate/up sender multicasts gate+up on the writer's NoC 1, which routes -y. Senders sit on the
  group's top row when (column + group base) is even, else on its bottom row (`sender_row_in_group`). A
  top-row sender's NoC-1 multicast has to wrap around the whole column torus (through the other groups'
  rows and the two non-worker rows) to reach the rectangle's NoC-1 start corner (its bottom row). On every
  2-row layout that is 6 of the 11 columns (even columns), on every group.
- Measured cost (r05-b04-a01, `analysis/wrap.py` on r04-b03-a01's UNPACK wait tallies), receivers' gate/up
  weight wait, wrapped column / non-wrapped column: kimi-k3 156.5 / 108.7 µs, dsv4-flash 44.2 / 21.6,
  minimax-m3 31.1 / 23.7, mistral 14.9 / 9.3, gpt-oss 10.4 / 2.4.
- GU is ~3x DN on every shape (dsv3 330 vs 112 µs per mean core), so the gate/up half is the larger one.
- Link model (r05-b02-a01's noc4.py with r06-b04-a01's corner placement, /tmp/r07b02/m/gufix.py): in gate/up
  the hottest NoC-1 worker-column link drops from 40 to 8-20 GB/s on 5x2 / 4x2 / 5+5 / 3+3+2+2 / 4+3+3, while
  NoC-0 worker-column links rise from 16-24 to 21-29. The DRAM-region maxima (54 / 52-63) are unchanged. The
  down fix had the same model signature and paid off, because a wrapped linked multicast holds a reserved
  ~12-hop path for the whole 50-110 KB transfer.

## Mechanism
One runtime predicate, identical on both RISCs: `wmc_gu = writer_mcasts_in1 && my_mt + 1 == group_rows`
(bottom-row senders, and any 1-row group, keep the writer's NoC-1 multicast exactly as today). A top-row
sender (`is_in1_sender && !wmc_gu`) multicasts gate/up from its reader on NoC 0, which starts at the
rectangle's NoC-0 corner, so nothing wraps. Top-row senders already multicast DOWN on the reader's NoC 0, so
such a column's whole weight stream is multicast by the reader, and its writer only reads (up tail + down tail).

Reader (`unified_routed_expert_ffn_reader.cpp`), gate/up sender section, top-row sender:
1. The gate (+ up head rows) DRAM read is issued first, into the sender's own ring slot (free by its own
   compute's pop), **before** waiting for the receivers' `in1_ready` acks. The acks only gate the multicast.
   (The legacy reader path waited for the acks before reading, which would serialize the read behind the
   slowest receiver.)
2. Read barrier, `up_done` wait (writer's NoC-1 up tail), then `in1_ready` wait, then the linked gate + up
   multicasts on NoC 0 and the local push.
3. The look-ahead read of block kb+1 (gate + up head, and the writer's up_go) is tried right after the
   multicasts are issued and before the flush, so the next block's DRAM read overlaps the multicast drain,
   as the writer-multicast path overlaps them today. Then flush and the in1_valid multicast.
4. No `mc_seq` / `mcast_go` / `mcast_done` traffic for top-row senders: the writer never multicasts for
   them, and the reader's own multicast is flushed before its slot can be refilled.

Writer (`unified_routed_expert_ffn_writer.cpp`): the gate/up multicast block runs only when `wmc_gu`. The
`up` tail read and `up_done` are unchanged.

To keep the h7168 programs inside the kernel-config buffer, the reader's slot multicast (data + linked
valid-sem) becomes one out-of-line helper shared by the new gate/up path and the existing down path.

## Why this is not a repeat
- r05-b04-a01 (hang) moved the gate/up senders to the bottom row and the down senders to the top row
  (different cores per phase, x owners on senders). Here placement, x owners and every core's role are
  unchanged; only which RISC/NoC carries a top sender's gate/up multicast changes. This is r05-b04-a01's
  repair option (c) applied to gate/up, and r06-b04-a01's next #1 option (a), with the read hoisted before
  the acks and the look-ahead under the multicast drain to avoid the serialization that reflection warned of.
- r03-b03-a02 / r02-b04-a02 moved whole READ streams onto one NoC (lost 5-20%). No read moves here: every
  DRAM read keeps its RISC, NoC, split (r05-b02-a01's 60/40) and slot.
- r01-b03-a01 / r01-b04-a01 issued the next read BEFORE the multicast; here it is issued after the multicast
  is in flight (their reflections' recommended order).

## Expected effect and risk
- Bit-identical outputs (same bytes into the same slots, same valid protocol).
- Expected: GU wait on even columns falls to the odd columns' level; GU -3..-10% per mean core, most on
  kimi-k3 / dsv4-flash / mm3 / gpt-oss (largest wrap/no-wrap gaps). Part of it may be conserved into DN on the
  DRAM-bound shapes, as r06-b04-a01 saw the other way. Guess +1..+3% geomean.
- Risks:
  - Reader serialization on top senders: the multicast is now on the reader's NoC, so the reader's next read
    issue waits for the multicast issue (not its drain). If GU gets slower on top-sender columns only, that is it.
  - NoC-0 column links carry the moved multicasts (model: 16-24 -> 21-29 GB/s, still well under the DRAM
    region links). The 10-row layout's NoC-0 column links rise to ~52 in the model (all 11 multicasts from
    y2); none of the benchmark shapes' plans use it as far as I know.
  - Hang if the reader and writer disagree on `wmc_gu` or `mc_seq`; both compute it from the same
    (is_sender, my_mt, group_rows).
  - Kernel-config size on h7168 (few hundred bytes of headroom): the reader gains a multicast path, paid by
    the shared helper.
- How to tell: per-row / per-column GU means vs r06-b04-a01 (even vs odd columns), mean / max core end.

## Addendum (before the eval): size
The first local build was reader +1.8 KB (by-reference captures pinned locals, and a captured Noc made its id a
runtime value). Final form: out-of-line helpers with by-value captures and a locally built Noc, shared by every
multicast of the same kind: reader column multicast + valid (gate, up, down), reader row multicast + valid
(x owner, activated sender), writer column multicast + valid (gate, up, down). Same operations in the same order
at every call site. Local -O2: reader -108 B, writer -276 B vs the parent.

## Addendum 2 (after run 1): repair of a hang
Run 1 hung on the first shape's first call (eval `hang`, no profile). Cause, found in the BH NoC sanitizer
(`debug_sanitize_check_linked_transactions`, tt_metal/hw/inc/internal/debug/sanitize.h): "Submitting a non-mcast
transaction if there's a linked transaction on any cmd_buf will cause a deadlock." Step 3 of the mechanism issued the
look-ahead DRAM reads of block kb+1 between the linked gate/up data multicasts and the (unlinked) valid-sem multicast
that closes the link, on the same NoC 0. Repair: the look-ahead is tried only after the valid sem, exactly where the
existing code tries it ("Block kb is pushed"), so the next read now waits for the multicast data to leave the core.
That is the serialization r06-b04-a01 warned about, reduced to the multicast drain only (the read itself was already
issued before the acks). No other change; this is the scored run.
