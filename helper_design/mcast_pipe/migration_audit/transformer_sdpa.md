# Migration audit — group: transformer + sdpa

Scope dirs swept:
- transformer/sdpa/device/kernels/dataflow/ (all .cpp + .hpp)
- transformer/sdpa_decode/device/kernels/dataflow/ (all)
- experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/ (all)

Rectangle-mcast + handshake Pipe block found in **5 files**, of which 3 are real refactor
targets, 1 is the existing abstraction (reference), 1 is marginal/defer.

## Per-kernel verdicts

| File | Block? | Verdict | Cost / Why |
|---|---|---|---|
| sdpa/.../chain_link.hpp | YES (mcast `forward`+`receive`) | **REFERENCE / clean** | Already the abstraction. Pipe should converge to this. Used only by ring_joint_reader.cpp. Unit = header. |
| sdpa/.../ring_joint_reader.cpp | YES (via ChainLink) | **CLEAN** | Already migrated to ChainLink (call sites L399-417, L475-492). No raw block. Validates the API shape. |
| sdpa/.../reader_interleaved.cpp | YES x2 (K,V) | **REFACTOR (medium)** | Open-coded duplicate of chain_link. Identical protocol. Blocker: cb_push timing differs between mcast/unicast paths; linked-companion ordering must be preserved; mask/Q reads interleaved between forward halves. |
| sdpa/.../exp_ring_joint_reader.cpp | YES x2 (K,V) | **REFACTOR (high)** | Open-coded duplicate, but embedded in ring/all-gather loop with a co-located `wait_min` ring sync (L271,L430) that is OUT OF SCOPE. Blocker: must cleanly separate Pipe (chain valid/sender sems) from ring per-link sems + mux-writer fabric path. |
| sdpa_decode/.../dataflow_common.hpp | YES (in `read_k`) | **REFACTOR (medium)** | DIFFERENT DIALECT: F1=BARRIER, linked=false, pre_handshake=NO, vertical column, INCLUDE-src. Unit = header function read_k. Blocker: the F1 fork (barrier vs flush) and linked=false vs linked=true diverge from the chain family — needs dual-path or a deliberate pick. |
| experimental/.../all_reduce/worker_writer.cpp | partial (sem-only rectangle L166-184) | **DEFER-RAW** | Data path is FABRIC mcast (cross-chip, different family). Only a data-less sem rectangle mcast is local; multi-range "charge dests once" + fabric coupling make it a poor target. |
| experimental/.../all_reduce/reduction_receiver.cpp | receiver of above | **DEFER-RAW** | Lone sem wait+reset (L91-92); pairs with the deferred worker_writer sem mcast. |
| experimental/.../all_reduce/worker_reader.cpp | NO | n/a | No mcast/handshake primitives. |
| sdpa/.../joint_reader.cpp, ring_joint_writer.cpp, exp_ring_joint_writer.cpp, joint_writer.cpp, writer_interleaved.cpp | NO rectangle mcast | n/a | Only barriers / unicast sem inc / fabric. exp_ring_joint_writer is fabric-MUX; ring_joint_writer uses trid barriers. OUT OF SCOPE. |
| sdpa_decode/.../writer_decode_all.cpp | NO | n/a | Unicast `noc_semaphore_inc` reduce-tree signalling (L341,L430), no rectangle mcast. OUT OF SCOPE. |
| sdpa/.../dataflow_common.hpp (sdpa, not decode) | NO mcast | n/a | Only `noc_async_write_barrier`/`flushed` in generic write helpers. No rectangle mcast. |
| sdpa/.../fused_op_receiver.hpp | NO | n/a | `wait_min` fused-op signal only (L43). Ring sync, OUT OF SCOPE. |

## Counts
- Files swept (3 dirs): 22 kernel/header files.
- Files containing the rectangle-mcast Pipe block: **5** (chain_link.hpp, reader_interleaved.cpp,
  exp_ring_joint_reader.cpp, sdpa_decode/dataflow_common.hpp, + marginal all_reduce/worker_writer.cpp).
- Distinct block instances (send+receive pairs): 7
  (chain_link 1 generic ; reader_interleaved K+V = 2 ; exp_ring_joint_reader K+V = 2 ;
   read_k 1 ; worker_writer sem-only 1).
- CLEAN/reference: 2 (chain_link.hpp, ring_joint_reader.cpp).
- REFACTOR: 3 (reader_interleaved medium, exp_ring_joint_reader high, read_k medium).
- DEFER-RAW: 2 (all_reduce worker_writer + reduction_receiver — fabric family).

## Headline blockers
1. **Two F1 dialects in one group.** sdpa chain family = FLUSH + linked=true (companion pair, NO
   barrier allowed between data and sem). sdpa_decode read_k = BARRIER + linked=false (barrier MUST
   separate data and sem). These are mutually exclusive contracts — the tune-helper must pick one or
   dispatch on a predicate. BH hang workarounds (#19201 elsewhere in this op, writer_decode_all L252)
   suggest BARRIER may be load-bearing on Blackhole → bake-off should cover BH.
2. **Two pre_handshake modes.** chain family = pre_handshake YES (dest reused, sender waits on
   sender_sem before write). read_k = pre_handshake NO (fresh slot, CB back-pressure only). Both must
   be expressible.
3. **The block lives in shared headers in 2 of 5 cases** (chain_link.hpp, sdpa_decode/dataflow_common.hpp).
   Migration unit is the header, and changes ripple to the single consumer each — lower blast radius
   but the API must satisfy both header shapes.
4. **Co-located non-Pipe sync** in exp_ring_joint_reader (ring wait_min) and worker_writer (fabric
   barrier) must not be absorbed into Pipe — clean separation required.
