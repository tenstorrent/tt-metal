# quasar/matmul mcast call sites — annotation (added by reconcile 2026-06-27)

**Group decision (recorded in ledger):** all 8 files → `status=deferred`, flag `quasar-metal2-port`
(the `..._block_sharded_metal2.cpp` file additionally carries flag `hang:#47797`). The subtree is an
**actively-churning Metal-2.0 API port** of the production matmul readers already censused as P1–P4;
migrating it now would triple the surface for the same four patterns with **zero new hazard coverage**.
Each file keeps its **intrinsic eventual target tag** below (clean ×those / refactor ×those /
defer for in1_ring_all_gather) so the durable record points at the right migration when the port stabilizes.

All 8 live under `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/`.
**All use the modern OBJECT API** (`Noc`, `Semaphore<>`, `MulticastEndpoint`, `NocOptions::MCAST_INCL_SRC`,
`CoreLocalMem`) — no raw free functions. `op_family: matmul` for every entry.

## `_metal2` relationship (host-binding-only fork; COEXISTS, not a replacement)
Each `_metal2` file is a fork of its non-metal2 sibling whose **algorithm body is byte-for-byte identical**
(stated verbatim in every metal2 header). The *only* delta is the host-binding surface:
- positional `get_compile_time_arg_val(N)` → named `get_arg(args::name)`
- positional `get_arg_val<uint32_t>(i)` → named RTAs
- in0/in1 tensor RTA + `TensorAccessorArgs` → typed `tensor::` binding
- named CB-index CTAs → `dfb::` tokens
- `Semaphore<>(get_compile_time_arg_val(id))` → `Semaphore(sem::name)` (via `SemaphoreBinding`)

The entire mcast+handshake structure (F1 fence / F2 staging / F3 loopback / pre_handshake KNOB) is
**unchanged** across each pair. Both generations are **live and co-referenced inside the SAME factory files**
`factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` and `..._2d_program_factory.cpp`: the metal2
forks are dispatched from the Metal-2.0 host-API path (mcast_1d ~L5182-5197, mcast_2d ~L3106-3121), while the
legacy-binding readers still serve the not-yet-ported sibling paths plus `sparse/` and `dram_sharded`.
⇒ From `mcast_pipe`'s perspective each metal2/non-metal2 pair is **one migration pattern with two
host-binding skins**, classified identically to its production census twin.

**Live DEBUG scaffolding (in-flux tell):** the metal2 forks carry hang-localization scaffolding that a
migration would race against — `reader_bmm_tile_layout_in0_sender_padding_metal2.cpp` includes
`api/debug/dprint.h` with a `// DEBUG: matmul layer3 hang localization (remove after)` note (L34); the
block_sharded_metal2 fork carries `[DEBUG #47797]` sender/receiver hang-localization comments (L372-375,
L127-128, L178-181) → flag `hang:#47797`.

---

## 1. reader_bmm_tile_layout_in0_sender_dram_sharded.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp`
- **Role:** HYBRID — three runtime-selected core types in one binary (sender-no-compute / sender∈rect / pure-receiver).
- **Tag (eventual):** refactor.  **metal2?** no (no metal2 fork emitted for the DRAM-sharded path).
- **is_call_site:** yes.  Twin = census P2 `refactor`.
- **Fork signature:**
  - **F1:** FLUSH, conditional — type-1 has NO flush between data and flag mcast (relies on same-noc/vc in-order); type-2 adds `noc.async_writes_flushed()` (L203). Final `async_write_barrier()` + `async_atomic_barrier()` (L236-237, atomic-barrier for the `sem.up` counter path).
  - **F2:** LEVEL FLAG with per-type INVALID reset. `receiver_sem.set(VALID)` pre-loop (L67); type-2/3 `set(INVALID)` per-iter (L147, L227); `wait(VALID)` (L213, L231). R→S side is a COUNTER: receivers `sender_sem.up(...)` (L210, L229), sender `sender_sem.wait(num_dests)` + `set(0)` (L87-88, L154-155).
  - **F3:** BOTH loopback modes in one file — type-1 EXCLUDE_SRC `noc.async_write_multicast(...)` (L115); type-2 INCLUDE_SRC `noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(...)` (L178) + INCLUDE_SRC flag (L194).
  - **KNOB pre_handshake:** YES — `sender_sem.wait` precedes data mcast (L87 / L154).
- **Migration-blocker audit:** H6/H7/H8 (mixed counter R→S + both loopback modes + per-type num_dests). Hand-rolled double-buffer of the mcast source addr (source ≠ `cb.get_write_ptr()`). Refactor-grade, same as twin.

## 2. reader_bmm_tile_layout_in0_sender_padding.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`
- **Role:** SENDER (data + flag mcast). Canonical P1 in0 sender.
- **Tag (eventual):** clean.  **metal2?** no — legacy-binding sibling of #3, COEXISTS.
- **is_call_site:** yes.  Twin = census P1 `clean`.
- **Fork signature:**
  - **F1:** FLUSH. Data mcast then flag mcast same noc/vc/cmd_buf in-order (comment L371); `noc.async_writes_flushed()` (L200, L377); final `async_write_barrier()` once at end (L424).
  - **F2:** LEVEL FLAG. `receiver_sem.set(VALID)` once (L157); R→S level `sender_sem.wait(in0_mcast_num_dests)` + `set(0)` reset (L190-191, L352-353).
  - **F3:** EXCLUDE_SRC (default `noc.async_write_multicast`, L358) + local self-fill. num_dests excludes self.
  - **KNOB pre_handshake:** YES — `sender_sem.wait` precedes data mcast (L352 before L358).
- **Migration-blocker audit:** Clean. Note R2 generality: a SECOND flag-only mcast (sparsity batch-valid, L190-202, `receiver_sem.set(VALID/IGNORE_BATCH)` + `set_multicast` (L193) + flush (L200) — no data payload) — the Pipe `send()` must support a data-less / flag-only mode. SKIP_MCAST compiles the whole block out (Pipe-of-width-1).

## 3. reader_bmm_tile_layout_in0_sender_padding_metal2.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp`
- **Role:** SENDER. Same canonical P1 sender as #2.
- **Tag (eventual):** clean.  **metal2?** YES — body byte-identical to #2; host-binding only (named `sem::in0_sender`/`sem::in0_receiver` (L126-127), `dfb::` CBs, `tensor::in0`). COEXISTS with #2 (Metal-2.0 factory path dispatches here).
- **is_call_site:** yes.  Twin = census P1 `clean`.
- **Fork signature:** identical to #2 — FLUSH (`async_writes_flushed` L201, L372; final barrier L415); LEVEL FLAG (`receiver_sem.set(VALID)` L160; `sender_sem.wait`+`set(0)` L191-192, L350-351); EXCLUDE_SRC (`noc.async_write_multicast` L356); pre_handshake YES (L350 before L356). Same R2 flag-only batch-valid mcast (L193-203).
- **Migration-blocker audit:** Clean (inherits #2). **In-flux tell:** DPRINT "matmul layer3 hang localization (remove after)" (L34) → flag `quasar-metal2-port`. With #8, one of the two safest eventual entry points.

## 4. reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
- **Role:** HYBRID — the R6 rotating-role STAR: one binary that is, per block, EITHER sender OR receiver, selected by `block_id == sender_id`; loopback when sender ∈ rect.
- **Tag (eventual):** refactor (high cost; the generality stress test).  **metal2?** no — legacy sibling of #5, COEXISTS.
- **is_call_site:** yes.  Twin = census P3 `refactor`.
- **Fork signature:**
  - **F1:** FLUSH, conditional — `noc.async_writes_flushed()` (L334) skipped only when rect core && num_cores==1; comment L327-332 is the clearest H1/H4 statement (flag mcast reads `receiver_sem`'s L1 as source; without flush the CPU overwrites it to INVALID before the NoC reads VALID → receivers hang). Final `async_write_barrier()` (L360).
  - **F2:** LEVEL FLAG. `receiver_sem.set(VALID)` pre-loop (L103); per-iter `set(INVALID)` for rect cores (L141); sender re-`set(VALID)` before flag mcast (L286, L314); receivers `wait(VALID)` (L343). R→S: `sender_sem.wait(num_dests-1 | num_dests)` (L224/L227) + `set(0)` (L229); receivers `sender_sem.up(...)` (L338).
  - **F3:** ALL THREE PATHS — EXCLUDE_SRC same-CB `num_cores-1` guarded by `num_cores>1` (L240); plain unicast `noc.async_write` degenerate `num_cores==1` (L259-266, INV5/H5); INCLUDE_SRC loopback `<NocOptions::MCAST_INCL_SRC>` (L270) + INCLUDE_SRC flag (L288); sender∉rect EXCLUDE_SRC `num_cores` (L300) + plain flag (L315). Explicit comment "noc_async_write_multicast[_loopback_src] may hang if called with 0 cores" (L237).
  - **KNOB pre_handshake:** YES — `sender_sem.wait` precedes data mcast (L224/227 before L232+).
- **Migration-blocker audit:** The sharpest case. **H12-star (M12b):** rotating-role STAR — the core's `data_ready` cell is mutable (its receiver turn clobbers it INVALID), so it needs a per-send re-assert of VALID before each flag `set_multicast` (the kernel does `receiver_sem.set(VALID)` at L286/L314). Per-block remote-sender coord table (`remote_sender_noc_x/y[block_id]`, used at L338) = rotating sender identity (R6); a fixed sender-object/receiver-object Pipe does NOT fit. Plus degenerate-unicast (H5) and send-only-core CB lockstep. Refactor (high cost) or defer, same as twin.

## 5. reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp`
- **Role:** HYBRID — same R6 rotating-role STAR as #4.
- **Tag (eventual):** refactor.  **metal2?** YES — body identical to #4 (rotating-role / all-3-loopback-paths / degenerate-unicast); host-binding only. COEXISTS with #4 (Metal-2.0 factory path dispatches here).
- **is_call_site:** yes.  Twin = census P3 `refactor`.
- **Fork signature:** identical to #4 — FLUSH conditional (`async_writes_flushed` L370; final barrier L420); LEVEL FLAG (`receiver_sem.set(VALID)` L123, per-iter `set(INVALID)` L195, sender re-`set(VALID)` L332/L359, receiver `wait(VALID)` L394); F3 all three (EXCLUDE_SRC L288, INCLUDE_SRC L316, plain flag L360, `sender_sem.up` L389); pre_handshake YES (`sender_sem.wait` L275/L278 + `set(0)` L280).
- **Migration-blocker audit:** Same H12-star/M12b + R6 rotating-sender + H5 degenerate-unicast as #4. **In-flux tell:** `[DEBUG #47797]` sender/receiver hang-localization comments (L127-128, L178-181, L372-375) → flags `quasar-metal2-port` + `hang:#47797`. This is the file being actively hang-debugged — strongest reason the subtree is deferred this round.

## 6. reader_bmm_tile_layout_in1_ring_all_gather.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp`
- **Role:** SENDER (borderline). Mostly DRAM-read + remote-CB; contains a degenerate flag-only mcast in `do_signaling()` (L46-70).
- **Tag (eventual):** defer.  **metal2?** no.
- **is_call_site:** yes, borderline (flag-only, no data half).  Twin = census P4 `defer`. Byte-identical body to the production twin.
- **Fork signature:**
  - **F1:** BARRIER — `noc.async_atomic_barrier()` after the signal (L68), plus caller `early_noc.async_write_barrier()` (L93) / final `async_write_barrier()` (L219) / `async_atomic_barrier()` (L217). No flush.
  - **F2:** COUNTER on the gather (`pv_sem.wait(target_sem_value)` L61), then LEVEL flag for the GO (`sig_sem.set(1)` L63 + `set_multicast(...)` L64). Hybrid.
  - **F3:** plain `set_multicast` (semaphore mcast, no INCLUDE/EXCLUDE choice). Privileged core sets own `sig_sem=1` then mcasts (loopback-by-local-set). Non-privileged: `pv_sem.up(...)` + atomic-barrier (L67-68).
  - **KNOB pre_handshake:** N/A — one-time op-level start barrier, not a per-block dest-reuse handshake.
- **Migration-blocker audit:** Flag-only collective start-barrier with NO data-block half → defer (R2 flag-only; a Pipe `send()` flag-only mode could express it but the lifecycle is a one-shot collective barrier, different from the producer/consumer data Pipe). `experimental::remote_cb_*` (out of family). Same verdict as P4 twin.

## 7. reader_bmm_tile_layout_in1_sender_writer_padding.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- **Role:** HYBRID — SENDER (reader portion, the P1 block appears TWICE: in1 + in3/bias) co-resident with an out-of-family WRITER portion in the same file.
- **Tag (eventual):** clean.  **metal2?** no — legacy sibling of #8, COEXISTS.
- **is_call_site:** yes.  Twin = census P1 `clean`.
- **Fork signature (identical for both in1 and in3 instances):**
  - **F1:** FLUSH. `noc.async_writes_flushed()` (L433, L558); in-order-NoC comment (L426, L552); final `async_write_barrier()` (L677).
  - **F2:** LEVEL FLAG. `receiver_sem.set(VALID)` (L214); `sender_sem.wait(in1_mcast_num_dests)` + `set(0)` (L407-408, L534-535).
  - **F3:** EXCLUDE_SRC (default `noc.async_write_multicast`, L413, L540) + self-fill via local reads.
  - **KNOB pre_handshake:** YES — `sender_sem.wait` precedes each data mcast (L407 before L413; L534 before L540).
- **Migration-blocker audit:** Clean for the mcast block (two structurally identical P1 sender instances). The WRITER portion (`noc.async_write` + barriers L630, output drain) is out-of-family — a Pipe migration touches only the reader half. Both blocks `#ifndef SKIP_MCAST`-guarded.

## 8. reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp
`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp`
- **Role:** HYBRID — same two P1 sender blocks (in1 + in3/bias) as #7.
- **Tag (eventual):** clean.  **metal2?** YES — body identical to #7; host-binding only (named `sem::`/`dfb::`/`tensor::`). COEXISTS with #7 (Metal-2.0 factory path dispatches here).
- **is_call_site:** yes.  Twin = census P1 `clean`.
- **Fork signature:** identical to #7 — FLUSH (`async_writes_flushed` L309, L380; final barrier L453); LEVEL FLAG (`receiver_sem.set(VALID)` L207; `sender_sem.wait`+`set(0)` L285-286, L357-358); EXCLUDE_SRC (`noc.async_write_multicast` L295, L367); pre_handshake YES (L285 before L295; L357 before L367).
- **Migration-blocker audit:** Clean (inherits #7). Note the metal2 fork does NOT serve the in1 DRAM-sharded reader paths (factory comment ~L499 / the host factory routes those to the legacy reader). **In-flux tell:** `quasar-metal2-port`. With #3, one of the two safest eventual entry points.

---

## Group migration verdict
**In scope** (object-API matmul, same handshake family as the censused production matmul P1–P4), but
**defer-as-a-group this round** — recorded `status=deferred`, flag `quasar-metal2-port` (×8), `hang:#47797`
(×1, file #5). Three concrete reasons:
1. **Actively churning** — mid-flight Metal-2.0 port (recent "port experimental/quasar ops to metal 2.0 api"
   commits); the metal2 forks carry live hang-debug scaffolding (`[DEBUG #47797]`, DPRINT "remove after")
   a migration would race against.
2. **Pure duplication** — every pattern exists 2× (legacy + metal2 host-binding skin) PLUS the canonical
   production twin already in the census; these add **zero novel forks** over P1/P2/P3/P4.
3. **Highest-value file is itself only conditionally migratable** — #4/#5 (block_sharded) is the R6
   rotating-role STAR needing M12b, and #5 is the one being actively hang-debugged.

When the port stabilizes and the DEBUG scaffolding is stripped, the metal2 forks become the single live
target; the clean P1 senders (#3, #8) are the safest entry points.
