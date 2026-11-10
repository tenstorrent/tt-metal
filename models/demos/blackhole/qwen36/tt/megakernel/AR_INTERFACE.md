# Megakernel AR: integration interface (spec v0.8.7, AR track)

Reference implementation: `oneshot_ar.py` `OneShotARP` + `kernels/mk_ps_{reader,writer,receiver,reduction,producer}.cpp`
(persistent: N all-reduces per program, fabric connections opened once). Tests: `tests/test_mk_oneshot_ar.py`
(`test_mk_oneshot_ar_persistent`, `_progressive`, `_rows`). Results: `AR_TRACK.md`.

Measured (d26u02, 1x8 ring, exact, identical on all 8 devices, persistent, burst release, rot order):
t0 (last producer release) -> x in CB_X: **R=32 20.2 us, R=4 11.05 us** (10.46 us with packed producer input, INPK).
Sum(x^2) partials land +0.5 (R=4) / +1.1 (R=32) us after x and are off the path (deferred RMSNorm, pack time).

## 0. Cores, roles, numbering

| Role | Logical cores | Phase type |
|---|---|---|
| Owner j = 2k + slot (chunk j = x tiles 10j..10j+9) | (10 + slot, k), k = 0..7 | 7 AR_RECV |
| Link senders, one per (link, direction) | (10,8) fwd L0, (11,8) bwd L0, (10,9) fwd L1, (11,9) bwd L1 | 6 AR_SEND |
| K-row k consumers (CB_X, CB_RSTD) | STREAM (0..9, k) | (MLP / PROJ phases) |

- Ring: 8 devices on cluster axis 1, FABRIC_1D_RING, 2 links per neighbour pair (all of them; nothing more is cabled).
- Chunk j's **owning device** is `j // 2`. Every device's core j holds that device's partial of chunk j; the owning
  device reduces it, every device ends with x chunk j on core j and in K-row j // 2's CB_X.
- Routing per device `d`: forward distance `nf = 4, nb = 3`, swapped for even `d` (`OneShotAR._dist`). Owner chunk
  link: `link_of(j) = (j % 2 + j // 2) % 2`.

## 1. Producer side (MLP / wo -> AR)

| Item | Value |
|---|---|
| Input buffer | **CB_XO on owner core j**, uniform L1 address: 10 tiles bf16 = this device's sum of the down (or wo) partials of chunk j. **No residual.** |
| R < 32 | rows 0..R-1 only. Either tile layout (the AR packs: 2 runs of R x 32 B per tile) or, with INPK=1 (v12 default), the **packed v0.8.2 stream at the CB_XO base**: tile t face f at `(2t + f) R 32` B, `640 R` B total |
| Release | the producer's dataflow kernel, once chunk j is complete: `noc_semaphore_inc(READY target, 1)`; target (x, y, sem address) from AR runtime args: the RS sender of (link_of(j), direction) for the 14 remote-owned chunks, core j's own AR receiver for the device's 2 owned chunks |
| Order | per device: chunks whose owning device is farthest around the ring first, own chunks last (FIRST / §15 rot). The AR reads the releases of a sender in its list order (count-based READY) |
| Buffer lifetime | CB_XO of layer L must stay intact until x chunk j of layer L is delivered (x flag >= L + 1 on K-row j // 2): both the remote RS read (sender) and the local copy (owner) happen before that |

Residual: added **once**, by the AR, in the owner reduce on chunk j's owning device (decision §15 v0.8.7). The owning
device keeps the bf16 result (packed PT pseudo-tiles at R<32) as the next layer's residual in the AR output buffer;
no other device or phase needs a residual copy.

## 2. Consumer side (AR -> next phase)

**One delivery path: the AR delivers** (v12's owner `O-X` step is removed in the engine; MK_PS_DELIVER=2 == v12
OPACK output format). After chunk j is in core j's output (owner reduce, or all-gather arrival), core j's AR receiver
multicasts it west in-row (NOC1) to K-row k = j // 2 (2 physical rectangles split at the DRAM column):

| R | Destination on each K-row core | Layout |
|---|---|---|
| 32 | CB_X `xbuf + slot x 20 KB` (10 tiles) | tile layout |
| < 32 | `xpk + slot x XPK_SLOT_B` (XPK_SLOT_B = 10 x 2 x 16 x 32 = 10240 B) | packed v0.8.2 stream; **the consumer unpacks locally** (owner-side unpack forbidden) |

then `x flag[slot]` (global-semaphore word on STREAM + owners, uniform address) is multicast to the same rectangles;
value = cumulative count (L + 1). Consumers wait `>= L + 1`.

**Sum(x^2) (CB_RSTD, per K-row core, uniform address, fp32):** after x, core j sends its chunk's row statistic
`P_j[r] = sum_{c in chunk j} x[r, c]^2 * 2^-12` to K-row k: 32 fp32 at `CB_RSTD + 128 x slot`, then a tag word
`L + 1` at `CB_RSTD + 2048 + 16 x slot`. For R < 32 the values are **already folded** (row r < R; entries >= R are 0;
R in {4, 8, 16, 32}, round up). Consumers: rstd' = rsqrt(sum over the 16 chunks + 1.25 eps) = rstd / NORM_FIX
(gamma x NORM_FIX folded into the weights). In v12 each core forwards its K-row's 2 partials to the column's RS rows.

## 3. Phase-table form

AR_SEND (type 6, the 4 sender cores) and AR_RECV (type 7, the 16 owners) records (16 u32 words, §3 table):

| Word | AR_SEND (6) | AR_RECV (7) |
|---|---|---|
| 0 | type 6 | type 7 |
| 1 | layer L (flags compare against L + 1) | layer L |
| 2 | R (32 or 4/8/16) | R |
| 3 | n_rs (chunks this sender reduce-scatters) | role: 2 owner (chunk owned by this device) / 1 gather |
| 4 | base of this sender's RS list in the per-pass arg block (per chunk: dest core xy, hops / direction, source core xy) | base of the owner arg block (sender xys, input segment, slot order) |
| 5 | n_ag (own chunks of this link, <= 1) + AG core xy | x destination address (xbuf / xpk + slot offset) |
| 6 | RS slot base on owners (160 KB region, uniform) | 2 row rectangles (index into the arg block) |
| 7 | CB_XO address (RS source) | CB_XO address (own partial) |
| 8 | AR output / residual buffer address | AR output / residual buffer address |
| 9-12 | semaphore addresses: READY, AG_READY, DONE, RS arrival | semaphore addresses: RS arrival, AG arrival, x flag[slot], CB_RSTD base |
| 13-15 | reserved (BFP8 flags: RS / AG) | reserved |

- **Fabric connections:** opened once per pass by the AR_SEND phase of layer 0 (`open_start` / `open_finish` with
  the packet-header CB), kept open across all AR phases and every other phase, closed after the last AR of the pass.
  One worker connection per (link, direction) (1D router limit: 2 sender channels), so no other phase may use these
  4 connections.
- **Slot region:** one dedicated 160 KB region on the owners (8 slots x 10 tiles x 2 KB; PT pseudo-tiles per slot
  at R < 32), shared by the attention-AR and the MLP-AR (dependency chain, §15), never aliased with SDPA / EXT.
- **Semaphores:** all global-semaphore words (uniform addresses), **monotonic counts, >= waits, reset only at
  pass end** (v0.8.7 rule; equality waits + per-iteration resets caused the HOPSEL hang):
  - READY (senders): `n_rs x L + i + 1` before reading RS chunk i; owner-local READY `L + 1`.
  - RS arrival (owners): `7 (L + 1)` (fused fabric write + increment).
  - AG_READY (senders): `n_ag (L + 1)`; AG arrival (gather cores): `L + 1`.
  - DONE (senders): `16 L` before layer L's RS (every output core of the device holds layer L - 1's chunk; protects
    the slot region and the AG staging).
  - x flag[slot], CB_RSTD tags: `L + 1`.
- **Per-layer runtime fields:** R and L only; everything else is static per pass.

## 4. Variants kept behind flags (default off)

| Flag | Effect | Status |
|---|---|---|
| MK_AR_RS_BFP8 / MK_AR_AG_BFP8 | BFP8 partials / all-gather payload | -3.2 / -2.3 us (-5.7 both); numerics -> output-length gate |
| MK_PS_SRED | reduce on arrival | 20.65 us (worse) |
| MK_PS_AGD | owner writes AG staging directly | 18.30 us (no gain) |
| MK_PS_HDRROT | rotating packet headers | no gain |
| MK_CH_* (chain_ar.py) | chain reduce-scatter | 17.8 us at best (ties one-shot) |

## 5. Chained test status

`test_mk_oneshot_ar_rows` / `_progressive` chain N all-reduces in one program on the 8-device ring with a synthetic
producer (`mk_ps_producer.cpp`: READY releases in rot order) and full delivery + Sum(x^2), checked exactly per AR.
A v12-MLP -> AR -> v12-MLP chain needs v12's owner sum (TRISC) and the AR reduce (TRISC) on the same owner cores,
i.e. one phase-sequenced compute kernel per core: that is the engine integration itself (mixed3 / -mk-int-mlp), not
buildable as two separate generic_ops on shared cores. The v12 single-chip stand-in (owner waits READY, writes static
x + P_j to the same CB_X / xpk / CB_RSTD addresses and flags) is the swap point for these kernels.

## 6. Semaphore inventory and reset rules (v0.9.13 drain-then-reset, v0.9.18 epoch for cross-device semaphores)

Root cause of the 2026-11-10 intermittent AR hang (tt-triage + per-RISC progress words, hunt run 5): devices are NOT
synchronised between consecutive programs (trace or not). Fast devices started program k+1 and their RS writes + fused
increments reached a slow device's owner before that device's program-k fini zeroed the arrival semaphore; the reset
wiped 4 of 7 arrivals and program k+1 waited forever. Rule: a semaphore incremented from ANOTHER DEVICE is never reset;
it carries a persistent per-core base word (EPOCH). Device-local semaphores may be drained and reset in fini (a device's
next program cannot start before its own previous one ends).

| Semaphore (core set) | Incremented by | Waited by | Per-AR target (AR index it) | Pass end |
|---|---|---|---|---|
| RS arrival `sem` (owner of chunk j, owning device) | 7 remote link senders (fused fabric write + inc) | owner receiver | base + 7 (it + 1) | **EPOCH**: never reset; base += 7 N |
| AG arrival `sem` (core j, the 7 other devices) | the owning device's AG multicast | gather receiver | base + (it + 1) | **EPOCH**: never reset; base += N |
| AG_READY (link senders) | owner receivers of this device | sender reader | n_ag (it + 1) | drain (n_ag N), reset |
| DONE (link senders) | the 16 output cores of this device | sender reader (before AR it + 1's RS) | 16 it | drain 16 N at the end, reset |
| READY (link senders; engine producer) | MLP / wo owner DM of this device | sender reader | n_rs it + i + 1 | drain n_rs N, reset |
| READY (owner, own chunks) | own producer | owner receiver | it + 1 | drain N, reset |
| GO (owners; synthetic producer only) | master sender reader of this device | producer | it + 1 | drain N, reset |
| x flag[slot] (STREAM + owners) | owner receiver (`f = f + 1`, multicast) | consumers | it + 1 | consumers' concern; never reset |
| CB_RSTD tag words (STREAM, in-buffer) | owner receiver | consumers | it + 1 | written, not counted |

- **EPOCH base word:** one global-semaphore word per arrival semaphore (same core set, uniform address), shared by every
  program that uses that arrival semaphore (in the engine: one per AR phase's arrival semaphore, allocated in the AR
  region). `init` reads `base`, every arrival wait is `(int32_t)(sem - (base + target)) >= 0` (wrap-safe: ~900
  increments per pass, 2^32 wraps after hours of serving), `fini` writes `base + wait_value x ARs done` and leaves the
  semaphore alone. Implementation: mk_ar_recv.hpp (`MKAR_SEM_BASE`, `mkar::wait_ge`), host `MK_PS_EPOCH=1`.
- **Gather-side release order (v0.9.12):** DONE is signalled only after the Sum(x^2) compute has read the chunk (the
  next all-gather overwrites it).

## 7. Engine wiring v1 (MK_REAL_AR=1, bench_mka2.py; MLP AR only)

- Per-device programs (`PROG_PER_DEV`): `ArPhase.engine_parts` adds the AR_SEND kernels on the 4 AR-only cores, their CBs
  and fabric-connection semaphores, the RS slot CB (a plain program CB on owners + senders, not an L1 tensor: the L1
  allocator is lockstep over all cores), and the owner CBs (OUT = RES = the residual stream, in place).
- Owner writer (EXT binary), at the v12 `AR_RECV(L)` hook after W-OREL: READY incs to the chunk's RS sender (or the own
  receiver) -> `mkar::Recv::iter(L)` -> +1 on the Q root's delivery count (`MK_AR_XCNT`). `fini()` at the end.
- Owner compute (EXT binary): `mkar::Reduce::iter()` right after C-OSUM (owning chunk only).
- x of layer L + 1 goes into the K-row slot (v_xk at R = 32, v_xpk packed at R < 32) with its own flag pair
  (`MK_AR_XF0/XF1`); the v12 stand-in delivery (v12_ox) keeps feeding the MLP after the attention / GDN AR stall.
- STREAM W-ARSTALL -> W-ARWAIT: wait this row's two flags >= L + 1 (still pumping the next attention MLP slice);
  the Q root waits 16 (L + 1) deliveries, then XRDY to all 3 full-grid rects (`mcast_word<NRF>`). The GDN-branch
  early XRDY is off under MK_REAL_AR.
- R = 32 (u8): `own_cb` / `MKAR_NOOWN`: no own slot in CB_RED (ring - 1 slots, -20 KB on the owners); the reduce adds
  the own partial in place from the producer CB (v12 CB_VXO); the senders get the destination slot in route bits 16+.
  The residual stream is v12's `v_x0` (unused with MK_OWNER_RESID=0). Owner CB budget is otherwise ~50 KB short.
- CB ids 50-55 grow every core's CB-config block; with the compute zones the u8 STREAM program was 192 B over the
  70656 B kernel config buffer, so MK_REAL_AR turns on MKA_LEAN_C.
- v1.1 (spec v1.21): x delivery moved to the owner READER on NOC1 (`MKAR_DLV_SIGNAL`: AR_RECV posts `MK_AR_DLV` = L + 1
  once the output chunk is complete; mka2_reader.cpp `v12_ardlv` multicasts it west inside the row, sets the row flag
  and counts on the Q root); the receiver's role block drops its delivery tail (`deliver_ext`). CB ids compacted to
  v12 STREAM-only ids 30/34-41/43/44 (`cb_ids`): the local CB-config block of every core is sized by the program's max
  CB index (16 B per index, dispatch.cpp finalize_cbs), so MKA_LEAN_C is no longer needed (`MK_AR_CBHI=1`: old ids).
- The pre-attention MLP AR (40-54 us vs ~21) is the next attention layer's MLP-slice pump (19.4 MB) sharing the NoC:
  the all-gather landing stretches, not the delivery. Rate-limiting the pump (`MK_AR_PUMPGAP`) or holding it
  (`MK_AR_NOPUMP`) restores ~21 us but the attention layer then loses 35-80 us; full-rate pumping is best.
- v1.2 (spec v1.25, engine rules: CB ids < 45; bulky per-core args out of runtime args): the owners' AR args (READY
  targets, Reduce [has_work, ring, PT], Recv common + role blocks) are a per-core L1 table, `ArPhase.arg_table(parts)`
  (uint32, width-sharded over the owners, 64 words / core, per-device contents via ShardTensorToMesh; layout
  [ar_on, n_ready, (xy, READY) x n | comp @16 | recv @20]). Kernels: `MKAR_ARGS_L1` (Recv carg / rarg),
  `Reduce::init_l1`, engine defines `MK_AR_BLK` / `MK_AR_BLK_COMP` / `MK_AR_BLK_RECV`. The owner writer / compute
  carry no AR runtime args. (The AR-only sender kernels keep theirs: the fabric connection args are runtime args, and
  the sender group is not the program's largest.)
- v1.3 MK_AR_GAMMA (spec §19, default off): the AG carries the un-scaled residual; every owner (owning + gather role)
  computes x_deliver = out ⊙ γ_next (layer L + 1's input_layernorm, √1.25 folded) in compute right after the AR:
  R = 32 bcast-rows multiply with the row-0 γ tiles; R < 32 element-wise with a host-built packed, row-replicated γ
  table (one page per (layer, norm, chunk)). γ lands in CB 32 = pbuf+0 (dead from C-OSUM(L) until the next MLP's
  partials; never across ARs: real wo aliases A / S there), read on trid 14 right after the READY incs (issuing it
  before them stalled every device's RS under the weight pump). x_deliver goes into v_xo (dead after the AR; the
  writer's pop of the partial gates the compute's reserve), then the writer posts MK_AR_DLV and the NOC1 reader
  delivers it. The γ table's DRAM address is arg-table word 19 (runtime data). Recv: MKAR_DLV_NONE; its own-chunk
  reads use MKAR_TRID (15) instead of a global read barrier. MK_AR_GEARLY (u1 only, L1): a dedicated γ CB filled at
  the MLP start (no gain measured).
