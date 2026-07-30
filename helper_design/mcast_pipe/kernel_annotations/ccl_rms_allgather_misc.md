# CCL misc senders (rms_allgather, all_to_all, llama_ag_matmul, selective_reduce_combine, all_gather_concat) — annotation

These are fabric CCL ops. The **intra-chip rectangle mcast leg** in each is IN SCOPE; the fabric/EDM cross-chip leg is OUT OF SCOPE (annotated only for the local block).

---
## rms_allgather/dataflow/rms_sender_reader.cpp — IN SCOPE (no fabric, pure intra-chip)
Two blocks:
1. **Reduce-sender sem handshake** (lines 80–85): set local sem VALID → `noc_semaphore_wait(receiver, num_blocks-1)` → reset → `noc_semaphore_set_multicast(reduce_sender_sem, noc_addr, num_blocks-1)`. F2=flag, F3=EXCLUDE_SRC, F1=none-here (handshake only).
2. **Post-reduce broadcast** (lines 130–155): `noc_async_write_multicast_loopback_src` (DATA, line 142–147) + `noc_semaphore_set_multicast_loopback_src` (SEM flag, line 133–134) + `noc_async_write_barrier` (135/148).
   - **F1 = barrier**, **F2 = flag** (VALID), **F3 = INCLUDE_SRC (loopback_src)** — sender is part of the receiver grid. **This is the loopback exemplar.**
   - pre_handshake = ABSENT.

## rms_allgather/dataflow/rms_writer.cpp — local mcast leg IN SCOPE (12 fabric refs)
`get_noc_multicast_addr` (97) + `noc_semaphore_set_multicast_loopback_src` (186). F3=INCLUDE_SRC, F2=flag, F1 surrounding barrier. Same profile as rms_sender_reader block 2.

---
## all_to_all_async_generic/all_to_all_sender_writer.cpp — local mcast leg IN SCOPE (37 fabric refs)
Lines 215–229 (core_id==0 && link_id==0): `noc_semaphore_wait(global_init_sem, num_devices-1)` → `noc_semaphore_set_multicast_loopback_src(local_init_sem, noc_addr, mcast_size, false)` → `noc_async_write_barrier` (228).
- **F1 = barrier**, **F2 = flag** (init sem), **F3 = INCLUDE_SRC (loopback_src)**. Sem-only mcast (no data block). pre_handshake = the `wait(num_devices-1)` is a cross-device gather (fabric side); local mcast is the release.

## selective_reduce_combine/reader.cpp + writer.cpp — local mcast leg IN SCOPE (28 fabric refs in writer)
- reader.cpp lines 108–119: `noc_semaphore_wait(sync_sem,1)` → `get_noc_multicast_addr(end,end,start,start,...,noc=1)` (**NOC1 coord swap**) → `noc_semaphore_set_multicast(..., bbox_size-1, linked=false, noc=1)` → `noc_async_writes_flushed(noc=1)`.
  - **F1 = flush**, **F2 = flag**, **F3 = EXCLUDE_SRC** (bbox_size-1). Sem-only. Receiver: `noc_semaphore_wait_min` (line 122).
- writer.cpp lines 240–244: same `noc_semaphore_set_multicast` on noc=1 (init sem). Same profile.

## llama_all_gather_matmul_async — local mcast leg IN SCOPE (no fabric in these files)
- reader_bmm_..._in1_ring_all_gather.cpp lines 54–58: `get_noc_multicast_addr | pv_semaphore` → `noc_semaphore_set_multicast(pv_semaphore, addr, num_signalling_semaphores)`. F2=flag, F3=EXCLUDE_SRC, sem-only.
- worker_receiver.cpp lines 47–78 (see separate detail): set local sem VALID (47) → `noc_semaphore_wait_min` gates (55,62) → **DATA** `noc_async_write_multicast_loopback_src` (72–73, INCLUDE_SRC) → **SEM** `noc_semaphore_set_multicast_loopback_src` (76–77) → `noc_async_write_barrier` (78).
  - **F1=barrier, F2=flag(VALID), F3=INCLUDE_SRC**, pre_handshake = PRESENT (wait_min on next-core's sem before mcasting — ring ordering).

## all_gather_concat_heads_fused/llama_all_gather_concat_writer.cpp — local mcast leg IN SCOPE (25 fabric refs)
Lines 197–216: two `get_noc_multicast_addr` + two `noc_semaphore_set_multicast` (203, 216). F2=flag, F3=EXCLUDE_SRC, sem-only, two dest rectangles.
(`llama_all_gather_concat_reader.cpp` / `llama_concat_reader.cpp`: NO mcast — receiver/wait side only.)

---
## Cross-cutting forks observed
- F3 splits cleanly: **loopback_src (INCLUDE_SRC)** when sender ∈ receiver grid (rms, all_to_all, llama worker_receiver); **plain (EXCLUDE_SRC)** when sender is a distinct coordinator (selective_reduce, concat_writer, in1 reader).
- F1 splits: **flush** when NOC1/local-value-reused (selective_reduce); **barrier** when terminal (rms, all_to_all, worker_receiver).
- F2 = flag everywhere in this set (VALID/wait_min). Counter only in deepseek_prefill.
- **NOC1 coord-swap** recurs (selective_reduce, sampling, mcast.hpp, moe tilize) — a mandatory Pipe responsibility.
