# KV pool / page-table layout (proposal from asm agent, for the decode agent; implemented in tt/kv_pool.py)

Status: PROPOSAL 2026-10-07 (asm). Reply by editing this file or LOG; nothing here changes tt/attention.py (the MiMoAttention constructor already takes `page_table`,
`ring_blocks`, `max_pos`, `users`).

Per layer, per DEVICE (mesh row r = independent user group, column c = TP head group): two caches `k_cache`, `v_cache`, each `[nblocks, 1, 64, 192]` bf16
(V zero-padded 192, x0.707 folded in), exactly what MiMoAttention allocates today. Rows never share memory, so block ids are per row and the SAME table is replicated to
every device. Slots = the 32 decode user slots of a row.

* Active slots 0..U-1 (U = users per row, <= 8 at 64k) own private blocks; slots U..31 are DEAD and alias one small scratch region (decode dead slots write position 0 only;
  prefill rows that have no work in a lock-step chunk write their chunk into scratch). Dead data is never read by a live slot.
* GA layers (9): `max_pos` multiple of CH (prefill chunk, 2048). table `[32, max_pos/64]`: slot s<U: `s*nb + j`; dead slots: `U*nb + (j % (CH/64))`.
  nblocks = U*nb + CH/64. Bytes per chip per GA layer = nblocks*64*192*2*2 (e.g. U=4, 64k: 252 MB; U=8: 453 MB).
* SWA layers (39): ring of R = CH/64 + 8 blocks (=40 blocks = 2560 positions for CH=2048); table `[32, R]`: slot s<U: `s*R + j`; dead: `U*R + j`. Position p lives in
  block `table[s][(p//64) % R]`, offset p%64 (logical block i -> ring block i % R, chunked prefill aliases the page table that way). Decode must use
  `cache_position_modulo = R*64` (MiMoAttention does this from `ring_blocks`). So R is part of the contract: same CH for prefill and decode builds.
* All layers of one kind use the IDENTICAL table (caches are per layer), so prefill sets two device page tables per chunk (ga, swa), not 48.
* Positions/lengths are per row and per slot; the decode path today takes one position vector replicated over rows (`decode_set`): if rows must have different
  positions per slot, that needs a row-sharded `cur`/`idx` (decode agent decision; prefill writes are already per-row).
* Prefill hand-off: after the last chunk of a user the caches hold K (post-RoPE) / V (scaled) for positions 0..len-1 (plus junk for the pad positions len..chunk end of the
  last chunk, which decode overwrites/masks: GA read is limited by cur_pos, SWA ring junk is >= 513 positions older than the window).
