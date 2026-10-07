# Hooks the DFlash spec path needs in shared files (PROPOSAL, no edits made; decode agent owns tt/decode.py, tt/layer.py decode methods)

1. Capture of the 5 target hidden states (outputs of layers 0,11,23,35,47, layer 47 PRE final norm; DFLASH_NOTES convention) for every token row of a decode/verify step:
   * `tt/decode.py` `Decoder._forward`: inside `for lyr in m.layers:` keep `h2` after layers in CAPTURE = {0,11,23,35,47} (a list of device tensors [1,1,32,4096], replicated over columns),
     after the loop `feat = ttnn.concat(caps, dim=3)` -> [1,1,32,20480], returned as a 4th persistent trace output (`vals, idx, lg, feat`); `Decoder.run` keeps `self.feat` on device (no host readback).
     No change to layer.py: the layer outputs are already the returned `h2`.
   * The drafter consumes `feat` on device: its fc is N-sharded over columns (each column: full K=20480 x 512 outputs) + all_gather over columns for hidden_norm when M is small (decode: 32 rows/row);
     the current tt/dflash.py K-shards fc and all-reduces (right for prefill-size M); both are in the module, select by M.
2. Prefill capture (drafter context for the prompt): in `tt/prefill.py` (mine) the chunk forward also returns the same 5 tensors per chunk ([1,1,CH,20480]) feeding `DFlashDrafter.append_ctx`.
3. Drafter KV: per user per layer ring [1024,128] K and V per column (20 KB per position over all 5 layers and 8 columns), written with the same paged_update/fill ops the target uses (contiguous ring, no paging needed).
4. Verify: see spec_verify.md (virtual users: batch 8U in the decode path; decode.py needs a `block` parameter that expands positions to pos+t and repeats page-table rows).
Files that would be touched: tt/decode.py (Decoder._forward, Decoder.run), tt/prefill.py (chunk forward), new tt/dflash.py, tt/spec.py. No change to tt/attention.py decode methods until the verify design is agreed.
# Decode-side spec-decode hooks (decode agent, branch ssinghal/mimo-attn) - interface for the assembly agent

All in `tt/decode.py` (class `Decoder`) + `tt/attention.py` / `tt/layer.py` (decode methods, new `block` / `upd` arguments). Verified: 4 layers, U = 1, 2, 4 users per mesh row (8, 16, 32 virtual rows), ragged real
GSM8K prompts, random draft tokens, on a freshly prefilled cache: argmax of every virtual user == argmax of the sequential plain decode at that position (100%), captured hidden states bit-identical to the
sequential decode (all 4 layers), K/V rows written bit-identical, two identical verify steps bit-identical (LOG entry E261007-* "verify vs sequential", tests/test_verify_decode.py).

## API
```
dec = Decoder(model, trace=True, U=U, block=8, capture=(0, 11, 23, 35, 47))   # U*block <= 32 rows per mesh row, so U <= 4 for block 8
argmax, best = dec.verify(tokens, positions, dead)    # tokens [rows,U,8] = [last accepted token, draft_1..draft_7]; positions [rows,U] = position of tokens[...,0]
                                                       # (= number of tokens already in the KV cache); dead [rows,U] bool. argmax [rows,U,8] long: the target's greedy token AFTER block
                                                       # position t (so argmax[...,0] is the next token if no draft is accepted). best = its logit.
n_acc = Decoder.accept_prefix(argmax, tokens)          # [rows,U] number of accepted drafts (0..7); emit argmax[..., :n_acc+1]; next block starts with argmax[..., n_acc] at position pos + n_acc + 1
dec.feat                                               # device tensor [1,1,U*8,len(capture)*4096] bf16 per device (replicated over columns), row u*8+t = hidden states after the captured layers
                                                       # (layer 47 BEFORE the final norm) of token t of user u; the accepted rows are 0..n_acc. Valid until the next verify/run. NOT read back to the host.
```
`Decoder(..., block=1, capture=...)` is the plain decode step with the same capture (feat [1,1,U,5*4096]). `Decoder.close()` frees everything (tensors + trace). One Decoder = one traced graph (capture once, replay per step); keep
one verify Decoder per (U, block) and plain Decoders as needed; Generator.reconfigure does not know about them (call close() first).

## How it works / what was learned
* Virtual users: row v = block*u + t feeds token t of user u at position pos_u + t; the page-table row of user u is repeated `block` times for SDPA/update (MiMoAttention._dec_state(B, block)); the 1-token paged SDPA decode op gives
  causality inside the block through cur_pos (token t sees positions <= pos+t including the speculative K/V of tokens < t written earlier in the SAME layer call). Sliding window 128 + sink + ring work unchanged.
* KV writes: ONE batched paged_update_cache over the virtual rows LOSES updates (the op is a read-modify-write of whole cache tiles and rows of the same user share a tile: probe tests/probe_paged_update_rows3.py: only 1-2 of 8 rows survive).
  Fix (default, MIMO_VERIFY_WRITE=skip): `block` calls in order, call t has update index -1 (= skip) for every row except the rows of token t, same sharded source tensor. Back-to-back calls on the same tile (eager and traced) all land
  (probe_paged_update_rows2.py). MIMO_VERIFY_WRITE=seq (strided slices, U rows per call) also works; =batch is the broken one.
* Rollback needs nothing: positions beyond the accepted one are invisible to later steps (GA masks by cur_pos, SWA window/ring) and are overwritten next round.
* MoE sees U*block rows per mesh row (<= 32, padded to 64 inside MiMoMoEDecode); dead slots follow MIMO_DEAD (default spread32).
* Cost: attention re-reads K/V once per virtual user (GA at 64k ~ 8x the plain decode attention of one user; SWA free), MoE/dense weights are read once per step.

# Device-resident verify round (perf agent, branch ssinghal/mimo-perf1; code `tt/verify_dev.py`, `Decoder(vdev=True)`, test `tests/test_verify_device.py`)

Goal: a whole spec round with NO host copy and NO host compute between the drafter and the next round. The verify trace owns its input prologue and its accept_prefix epilogue, all on device.

```
dec = Decoder(model, trace=True, U=U, block=8, capture=(0,11,23,35,47), vdev=True, eos=eos_ids, max_pos=max_pos, chunk=N)   # needs MIMO_VERIFY_WRITE=skip (default)
dec.set_verify_state(anchor, pos, alive)   # host write ONCE per generation: long [rows,U] anchor token, long [rows,U] position of the anchor (= tokens in the KV cache), bool [rows,U]
                                           # also captures the trace (eager warm-up + capture) and restores this state, so call it before writing any drafts
dec.write_drafts(d)        # device op(s): blk_in[..., 1:8] = d, d = device tensor [..,U,7] (any numeric dtype / layout; reshaped to [1,1,U,7]); usable inside your own trace
dec.verify_enqueue(n)      # n non-blocking trace replays (one round each); state advances on device
post, commit = dec.read_rounds(n)   # optional bulk host read of the last n rounds from a ring of N rounds (post long [n,rows,U,8], commit long [n,rows,U] = k+1, 0 for dead users)
```
Persistent device tensors (float32 TILE, row sharded = one copy per mesh row = that row's users, identical over the 8 columns; ids / positions < 2^24 are exact). Write them with ttnn.copy / ttnn ops, read them as inputs:
* `dec.blk_in`  [1,1,U,8]  the NEXT verify block. col 0 = anchor: written by the epilogue of the previous round (= bonus token, 0 for dead users) - you do NOT write it. cols 1..7 = drafts: written by you (`write_drafts`).
* `dec.vpos`    [1,1,U,1]  position of block row 0 of the next round (advanced by the epilogue: += k+1 for live users).
* `dec.valive`  [1,1,U,1]  1 = user still generating. Epilogue clears it when an `eos` id is among the committed tokens (rows 0..k) or when pos + 9 >= max_pos (same rules as spec.py).
Outputs of the LAST round (valid until the next replay; use them for the drafter append, in the same device stream):
* `dec.vpost`   [1,1,U,8]  target argmax after every block row (0 for users that were dead at round start)
* `dec.vk`      [1,1,U,1]  accepted drafts k (0..7), `dec.vcommit` [1,1,U,1] = k+1 (0 for dead users)
* `dec.vvalid`  [1,1,U,8]  1 for the committed rows 0..k of live users (the drafter `valid` mask), `dec.vposrows` [1,1,U,8] = position of every block row (the drafter append positions)
* `dec.feat`    captured hidden states (unchanged: [1,1,U*8,len(capture)*4096] bf16, row u*8+t)
The round counter `dec.vcnt` and the rings `rpost`/`rcommit` ([1,N,U,8] / [1,N,U,1]) are internal; `read_rounds` reads them (N rounds, wraps; the host mirror `dec.c0` must stay in step: call read_rounds(n) for every n replays or call set_verify_state again).
Semantics = Decoder.verify + Decoder.accept_prefix + spec.py bookkeeping: block row 0 = anchor, draft j accepted iff argmax[j-1] == block[j] for all earlier drafts too; committed tokens = argmax[:k+1]; next anchor = argmax[k] (bonus), position += k+1.
Dead users feed token 0 / position 0 / update index 0 and are marked dead in the MoE live mask (same as set_inputs). The host-fed `verify(tokens, positions, dead)` path is unchanged.
Verified by tests/test_verify_device.py (device rounds vs host verify+accept_prefix, 4 layers, U=1,2,4, ragged acceptance, eos pass): see LOG.
# Device-resident spec round (spec agent, tt/spec_dev.py, commit 697008e0aad) - design as built + what changes when the decode agent's vdev verify lands

One trace per round, enqueue only. Everything between "drafter inputs" and "state update" is device ops on persistent fp32 state [1,U,..] per mesh row: pos, alive, anchor, cnt (stream length), out (committed token stream,
scatter by one-hot masks at cnt+t), ring_pos (drafter ring slot -> position), rcount (alive rounds). Per round:
1. block positions pos+iota8 -> uint32 -> embedding lookup of cos/sin tables (positions up to max_pos+16) -> rope rows for the drafter block (rows (head,tok)) and for the ring append rows.
2. window mask = (1 - [ring_pos in [pos-1024, pos) and >= 0]) * NEG concat [8 zeros, NEG pad] (row mask, broadcast over the 64 q rows; the per-head sink is a constant per-layer tensor).
3. drafter forward (tt/dflash.py `_draft_forward(inp=...)`), cross-column argmax on device (all_gather of the per-column (max, idx), argmax over columns = lowest vocab id on ties) -> block = [anchor, d1..d7].
4. verify inputs written in place into the Decoder's persistent tensors with ttnn.assign: tok (uint32 [1,B]), pos["cur"] (int32 [B]), pos["idx"], upd[t] (index = position for the rows of token t, -1 otherwise = the decode agent's
   MIMO_VERIFY_WRITE=skip pattern), live mask (alive expanded to the B virtual rows through a [B,U] 0/1 matmul, padded to 64 rows). Dead rows (alive=0): token 0 / position 0. Then `dec._forward()` (the decode agent's graph, unchanged).
5. accept on device: eq(argmax[:7], block[1:8]) -> triangular matmul of the mismatch vector -> leading-ones count k; bonus = argmax[k] by one-hot sum; valid rows = (iota8 <= k) * alive; eos hit = any valid committed row equals an EOS id;
   alive_next = alive * !eos * (cnt + k + 1 < max_new) * (pos + k + 1 + 9 < max_pos).
6. drafter ring append from dec.feat rows 0..k: slot = pos+t mod 1024 (fp32 floor), S = (iota1024 == slot) * valid, keep = 1 - sum(S), ring = ring*keep + S@new (bf16 one-hot matmul, exact), ring_pos likewise (elementwise, fp32 exact).
7. state update with ttnn.assign; host loop = execute_trace(blocking=False) x chunk (4) then ONE blocking read of (alive, cnt, out, rcount) per chunk. Trace capture order here: prefill trace, then the single round trace
   (drafter + verify inside it), so the 'drafter before verify' hang cannot occur. Warm-up round runs eagerly on a snapshot (clones of all state + rings) which is restored before capture.
Interface used from decode.py: Decoder(block=8, capture=...), persistent dec.tok / dec.pos / dec.upd / dec.live.t, dec._forward() -> (vals, idx, _, feat); nothing else. To switch to the decode agent's `vdev` prologue/epilogue the round would
keep steps 1-3 and 6-7 and replace 4-5 by write_drafts + verify_enqueue; vpost/vk/vcommit/vvalid/vposrows replace my steps 5 inputs (the ring append consumes vvalid and vposrows).

## Prefill: per-chunk HOST work today (tt/prefill.py set_step) and the device version (proposal, not implemented)
Today per step: (a) tokens [rows,CH] host->device (needed, the only real input); (b) per attention kind (ga, swa) four host-built row-sharded tensors copied to the device: rope_idx = arange(start..start+CH) per row,
fill_pt = the chunk's page-table slice, sdpa_pt = the user's full page-table row (padded to 8), start = chunk_start; (c) python scheduling of (slot, start, last-chunk) per row per step; (d) after the last chunk of a user the host reads
the whole hidden chunk of that row to the host, picks the last real row, and runs final norm + lm_head separately; (e) pad-token construction (cyclic prompt repeat) on the host.
Device version: upload once per BATCH (not per chunk) the schedule tensors [steps,rows] (slot, start, is_last) plus the page tables [slots,nb] (already constant), and the token matrix [rows, steps*CH] padded; a device step counter selects row `step`
of each (embedding lookup / slice by counter): rope_idx = start + iota(CH) (ttnn iota arithmetic), fill_pt = embedding(page_table_row(slot), block index start/64 + iota(CH/64)) (gather), sdpa_pt = embedding(slot, table), start = scheduled start; counter += 1 at the end of the trace.
The last real hidden row of each user is gathered on device (one-hot over positions via (iota == len-1-start) * is_last, matmul into a per-user [rows,U,4096] buffer), final norm + lm_head + the cross-column argmax (the same code as in spec_dev.global_argmax) run once after the last step,
and the first tokens land in a device buffer read once. Pad tokens can be produced on device from the prompt (cyclic index = iota % len). Host per chunk then = enqueue only. Cost: one trace per (CH), schedule size is tiny; capture order stays prefill first.
