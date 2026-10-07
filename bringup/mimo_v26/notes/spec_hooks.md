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
