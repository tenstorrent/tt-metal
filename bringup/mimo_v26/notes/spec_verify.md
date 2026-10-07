# Speculative VERIFY step: design + measured op probes (spec agent, 2026-10-07; nothing integrated yet)

Verify = target forward over 1+7 tokens per user (positions pos..pos+7, causal inside the block), KV of the block written speculatively, then accept/rollback.

## Attention: which op (probe `tests/probe_verify_ops.py`, one BH chip, per-chip TP8 shapes: 8 Q heads, 1 KV head, QK 192, V padded 192, bf16, commit 03832654772, host .48, log mimo-logs/spec_verify_probe_03832654772.jsonl)
* **V1 "virtual users" (RECOMMENDED, works today, no C++)**: call the existing 1-token paged decode op (`paged_scaled_dot_product_attention_decode`, pad-V-192) with 8 virtual users per real user:
  q rows (user,t) -> batch index 8*u+t, page-table row repeated 8x, `cur_pos = pos_u + t`. Causality inside the block falls out of cur_pos (token t sees positions <= pos+t, which
  includes the speculative KV of tokens < t that was written before the call), sliding window 128 + sink + ring (`cache_position_modulo`) behave as in 1-token decode.
  Correctness vs torch fp32 (B=1/4 users x 8 tokens, positions incl. 127/128/1999/2030, cache 2048): SWA PCC min 0.999581-0.999648, GA 0.999362-0.999621.
  Cost: K/V are re-read once per virtual user (8x traffic for GA). Measured per layer per chip: GA 64k cache B=1: 1.005 ms (1-token decode 0.222 ms), B=4: 3.81 ms, B=8: 7.71 ms;
  SWA (cache 2048, ring-like window): 0.044 / 0.121 / 0.077 ms for B=1/4/8 (1-token B=1 0.074 ms: floor-limited, free). So GA verify attention ~ 4.5x (B=1) to 8x of a decode step's attention.
  9 GA layers at 64k, U=4 users/row: ~9 x 3.8 = 34 ms extra per verify step (decode step today 143 ms at 64k); U=8: ~69 ms.
* **V2 chunked prefill SDPA with a 32-row q tile per user: REJECTED**: the op requires `chunk_start_idx % q_chunk_size == 0` (TT_FATAL; q_chunk 32): the block start `pos` is arbitrary.
  Only aligned positions work (PCC 0.9997 at pos 1024/96/128/1984/2016). Also GA 64k is 6.07 ms at B=1 (27x the 1-token op).
* **Fold the 8 tokens into the head dimension (q [1,B,64,192]) with MLA decode / the decode op**: one K read, but all 8 tokens would see the same cur_pos (no causal mask inside the block) and
  the op returns no softmax statistics, so the in-block part cannot be merged: needs a C++ change (LSE output or a per-row mask) -> only as an optimisation if V1 is too slow at 64k.
## KV writes and rollback
* Write the 8 speculative K/V rows with `paged_update_cache` using the same virtual-user trick (batch 8U, page-table row repeated, update idx pos+t): batch <= 32 so U <= 4 per call, more users = more calls (or
  per-t calls of U users, 8 calls). Not probed yet (the update op is the one the decode path already uses).
* Rollback = nothing to do: positions > accepted are invisible (GA masks by cur_pos, SWA window/ring) and are overwritten by the next round's writes. SWA ring (2560 positions) vs 8 speculative rows: no clobbering of
  the 128-window.
## MoE
* 8U tokens per row (U=4: 32, = today's decode batch; U=8: 64; U=16: 128). The unified op path handles N>=64 (moe_decode pads 32 -> 64 today); dispatch capacity (cf=4) must be asserted per call. NOT probed here.
  Router exact on 8U tokens is a plain matmul+topk. Decode-sized unified op was measured by the moe agent at about 2.3-2.8 ms per layer for 512 tokens/row, decode-sized (64) times are in LOG E261007-0346-*.
## Costs per verify step (estimate from the probes, not a measurement of the whole step)
Decode today: 143 ms at 64k (GA attention 9 x 3.3 ms = 30 ms of it at B=32 per chip). Verify at U=4: attention +34 ms, MoE/dense ~ same weights read once (batch 32 tokens vs 4 live users) -> step ~ 180-200 ms for 8 tokens
per user; with mean accepted length ~2.8-3.4 (LOG E261007-0616-6ca upper bound) that is ~2.3x tokens per step minus drafter time. At short context (SWA-dominated) verify is free.
