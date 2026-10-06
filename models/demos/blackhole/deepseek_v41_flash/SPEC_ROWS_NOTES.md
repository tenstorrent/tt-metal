# Spec decode beyond 32 rows per mesh row (branch ssinghal/dsv4p1-spec-rows) - STATUS: partial, paused

## Row limits per op (T = token rows per mesh row = U*(1+k); drafter 5*U in chunks of 4 users)
| op | limit | evidence | handled by |
|---|---|---|---|
| mHC mixes ProjPlan | T%4==0, T<=32 (T=12/20 wrong -> pad to 8; T=5..7 asserted) | mhc_mixes2.py:32, mhc.py mixes() | pad T=5..7 -> 8 (B=4); >32: per-32-row chunks |
| mHC collapse_norm / expand | T<=32 (fused cn only T=4/16/32, else composite +25..40 ms/round) | mhc_collapse.py:58,213; mhc_expand*.py:17 | per-chunk lists (layer.forward_chunks) |
| router_select / gate | T<=32 | router_select.py:22 | DSV41PrefillMoE runs gate per 32-row slice |
| moe_compute + tail | decode view <=32 rows; tail fused only <32 | moe_block.py _TailDecode | _MoEBig = DSV41PrefillMoE(g=T/32), measured 0.75/1.15/1.5 ms per layer at 32/64/128 rows (test_moe_tokens_scaling) |
| shared expert v2 | per_core_M=1 (32 rows) | shared_expert_v2.py | per chunk |
| attention qkv/o linears | tuned 1D configs per_core_M=1 | attention.py _lin | auto 8x8 grid when rows>32 |
| nlp_create_qkv_heads_decode / nlp_concat_heads_decode | <=32 users | TT_FATAL num_users_supported | per 32-row slice |
| fused rope kernel | 2 cores per row -> <=32 rows (grid 120 cores) | attn_fused.py:40 | per 32-row slice |
| paged_kv_step | one core/row, 8x8 grid (<=64), wide grid hits dispatch cores; lat assert rows<=32 | paged_ops.py | assert relaxed to 128, 64 rows per call with user offset (ring_base, page-table slice) |
| q shard cfg, SDPA | <=64 cores | spec_model._ucfg | capped; chunked path does not use it |
| head / sampling / accept | any T (matmul, argmax) | - | unchanged |
| drafter | 5 rows/user, <=32 | mtp.ChunkedDrafter | already chunked (U/4 chunks, ~9 ms each) |

## Cost model (plain decode ms/token from GRID_SUMMARY: B=32 53, 64 67.5, 128 100 at 4k; round = verify(R rows) + drafter chunks + host)
Marginal row cost ~1.8-2 ms/row (R 16->32); MoE grows sub-linearly, mHC linearly.
* B=64 k=3 (R=64): est. verify 120-170 ms + 4 drafter chunks 36 + host 8 = 165-215 ms. Plain 67.5. Break-even accepted/round = (round/plain)-1 = 1.4-2.2. GSM8K (2.3) wins 1.0-1.4x; long docs (1.0) lose (0.6x).
* B=128 k=3 (R=128): est. 280-390 ms vs plain 100: break-even 1.8-2.9 of max 3 -> at best ~1.2x on GSM8K, loses on documents. k=1 (R=64) needs 1.5 > 1.0 max accepted: cannot win.
* B=4/8 (U=1,2): rows 4-10, round ~ plain+25%; k=3 GSM8K ~2x (B=8 measured 2.04x).

## Results so far (4-layer smoke only, DSV41_LAYERS=0-3; spec numbers meaningless for acceptance)
* B=4 k=3 pad fix: runs (381 rounds). B=64 k=3 (64 rows) and B=128 k=3 (128 rows, fp8 pool) run end to end, first token spec==plain 64/64; round 72.6 ms / 130 ms at 4 layers vs plain 28 / 27 ms.
* 40-layer runs (b64/b4/b128) were queued behind the 5-slot build cap (waited >40 min) and were killed on request; NOT measured. Exactness at 40 layers not verified.
Enable with DSV41_SPEC_ROWS=1 (T>32, T%32==0). Run: ./go.sh-style session, gsm8k_b64 / gsm8k_b128 (DSV41_POOL_DTYPE=fp8) with DSV41_SPEC=3.

## Update (2026-10-06): 40-layer results, RoPE finding, open items
* RoPE: the fused rows-layout kernel (attn_fused.rope_inplace rows_layout=True) rotates every row by table row 0 (tests/test_attn_fused_rope.py). Wrong for spec verify blocks (rows at different positions). Split: spec verify views (`per_row_rope=True` on SpecPagedCompressedAttention / SpecCompressedAttention) use the per-row addcmul path; plain decode keeps the fused kernel (unfused costs +4.6 ms/token at B=16). DSV41_ROPE_ROWS_FUSED=1/0 forces fused/per-row everywhere.
* B=64 k=3 (DSV41_SPEC_ROWS=1), 40 layers: gsm8k 2.32 accepted/round, round 230 ms, 14.4 vs 12.9 tok/s/user = 1.11x, 24/64 users identical to plain (rest near-ties); isl4k 1.13 accepted, 0.62x, 48/64 identical (divergences at token 1, plain gap 0.205). Before the RoPE fix isl4k was 0/64 identical.
* B=128 k=3 (128 rows, bf16 pool, isl4k, pre-RoPE-fix): runs, round 435 ms vs plain 98.6 ms = 0.60x (1.65 accepted); break-even would need ~3.4 accepted > k: cannot win. A second scenario in the same process fails (static CB / L1 clash in prefill SDPA after the spec runner): one scenario per process at B=128.
* B=4 k=3: runs (T=5..7 pad), verify exact, but drafter acceptance ~0.05-0.09 (target 2.3): OPEN. mHC at T=5/6/7 with the T=8 pad passes PCC >= 0.9999 (test_mhc_flags), so the mHC is not the cause; suspect another U=1 path of the drafter (mtp.py: sample_global on a 1-row tile, reshape [1,1,U,256], draft attention rows T_d=5, ucfg). Next step: tests/test_spec_mtp.py with users_per_row=1 vs 2. The b4_pad8 40-layer run hung silently (>5 h) and was killed.
* Not run: B=64 at 32k, B=128 at 16k, B=4 k=2 (scope narrowed to gsm8k).
