# RESULT_PFX: row-local prefix cache for Hy4-Preview (agent pfx, host .45 all-slot claim 'pfx', 2026-11-17; branch ssinghal/hy4-preview-pfx = wt/hy4_pfx, from product12 5c56ae5db40; nothing pushed)
## VERDICT: PASS, behind a flag (HY4_PFX=1, default OFF). No C++, no extra DRAM, no plugin or vLLM change.
Prefix reuse hits are bit-identical to cold prefill (first-token logits md5 + greedy tokens) on 9 prompts incl. the 12288 hybrid boundary; a returning turn of a long conversation costs 8-11 s of prefill instead of 70-135 s (cold at 13k-25k) / 165 s (34k); needles, 12-turn and 4-session multi-turn, and a 166-request soak are clean on the same boots.

## 1. Inventory: what defines a prefix on this stack (CPU/reading, step 1)
| item | finding |
|---|---|
| per-user state | ONLY the pool pages of the user's mesh row: 78 SCALED_FP8 KV pools (704 B rows) + 21 bfp8 indexer-key pools + the MTP layer's own kv/key pool pair (`Hy4SpecRunner.mtp_kv/mtp_keys`), page = 512 tokens, striped over the 8 columns (64-token slices), identity page table, one independent pool per mesh row (B=4: user = row). Confirmed against kv-design s2.3: no halo, no compressor/Engram/n-gram state, no SWA ring; IndexCache ids are consumed inside one chunk (`ids_by_layer` in `Hy4Chain.prefill`) so a chunk that starts at a hit boundary recomputes them from the pages. iHC streams are in-layer. RoPE positions are a function of the chunk start (`start` argument of `ch.prefill`). |
| decode/spec state | host mirror + device tensors per row (pending token, position, drafts, cap, sampling params). It is set by the seed round after every prefill, so a hit only has to produce the same first token / hidden / pages. |
| MTP | MTP prompt KV is written per chunk by `mtp_prompt_chunk` from the main model hidden of the chunk (pair (h_t, token t+1)); the pair of the LAST prompt position is written by the seed round with the first generated token. After a hit the pair at the boundary position would be wrong (next token differs), so the last chunk of the hit is recomputed (HY4_PFX_REDO=1, cost 1 chunk = ~2.6 s; makes the MTP state exact, not just the main state). |
| the trap found | a parked (idle) row runs (PAD token, position 0) through every spec round and every seed round and WRITES KV/key rows at positions 0..K of page 0 of its own pool: any retained prefix in an idle row is corrupted within one round. Fix: the parked position is per row = length of the retained prefix (`MTPRoundTail.parkvec`, `Hy4SpecRunner.set_park(park, park_pos)`): junk lands just past the retained blocks and the parked row only attends to initialised keys. No spare page and no scratch region needed. |
| who picks the row | vLLM picks the logical slot; the model owns logical -> physical row (`_Slots`). A new request is steered to the FREE row that retains the longest matching prefix by swapping two entries of that map (both rows idle), so no plugin change and no cross-row copy. |
| prefill structure | all 4 rows in lock step with one global chunk start; rows not being prefilled are masked (`wr_mask`). Pool-path chunks (512 rows) can start at any multiple of 512 (they read earlier pages from the pool). The hybrid chain path (<= 12288 tokens, replicated cache) cannot start mid-prompt, so a call with a hit uses the pool path. |
| DeepSeek APC (reference only) | the DSV4 plugin APC (BlockIndex, refcounted pages, halo/zone snapshots, cross-row d2d copy, hold/row-match scheduling) was the model; Hy4 needs none of the halo/zone machinery (pages are the whole state). kv-design `apc_blocks.Hy4BlockIndex` (70 CPU tests) is NOT used here: the row-local design needs no block table. |

## 2. Design as built (flag HY4_PFX=1, spec mode only)
* Per mesh row `ret[row]` = np.int32 token array of the prompt blocks whose pages are valid (length % 512 == 0, full prompt blocks only: never the decode-written tail, so a hit reuses only bytes written by prefill itself). Set after a successful prefill, cleared at the start of one (an aborted prefill retains nothing), kept across request end and across decode of other rows.
* Hit: `match_len` (vectorised compare) -> h = floor(min(match, len-1) / 512) * 512 (at least one token left to prefill); start = h - 512 when REDO (else h); the prefill loop starts at chunk start/512 (`Hy4PoolPrefill.prefill(start=...)`). Lock step: start = min over the rows of the call (a row with a longer hit recomputes down to it, same bytes). chain (hybrid) prefill only when start == 0.
* Admission: `assign_rows` (engine) + swap in `prefill_forward` (adapter): longest match among free rows; no match -> the free row with nothing retained / least recently used, so other sessions' prefixes survive; live rows never.
* Eviction = overwrite: a row holds one chain (its last prompt). 4 rows = 4 cached sessions at B=4. Same-prefix requests on two rows do not share (no cross-row copy).
* Live switch: file HY4_PFX_FILE (0 / 1, read per prefill call; 0 = no hits and the plugin's own row mapping, bookkeeping continues). Boot flag default OFF: with HY4_PFX unset the graph and code paths are unchanged.
* Logging: every prefill call logs `lg_md5` (md5 of the fp32 first-token logits row), `pfx_start`, `hits`, `pfx_stats` (calls, hit_calls, hit_tokens, prompt_tokens, computed_tokens).
* Files: `tt/hy4_serve_engine.py` (ret, hit/assign, park positions, prefill start), `tt/hy4_pool_prefill_model.py` (start), `tt/hy4_spec_trace.py` (parkvec), `tt/generator_vllm.py` (row swap), `tests/test_hy4_pfx_cpu.py` (5 CPU tests: hit alignment, row assignment, LRU, park pos; pass). Launch: `program/pfx/run_pfx_server.sh <tag> HY4_PFX=1 HY4_PFX_FILE=/out12/pfx_file` (= product12 launch + the flags; `SPEC_K=3 MAXLEN=1048576`). Memory: zero extra DRAM (the pool is the cache; per row up to max_model_len tokens retained = the whole 1M pool of a row); persistent L1 unchanged (one extra 1-float row tensor `parkvec` in the spec tail). No C++ rebuild. Boot: ~12 min (JIT of the tiny new tail ops), then as product12.

## 3. Gates
Boot a1 (strict1, multi-turn 12, 4-session) and boot a2 (restart with the live-off row-policy fix; strict2/3, needles, soak). Both: SPEC_K=3, MAXLEN 1M, B=4, hybrid cap 12288 (switched to 0 = pool path for the strict gate so cold and hit use the same path), live files in program/pfx/.
### 3.1 hit == cold (greedy, spec on, 48 new tokens; cold = pfx file 0 so no reuse; same mesh row)
Prompt = natural English text (tech reports), base N tokens, extended prompt E = N + 700. Sequence: cold(E) x3 (pfx off), warm(B) (pfx on, 1 token), hit(E) x2. PASS = first-token logits md5 equal across ALL runs AND tokens equal to all three cold runs AND the hit really started > 0. Script `pfx/pfx_test2.py`; log analysis `ev_strict2_analysis.txt`; server log `pfx/logs/server_a2.log`.
| E tokens | md5 cold x3 / hit x2 | 48 tokens hit == cold x3 | hit start | cold prefill | hit prefill | note |
|---|---|---|---|---|---|---|
| 1724 | equal | yes | 512 / 1024 | 10.2 s | 7.6 s | |
| 3260 | equal (4/4) | cold NOT deterministic (see 3.2) | 2048 | 23.5 s | 9.0 s | md5 gate only |
| 5700 | equal | yes | 4096 / 5120 | 31.9 s | 10.8 s | |
| 8700 | equal | yes | 7168 | 52.4 s | 8.5 s | |
| 12700 | equal | yes | 11264 | 70.3 s | 8.6 s | crosses the 12288 hybrid cap |
| 12988 | equal | yes | 11776 / 12288 | 69.6 s | 8.2 s | base prompt exactly 12288 |
| 13500 | equal | yes | 12288 / 12800 | 72.5 s | 8.2 s | base 12800 |
| 16700 | equal | yes | 15360 | 94.2 s | 8.2 s | |
| 20700 | equal | yes | 19456 / 19968 | 110.6 s | 8.2 s | |
| 24700 | equal | yes | 23040 / 24064 | 133.1 s | 10.1 s | |
=> 10 prompts bit-identical in first-token logits (hit == cold, also hit2 == hit); 9 of them also token-identical over 48 greedy tokens (the 10th has a non-deterministic cold reference). Earlier boot a1 (strict1, cold runs on different rows): logits md5 hit == cold for 9/9 more prompts (E = 1724 .. 20700).
### 3.2 caveat: the product decode is not bit-deterministic run to run (pre-existing, not the cache)
cold vs cold on the SAME row with the cache switched off differs in the 48-token greedy stream for some prompts (E=3260 in boot a2; cold rows 0 vs 1 differ in 4 of 9 prompts in boot a1) although the prefill logits md5 is identical: the spec path (draft acceptance pattern changes the verify batch composition; near-tie logits) is not reproducible across runs. Therefore the token gate is meaningful only where cold is deterministic (9 of 10 prompts above, all token-equal), and the logits md5 is the primary bit-exact gate. Row choice per se is not the cause (a2 cold x3 on one row).
### 3.3 other gates
| gate | result | file |
|---|---|---|
| >= 8 prompts incl. > 12288 boundary | PASS (10 prompts md5, 9 token-identical; boundary cases 12700 / 12988 / 13500 / 16700 / 20700 / 24700) | ev_strict2.out, ev_strict3.out, ev_strict1.out |
| stray CJK | 0 in every strict output, in the 12-turn (a1), in 4 x 8-turn sessions x 3 sets (a1, a2), 48 mix requests, soak | ev_*.out |
| multi-turn 12 turns (boot a1, single session, product12 `multiturn.py`: planted codes + recall of an earlier code, +2.8k tokens per turn, hybrid on) | 12/12 recall PASS; per-turn TTFT 16.1 / 20.3 / 17.7 / 20.2 / 17.7 / 20.1 / 17.6 / 20.2 / 19.0 / 20.0 / 17.4 / 21.3 s for prompts of 2.8k .. 33.9k tokens (product12 without cache: 13.7 s at 2.8k, 50.6 s at 11.3k, 89.5 s at 17k, 138.7 s at 28k, ~165 s at 34k). Flat because the new part of every turn is ~2.8k tokens (+ the redo chunk). | ev_mt12_pfx.out |
| 4 concurrent sessions x 8 turns (a1) | 4/4 PASS, rows pinned per session (hits per row in the log, hits {0:5632,2:5632,3:5120} etc.); TTFT 37-40 s in lock step (4 x 2.8k suffix prefilled together) vs ~20 s alone | ev_mt_c4_s*.out |
| needle 64k (a2) | PASS cold (63890 tok, key 35409, 320.6 s) and PASS again as a full prefix hit 5.7 s | ev_needle64k_1/2.out |
| needle 200k (a2) | PASS cold (199988 tok, key 75854, 1002 s) and PASS as a full prefix hit 5.9 s | ev_needle200k_*.out |
| soak (a2): 166 soak requests = 2 x (4 concurrent 8-turn sessions, 64) + accept_probe all T=0 and T=0.7 (54, 8 chat + 8 gsm + 8 code + 3 story each) + 48 random prefix-sharing mix (6 docs, 600-9000 tokens, T 0/0.7, 4 threads) | 166/166 ok, 0 errors, 0 CJK, all 8 sessions recall PASS; server log of the boot: 224 POSTs all 200, 0 Traceback / TT_FATAL / TT_THROW, 0 'mirror != plugin block' reloads, 0 recompute warnings; end counters pfx_stats calls 187 / hit_calls 67 / 1,006,592 of 1,912,698 prompt tokens reused (53%) | ev_soak_*.out, soak.log, logs/server_a2.log |
| clean shutdown | boot a1 TERM -> RC=0; no crash/hang, so NO board reset and no post_reset_gate in this task | logs/server_a1.log |

## 4. Costs and limits (numbers)
* TTFT of a hit = (suffix + the 512-token redo chunk, rounded to chunks) x ~2.6 s per 512 tokens (~5 ms/token chain-equivalent) + 0.15 s seed: 5.3 s (suffix <= 1 chunk) .. 10 s (700-token suffix) at any context depth up to the 33.9k tested; cold prefill is ~5.2 ms/token (133 s at 24.7k). The redo chunk costs 2.6 s per hit (HY4_PFX_REDO=0 skips it at the price of an inexact MTP pair at the boundary, which only changes draft acceptance).
* A hit always uses the pool path (the chain path cannot start mid-prompt); cold prompts <= 12288 still use the chain (hybrid) path. Chain-cold vs pool-hit therefore differ numerically for short prompts (the 'strict' gate switches the cap to 0 to compare like with like; the production multi-turn run crosses both paths and passed 12/12 recall, 0 CJK). The stray-CJK risk of the pool path for GPQA-style thinking prompts that motivated the hybrid cap was not re-measured with hits (only 4 q GPQA in RESULT12); keep HY4_PFX off for such workloads or re-run the GPQA gate with the cache on before shipping it on.
* Lock step: a call with rows of different hit length starts at the minimum (unequal hits waste up to a few chunks, seen 5632/5120 -> start 4608).
* One chain per row: a 5th concurrent conversation evicts the least recently used row; two conversations with a common system prompt on different rows do not share it. 1M retained per row costs nothing extra but a 1M prefix hit was not measured (needle 200k only).
* Spec mode only (HY4_SPEC_K > 0); plain decode (K=0) ignores the flag.
* Nondeterminism of the product decode (3.2) predates this change.

## 5. Next steps if wanted (not done)
1. Block-level reuse across rows (two sessions sharing a system prompt): needs the d2d row copy (kv-design s6 extrapolation 0.55 s per 64k SCALED_FP8; DSV4 rows-mode op; C++/kernel work) + Hy4BlockIndex wiring: P5 in the charter.
2. Reuse the decode-written tail of the previous turn (the assistant answer is in the next prompt): needs the verify-written bytes to equal prefill bytes (they do not bit-match), so only a recompute-from-block-boundary is exact; saves ~300-500 tokens per turn.
3. Run the thinking-GPQA / tau3 gates with HY4_PFX=1 before making it a catalog default; add `supports_prefix_caching` accounting only if the plugin should report cached tokens (`--enable-prompt-tokens-details` currently shows 0).
4. Make the min-start lock step per-row (needs per-row chunk starts in the PF kernels: C++).

## 6. Reproduce
`bin/slot_claim.sh host-acquire 10.82.97.45 pfx 480`; boot: `ssh 10.82.97.45 "cd program/pfx && SPEC_K=3 MAXLEN=1048576 ./run_pfx_server.sh a2 HY4_PFX=1 HY4_PFX_FILE=/out12/pfx_file"`; gate: `TAG=a2 python3 pfx_test2.py 10.82.97.45 strict 1024,5000,12288,12800,20000,24000`; soak `soak_pfx.sh 10.82.97.45`; analysis `analyze.py`/`ev_strict2_analysis.txt`. CPU tests: `PYTHONPATH=. python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_pfx_cpu.py`. All under /mnt/tt-data/ssinghal/hy4-logs/program/pfx/.
