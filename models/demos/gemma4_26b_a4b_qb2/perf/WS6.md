# WS6 · MTP speculative decoding, batch 1 (A, 16 Nov 2026)

Branch `ws6-mtp` (base `integration` b1d5cbf206a + the RouterTopk speedup + the quick_check change). Greedy, batch 1,
drafter `google/gemma-4-26B-A4B-it-assistant` (revision 6e5aaaf4c4). Not landed: the stage gate (numerics, decode
timing, agreement, serving, GPQA) has not been run, and the token-identity proof needs a decision (see below).

## Result

10 chat prompts (`perf/tools/mtp_prompts.json`), 192 new tokens each, chat template, traced, `perf/tools/mtp_check.py`
(`results/WS6/mtp_k_sweep.json`). Plain = the integration batch-1 traced greedy decode in the same process.

| K | accepted drafts / round | tok/s/user (mean of prompts) | aggregate tok/s | ms / round | vs plain 86.4 |
|---|---|---|---|---|---|
| 3 | 1.96 | 147.9 | 144.7 | 20.3 | 1.71x |
| 4 | 2.34 | 156.6 | 151.4 | 21.9 | 1.81x |
| **5** | **2.63** | **160.7** | **153.6** | 23.4 | **1.86x** (1.87x vs the WS1 proof 85.9) |
| 6 | 2.76 | 157.7 | 149.0 | 25.0 | 1.83x |

K=5 is the default. Per prompt (K=5): 107 to 215 tok/s (acceptance 1.58 to 4.08); the short story is the slowest.
tok/s = (generated tokens - 1) / decode wall time after the first token, the same definition as the plain
`decode_tps`.

Proof line "effective batch-1 tok/s/user >= 1.5x integration": **met** (1.86x mean, 1.78x aggregate).

Proof line "token-identical to plain greedy on 10 prompts": **not met as written**, 0/10 at 192 tokens. What holds:

- The speculative output is token-identical, 10/10 at every K = 3..6, to plain greedy decoded one token per step
  through the **batched** decode path that the verify uses (`SpeculativeDecoder.generate_plain_rows`: 2 rows, the
  second inactive). So acceptance, K/V handling and the seed are exact; the first divergence from batch-1 plain is at
  the same token for every K.
- The batched path itself (WS1, landed with 99.6 % teacher-forced agreement) and the batch-1 path disagree at near
  ties: first difference at tokens 34-145, top-1 minus top-2 logit margin 0.0-1.0 there (four of ten 0.0-0.125).
  Device batch-1 plain and the speculative output also both leave HF CPU bf16 greedy at the same early tokens (3 and
  11 on two of four prompts).

A verify that is bit-identical to the batch-1 path would need the batch-1 numerics for K+1 rows (TP experts, packed
bf16 all-reduces, BFP8 collectives off), i.e. K+1 batch-1 forwards, which removes the gain. Options for the lead:
accept "identical to plain greedy through the batched path" as the proof, or compare against the batch-1 decode
with teacher forcing as the agreement gate does (spec tokens are the batched path's greedy, which WS1 already gated).

## Acceptance vs the HF reference

`perf/tools/mtp_host_reference.py`: transformers' own MTP path (`SinglePositionMultiTokenCandidateGenerator`, K=5
constant) on CPU, first 4 prompts x 96 tokens: **2.65** accepted per round (2.28 / 4.59 / 2.96 / 1.79,
`results/WS6/host_reference.json`). Device, gemma4 demo drafter, untraced loop, same prompts: **3.03**
(3.04 / 4.59 / 3.26 / 2.03, `device_acceptance_4prompts.json`). Device acceptance is not below the host, so the
drafter's Q heads and the port's KV heads line up (sliding: 4 q / 2 kv per chip; full: 4 q / 1 kv, the GQA head
of the chip's q heads, the same `kv_replicated` mapping as the demo loader). The trajectories differ (device and
CPU greedy part at tokens 3 / 11 on two prompts), and HF's first round seeds the drafter with the hidden at
position 0 (an HF quirk), so only the order of magnitude is comparable.

## How it works (`tt/mtp/`)

Round r: anchor token `a` at position `p` (committed, K/V not yet written), seed `h` = the target's post-final-norm
hidden at `p - 1`, as in HF:

1. drafter, K steps at the fixed position `p`: `(a, h) -> d1`, `(d1, h1) -> d2`, ...; device argmax chains each
   draft into the next step;
2. verify: one batched target decode of K+1 rows `[a, d1..dK]` at positions `p..p+K` on the user's page-table row,
   logits and post-norm hidden of every row; argmax with the plain decode's sampler (same tie rule);
3. host reads the 2K+2 ids, accepts the longest prefix with `d(i+1) == argmax(row i)`, commits `d1..dn` and the
   target token `g`; next anchor `g` at `p+n+1`, next seed = verify hidden row `n` (a device copy).

Steps 1-2 are one trace per K (`SpeculativeDecoder.generate_traced`); per round the host refreshes 11 small inputs
(0.17 ms). The drafter's SDPA attends positions `<= p` when slot `p` holds K/V (a rejected draft's, as HF), else
`<= p - 1`. Rejected rows leave K/V beyond the committed length; the next verify rewrites every slot it attends
before SDPA reads it, and SDPA never reads past a row's position (checked: output equals the one-token-per-step
reference).

| Piece | What | Cost (`results/WS6/components.txt`) |
|---|---|---|
| `lean_drafter.py` | The it-assistant on the port's ops: q projection padded with one zero k and v head so `QkvHeadsNormRope` does head split + q norm + RoPE in one program; the target layer's (28 / 29) paged SDPA config and cache; `ConcatHeadsDecode`; TP4 over heads and the MLP, two persistent-buffer all-reduces per layer straight into an 8x4 width-sharded residual; sharded RMSNorms; 1D-mcast matmuls; BFP8 weights and LM head; RoPE rows gathered once per round | 95 programs, 0.97 ms / step traced (demo `Gemma4AssistantModel`: 231 programs, 1.70 ms); acceptance 2.63 vs 2.68 for the bf16 demo drafter |
| `argmax.py` | Row-0 argmax of the vocab-sharded drafter logits: transposed two-stage max, `>=` mask, one batched matmul against a small-integer index table, combined over chips (exact for a unique max; a tie gives an in-range draft that the verify rejects) | 0.21 ms (sampler 0.52) |
| verify K/V writes | Rows share one page: one batched `paged_update_cache` races on the tile. The gemma4 `PagedUpdateRows` op writes all rows of the user per tensor (unit tests 71/71 pass) | +0.25 ms per verify (one update per row: +1.3 ms) |
| `drafter.py` | The demo drafter on this mesh (gemma4 CCL manager, `MeshConfig(tp=4)`), kept for A/B (`lean=False`) | |

Round at K=5: verify 15.25 ms + its argmax 0.52 + 5 x (0.97 + 0.21) + host ~0.3 = 23.4 ms (first traced version
29.1 ms, 127.6 tok/s, `k_sweep_first_version.json`). Device profile of one round: 2261 programs, 20.2 ms kernel
time, of which the verify is 1577 programs / 15.1 ms.

Switches: `GEMMA4_26B_MTP_KV_WRITES=loop`, `GEMMA4_26B_MTP_DRAFT_ARGMAX=sampler`, `GEMMA4_26B_MTP_WEIGHTS` /
`GEMMA4_26B_MTP_LM_HEAD` (`bf16`, `bfp8` default, `bfp4`: BFP4 head measured no faster, acceptance 2.60).

## Decode-path hooks (each its own commit)

- `tt/model.py`: `logits()` split into `final_norm()` + `head_logits()` (same ops); `prefill_forward` / `decode_forward`
  take `return_hidden` (logits plus post-norm hidden) and `decode_forward` takes `kv_write_rows` (verify K/V write
  order; set on each layer's attention for the call). Defaults unchanged.
- `tt/multichip_decoder.py` `_LocalAttention.decode` (batched rows): with `kv_write_rows` a list of masked position
  tensors, one `paged_update_cache` per row; with an object (`positions`, `page_table`), `PagedUpdateRows`. Without
  it the same two updates as before.
- `tt/generator.py`: `Gemma4Generator.generate_speculative(prompt, max_new_tokens, draft_len=5)`.

`perf/tools/quick_check.py --batches 1,32 --check-batch 2,8,32` against `results/WS0/quick_baseline.json`: **PASS**,
B1_layers_ms 11.92, B32_layers_ms 27.98, KV equal (`results/WS6/quick_check.json`). B32 varies 26.7-28.0 run to run on
both this branch and its base (base c420d1f22eb: 26.66, 27.86; this branch: 27.92, 27.98, 26.80, 27.11); the hooks
add no device op to plain decode.

## Known limits / left

- Token identity vs batch-1 plain greedy (above).
- Batch 1, greedy only. vLLM integration (`generator_vllm.py`) and batch > 1 (K+1 rows per user, up to 32 rows: 5
  users at K=5) are not started.
- The verify is the WS1 batched decode: 15 ms for 6 rows vs 11.6 ms for batch 1 (expert-parallel MoE over the union
  of the rows' experts). Faster batched decode lifts WS6 directly; so would a verify that reads the KV once for the
  K+1 rows.
- Prompts with low acceptance gain less (short story: 1.58 accepted, 107 tok/s). No adaptive K yet.
- Context: measured at ~250-450 tokens; the drafter attends the target's unbounded caches (`max_seq_len` 4096 in the
  tool). Longer contexts are untested.
- The round trace is recaptured per K and whenever the generator reallocates its cache (longer request).
