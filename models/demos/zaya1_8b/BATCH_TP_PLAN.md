# ZAYA1-8B multi-user / multi-chip performance plan

Diagnosis: single-chip decode is **op/dispatch bound** (99.6 ms/tok, ~5000 op
launches; bfp8 experts → only 3%). Multi-chip TP/EP does not cut per-chip op count, so
it does **not** reduce single-token latency (2-chip TP lm_head = 1.08×). The levers, in
the order we execute them:

## Phase 1 — single-chip batch-on-M decode (throughput)  [CSEJ tracking TBD]
Serve B users per step on one P150a; throughput ~B× until compute/dispatch-bound; each
user's token stream must equal its own B=1 result.

Layout reconciliation (the crux):
- Residual stream / RMSNorm / **MoE** keep tokens on the M axis: `hidden [1, B, 2048]`
  (B in the old "S" slot). MoE batched experts become `[16,B,2048]@[16,2048,4096]` — **no
  code change** (B just replaces S).
- **CCA attention** needs B as a real batch dim with per-user KV. At the CCA boundary
  `permute [1,B,2048] -> [B,1,2048]`, run batched decode attention (`[B,nh,1,128]`,
  per-user KV `[B,nkv,pos/MAX,128]`, per-user pos), then `permute back -> [1,B,2048]`.

batch=1 hardcodes to fix:
- `to_heads`/`from_heads`/`_l2_h`: leading `1` and slice stops → use the real batch dim B.
- `cache.k/v[layer]`: `[1,nkv,pos,128]` → `[B,nkv,pos,128]`; conv_state/prev_hs → `[B,1,*]`.
- `model.decode_step`/`_argmax_id`: accept B token ids, return B ids (argmax per row).
- `trace.py TraceState`: `kc/vc[B,nkv,MAX,128]`, `onehot/amask/cos/sin` per-user; `hin/hout [1,B,2048]`.

Steps: (1) eager batched decode + per-user KV, validate B=2 (same-prompt → identical to
B=1; different-prompt → each matches its own single-run golden); (2) batched trace; (3)
bench B=1/2/4/8 (ms/step, tok/s/user, aggregate). Expect near-free batching (qwen36: B=8 +4ms).

## Phase 2 — 4-chip expert-parallel MoE (+ replicated CCA)  (throughput/capacity)
`(1,4)` sub-mesh + FABRIC_1D. Shard 16 experts across 4 chips (4/chip) on the expert dim;
replicate CCA/router/embed. **First cut: expert-sharded DENSE + `all_reduce` of the
gated sum** — token-exact, no all-to-all, simplest correct EP. Validate vs single-chip,
bench. Stretch: true sparse all-to-all dispatch/combine (drops the dense-16 waste).
Note: the 8×P150a host is a **2×4 grid (not a torus)** — no free Ring all_reduce; use a
1×4 line sub-mesh for collectives.

## Phase 3 — CCA op-fusion (single-chip latency)
Break the dispatch bound: `to_heads`/`from_heads` → `ttnn.experimental.nlp_create_qkv_heads_decode`
/ `nlp_concat_heads_decode`; evaluate SDPA-decode while preserving the fp32-softmax
token-exactness; fewer rope/`_assemble` slices. Re-profile + bench, keep token-exact.

Validated infra: FABRIC_1D + tt_ccl all_gather work on 1..8 chips (2×4 grid). Run via
`run_zaya_multi.sh` (TT_DEVICES=...).
