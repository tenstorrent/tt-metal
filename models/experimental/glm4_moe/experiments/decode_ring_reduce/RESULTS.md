# Track 1 — DeepSeek fused ring TP reduce (decode)

Replace the decode MoE **TP=8 (cluster_axis=0)** all-reduce with the DeepSeek ring ops:

```
concat([shared*(1/DP), routed], dim=0)                       # [2,1,32,5120] per device
 -> deepseek_moe_fast_reduce_nc(dim=0, split_size=5120/8)     # local sum + split list
 -> deepseek_moe_reduce_scatter(dim=3, cluster_axis=0, Ring)  # cross-device ring RS -> [1,1,32,640]
 -> all_gather(dim=3, cluster_axis=0, Ring)                   # restore [1,1,32,5120] replicated
```
The DP (cluster_axis=1) reduce is unchanged. Gated by `GLM4_MOE_MOE_RING_REDUCE=1` (default off).

## Op-level validation — PASSED (Mesh(8,4), hidden=5120)

`test_ring_reduce_glm4.py` vs a torch reference (sum of `shared/DP + routed` over the 8 TP rows):

| check | PCC |
|---|---|
| ring reduce vs torch ref (col 0..3) | 0.99977 |
| all_gather row-replication | 1.000000 |

Trace-capable fabric (`FABRIC_1D_RING`), 32 s, no weights. Confirms the op sequence and the
`three_links_partial` config (rs-input NdShardSpec `[1,1,32,128]` × 5 cores `(0,0)-(4,0)`, 3 links,
L1-interleaved output) are numerically equivalent to today's 2-step all-reduce for GLM4's geometry.

Run:
```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
./python_env/bin/python -m pytest -svq \
  models/experimental/glm4_moe/experiments/decode_ring_reduce/test_ring_reduce_glm4.py
```

## Code

- `tt/moe_tt.py`: `ring_tp_reduce_supported()` (shape/geometry guard) and
  `ring_tp_reduce_combined()` (the fused reduce).
- `tt/decoder_layer_tt.py`: `_moe_forward` fused-reduce block — ring branch behind the flag,
  full fallback to `add + _simple_all_reduce` otherwise. Prefill excluded.

## Full-model A/B — DONE (B1, FUSE off, decode-winner env)

| run | decode mean (ms) | min/max | tok/s | output |
|---|---|---|---|---|
| baseline (ring off) | 140.7 | 138.5 / 144.1 | 7.10 | coherent |
| ring (`MOE_RING_REDUCE=1`) | 140.6 | 138.6 / 142.1 | 7.11 | coherent |

**Correct but perf-neutral (~0.1 ms, within noise).** The ring path *did* execute: under
deterministic greedy the ring run's output wording differs from baseline (same coherent answer),
i.e. the reduce numerics changed as expected from the op-level PCC 0.9997 — not a silent fallback.

**Why no speedup:** GLM4 restores full-hidden replication with an `all_gather` right after the
ring `reduce_scatter`, so it trades one `all_reduce` for `reduce_scatter + all_gather` — roughly
equal work. DeepSeek's win comes from **keeping the output TP-sharded** (no all_gather) and
carrying a sharded residual stream, only reverting TP at select points. Capturing a real decode
gain here therefore needs the larger **sharded-residual restructuring** (attention O-proj + norms +
residual add operating on `hidden/8`-sharded activations), not just swapping the reduce op.

The ring code stays in (default-off, validated) as the reusable building block for that work.

## Not pursued yet — full-model A/B for the other tracks (needs Galaxy device)

Baseline vs ring, `FUSE_EXPERTS_GATE_UP` OFF, otherwise the decode winner env:
```bash
# ... production env (see plan) with:  GLM4_MOE_NORM_L1=1
#     GLM4_MOE_MOE_SPARSE_FIDELITY=lofi GLM4_MOE_ATTN_FIDELITY=lofi
# A) baseline:  (leave GLM4_MOE_MOE_RING_REDUCE unset)
# B) ring:      export GLM4_MOE_MOE_RING_REDUCE=1
./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
  --mesh-rows 8 --mesh-cols 4 --model-id cerebras/GLM-4.7-REAP-218B-A32B \
  --batch-size 1 --max-new-tokens 128 --min-cache-tokens 256 --kv-cache-dtype bf8 \
  --enable-trace --trace-mode sampling --warmup-decode-trace \
  --prompt "Explain in two short paragraphs how mixture-of-experts models route tokens to experts, and why that can improve efficiency."
```
Gate: coherent output matching baseline AND lower `subsequent: mean=` decode_mean_ms.
Also run at `--batch-size 8`. The ring RS input requires token block ≤ 32 (decode); the guard
`ring_tp_reduce_supported()` falls back automatically otherwise.
