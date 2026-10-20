# GLM-5.3-Flash functional decoder (stage 1, single Blackhole chip)

Design of record: `~/ashai/GLM_5.3_DESIGN.md` (Phase 1 = this stage). Code: `models/demos/glm53_flash/`.
Host: `sjc3-glx-a04u07`, one chip of the 32-chip Blackhole Galaxy (`TT_VISIBLE_DEVICES=0`, `MeshShape(1,1)`),
tt-metal main `8cfcaf754d3` + the commits on `arg/glm53_flash`.

## Contract

`tt/functional_decoder.py::FunctionalDecoder` is one decoder layer of any kind with device-owned state.

| Call | Input | Output | State effect |
|---|---|---|---|
| `from_state_dict(state_dict, hf_config=cfg_or_config_json, layer_idx, mesh_device, max_batch, max_seq_len)` | raw checkpoint tensors of `model.language_model.layers.{i}.` (prefix stripped, fp8 groups dequantized by `reference/checkpoint.load_layer_raw`) | decoder | allocates the KDA user slots / the DSA paged latent + pooled-key index caches |
| `prefill_forward(streams, user, seq_len)` | packed mHC streams `[1,1,T_pad,4*4096]` fp32, `T_pad % 32 == 0`, rows past `seq_len` padding | `[1,1,T_pad,4*4096]` fp32 | KDA: writes slot `user`; DSA: fills the user's cache pages `[0, T_pad)`, pooled keys, tail ring |
| `decode_forward(streams, positions)` | `[1,1,B,4*4096]` fp32 (B == max_batch user slots); `positions` = `DecodePositions` from `make_decode_positions(host_positions)` / `update_decode_positions` (persistent `[B]` int32: position, position // 4, position % 4, + host values) | `[1,1,B,4*4096]` | KDA state updated in place; DSA latent appended and the pooled-key index maintained (dense and sparse); sparse path when any host position >= 2048 |
| `capture_decode_trace(streams, positions)` / `replay_decode(trace_id)` | persistent device inputs | `(trace_id, output tensor, warm-up output)` | dense decode is trace-replayable (state and index updated inside the captured graph); refused for DSA decoders with max_seq_len > 2048 unless `allow_sparse_range` |

Residual convention: HF `[B, S, 4, 4096]` <-> packed `[1, 1, B*S, 16384]` (`host_streams_to_device` /
`device_streams_to_host`, test boundary only).

## Modules

| File | Content | Reuses |
|---|---|---|
| `tt/config.py` | frozen config from `config.json` | |
| `tt/mhc.py` | mHC hyper-connection (fused Sinkhorn parametrization, fp32) | `deepseek_v3_d_p/tt/mhc/TtMHCWrap` |
| `tt/kda.py` | KDA: prefill via `ttKDA` (chunked kernels), decode = composite recurrence on `[B*H,128,128]` fp32 with in-place state writeback | `deepseek_v3_d_p/tt/kda` |
| `tt/dsa.py` | MLA (NoPE): dense causal SDPA prefill (<= 2048), paged latent cache, absorbed `paged_flash_multi_latent_attention_decode`; sparse prefill/decode via the indexer + `sparse_sdpa` | paged cache ops, flash MLA decode, sparse SDPA |
| `tt/indexer.py` | GLM k-pool indexer: pooled-key cache, `indexer_score_dsa(key_compression_ratio=4)`, `topk_large_indices` over 4x-expanded pool scores, tail ids | |
| `tt/moe.py` | router (exact fp32 composite top-8) + `sparse_matmul` experts per 32-token block + shared expert | gpt_oss sparse-matmul pattern |
| `tt/mlp.py` | clamped SwiGLU (dense layers, shared expert) | |
| `tt/decoder_layer.py` | mHC(attention) -> mHC(FFN) composition | |
| `reference/` | checkpoint access + fp8 dequant, HF layer reference runner (dumps), KDA reference equivalence check | transformers 5.20 `glm5_next` |

## Evidence (real weights, PCC vs HF)

| Test | What | Result |
|---|---|---|
| `test_kda.py` | layer 0 KDA, T=96 / 1000 + 3 decode | prefill 0.99994 / 0.99995, state 0.9999, decode 0.9999 |
| `test_moe.py` | layer 4 MoE, T=32 / 256 | router sets identical 32/32, 255/256; output 0.99993 / 0.99995 |
| `test_dsa.py` | layer 3 MLA, T=1, 2, 3, 31, 33, 96, 1000 + 3 decode | prefill 0.99989 (T=1) .. 0.99961 (T=1000), latent cache 0.99997, decode 0.9991-0.9999 |
| `test_decoder_layer.py` | full layers 0 / 3 / 4 vs HF dump, T=256 + 4 decode; T=200 non-aligned live HF | 0.99984 / 0.99992 / 0.99997 prefill, 0.9999 decode; 0.99980 / 0.99993 |
| `test_decode_trace.py` | traced decode replay vs the HF dump (layers 0/3/4); two users (100 + 37 tokens, slots) | replay PCC 0.9999 (same as the eager run of the same steps); per-user 0.9998-0.99998 |
| `test_dsa_sparse.py` | layer 3 at 4097 tokens (repeated prompt): dense rows [0,2048) + sparse rows, decode 4097..4100 | prefill 0.99992 (rows<2048 0.99994, rows>=2048 0.9999), decode 0.9999 |
| `test_dsa_long.py` | random-token prompt: 4097 + 12 sparse decode steps with TT/HF selected-id sets (contiguous-prefix contract asserted) and the pooled-key cache vs a host recomputation; 2040-token dense prefill (slot 1, permuted page table) then decode 2040..2063 with the short user 0 checked every step; 16385-token prefill + 2 decode; prompts 2042/2043 (tails 2, 3) + 12 decode steps with the pool-cache check | prefill 0.99994 (rows<2048 0.99988, >=2048 0.99986), decode 0.9997+, 98-99 % ids common, pool cache 0.99999; transition >= 0.9996 both users; 16K 0.99983 (0.99988 / 0.99980); tails 2/3: prefill 0.9987-0.9990, decode >= 0.9978, pool cache 0.99999 |
| `test_batch8.py` | 8 user slots (lengths 1..200, permuted page table), layers 3 and 4, 3 batched decode steps | prefill >= 0.9968 (1-token user), decode worst per user 0.9994 |
| `test_determinism.py` | layers 0, 3: repeated prefill and traced-decode replay with `throw_exception_on_fallback`; sparse prefill + sparse decode repeated | bit-identical outputs |
| watcher | `TT_METAL_WATCHER=10` on the layer-3 dump test (`~/ashai/glm53/_logs/watcher_layer3`) | PASSED, 10 dumps, no watcher exceptions |
| `test_all_layers.py` | every layer 0..44 on its own HF input (`L45_S128` dump: 128 random tokens + 2 decode steps through the full HF stack, produced layer by layer with `hf_layer_reference.py --stream`), prefill S=128 + 2 decode vs the live HF layer, router top-8 sets with near-tie margins, attention / FFN sub-block PCCs | 45/45 pass: prefill 0.99886 (layer 23) .. 1.0, decode >= 0.99786 (layer 27, one 4e-4-margin expert flip), per-row prefill min 0.9755 (layer 23 row 125); router sets 109-126/128 identical, every flipped prefill row below the median 8th-9th margin except 3 rows of layer 43; sub-blocks: attention >= 0.99915, FFN >= 0.99589 (prefill); HF live == dump (1.000000) for all 45 layers; 4 chips, ~2 min/layer |
| `test_perf_layer.py` | warmed prefill S=1024 (wall), traced decode B=1 (one replay, tt-perf-report `Device Time`) | prefill: layer 3 314 ms, layer 4 298 ms (device 293 ms, 1002 ops); decode: layer 3 10.87 ms/step device (435 ops, wall 11.31), layer 4 10.76 ms/step device (471 ops, wall 11.2) |

Context contract: `doc/context_contract.json`. Full log: `doc/functional_decoder/work_log.md`.
Profiler note: Tracy device zones of the SDPA prefill kernels (dense and sparse share `compute_streaming.hpp`: `:632` vs
`:1769`, hash 0x6835) collide on main 8cfcaf754d3, so prefill tables exist for layer 4 only; decode tables for both layers
come from one-replay runs (`GLM53_PERF_DECODE_STEPS=1`, layer 3 with `GLM53_PERF_SEED_ONLY=1`), tables under `doc/functional_decoder/tracy/`.

## Running

```bash
source ~/ashai/glm53/env.sh
TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests -s           # all device tests (one chip)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0,1,2,3,4 --seq-len 256 --num-decode 4 \
    --out bringup/references/glm53_flash/T256                            # regenerate the HF dumps (CPU)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0-44 --seq-len 128 --random-ids 45 \
    --num-decode 2 --stream --out bringup/references/glm53_flash/L45_S128   # all-layer dump (layer by layer, ~25 min)
TT_VISIBLE_DEVICES=0 GLM53_LAYERS=0-11 pytest models/demos/glm53_flash/tests/test_all_layers.py -s --timeout 1800
                                                                         # every layer vs HF (shard GLM53_LAYERS over chips)
```

## Known limits (single chip, v1)

- Prefill is single-shot (one chunk from position 0 per user): the design's chunked prefill (``actual_start``
  carries exist in ttKDA and the layer plumbs them, the DSA cache is filled from 0) is not wired through the public
  API yet. The sparse prefill materializes a `[T - 2048, 4 * pools]` bf16 score matrix per DSA layer (~0.5 GB at 16K),
  so the practical single-chip prefill bound is that quadratic intermediate, not DRAM.
- Runtime audit method: `grep -nE 'from_torch|to_torch|torch\.'` over `tt/*.py` forward paths (hits are
  construction-time or first-use-cached constants only) plus `throw_exception_on_fallback=True` runs
  (`test_determinism.py`); the ring tail rows and the KDA scalars use tensors created once per geometry/value.

- Performance is the design's Phase 2/3 work: the KDA decode recurrence and the per-block MoE are composites,
  the sparse decode untilizes the user's latent prefix per step, and the decode SDPA runs on 4 cores per user.
- The sparse DSA decode path takes host positions (chunk geometry) and is therefore not captured in the trace;
  the pooled-key index is nevertheless maintained inside the traced dense graph, so a user can cross 2048.
- Indexer scores are ranked at bf16 resolution (`indexer_score_dsa` + `topk_large_indices`): 98-99 % of the
  HF-selected ids per step on a random prompt, layer PCC >= 0.9997.
- `moe_grouped_topk` is not used (bf16 ranking swaps near-tied experts); the exact fp32 composite costs ~8x5 ops.
