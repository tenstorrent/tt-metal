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
| `decode_forward(streams, positions, host_positions=None)` | `[1,1,B,4*4096]` fp32 (B == max_batch user slots), `positions` int32 `[B]` = each user's context length | `[1,1,B,4*4096]` | KDA state updated in place; DSA cache appended at `positions[b]`; sparse path when `host_positions` has a value >= 2048 |
| `capture_decode_trace(streams, positions)` / `replay_decode(trace_id)` | persistent device inputs | output tensor at a fixed address | dense decode is trace-replayable (state updated inside the captured graph) |

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
| `test_dsa.py` | layer 3 MLA, T=96 / 1000 + 3 decode | prefill 0.99964 / 0.99961, latent cache 0.99997, decode 0.9991-0.9999 |
| `test_decoder_layer.py` | full layers 0 / 3 / 4 vs HF dump, T=256 + 4 decode; T=200 non-aligned live HF | 0.99984 / 0.99992 / 0.99997 prefill, 0.9999 decode; 0.99980 / 0.99993 |
| `test_decode_trace.py` | traced decode replay (layers 0/3/4); two users (100 + 37 tokens, slots) | replay == eager (0.9999); per-user 0.9998-0.99998 |
| `test_dsa_sparse.py` | layer 3 at 4097 tokens (sparse path) + decode 4097..4100 | prefill 0.99977 (selected rows 0.99990), decode 0.9999 |
| `test_determinism.py` | layers 0, 3: repeated prefill and traced-decode replay, `throw_exception_on_fallback` raised | bit-identical outputs, no host fallback |
| watcher | `TT_METAL_WATCHER=10` on the layer-3 dump test (`~/ashai/glm53/_logs/watcher_layer3`) | PASSED, 10 dumps, no watcher exceptions |
| `test_perf_layer.py` | warmed prefill S=1024, traced decode B=1 (wall clock) | layer 3: 314 ms prefill, 10.95 ms/step; layer 4: 300 ms, 10.96 ms/step |

Context contract: `doc/context_contract.json`. Full log: `doc/functional_decoder/work_log.md`.
Profiler note: Tracy device zones of the dense SDPA prefill kernel collide (`compute_streaming.hpp:632` vs `:1769`,
hash 0x6835) on main 8cfcaf754d3, so tt-perf-report tables come from layer 4 (no SDPA); layer 3 keeps wall-clock numbers.

## Running

```bash
source ~/ashai/glm53/env.sh
TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests -s           # all device tests (one chip)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0,1,2,3,4 --seq-len 256 --num-decode 4 \
    --out bringup/references/glm53_flash/T256                            # regenerate the HF dumps (CPU)
```

## Known limits (single chip, v1)

- Performance is the design's Phase 2/3 work: the KDA decode recurrence and the per-block MoE are composites,
  the sparse decode untilizes the user's latent prefix per step, and the decode SDPA runs on 4 cores per user.
- The sparse DSA decode path takes host positions (chunk geometry) and is therefore not captured in the trace.
- `moe_grouped_topk` is not used (bf16 ranking swaps near-tied experts); the exact fp32 composite costs ~8x5 ops.
