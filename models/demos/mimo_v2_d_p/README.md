# MiMo-V2.6-Flash — disaggregated chunked prefill (layers)

Layer-level bring-up of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
for the common/prefill engine (tt-d-gen), in the `deepseek_v3_d_p` / `gemma4_d_p` style: sequence
block-cyclic over SP mesh rows, heads over TP cols, experts over all chips, KV in block-cyclic DRAM caches that
decode reads by address. The full 48-layer model does not fit a 4-chip QuietBox; the code runs any contiguous
layer range and is laid out for a BH Galaxy (8x4).

## Architecture (text backbone)

| | GA (global attention, 9 layers) | SWA (39 layers) |
|---|---|---|
| Q heads / KV heads | 64 / 4 | 64 / 8 |
| qk / v head dim | 192 / 128 | 192 / 128 |
| rope | first 64 dims, theta 1e7 | first 64 dims, theta 1e4 |
| window / sink | — | 128 / per-head sink bias |
| FFN | L0 dense SwiGLU 16384, else 256-expert top-8 sigmoid MoE (2048) | MoE |

V is scaled by 0.707 (folded into the V projection); scale 192^-0.5.

## Layout

```
reference/   config.py, remote_st.py (HF range reads), weights.py (fp8 / mxfp4 dequant), hf.py (HF golden)
tt/          attention/{attention,sdpa,kv_cache}.py, rope.py, ffn.py (dense MLP, gate, EP MoE), decoder.py,
             model.py, tt_prefill_runtime.py, mm_configs.py, ccl.py
tt/runners/  adapters/mimo_v2.py (PREFILL_MODEL=mimo_v2_d_p), kv_chunk_table.py, manifests/ (Gate 1 on 2x2)
tests/       unit/ (attention, decoder layer, host-side TP/EP layout for TP 1..8), test_model_prefill.py
             (layers 0-5 stitched vs the HF chain), test_runtime_contract.py (dgen contract), perf/
```

Checkpoint gotchas (see `reference/weights.py`): the fused fp8 `qkv_proj` is **pre-sharded in 4 chunks**
`[Q_r|K_r|V_r]` with per-chunk 128x128 scales (vLLM `_shard_fp8_qkv_proj`); experts are MXFP4. The MoE gate's
`e_score_correction_bias` (~1-2) must stay fp32 (bf16 flips ~16% of tokens' expert choice).

## Parallel layout (2x2 QuietBox -> 8x4 Galaxy)

* SP rows: chunk block-cyclic, `chunk/SP >= 128` (one-hop sliding halo), chunk-aligned starts for SWA.
* TP cols: column-parallel qkv (per col `[q | k | v]`), row-parallel o_proj + all-reduce; KV heads per col
  = n_kv/TP (2x2: GA 2, SWA 4; Galaxy: GA 1, SWA 2), duplicated when TP > n_kv.
* EP: 256 experts over all chips (2x2: 64/chip, Galaxy: 8/chip), dispatch along SP, capacity 2x expected
  load (`moe_capacity_factor`: 4 on 2x2, 2 on Galaxy).
* `tests/unit/test_galaxy_layout.py` checks the TP layouts for TP 1/2/4/8 and the EP sizing for 32 chips on host.

## Running

All device tests via `scripts/run_safe_pytest.sh` with a clean env
(`env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD`):

```
tests/unit/test_attention_block.py      # GA / SWA attention, 3 chunks
tests/unit/test_decoder_layer.py        # L0 GA+dense, L1 SWA+MoE, L5 GA+MoE
tests/test_model_prefill.py             # layers 0-5 on a real prompt, 2 x 4k chunks: hidden + KV vs HF
tests/test_runtime_contract.py          # adapter -> runtime -> prefill_chunk (pad tail, slot 1), acks, KV, table
tests/perf/test_attention_perf.py       # --profile; analyze_attention.py <csv> (SDPA FPU util)
tests/perf/test_layer_perf.py           # --profile; analyze_tags.py <csv> --ops
```

dgen Gate 1 (mock migration + producer KV PCC through the chunk table):

```
# runner (bare python, exports the binding's global_env) — see tt/runners/manifests/mimo_v2_binding_2x2_mock.yaml
python -m models.demos.common.prefill.runners.prefill_runner
python -m models.demos.common.prefill.runners.prefill_producer --manifest models/demos/mimo_v2_d_p/tt/runners/manifests/mimo_v2_producer_mock_pcc.yaml
```

## Results (BH QuietBox 2x2, 2026-09-25)

### Correctness (vs the HF module in fp32)

| test | result |
|---|---|
| attention block, 3 chunks (GA / SWA, HiFi2 SDPA) | PCC 0.99999 / 0.99999 (also with a padded KV shard) |
| decoder layer L0 GA+dense / L1 SWA+MoE / L5 GA+MoE (embedding input) | 0.99999 / 0.9996 / 0.991 |
| layers 0-5 stitched, real prompt, 2 x 4k chunks | hidden >= 0.9991, KV >= 0.9974 |
| dgen runtime contract (pad tail, slot 1, 12 acks, chunk table) | KV >= 0.9974 |
| dgen Gate 1: bare runner + prefill_producer (H2D, 24 acks, device-less KV read via table) | PASSED, min 0.9974 |
| host TP layout TP=1/2/4/8 (Galaxy TP=4), EP sizing 4 / 32 chips | PASSED |

bf4 experts (default, as DeepSeek) cost the MoE output ~0.98-0.99 PCC per layer; `MIMO_EXPERT_DTYPE=bf8`
gives ~0.997 at ~+0.9 ms per MoE layer on 2x2.

### GA SDPA FPU utilization (HiFi2, vs 4096 FMA-FLOP/cycle/core x 100 SDPA cores @ 1.35 GHz)

| tokens/chip | 8-9k ctx | 33k ctx | 66k ctx |
|---|---|---|---|
| 640 (Galaxy chunk 5120 / SP8) | 39.7% | 51.6% | 55.9% |
| 2048 | 43.0% | 57.4% | 60.0% |

Ceiling analysis: (head, q-chunk) work units over 100 cores balance to at most 80% (640/chip) / 85%
(2048/chip); the plain single-chip SDPA kernel reaches 62.8% at DH 192 (~74% per busy core). The ring path is
within ~10% of that. Further gains need split-K scheduling or kernel-level matmul efficiency.

### Attention layer (device time, per chip, 640 tokens/chip, 33k ctx)

| | baseline | now |
|---|---|---|
| SWA layer | 1041 us | 648 us |
| GA layer | 3940 us | 3531 us |

### Decoder layer (device time, 33k ctx)

| | 640 tok/chip: first profile -> now | 2048 tok/chip |
|---|---|---|
| L0 GA + dense | 4.49 -> 4.44 ms | 12.9 -> 12.7 ms |
| L1 SWA + MoE | 9.24 -> 7.21 ms | 17.6 -> 14.0 ms |
| L5 GA + MoE | 15.5 -> 12.2 ms | 28.6 -> 23.9 ms |

On 2x2 the MoE dominates (64 experts/chip: expert weights ~0.85 GB bf4 per layer per chunk, dispatch /
combine ~16 MB/chip of fabric traffic); a Galaxy holds 8 experts/chip.

### Optimizations

* matmul: 2D-mcast configs, full-width grid, widest L1-fitting in0 block (`tt/mm_configs.py`): qkv 315 -> 188 us,
  o_proj 186 -> 105 us (640/chip).
* `nlp_create_qkv_heads` / `nlp_concat_heads`: per-head NoC batching instead of a barrier per tile
  (105 -> 63 us, 57 -> 35 us).
* `rotary_embedding_indexed(in_place=True)` with a 64-wide cos/sin: partial rope rotates 2 of 6 tiles in place
  (80 -> 32 us).
* ring sliding SDPA: V at 128 (the op's DH == VDH check relaxed to VDH <= DH; kernels are vDHt-generic),
  lazy halo sync (only Q chunks that read the predecessor tail wait): 125 -> 90 us.
* GA SDPA q128 / k1024 (sweep), padded last K chunk rather than small k chunks.
* MoE: fp32 gate logits + bias (accuracy), bf4 experts, dispatch capacity 2x expected load, 3 fabric links
  (dispatch 2.59 / 1.31 / 0.88 ms at 1 / 2 / 3 links).

Next steps: GQA K/V sharing in the sliding ring path (redundant K/V reads, ~50-150 us per SWA layer), split-K
work units for GA SDPA balance, fused rope + head split, keeping the residual stream in L1 at 640 tokens/chip.
