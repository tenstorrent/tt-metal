# Functional decoder — Qwen3-Coder-30B-A3B-Instruct on Blackhole

Stage 01 of the repo-local TTNN autoport pipeline: one `Qwen3MoeDecoderLayer`
implemented in TTNN and validated against a layer-only HuggingFace reference.

| | |
|---|---|
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Device | Blackhole `p300c`, 1×1 mesh, 32 GB/die |
| Scope | decoder layer 0, batch 1 |
| Implementation | `tt/functional_decoder.py` |
| Reference | `tests/reference.py` (single layer, real checkpoint weights) |

## Architecture

```
hidden 2048 | 32 Q heads | 4 KV heads (8:1 GQA) | head_dim 128
128 experts, top-8, moe_intermediate 768, SwiGLU
QK-norm on Q and K | no attention bias | no sliding window | every layer MoE
```

`32 × 128 = 4096 ≠ hidden`, so `head_dim` is an independent config field, not
`hidden / n_heads`.

## Results

### Correctness (PCC vs HuggingFace)

| path | lengths | PCC |
|---|---|---|
| RMSNorm | 32, 128 | 0.99999 |
| attention prefill | 32 / 128 / 512 | 0.9997 / 0.9996 / 0.9994 |
| router dense weights | 32 / 128 | 0.99999 / 0.9949 |
| experts (reference routing) | 32 / 128 | 0.99981 / 0.99980 |
| MoE block end-to-end | 32 / 128 | 0.99981 / 0.99811 |
| **decoder layer, prefill** | 32 / 128 / 512 | **0.9991 / 0.9995 / 0.9994** |
| **decoder layer, non-aligned** | 33 / 100 / 257 | **0.9990 / 0.9995 / 0.9994** |
| decode attention (contiguous) | pos 32 / 128 | 0.99953 / 0.99941 |
| decode attention (paged 32/64) | pos 32 / 128 | 0.99953 / 0.99945 |
| decoder layer decode, multi-step | pos 32/33/34 | 0.9947 / 0.9996 / 0.9999 |
| traced decode vs eager | — | **1.0 (bit-exact)** |

Acceptance bar is PCC ≥ 0.995 for the composed layer; met at every length.
Sub-module tests use 0.99 because a single token compares only 2048 values and
is correspondingly noisier.

### Performance (baseline for stage 02)

Warmed, device-synchronised, median of N iterations.

| prefill | median | µs/token |
|---|---|---|
| S=128 | 69.16 ms | 540.3 |
| S=512 | 274.71 ms | 536.6 |
| S=1024 | 548.69 ms | 535.8 |
| S=2048 | 1098.21 ms | 536.2 |

| traced decode | median | tok/s/layer |
|---|---|---|
| ctx 128 | 1.565 ms | 639.1 |
| ctx 1024 | 1.661 ms | 601.9 |
| ctx 4096 | 1.993 ms | 501.7 |

Prefill is flat at ~536 µs/token, i.e. strictly linear in sequence length: cost
is dominated by per-token expert compute, not by attention.

### Op profile

| | prefill S=512 | decode (paged) |
|---|---|---|
| `SparseMatmulDeviceOperation` | **96.79%** (265.45 ms, 48 calls) | **92.6%** (17.64 ms, 6 calls) |
| everything else | 3.21% | 7.4% |

Attention is 0.08% of prefill. The MoE experts are the whole cost.

## Known limitations

1. **Expert weights are bf16, not `bfloat8_b`.** Deliberate: a PCC miss during
   bringup should mean a bug, not quantisation. `bfloat8_b` is the production
   dtype and a measurable stage-02 step.
2. **Prefill computes all 128 experts per chunk** (`active=128/128` in the
   profile) and masks after the down-projection. Correct and trace-friendly,
   but ~16× more expert arithmetic than the top-8 requires. Decode already uses
   real sparsity.
3. **Expert matmuls reach ~5.4% of peak FLOPs** and occupy 24 of 110 worker
   cores for gate/up (64 for down). Program-config and core-grid tuning is
   untouched — stage 02 territory.
4. **PCC is verified to 512 tokens.** Longer lengths are capacity/liveness
   probes only, because a 262144-token torch reference does not fit in host
   RAM. This is a limit of the reference, not of the decoder.
5. **Single device, batch 1.** Multi-device is stage 03.
6. `fp32_dest_acc_en` is off for expert matmuls — mandatory on Blackhole, see
   tt-metal #49068.

## Context

Full HF context, no reduction. See `../context_contract.json`.

| | tokens | evidence |
|---|---|---|
| HF advertised | 262144 | `config.max_position_embeddings` |
| decode supported | 262144 | step at position 262143, 2.42 s |
| prefill supported | 262144 | single-shot, 192.23 s |

## Reproduce

```bash
source python_env/bin/activate

# full correctness suite
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q

# watcher-clean run
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DUMP_ALL=1 \
  pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/ -q

# performance (writes perf_prefill.csv / perf_decode.csv here)
pytest models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_perf.py -q

# op-level profile
python -m tracy -v -r -p --sync-host-device -o <outdir> -m pytest \
  "models/autoports/qwen_qwen3_coder_30b_a3b_instruct/tests/test_decoder_layer.py::test_decoder_layer_vs_reference[blackhole-s512-1x1]" -q
tt-perf-report <outdir>/reports/*/ops_perf_results_*.csv

# context contract
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct --require-contract
```

## Artifacts

| file | contents |
|---|---|
| `perf_prefill.csv` | warmed prefill latency by sequence length |
| `perf_decode.csv` | traced decode latency by context length |
| `ops_perf_prefill_s512.csv` | raw op profile, prefill S=512 |
| `ops_perf_decode_paged32.csv` | raw op profile, paged decode |
| `tt_perf_report_prefill_s512.txt` | formatted report incl. stacked summary |
| `tt_perf_report_decode_paged32.txt` | same, decode |
| `work_log.md` | chronological log, decisions, failures, fixes |
| `../context_contract.json` | context contract (validated by the checker) |
