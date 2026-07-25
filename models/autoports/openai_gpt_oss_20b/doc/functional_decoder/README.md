# GPT-OSS-20B IR-derived functional decoder

## Status

This directory contains the functional, single-device translation of one representative middle decoder layer from the supplied GPT-OSS-20B TTNN IR. Both forward paths present in the capture are implemented: full-sequence prefill and single-token decode with a persistent KV cache. This stage does not contain optimized-decoder, multichip, full-model, generator, or vLLM work.

The implementation uses canonical full, unsharded Hugging Face layer weights on a `1x1` mesh. The source graphs used a `1x4` tensor-parallel mesh; compiler collectives and layout glue were collapsed to equivalent dense single-device math and validated against the dense Hugging Face layer.

## IR provenance and classification

IR root:

`/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_1784000876774`

`scripts/classify_graphs.sh` reported:

| MLIR file | Role | `fill_cache` | paged update | decode SDPA | Runtime wrapper |
| --- | --- | ---: | ---: | ---: | --- |
| `ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g0_1784000876774.mlir` | prefill | 48 | 0 | 0 | no |
| `ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g1_1784001120946.mlir` | decode | 0 | 48 | 24 | no |
| `ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g2_1784004518761.mlir` | prefill | 48 | 0 | 0 | no |
| `ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g3_1784004584320.mlir` | decode | 0 | 48 | 24 | no |
| `ttnn_runtime_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g0_1784000876814.mlir` | prefill | 48 | 0 | 0 | yes |
| `ttnn_runtime_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g1_1784001120978.mlir` | decode | 0 | 48 | 24 | yes |
| `ttnn_runtime_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g2_1784004518799.mlir` | prefill | 48 | 0 | 0 | yes |
| `ttnn_runtime_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_runc01f_g3_1784004584352.mlir` | decode | 0 | 48 | 24 | yes |

The selected compiler, non-runtime graphs were `g0` for prefill and `g1` for decode. Their SHA-256 values are respectively `95a2eac6f9188ce1d88f93ef23d9bf768b2c1a967990402e282c3bdacaeaf338` and `62855b105ebba55addcfbb11fd4564e35ed7c3c338d3a663a45e229a0f2b3ae8`.

The shipped `ir_to_emit.sh` converter rejected the IR's signed `si32` spelling for the `topk` and `scatter` dimension attributes. Temporary copies in `/tmp/openai_gpt_oss_20b_ir_emit` normalized only those attribute types to `i32`; the model operations, tensors, and original MLIR files were unchanged. The converter then produced flat Python emits of 13,160 lines for prefill and 12,021 lines for decode.

The flat graph contains 24 repeating decoder layers plus the final norm. Layer 12 was selected as the representative middle layer. In the prefill emit its input RMSNorm begins at `ttnn_rms_norm_24`, its post-attention RMSNorm is `ttnn_rms_norm_25`, and the next layer starts at the following input norm. Decode has the same `24`/`25` pair. The raw graph shapes establish the captured workload:

- prefill hidden state: batch 1, sequence 17, hidden size 2880;
- decode hidden state: batch 1, one token, hidden size 2880;
- per-TP-rank caches: `[1, 2, 128, 64]`;
- mesh: `1x4`, giving 16 local query heads and 2 local KV heads per rank;
- dense translation: 64 query heads, 8 KV heads, 32 experts, and full `[1, 8, max_cache_len, 64]` caches.

## Translation details

`FunctionalDecoder.from_state_dict` performs graph const-eval work on the host: it fuses transposed Q/K/V weights and biases in Q-K-V order, dequantizes official MXFP4 expert blocks when dense expert tensors are not already supplied, builds RoPE tables, and transfers canonical full weights. There are no collectives in the runtime because full dense weights make each TP partition plus all-reduce equivalent to a single dense projection.

The runtime follows the emitted layer topology:

1. RMSNorm, fused QKV projection, Q/K/V head split, YaRN RoPE, attention sink, output projection, and residual add.
2. RMSNorm, 32-way router, top-4 softmax routing, interleaved gate/up expert projection, clipped SwiGLU, down projection, weighted expert reduction, and residual add.

The non-tile-aligned captured prefill requires the emitted post-RoPE logical slice from physical sequence 32 back to sequence 17. Attention sinks are divided by the attention scale before transfer because TT SDPA applies the scale inside the sink exponential path. The emitted router's FP32 input/logit boundary is retained, and the high-fidelity compute configuration uses FP32 destination accumulation at emitted norm, projection, attention, and expert matmul boundaries. Other activations, weights, caches, and outputs default to BF16, TILE layout, and DRAM.

## Runtime contract

Construction:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    batch=1,
    max_cache_len=128,
)
```

The mesh must have shape `1x1`, the preserved emitted batch is exactly 1, and the default retains the emitted cache length of 128. A bounded complete-prefill capacity sweep established a supported maximum of 21,248, so construction enforces `1 <= max_cache_len <= 21248`.

Prefill:

```python
decoder.prefill_forward(
    hidden_states,          # [1, 1, seq_len, 2880], BF16/TILE/DRAM
    key_cache=key_cache,    # [1, 8, max_cache_len, 64]
    value_cache=value_cache,
)                         # -> [1, 1, seq_len, 2880]
```

Prefill requires `1 < seq_len <= max_cache_len` and fills cache positions beginning at zero.

Decode:

```python
decoder.decode_forward(
    hidden_states,                # [1, 1, 1, 2880], BF16/TILE/DRAM
    key_cache=key_cache,
    value_cache=value_cache,
    cache_position=position,      # host integer used for validation and RoPE
    cache_position_tensor=index,  # [1], int32/ROW_MAJOR/DRAM
)                               # -> [1, 1, 1, 2880]
```

Decode updates K and V in place and invokes decode SDPA against the persistent cache. `cache_position` and `cache_position_tensor` are two representations of the same position and must be equal: the host integer selects the RoPE row, while the device tensor selects the cache write and decode-SDPA position.

## Correctness evidence

Blackhole, `TT_VISIBLE_DEVICES=2,3`, one-device mesh:

| Test | Reference | PCC threshold | Observed PCC |
| --- | --- | ---: | ---: |
| synthetic prefill, sequence 17 | dense HF layer 12 | 0.99 | 0.9999945678956855 |
| synthetic prefill, sequence 128 | dense HF layer 12 | 0.99 | 0.9999974798163773 |
| synthetic prefill, sequence 256 | dense HF layer 12 | 0.99 | 0.9999979815485808 |
| official real-weight prefill, sequence 17 | dense HF layer 12 | 0.99 | 0.9997057201235178 |
| official real-weight decode, cache position 17 | one dense HF decode step after the same prefill | 0.99 | 0.9996046254703057 |

The final model-local JUnit artifact is `doc/functional_decoder/test_results.xml`: 6 tests passed with no failures, errors, or skips, and its captured output preserves every exact PCC line. The tests also statically inspect every runtime method for `torch`, `from_torch`, and `to_torch` fallback calls.

## Context-capacity evidence

The capacity probe constructs the full translated layer, runs attention plus the dense 32-expert MoE, synchronizes the output, and exits normally for each passing length. Every invocation used a fresh process and hardware commands were serialized on `TT_VISIBLE_DEVICES=2,3`.

| Outcome | Sequence lengths |
| --- | --- |
| complete prefill pass | 512, 1024, 2048, 4096, 8192, 16384, 20480, 20992, **21248** |
| device-DRAM failure | **21249**, 21280, 21312, 21376, 21504, 22528, 24576, 32768 |

The exact first failing logical length, 21,249, pads to a physical TILE length of 21,280. Its MoE up-value slice requested 3,922,329,600 bytes total (490,291,200 per bank); the allocator reported 566,160,768 bytes free per bank but a largest contiguous block of 457,193,856 bytes. Thus 21,248 is both the largest observed complete prefill and the enforced functional-stage bound. The larger failure points were collected during the bounded search; after the boundary was established, the constructor was changed to reject larger lengths before allocating weights.

## Reproduction commands

```bash
.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_gpt_oss_20b_tp_batch_size_1_qb2_bs1_isl128_1784000876774

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /tmp/openai_gpt_oss_20b_ir_emit/prefill.normalized.mlir \
  /tmp/openai_gpt_oss_20b_ir_emit/prefill.py

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /tmp/openai_gpt_oss_20b_ir_emit/decode.normalized.mlir \
  /tmp/openai_gpt_oss_20b_ir_emit/decode.py

TT_VISIBLE_DEVICES=2,3 python \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py 21248

TT_VISIBLE_DEVICES=2,3 pytest -q --capture=tee-sys -o junit_logging=all \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/functional_decoder/test_results.xml \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py

python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/openai_gpt_oss_20b \
  --hf-model models/demos/gpt_oss/configs/gpt-oss-20b \
  --stage functional_decoder --require-contract --strict-caps
```

## Limitations

- Functional correctness only; no performance claims or program-config tuning are made.
- Single-device `1x1` mesh and preserved batch 1 only.
- The representative validated layer is layer 12, a sliding-attention layer. The constructor accepts other valid layer indices with the same GPT-OSS-20B tensor shapes, but they were not the real-weight PCC target for this stage.
- The current supported context is 21,248, below the Hugging Face advertised 131,072. The correctness-first dense TP collapse evaluates all 32 experts simultaneously and first fails at 21,249 from device-DRAM fragmentation during the MoE up-value slice. At the advertised length, its gate/up intermediate alone would be `32 * 131072 * 5760 * 2 = 48,318,382,080` bytes (45 GiB), exceeding the local Blackhole single-chip DRAM view of `8 * 4,278,190,080 = 34,225,520,640` bytes (31.875 GiB) before weights, attention tensors, or caches. The per-channel value is recorded by `tt_metal/soc_descriptors/blackhole_140_arch.yaml`. Expert dispatch or chunking can remove this functional-stage allocation, but belongs to a later stage.
- The real-weight test discovers the official layer-12 shard under `HF_HOME`, or accepts `GPT_OSS_20B_REAL_WEIGHT_SHARD`; it skips with an explicit reason when that artifact is unavailable.
