# Functional-decoder work log

## Scope and starting point

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/model/qwen-qwen2.5-coder-32b-instruct`
- Starting HEAD: `66533e5bc32`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- IR directory: `/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995`
- Stage ownership: `models/autoports/qwen_qwen2_5_coder_32b_instruct` only

No optimized-decoder, multichip, full-model, or vLLM artifacts were started.

## Classification and lowering

Commands:

```bash
.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995/ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995/ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_runc0ba_g0_1784014672995.mlir \
  /tmp/qwen_qwen2_5_coder_32b_instruct_ir/prefill.py

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995/ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_1784014672995/ttnn_qwen_2_5_coder_32b_instruct_tp_qb2_bs32_isl128_runc0ba_g1_1784015214915.mlir \
  /tmp/qwen_qwen2_5_coder_32b_instruct_ir/decode.py
```

The classifier identified g0 as prefill (`fill_cache=4096`) and g1 as decode (`paged_update_cache=128`, decode SDPA=64). Both are compiler, non-runtime, non-logits graphs. Lowering produced flat Python emits of 65,574 and 21,650 lines respectively.

The raw MLIR and lowered emits show a 64-layer, 1x4 TP graph. Layer 32 was segmented as the representative middle layer. The emitted workload is batch 32; the prefill input has sequence 17, the decode input sequence 1, and the persistent source cache length is 128. TP-local Q/KV heads are 10/2 and are collapsed to dense single-device 40/8-head math using full HF weights. The two all-reduces per layer become equivalent dense attention-output and MLP-down projections.

## Implementation decisions

- Preserved Qwen2 input/post-attention RMSNorm, biased Q/K/V projections, RoPE, 5:1 grouped-query attention, causal prefill SDPA, decode SDPA, in-place cache updates, residuals, and SwiGLU MLP.
- Preserved the emit's QKV constant transform: const-eval receives `[V, K, Q]`, then slices, transposes, and reverses to runtime Q-K-V order. The dense translation concatenates the full HF Q/K/V weights and biases in that final order.
- Preserved the emitted post-RoPE logical-head slices. This was essential for dense 40-head decode because the physical Q extent pads to 64 heads; slicing restores the intended 5:1 GQA grouping before SDPA.
- Recorded the emit precision policy: fused QKV, O, gate, up, and down weights become `BFLOAT8_B`, while normalization weights and QKV bias remain BF16.
- Intentionally normalized all weights, biases, and activations to BF16/TILE/DRAM defaults for this correctness stage; precision selection belongs to a later datatype stage.
- Removed source mesh transforms, TP collective calls, and compiler program configurations from runtime.
- Retained only required decode L1 layouts: 40 Q-head compute cores, 32 user/cache-update cores, and a 64-row tile-padded concat-input shard.

Runtime forwards are device-only. Static source inspection rejects `torch`, `from_torch`, and `to_torch` in `_qkv_forward`, `_mlp_forward`, `prefill_forward`, and `decode_forward`.

## Hardware and correctness evidence

`timeout 60 tt-smi -ls --local` reported four Blackhole p300c devices. Hardware tests were serialized on one 1x1 mesh selected from the paired `TT_VISIBLE_DEVICES=2,3` board functions. Exposing only device 2 was rejected by UMD as an incomplete p300 custom cluster; exposing the pair and opening a 1x1 mesh passed. No reset was needed, and no watcher or profiler ran concurrently.

Synthetic test result:

| Path | Case | Output PCC | K PCC | V PCC |
| --- | --- | ---: | ---: | ---: |
| prefill | seq 4 | 0.9994472112 | 0.9998692739 | 0.9998704918 |
| prefill | seq 17 | 0.9995370097 | 0.9998690415 | 0.9998728927 |
| prefill | seq 128 | 0.9996479480 | 0.9998651750 | 0.9998713349 |
| decode | position 17 | 0.9995728532 | 0.9998501798 | 0.9998522710 |

Real Qwen2.5-Coder-32B-Instruct layer-32 result:

| Path | Case | Output PCC | K PCC | V PCC |
| --- | --- | ---: | ---: | ---: |
| prefill | seq 17 | 0.9987814814 | 0.9998766862 | 0.9998666144 |
| decode | position 17 | 0.9989285759 | 0.9998826504 | 0.9998565872 |

The real checkpoint was read lazily from `model-00007-of-00014.safetensors`; only layer 32 was loaded. Replay command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
pytest -q -s models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_functional_decoder.py
```

## Debugging ledger

The first decode execution exposed a tile-alignment contract at head concatenation: 40 logical Q heads require a 64-row physical shard. After that layout correction, decode ran but reached only 0.982963 output PCC while the appended K/V were above 0.99985. Reading the layer-32 emit localized the cause: the compiler slices the post-RoPE Q/K tensors back from padded heads to logical 10/2 TP heads. Adding the dense 40/8 slices restored the intended GQA ratio and raised synthetic decode output to 0.999573. This was a bounded, source-proven correction, so autofix escalation was not needed.

Device shutdown reports pre-existing nanobind reference-leak warnings after pytest; devices close normally and subsequent isolated runs succeed. This is recorded as a non-functional harness anomaly, not dismissed as a decoder result.

## Context-capacity evidence

One length was run per process with full-shaped BF16 weights and cache allocations, and the result tensor was copied to the host to force completion. Batch-32 prefill passed at 3,584, 3,840, 3,968, 3,984, 3,992, 3,996, 3,998, and 3,999. It failed with allocator OOM at 4,000, 4,032, and 4,096. An initial assumption that 3,969 would pad to a 4,000-token class was explicitly refuted by a passing probe; the flattened batch-32 matmul height remains tile-aligned. The final boundary therefore uses directly adjacent logical lengths.

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN2_5_CODER_32B_CONTEXT_PROBE_LEN=3999 \
pytest -q -s models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_context_capacity.py

TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN2_5_CODER_32B_CONTEXT_PROBE_LEN=4000 \
QWEN2_5_CODER_32B_CONTEXT_EXPECT_OOM=1 \
pytest -q -s models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_context_capacity.py
```

The 3,999 run passed with output shape `[1, 32, 3999, 5120]`. The adjacent 4,000 run reproduced and verified an expected allocator failure: a 1,310,720,000-byte buffer was requested when the largest free block was 160,768,000 bytes. Both final evidence commands exited successfully.

## Gates and review

Completed pre-review gates:

- `python -m py_compile` passed for the decoder and both test files.
- Black's Python 3.12 format check passed for all three Python files.
- `python -m json.tool doc/context_contract.json` passed.
- `.agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen2_5_coder_32b_instruct` passed with target 32,768 and DRAM-limited support 3,999.
- Final combined functional test: 3 passed in 16.03 seconds, including the runtime audit, synthetic prefill/decode, and real-weight prefill/decode. Exact PCC output is in `final_test.log`.
- Both final context boundary commands passed: the 3,999 workload completed and the 4,000 test verified the expected allocator OOM.
- Final device inventory listed all four Blackhole p300c boards normally.

Fresh independent review `/root/stage_review_qwen25_coder32` returned `Verdict: clean-pass` with no required work. It independently re-derived graph classification, layer-32 semantics, constant-eval weight mapping, TP collapse, HF/checkpoint provenance, fallback audit, context arithmetic, and static gates without opening devices. Its full recorded summary is `stage_review.md`.

The reviewer noted that the three evidence logs match the repository-wide `*.log` ignore rule; they are intentionally force-added to the checkpoint. It also classified the paired-p300 selector failure, nanobind teardown warnings, fixed decode head-padding issue, corrected context-padding assumption, and measured 4,000-token OOM as controlled or fixed rather than stage blockers.

Reviewed stage checkpoint: `72f6e948c5e9e0d9794a98c9e9ac4088a67bd800` (`Add Qwen2.5 Coder 32B IR functional decoder`). The three `.log` artifacts were force-added as required. No push was performed.

## Multichip provenance

`multichip_provenance.json` retro-records the complete layer-32 sharding prior from the selected compiler prefill and decode IR. The source mesh is 1x4 with TP degree 4 on mesh/cluster axis 1; the segmented layer's collective set is ring-sum `ttnn.all_reduce` after the attention O projection and MLP down projection in both paths. This provenance pass was pure IR analysis and did not open a TT device.
