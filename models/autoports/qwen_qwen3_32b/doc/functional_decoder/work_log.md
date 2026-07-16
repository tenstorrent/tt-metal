# Functional-decoder work log

## Scope and starting point

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/model/qwen-qwen3-32b`
- Starting HEAD: `66533e5bc32`
- Model: `Qwen/Qwen3-32B`
- IR directory: `/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090`
- Stage ownership: `models/autoports/qwen_qwen3_32b` only

No optimized-decoder, multichip, full-model, or vLLM artifacts were started.

## Classification and lowering

Commands:

```bash
.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090/ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090/ttnn_qwen_3_32b_tp_qb2_bs32_isl128_runcf4a_g0_1784008145090.mlir \
  /tmp/qwen_qwen3_32b_ir_emit/prefill.py

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090/ttnn_qwen_3_32b_tp_qb2_bs32_isl128_1784008145090/ttnn_qwen_3_32b_tp_qb2_bs32_isl128_runcf4a_g1_1784008729649.mlir \
  /tmp/qwen_qwen3_32b_ir_emit/decode.py
```

The classifier identified g0 as prefill (`fill_cache=4096`) and g1 as decode (`paged_update_cache=128`, decode SDPA=64). Both are compiler, non-runtime, non-logits graphs. Lowering produced flat Python emits of 67,613 and 18,825 lines respectively.

The raw MLIR and lowered emits show a 64-layer, 1x4 TP graph. Layer 32 was segmented as the representative middle layer. The emitted workload is batch 32; the prefill input has sequence 17, the decode input sequence 1, and the persistent source cache length is 128. TP-local Q/KV heads are 16/2 and are collapsed to dense single-device 64/8-head math using full HF weights. The two all-reduces per layer become their equivalent dense attention-output and MLP-down projections.

## Implementation decisions

- Preserved Qwen3 input/post-attention RMSNorm, per-head Q/K RMSNorm, RoPE, grouped-query attention, causal prefill SDPA, decode SDPA, in-place cache updates, residuals, and SwiGLU MLP.
- Preserved the emit's QKV constant transform: the const-eval operand order `[V, K, Q]` is reversed, transposed, and concatenated Q-K-V.
- Recorded the emit precision policy: fused QKV, O, gate, up, and down weights are cast to `BFLOAT8_B` after load-time transforms, while normalization weights remain BF16.
- Intentionally normalized all weights and activations to BF16/TILE/DRAM defaults for this correctness stage; the emitted BF8 weight policy is provenance, not a precision selection for this functional baseline.
- Removed source mesh transforms, TP collective calls, and compiler program configurations from runtime.
- Retained only required decode L1 layouts: 64 Q-head cores, 32 user/cache-update cores, and a distinct concat-input shard height.
- Moved decode Q/K RMSNorm through DRAM because the default TTNN RMSNorm kernel rejects the required height-sharded head tensors without a compiler program configuration. This retains device-native math and avoids importing compiler layout glue.

Runtime forwards are device-only. Static source inspection rejects `torch`, `from_torch`, and `to_torch` in `_mlp_forward`, `prefill_forward`, and `decode_forward`.

## Hardware and correctness evidence

`timeout 60 tt-smi -ls --local` reported four Blackhole p300c devices. No stale model, profiler, watcher, or vLLM process was present. Hardware tests were serialized on one 1x1 mesh selected from `TT_VISIBLE_DEVICES=2,3`; no watcher or profiler ran concurrently. A mesh open/close smoke passed before model execution.

Synthetic test result:

| Path | Case | Output PCC | K PCC | V PCC |
| --- | --- | ---: | ---: | ---: |
| prefill | seq 4 | 0.9993484856 | 0.9999043628 | 0.9998671614 |
| prefill | seq 17 | 0.9994795918 | 0.9999030123 | 0.9998687840 |
| prefill | seq 128 | 0.9996403697 | 0.9999008726 | 0.9998666644 |
| decode | position 17 | 0.9995784204 | 0.9999023038 | 0.9998504092 |

Real Qwen3-32B layer-32 result:

| Path | Case | Output PCC | K PCC | V PCC |
| --- | --- | ---: | ---: | ---: |
| prefill | seq 17 | 0.9988867238 | 0.9999004048 | 0.9998653978 |
| decode | position 17 | 0.9985761292 | 0.9999019711 | 0.9998545239 |

The real checkpoint was read lazily from `model-00009-of-00017.safetensors`; only layer 32 was loaded.

Final correctness command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
pytest -q -s models/autoports/qwen_qwen3_32b/tests/test_functional_decoder.py
```

## Debugging ledger

The first decode execution exposed three explicit TTNN kernel contracts. Each was reproduced in isolation and corrected directly: default RMSNorm rejected height-sharded heads, paged cache update required exactly 32 user shards, and decode head concatenation required a distinct shard height of 64. The final layout split is documented above, and both synthetic and real decode now pass. No host fallback or unproven workaround was retained, so the autofix escalation was not needed.

Device shutdown reports pre-existing nanobind reference-leak warnings after pytest; devices close normally and subsequent isolated runs succeed. This is recorded as a non-functional harness anomaly, not dismissed as a decoder result.

## Context-capacity evidence

One length was run per process with full-shaped BF16 weights and cache allocations, and the result tensor was copied to the host to force completion. Batch-32 prefill passed at 1,024, 2,048, and 4,096. It failed with allocator OOM at 4,128, 4,224, 4,352, 4,608, 5,120, 6,144, and 8,192. The final adjacent probe uses sequence 4,097, which pads to the failing 4,128 allocation class.

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_CONTEXT_PROBE_LEN=4096 \
pytest -q -s models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py

TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_CONTEXT_PROBE_LEN=4097 QWEN3_32B_CONTEXT_EXPECT_OOM=1 \
pytest -q -s models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py
```

The exact pass and expected-OOM logs are stored beside this work log. `doc/context_contract.json` records the 4,096-token limit and the advertised-context allocation proof.

The final 4,096 run passed with output shape `[1, 32, 4096, 5120]`. The adjacent 4,097 run reproduced the expected allocator failure after TILE padding: a 1,352,663,040-byte buffer was requested when the largest free block was 98,279,424 bytes. The opt-in test caught and verified that exact OOM, so both evidence commands exited successfully. A subsequent `tt-smi -ls --local` listed all four boards normally.

## Gates and review

Completed pre-review gates:

- `python -m py_compile` passed for the decoder and both test files.
- Black's Python 3.12 format check passed for all three Python files.
- `python -m json.tool doc/context_contract.json` passed.
- `.agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_32b` passed with target 40,960 and DRAM-limited support 4,096.
- Final combined functional test: 3 passed in 16.17 seconds, including the runtime audit, synthetic prefill/decode, and real-weight prefill/decode. Raw output is in `final_test.log`.
- Both final context boundary commands passed: the 4,096 workload completed and the 4,097 test verified the expected allocator OOM.
- Final device inventory listed all four Blackhole p300c boards normally.

First independent review (`/root/stage_review_qwen32`) returned `more-work-needed` with no implementation or correctness defect. Its two P2 documentation findings were corrected: the replayable lowering commands now include the nested IR directory, and source BF8/BF16 precision provenance is recorded separately from the functional all-BF16 policy.

Fresh rereview (`/root/stage_rereview_qwen32`) returned `Verdict: clean-pass`. It directly verified both corrected source paths and their emit line counts, confirmed the layer-32 BF8/BF16 source policy from both emits, rechecked the implementation, raw logs, scope, and context evidence, and reported no required work or blocking hard-check gap.

Stage checkpoint: `cbf05453488c69df82e2efbb5e4109fcd002654e` (`Add Qwen3 32B IR functional decoder`). The three raw `.log` artifacts were intentionally force-added because the repository's broad `*.log` ignore rule would otherwise omit required evidence. No push was performed.

## Multichip provenance

The structured layer-32 sharding prior is in `multichip_provenance.json`. It records the source 1x4 mesh (TP degree 4), column-sharded Q/K/V and gate/up projections, row-sharded O/down projections, KV-head ownership, replicated layer boundaries, and the two `all_reduce(sum)` sites on mesh axis 1 in each of the prefill and decode graphs. No other collective occurs inside the segmented layer.
