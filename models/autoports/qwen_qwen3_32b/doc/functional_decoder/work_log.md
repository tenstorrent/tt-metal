# Functional-decoder work log

## Scope and starting point

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/models/v2/qwen-qwen3-32b`
- Starting HEAD: `42a961bdbf7`
- Model: `Qwen/Qwen3-32B`
- Pre-generated EmitPy package: `/home/mvasiljevic/emit-qwen3`
- Stage ownership: `models/autoports/qwen_qwen3_32b` only

No EmitPy generation, MLIR conversion, fused-decoder, optimized-decoder,
multichip runtime, full-model, generation, or vLLM work was performed.

## Source audit and segmentation

The following pre-generated files were read directly:

- `/home/mvasiljevic/emit-qwen3/g0_prefill/main.py`
- `/home/mvasiljevic/emit-qwen3/g0_prefill/consteval.py`
- `/home/mvasiljevic/emit-qwen3/g1_decode/main.py`
- `/home/mvasiljevic/emit-qwen3/g1_decode/consteval.py`

`g0_prefill` is identified by sequence 17 plus cache-fill calls;
`g1_decode` is identified by sequence 1 plus paged cache updates and decode
SDPA. The emit is a flat 64-layer, batch-32, `[1, 4]` TP graph. Layer 32 was
selected as the representative middle layer using its norm sites and
`model.layers.32` weight keys. Its source spans are prefill lines 30450-31311
and decode lines 5964-6069. Adjacent layer-33 RMSNorm starts at lines 31314 and
6073, respectively.

The representative op sequence matches the installed HF Qwen3 decoder:
input RMSNorm, fused QKV, per-head Q/K RMSNorm, RoPE, K/V cache write, GQA
attention, O projection, residual, post-attention RMSNorm, SwiGLU gate/up/down,
and residual. Both all-reduces occur at TP row-parallel boundaries: after O
and after down. `g0_prefill` placements are lines 31278 and 31306;
`g1_decode` placements are lines 6032 and 6064.

The source consteval reverses its `[V, K, Q]` operands into Q-K-V order,
transposes the projections, and casts fused QKV/O/gate/up/down to
`BFLOAT8_B`. RMSNorm and Q/K norm weights remain BF16. The full provenance is
in `multichip_provenance.json`.

## Implementation decisions

- Loaded full canonical HF layer-32 weights and collapsed TP4 column/row
  sharding to dense 64-Q-head, 8-KV-head, intermediate-25600 computation.
- Replaced each pair of per-rank partial projections plus all-reduce with the
  equivalent dense full-weight projection; runtime contains no collective.
- Preserved input/post RMSNorm, per-head Q/K RMSNorm, RoPE, GQA, causal
  prefill, decode SDPA, in-place cache writes, both residuals, and SwiGLU.
- Used BF16/TILE/DRAM correctness defaults instead of inheriting the source
  BF8 weight policy.
- Retained only API-required decode L1 sharding. Decode Q/K RMSNorm is moved
  through DRAM because the default kernel rejects the required height-sharded
  tensors without compiler-specific program configuration.
- Kept all Torch conversion and preprocessing at construction/test boundaries.
  `_mlp_forward`, `prefill_forward`, and `decode_forward` are device-only.

## Device procedure and anomalies

`tt-smi -ls --local` reported four Blackhole p300c devices before testing.
The first 1x1 smoke used `TT_VISIBLE_DEVICES=2` and failed during device open
because this p300 topology requires the paired chips in the visible set. It
did not reach model code. Retrying with `TT_VISIBLE_DEVICES=2,3` and opening a
1x1 mesh passed. All model and capacity commands were then serialized with
the paired visible set; no watcher or profiler was run concurrently.

Pytest shutdown reports nanobind reference-leak warnings from the binding
layer. Every command exits normally, mesh fixtures close, subsequent isolated
runs work, and a final `tt-smi -ls --local` still listed all four boards. No
device reset was needed.

## Correctness evidence

Final command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
timeout 300 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_functional_decoder.py
```

Result: `3 passed in 19.48s`.

| Weights | Path | Case | Output PCC | K PCC | V PCC |
| --- | --- | --- | ---: | ---: | ---: |
| synthetic | prefill | seq 4 | 0.9993484856 | 0.9999043628 | 0.9998671614 |
| synthetic | prefill | seq 17 | 0.9994795918 | 0.9999030123 | 0.9998687840 |
| synthetic | prefill | seq 128 | 0.9996403697 | 0.9999008726 | 0.9998666644 |
| synthetic | decode | position 17 | 0.9995784204 | 0.9999023038 | 0.9998504092 |
| real layer 32 | prefill | seq 17 | 0.9988867238 | 0.9999004048 | 0.9998653978 |
| real layer 32 | decode | position 17 | 0.9985761292 | 0.9999019711 | 0.9998545239 |

The real checkpoint test reads only layer 32 from the local Qwen3-32B
safetensors snapshot. `final_test.log` preserves the raw result.

## Context-capacity evidence

One probe was run per process using the full batch-32 dense layer, full-shaped
BF16 weights, and cache allocation. The result was copied to the host to force
completion.

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_CONTEXT_PROBE_LEN=4096 \
timeout 300 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py

TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_CONTEXT_PROBE_LEN=4097 QWEN3_32B_CONTEXT_EXPECT_OOM=1 \
timeout 300 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_context_capacity.py
```

The 4,096 run passed with output shape `[1, 32, 4096, 5120]`. The adjacent
4,097 run produced the expected allocator failure after TILE padding: a
1,352,663,040-byte buffer was requested while the largest free block was
98,279,424 bytes. The opt-in test verified the OOM and exited successfully.
Concise command/result evidence is in `capacity_4096_pass.log` and
`capacity_4097_oom.log`; `final_test.log` is the verbatim functional-suite
console capture.

## Gates and review

Pre-review gates completed:

- Python compilation for the decoder and both tests.
- Black Python-3.12 formatting check.
- JSON parsing for the context contract and multichip provenance.
- `.agents/scripts/check_context_contract.py` with target 40,960 and measured,
  DRAM-limited support at 4,096.
- Functional suite: 3 passed, including synthetic and real-weight prefill and
  decode plus the runtime-fallback grep audit.
- Context boundary: 4,096 passed; 4,097 verified expected OOM.
- Post-test device inventory: all four p300c chips visible.

Independent stage-review results and local checkpoint SHAs are appended after
their gates complete. The three `.log` artifacts are force-added because the
repository ignores `*.log` globally.

### Independent review

The first fresh reviewer (`/root/stage_review_qwen3_32b_functional`) returned
`Verdict: more-work-needed` with one P2 structured-provenance defect: decode Q
and K/V shapes place heads on axis 2, but the shared `head_axis` field said 1.
The implementation and correctness evidence had no blocking finding.

The provenance now records `prefill_head_axis: 1` and
`decode_head_axis: 2` separately for both query and K/V tensors. JSON parsing,
the strict context checker, and diff hygiene passed after the correction.

A different fresh reviewer (`/root/stage_rereview_qwen3_32b_functional`)
returned `Verdict: clean-pass`. It directly checked both path-specific axes
against the recorded shapes and EmitPy source, and rechecked source
collectives, the zero-runtime-collective condition, runtime fallback audit,
context evidence, syntax, JSON, and stage scope. No required work remained.

### Checkpoints

- Starting checkpoint: `42a961bdbf7`
- Functional-decoder stage checkpoint:
  `606ab816a33692f0b1f0c22105dadc235b37ba12`
