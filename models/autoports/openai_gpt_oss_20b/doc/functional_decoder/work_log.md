# Functional decoder work log

Date: 2026-07-22 UTC

## Translation

- Read the pre-generated `/home/mvasiljevic/emit-gptoss/g0_prefill/{main.py,consteval.py}` and `g1_decode/{main.py,consteval.py}` directly.
- Did not run `ir_to_emit.sh`, inspect an MLIR graph as the translation source, or regenerate EmitPy.
- Counted 49 RMSNorm sites per flat graph (two per decoder plus final norm) and segmented representative middle layer 12.
- Cross-checked eps `1e-5`, attention scale `0.125`, Q-K-V fusion, raw/prescaled sink handling, cache ops, SwiGLU clamp/fusion, FP32 router, and residual placement against the Hugging Face GPT-OSS decoder.
- Collapsed the emitted TP4 Q/K/V/O shards, local eight-expert tensors, attention/expert all-reduces, and routing mesh partition to full dense single-device tensors.

## Commands and results

Static/runtime audit:

```text
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_runtime_forwards_have_no_host_or_collective_fallback
PASS (1 test)
```

Device health and topology:

```text
timeout 60 tt-smi -ls --local
PASS: four local Blackhole chips enumerated; no reset performed.

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
python <open 1x1 mesh and query ttnn.get_memory_view>
PASS: 8 DRAM banks, 4,272,341,376 bytes per bank (34,178,731,008 bytes total).
```

Synthetic device evidence:

```text
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_synthetic_prefill_pcc_small_and_emitted_sequence
PASS
```

Real-weight device evidence:

```text
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_real_weight_prefill_and_decode_pcc
PASS
```

Initial complete stage suite before independent review:

```text
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py
PASS: 3 tests in 6.69s
```

Batch-parameterization review fix:

```text
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_synthetic_non_emitted_batch_prefill_and_decode
PASS: batch-2 prefill and decode (1 test in 11.13s)
```

Post-remediation complete stage suite:

```text
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py
PASS: 4 tests in 8.49s
```

PCC evidence:

| Test | Output | K cache | V cache |
| --- | ---: | ---: | ---: |
| synthetic prefill S=4 | 0.9998262341 | 0.9999468898 | 0.9999490358 |
| synthetic prefill S=17 | 0.9997995855 | 0.9999447429 | 0.9999508064 |
| real prefill S=17 | 0.9991933121 | 0.9999477953 | 0.9999524116 |
| real decode position 17 | 0.9992976003 | 0.9999464908 | 0.9999486483 |
| synthetic batch-2 prefill S=4 | 0.9998323648 | 0.9999442672 | 0.9999495850 |
| synthetic batch-2 decode position 4 | 0.9998001953 | 0.9999441058 | 0.9999489098 |

## Correctness investigation

The first real test used BF16 routing end to end and produced prefill PCC 0.9765. Isolating the path showed attention PCC 0.99933 and K/V PCC above 0.9999, proving the mismatch was in MoE routing. Restoring the emitted FP32 router promotion and HiFi4/FP32-accumulating correctness policy raised prefill above threshold. The g1 path was separately isolated at decode attention PCC 0.99959 with append-cache PCC above 0.9999. The final deterministic real-weight seed uses float32 random generation followed by BF16 conversion so top-k margins are stable across CPU and device rounding.

The first batch-2 prefill attempt exposed that TTNN `fill_cache` accepts an input tensor with batch dimension one. The runtime now slices each user row on device and calls `fill_cache` with the matching `batch_idx`; the rerun passed batch-2 prefill and decode without host fallback.

## Independent review remediation

The first independent stage review returned more-work-needed on four items. Remediation:

- Lowered `current_supported_context` from the allocated cache extent 128 to the largest validated prefill S=17; clarified that the dense-mask calculation proves the HF target infeasible but does not establish 128 as a physical boundary.
- Removed the batch-one rejection, generalized sink/cache/layout handling, and added passing batch-2 prefill/decode PCC coverage.
- Replaced the prose TP summary with structured mesh axes, per-tensor parallel axes/dtypes, boundary activations, K/V cache, RoPE/mask/index constants, and every representative runtime/consteval collective including reduce type and placement.
- Confirmed with `git status --ignored` that downstream-stage directories were pre-existing ignored artifacts; they remain untouched and outside the explicit functional-decoder checkpoint.

## Stage bookkeeping

- Context contract: PASS after review remediation; target 131072, largest validated prefill 17, DRAM-limited dense-mask target.
- Independent stage review: first review MORE WORK NEEDED; all four findings fixed; rereview CLEAN PASS.
- Local implementation commit SHA: `f972e13d4d699e7bfa3ca6bde66a9dc069aa8993`.
