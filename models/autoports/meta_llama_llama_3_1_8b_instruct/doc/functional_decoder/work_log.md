# Functional decoder work log

## 2026-07-15: IR classification and translation

HF model: `meta-llama/Llama-3.1-8B-Instruct`

IR root: `/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476`

Commands:

```bash
.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476/ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476/ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_run8032_g0_1784013027476.mlir \
  /tmp/meta_llama_llama_3_1_8b_instruct/prefill

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476/ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_1784013027476/ttnn_llama_3_1_8b_instruct_tp_qb2_bs32_isl128_run8032_g1_1784013181786.mlir \
  /tmp/meta_llama_llama_3_1_8b_instruct/decode
```

Classification selected g0 as prefill (`fill_cache=2048`, no paged update or decode SDPA) and g1 as decode (`paged_update_cache=64`, decode SDPA=32, no fill). Both are compiler/non-runtime graphs without the trailing logits path.

The raw IR records batch 32, prefill sequence 18, single-token decode, cache shape `[32, 2, 128, 128]` per TP shard, and a `1x4` mesh. AutoConfig records hidden size 4096, 32 Q heads, 8 KV heads, head dimension 128, intermediate size 14336, 32 layers, RMS epsilon `1e-5`, and advertised context 131072. Layer 16 was segmented as the representative block. The full dense cache shape after collapsing TP is `[32, 8, 128, 128]`.

Load-time const-eval analysis established Q/K/V fusion order, all weight transposes, and the emitted BF8/BF4 casts. Runtime translation uses canonical full HF weights and removes TP all-reduces by replacing shard-local projection math with equivalent dense matmuls.

## Validation commands and evidence

Static checks:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/functional_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py

pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k runtime_forwards_have_no_host_fallback
```

The syntax check and runtime-fallback audit pass.

Real layer-16 weights are available locally in the HF safetensors shard selected through `LLAMA_31_8B_REAL_WEIGHT_FILE`. The intended hardware command is:

```bash
LLAMA_31_8B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/model-00002-of-00004.safetensors \
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py
```

PCC status:

| Path | Weight kind | Batch | Case | PCC |
| --- | --- | ---: | --- | --- |
| prefill | synthetic BF16 | 32 | seq 4 output | 0.9988648701 |
| prefill | synthetic BF16 | 32 | seq 4 K / V cache | 0.9999016043 / 0.9999037321 |
| prefill | synthetic BF16 | 32 | seq 18 output | 0.9984298943 |
| prefill | synthetic BF16 | 32 | seq 18 K / V cache | 0.9998981935 / 0.9999029896 |
| prefill | synthetic BF16 | 13 | seq 4 output | 0.9989121824 |
| decode | synthetic BF16 | 13 | position 4 output | 0.9986191212 |
| decode | synthetic BF16 | 13 | position 4 K / V append | 0.9999038387 / 0.9999055154 |
| prefill | real layer 16 BF16 | 32 | seq 18 output | 0.9999855315 |
| decode | real layer 16 BF16 | 32 | position 18 after prefill output | 0.9999878148 |
| decode | real layer 16 BF16 | 32 | position 18 K / V append | 0.9998891691 / 0.9998918704 |

All PCC rows use the official Transformers `LlamaDecoderLayer` loaded with the same layer state and `DynamicCache` for prefill/decode state. The final watcher-enabled suite collected four tests with zero failures or skips.

## Device recovery evidence

Four local Blackhole p300c ASICs are visible and management telemetry is healthy. TTNN cannot initialize the UMD, before model code or a device fixture opens:

```text
Expected sysmem to be mapped at NOC address 0x1000000000000000,
but it was mapped at 0x1000000040000000.
```

No live process visible in this process namespace owns a `/dev/tenstorrent/*` file descriptor. The recovery sequence followed the TT device-use policy:

1. Confirmed the failure with a minimal `ttnn.open_mesh_device(ttnn.MeshShape(1, 1))` smoke.
2. Listed all four devices with `tt-smi -ls --local`.
3. Issued `tt-smi -r` and confirmed all devices reappeared.
4. Removed confirmed-stale `/dev/shm/TT_UMD_LOCK.*` entries only after checking for a live owner.
5. Repeated reset, list, and minimal open; the identical sysmem mapping error remained.

AutoTriage showed that the address delta is exactly one 1 GiB UMD sysmem channel and that the conflict belongs to host/sibling-container state invisible from this container. Restricting UMD to one ASIC does not form a valid P300 topology, but selecting the free endpoint pair works:

```bash
TT_VISIBLE_DEVICES=2,3 python -c \
  'import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print(m.get_device_ids()); ttnn.close_mesh_device(m)'
```

That smoke opened and closed device 0 in the restricted topology. All hardware tests subsequently used `TT_VISIBLE_DEVICES=2,3`.

A post-test allocator probe reported 8 DRAM banks with 4272341376 bytes per bank, or 34178731008 bytes total. Advertised-context prefill at batch 32 would require 34359738368 bytes for the BF16 input and 17179869184 bytes for the BF16 K/V cache alone (48 GiB combined), so the reduced functional context has direct device-capacity evidence.

## AutoFix report

### Starting evidence

- Initial UMD failure: fixed-base sysmem mapping shifted by exactly 1 GiB before any model operation.
- First prefill run: SDPA rejected rotated K with padded sequence 32 against V sequence 4.
- First decode run: `nlp_concat_heads_decode` raised `bad optional access` for a multi-range input shard grid without an explicit sub-core compute grid.

### Hypothesis experiments

1. Hypothesis: another physical endpoint pair is free. Experiment: restrict UMD first to one endpoint, then to the complete `2,3` P300 pair. Result: one endpoint produced the expected invalid-custom-topology error; the full pair opened a 1x1 mesh. Verdict: verified. Fix: run stage tests on `TT_VISIBLE_DEVICES=2,3`.
2. Hypothesis: tiled RoPE padding must be sliced away as in the flat prefill emit. Experiment: inspect the emitted layer-16 RoPE-to-SDPA segment and run the synthetic seq-4 case. Result: the emit slices rotated K/Q from padded 32 to true sequence 18; adding a sequence-parameterized slice removes the mismatch. Verdict: verified. Verification: both synthetic prefill lengths pass with PCC above 0.9984.
3. Hypothesis: the concat failure is caused by its sub-core-grid selection contract. Experiment: inspect `nlp_concat_heads_decode_device_operation.cpp` and its unit test. Result: more than one input core range selects the sub-core factory, which dereferences `sub_core_grids`; the unit test uses a single 8x4 range for batch 32. Verdict: verified. Initial fix: use one regular 8x4 height-shard range and let the op derive its width-sharded output. Verification: the real decode smoke passed.

Independent stage review then required four remediations: use the official HF decoder layer rather than a handwritten oracle; prove the accepted non-32 batch API; clarify watcher wording; and cover a non-factorable logical batch. The oracle now instantiates `LlamaDecoderLayer`; decode derives one input core per logical user, supplies an explicit 32-head compute sub-core grid to head concatenation, and slices the padded result back to the logical batch. Batch-13 prefill/decode passes, including K/V append PCC above 0.99990, and the README distinguishes watcher correctness from unmeasured Tracy/performance work.

The final fresh-context review at `/root/functional_decoder_stage_final_review` independently rechecked the skill contracts, selected IR and flat emits, implementation, official-HF tests, final JUnit timestamps, context contract, documentation, and scope. Its verdict was `clean-pass` with no required work.

### Final status

Fixed. The focused synthetic and real-weight gates pass under watcher/assert instrumentation. The full-suite command (four tests, zero skips) is:

```bash
TT_VISIBLE_DEVICES=2,3 \
LLAMA_31_8B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/model-00002-of-00004.safetensors \
scripts/run_safe_pytest.sh --dev --run-all \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py -q -s
```

The unavailable endpoint pair remains an external host-state limitation, not a stage correctness issue. Warmed performance remains unmeasured by design for this functional stage.

## Local checkpoint

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/agentic-skills`
- Functional-decoder checkpoint: `9a33ac6ee68605e4e897f2f2985dfb59aa6f7443`
- Remote action: none; the checkpoint was not pushed.
