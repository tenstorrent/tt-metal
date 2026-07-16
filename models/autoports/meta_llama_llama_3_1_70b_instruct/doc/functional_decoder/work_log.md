# Functional decoder work log

## 2026-07-15/16: IR classification and translation

HF model: `meta-llama/Llama-3.1-70B-Instruct`

IR root: `/home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663`

Commands:

```bash
.agents/skills/forge-functional-decoder-from-ir/scripts/classify_graphs.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663/ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663/ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_runbb45_g0_1784014705663.mlir \
  /tmp/meta_llama_llama_3_1_70b_instruct_ir/prefill

.agents/skills/forge-functional-decoder-from-ir/scripts/ir_to_emit.sh \
  /home/mvasiljevic/qb2-irs/ttnn-mlir-ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663/ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_1784014705663/ttnn_llama_3_1_70b_tp_qb2_bs32_isl128_runbb45_g1_1784015812679.mlir \
  /tmp/meta_llama_llama_3_1_70b_instruct_ir/decode
```

Classification selected g0 as prefill (`fill_cache=5120`, no paged update or decode SDPA) and g1 as decode (`paged_update_cache=160`, decode SDPA=80, no fill). Both are compiler/non-runtime graphs without trailing logits. Compiler g2/g3 add logits; runtime g0-g3 mirror all four roles.

The raw IR records batch 32, prefill sequence 18, single-token decode, per-chip cache shape `[32, 4, 128, 128]`, and a `2x2` mesh. The local HF config records hidden size 8192, 64 Q heads, 8 KV heads, head dimension 128, intermediate size 28672, 80 layers, RMS epsilon `1e-5`, RoPE theta 500000 with Llama 3 scaling, and advertised context 131072.

Layer 39 was segmented as the representative middle block. Its full cache shape after TP collapse is `[32, 8, 128, 128]`. Load-time const-eval analysis established projection transposes and fused Q/K/V order. Runtime translation uses full canonical HF weights and replaces TP shard-local projections plus both collective axes with dense matmuls.

## Validation commands and evidence

Static checks:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tt/functional_decoder.py \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_functional_decoder.py

pytest -q models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_functional_decoder.py \
  -k runtime_forwards_have_no_host_fallback
```

Watcher-enabled synthetic gate:

```bash
TT_VISIBLE_DEVICES=2,3 scripts/run_safe_pytest.sh --dev --run-all \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_functional_decoder.py \
  -k synthetic_prefill_and_decode_pcc -q -s
```

Watcher-enabled real-weight gate:

```bash
TT_VISIBLE_DEVICES=2,3 \
LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors \
scripts/run_safe_pytest.sh --dev --run-all \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_functional_decoder.py \
  -k real_weight_prefill_and_decode_pcc -q -s
```

PCC evidence:

| Path | Weights | Batch | Case | PCC |
| --- | --- | ---: | --- | ---: |
| prefill | synthetic BF16 | 32 | seq 4 output | 0.9973702517 |
| prefill | synthetic BF16 | 32 | seq 4 K / V cache | 0.9998549707 / 0.9998574937 |
| prefill | synthetic BF16 | 32 | seq 18 output | 0.9965384261 |
| prefill | synthetic BF16 | 32 | seq 18 K / V cache | 0.9998535517 / 0.9998588139 |
| decode | synthetic BF16 | 32 | position 18 output | 0.9959452174 |
| decode | synthetic BF16 | 32 | position 18 K / V append | 0.9998607198 / 0.9998629106 |
| prefill | real layer 39 BF16 | 32 | seq 18 output | 0.9999926657 |
| decode | real layer 39 BF16 | 32 | position 18 output | 0.9999941847 |
| decode | real layer 39 BF16 | 32 | position 18 K / V append | 0.9998496644 / 0.9998578782 |

Every reference is the official Transformers `LlamaDecoderLayer`, loaded with the identical layer state and a `DynamicCache`. Both hardware commands completed with watcher and lightweight assertions enabled through the safe wrapper.

## Device and context-capacity evidence

`timeout 60 tt-smi -ls --local` reported four local Blackhole p300c ASICs. A bounded 1x1 open/close smoke passed on the free complete endpoint pair selected with `TT_VISIBLE_DEVICES=2,3`; all subsequent device commands used that pair and were serialized.

The post-test allocator probe reported:

```text
num_banks 8
total_bytes_per_bank 4272341376
total_bytes_allocated_per_bank 0
total_bytes_free_per_bank 4272341376
largest_contiguous_bytes_free_per_bank 4272341376
```

Total allocator-visible DRAM is 34178731008 bytes. Advertised-context prefill at batch 32 requires 68719476736 bytes for the BF16 hidden-state input plus 17179869184 bytes for the BF16 K/V cache, or 85899345920 bytes before weights, outputs, residuals, RoPE tables, and operation temporaries. This is the direct capacity evidence for the reduced functional context contract.

The platform reports firmware bundle 19.8.0, newer than the latest fully tested 19.5.0, and an unknown `B850M-C` motherboard fallback. These warnings did not prevent watcher-enabled correctness tests or allocator probes.

## AutoFix report

### Starting evidence

Synthetic prefill passed at both sequence lengths, but dense decode Q `[1,32,64,128]` with caches `[32,8,128,128]` failed before execution:

```text
Program size (70976) too large for kernel config buffer (70656) on TENSIX
```

Fresh source-only diagnosis is preserved in `AUTODEBUG.md`.

### Hypothesis experiments

1. **Bound the default 11x10 grid to 8x8.** Source inspection showed that missing `program_config` selected the full Blackhole grid. Adding the canonical 8x8 config reduced serialized size to 70848 bytes but remained 192 bytes over. Verdict: grid involvement verified; 8x8 fix refuted.
2. **Use the non-paged 70B 8x4 grid.** Source allocation proved it legal for batch 32, with all 32 cores active. Hardware produced size 70864 bytes, still 208 bytes over. Verdict: legality verified; descriptor-fit hypothesis refuted.
3. **Restore the emitted explicit mask signature.** The raw IR and flat emit compare scalar position against `[0, ..., 127]`, select zero or `-inf`, repeat across the local Q heads, and call decode SDPA with `is_causal=False`, an attention mask, and no cur-position tensor. Source inspection showed the causal tensor variant allocates two extra circular buffers. The dense translation now constructs `[1,1,64,128]` with TTNN operations and retains update indices only for cache appends. Verdict: verified and fixed.
4. **Stage-review ablation of the retained 8x4 config.** The first passing run changed the mask while retaining 8x4, so the reviewer correctly found that the grid exception lacked isolated evidence. Removing both the config object and call argument left the watcher-enabled synthetic result bit-for-bit unchanged, including decode PCC 0.995945. Verdict: default emitted program configuration verified; explicit grid removed.

### Final status

Fixed. The original synthetic command passes with decode output PCC 0.995945 and no watcher failure. The real layer-39 command passes with prefill/decode PCC above 0.99999. Runtime forward methods remain free of torch and host conversion.

The final aggregate suite command is:

```bash
TT_VISIBLE_DEVICES=2,3 \
LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors \
scripts/run_safe_pytest.sh --dev --run-all \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_functional_decoder.py -q -s
```

After removing the unneeded program config, the final aggregate watcher-enabled suite completed in 22.65 seconds with three tests passed, zero failures, and zero skips. Its JUnit report is `generated/test_reports/most_recent_tests.xml`.

Independent stage-review evidence is appended after that final gate.

## Independent stage review

The first fresh-context review at `/root/functional_decoder_stage_review` returned `more-work-needed` for one evidence-isolation gap: the first passing emitted-mask run still retained an explicit 8x4 SDPA config, so its necessity had not been ablated. The config was removed, the focused watcher gate passed with identical PCC, the docs were corrected, and the complete synthetic/real suite was rerun.

The second fresh-context review at `/root/functional_decoder_stage_rereview` independently rechecked the selected raw IR and lowered emit, implementation, official-HF tests, post-edit JUnit timestamps, context contract, documentation, and functional-only scope. Its verdict was `clean-pass` with no required actions. It specifically confirmed that the final decode path has no `SDPAProgramConfig` or `program_config` argument and matches the emitted masked, non-causal SDPA signature.

## Local checkpoint

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/model/meta-llama-llama-3.1-70b-instruct`
- Functional-decoder checkpoint: `0d020f9ea7221dc89ae32d205421e6bdf4bc5493`
- Remote action: none; the checkpoint was not pushed.

## Multichip provenance

The retro-generated structured sharding record is `multichip_provenance.json`. It captures representative layer 39 on the compiler IR's `2x2` mesh (TP degree 4): residual/hidden shards use cluster axis 0, Q/K/V head and MLP-intermediate ownership use cluster axis 1, and KV caches own 4 of 8 heads per axis-1 coordinate. The complete collective set is two prefill RMS-stat `all_gather` operations and two decode `distributed_rms_norm` operations on axis 0, plus Q/K/V/gate/up `all_reduce` sums on axis 0 and O/down `all_reduce` sums on axis 1 in both paths. This was pure IR analysis; no device was opened or queried.
