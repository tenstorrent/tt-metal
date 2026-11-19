# Isolated ordered Q/Z projection control

[Candidate](v5_ordered_qz.py.snapshot), SHA256 `8f7fa4de6b07a93224fb7899cb144fedb7cd6afd162620dec018d8c85182a67e`, extends qualified V5's existing ordered KVAB projection to `linear(..., "output_only_packed")`. It is an unselected diagnostic; no device, model-quality, capacity or performance result is claimed.

The [actual layer34 counterfactual](AUTOFIX_v5_layer34_component_counterfactual.md) found exact normalized inputs but different native Q and Z outputs at137/165/181. At181, substituting either HF core or HF Z alone did not restore the captured route, while both plus canonical downstream arithmetic did. This motivates testing precise Q/Z; it does not prove Q/Z alone is sufficient or that every failing position has the same cause. This snapshot does **not** also apply the separate ordered-postnorm control or change the native gated norm.

## Scope and arithmetic

Setup concatenates the actual BF16 Q2048 and Z4096 checkpoint weights, transposes K/N, and applies the same adjacent-pair odd/even K order as KVAB. One additional `(1,1,2048,6144)` BF16 DRAM tensor costs exactly **25,165,824 bytes (24 MiB)**. Existing native Q/Z weights/sharding remain allocated so this is a narrow control, not an accounting claim that an old allocation was removed. Full-context capacity validation remains pending.

Only the two ordered `linear` branch guards admit the additional key. The singleton branch selects width6208/KVAB weight or width6144/QZ weight; multiplication, noncompensated FP32 cumsum, terminal-band/row extraction, sequential block reduction and final BF16/FP32 cast are otherwise unchanged. Public decode uses K512; the existing serial-prefill recursion explicitly uses K1024, including nonaligned S33. Output remains `(1,S,6144)` TILE with Q first and Z second. A Q/Z product or prefix occupies100,663,296 bytes, slightly less than the existing KVAB path. Execution is sequential, but no peak-memory/capacity pass follows from this geometry alone.

[Full-module AST proof](v5_ordered_qz.py.provenance.json) restores the two guards, local width/weight choices and one setup assignment and then matches the entire selected V5 AST. KVAB uses the same width6208, same weights, original arithmetic and original recursion. Full-attention and other projection roles are unchanged. There is no functional-decoder or host runtime fallback.

## Complete projection probe

[Probe](../../../tests/v5_ordered_qz_probe.py) uses actual layer34 QKV/Z weights with synthetic normalized BF16 input rows. It executes each snapshot's real `linear` method. Baseline native Q/Z sharding/program configuration is extracted from the baseline setup AST; the candidate ordered weight upload executes the candidate's actual setup assignment. Setup readback verifies ordered checkpoint weights. The fixture excludes unrelated decoder allocations and is not a full-model capacity test.

The probe covers singleton K512 and K1024, S33 and S128, with BF16 and FP32 outputs. It reports all6144 output-code/PCC comparisons against HF-style CPU QKV/Z linear outputs separately for native and ordered paths, plus Q/Z splits. An independent selected-column CPU reference uses explicit FP32 product accumulation with the selected K/block order. CPU HF references are computed separately at each actual M, avoiding silently reusing an M128 matmul for an M1 reference. The native and ordered outputs are **not** required to be identical; trace replay is checked against each path's own eager output. Production BF16-output eager and trace timings include every adapter, per-row loop, concat and final tilize.

[Final CPU proof](v5_ordered_qz_cpu_final.json), with [run provenance](v5_ordered_qz_cpu_final.json.provenance.json), passed without importing TTNN. All10 selected columns match canonical BF16 Q/Z codes for all four cases (1630 total values); this is not an all-column or device result. Setup-only AST execution using a CPU upload stub also reproduced the ordered checkpoint weights bitwise. The initial CPU run is preserved in `v5_ordered_qz_cpu.json`; final proof uses the corrected per-M reference construction. Black/isort and repository pre-commit pass. Device execution remains exclusively parent-owned.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_ordered_qz_probe \
  --baseline-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/selected_fused_decoder_v5.py.snapshot \
  --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz.py.snapshot \
  --layer 34 --seed 154 --lengths 1 33 128 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz_device.json
```

If useful after the component probe, the same-stream model discriminator is:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz_layer34_seed154.json
```

Cached-oracle, one-repeat results are diagnostic; quality thresholds and final acceptance requirements are unchanged.

## Completed component and isolated model results

The parent-owned [device projection report](v5_ordered_qz_device.json) completed and closed the device. All selected-column FP32 ordered-reference comparisons are bit exact. Across all6144 output columns, candidate BF16 versus HF code differences are:

| Case | Native differences | Ordered differences | Traced native → ordered ms |
| --- | ---: | ---: | ---: |
| S1, K512 |955 |0 |0.065148 →0.283829 |
| S1, K1024 |955 |1 |0.064791 →0.255920 |
| S33, K1024 |29012 |0 |0.089104 →7.975773 |
| S128, K1024 |111947 |0 |0.094133 →30.872714 |

The single S1/K1024 difference is packed column5602 (Z column3554): HF float-encoded BF16 bits0x37780000 versus candidate0x37730000, absolute difference2.980232238769531e-7. This cross-policy singleton compares K1024 accumulation against HF's actual M1 kernel; it prevents claiming universal all-column exactness. FP32 diagnostic outputs also differ from ordinary FP32 CPU matmul because the accumulation schedules differ; the explicit ordered reference matches all sampled columns. Eager/trace outputs agree within each device path. Complete eager costs likewise increase (S1/K5120.108277→0.370028 ms, S1280.112475→31.085267 ms). These are complete projection costs, not whole-decoder latency.

[Component summary and evidence hashes](v5_ordered_qz.component_summary.json) preserve the raw comparisons/timings. The isolated Q/Z [layer34 seed1548192 run](v5_ordered_qz_layer34_seed154.json) **fails** the unchanged gate with282 failing rows and minimum contribution PCC0.91251756094; positions137 and181 remain failures. Q/Z accuracy improves locally but does not alone repair this failure family. No threshold waiver, model pass or speed improvement is claimed.

## Prepared Q/Z plus ordered-postnorm composition

[Combined diagnostic](v5_ordered_qz_postnorm.py.snapshot), SHA256 `2c66df4dbb29378764175edf6bb7591e96807f4d806c1de2207538cdbcc56fc2`, adds exactly the ordered-postnorm guard from [its isolated donor](v5_ordered_postnorm.py.snapshot) to the Q/Z snapshot. [Composition provenance](v5_ordered_qz_postnorm.py.provenance.json) proves that restoring this guard reproduces the entire Q/Z-base AST, and that the combined norm method is identical to the ordered-postnorm donor. Q/Z/KVAB projection code and V5 native output gated norm remain unchanged. The combination retains the24 MiB extra Q/Z weights and adds no further persistent allocations. Its model result is pending; it is not selected.

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz_postnorm.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_qz_postnorm_layer34_seed154.json
```

Raw device-report SHA256: `272dcdd650793fc5dde0e3dae7d71e554d1e0b787c6959b24db5b02391c6fb4a`. Isolated8192 model-report SHA256: `7db8aff4c7bca3dcdc9fe53e5fe43fca44250e4900ab733d28c020bb331ebf40`.
