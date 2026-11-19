# Current-source decode native gated-norm candidates

Four isolated snapshots use frozen v4 `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`. Production is unchanged. CPU syntax checks and AST comparison establish that only the decode output-normalization tail changes: setup, projections, recurrence/state copy, prefill, full attention and MoE are unchanged. [Source proof](v4_decode_native_norm_source_proof.json) binds every snapshot hash. [CPU probe validation](v4_decode_native_norm_cpu_v2.json) binds the final probe, baseline and candidates. The first CPU report is retained; v2 corrects probe fixture placement to FP32 L1 core and FP32 gamma, matching current production setup.

| Snapshot | Native geometry | Native output | External final product |
|---|---|---|---|
| [heads BF16](v4_decode_native_norm_heads_bf16.py.snapshot) | 32 heads, T32 with promoted physical padding | BF16 | Native result times FP32 Z, rounded to BF16 |
| [heads FP32](v4_decode_native_norm_heads_fp32.py.snapshot) | 32 heads, T32 with promoted physical padding | FP32 | Native result times FP32 Z, rounded to BF16 |
| [rows BF16](v4_decode_native_norm_rows_bf16.py.snapshot) | One head, 32 head-as-token rows | BF16 | Native result times FP32 Z, rounded to BF16 |
| [rows FP32](v4_decode_native_norm_rows_fp32.py.snapshot) | One head, 32 head-as-token rows | FP32 | Native result times FP32 Z, rounded to BF16 |

All four preserve the initial BF16 core cast and use native L1 interleaved output. The compact variants retain the current sharded head-concat adapter and its metadata crop. The true-head variants still perform actual head-first expansion, then promote its physical H32 padding through explicit logical/padded metadata. They promote the existing Q/Z gate operand from logical H1 to H32 with width6144 and offset2048. Native output already has packed heads; its first token is selected by metadata and multiplied by the existing Z slice. This removes explicit pad/crop operations from the historical true-head adapter, and avoids the final head-concat used in the compact geometry. It does not claim head-plane expansion is free.

Native validation permits both output dtypes and interleaved L1 memory. Input must be rank-three `[B*H,T,V]`, gate rank-three `[B,T,W]`, T divisible by32, gate offset divisible by32, with sufficient gate width. The true-head promoted input has physical shape `[32,32,128]`; its head-first reshape allocates the required128 tiles before promotion. The compact input is `[1,32,128]` with one head, only four physical tiles. Both are valid native geometries. Runtime memory placement and installed binding support still require the parent-owned device probe.

The native gate input is BF16, with internal FP32 sigmoid and normalization buffers. Neither output policy recreates current intermediate BF16 normalization/gamma boundaries or the current multiplication association. BF16 native output adds rounding before the external Z product; FP32 native output defers that rounding. These differences are reported explicitly and **are not new acceptance gates**. The unchanged functional/model PCC bar, representative-layer tests and 8192-token streams decide quality. Historical .9 ms controls predate the current precise graph: their losses remain true historical evidence, not a current-source rejection. Current whole-decoder speed must be measured against b163, including all adapters.

The bounded [probe](../../../tests/v4_decode_native_norm_probe.py) extracts each actual tail via AST and returns its merged output before output projection. It measures all norm, gate, cast, reshape, sharding and head-concat work present in that tail. It uses three deterministic synthetic finite core/QZ fixtures (seeds42/154/913), real checkpoint gamma/epsilon, FP32 L1 core, DRAM QZ and FP32 gamma. It reports code differences/PCC/relative L2 against current b163 without rejecting candidates for those diagnostic differences. Trace replay must reproduce each candidate's own output. Independently supplied true-head physical rows contain verified +17, -23 or NaN sentinels; consumed row zero must match the actual metadata adapter. That assertion checks geometry and row independence, not an extra model arithmetic gate. Synthetic fixtures do not replace model stress.

The initial preparation was CPU-only. The parent subsequently completed the isolated device probe below; whole-decoder benchmarks and cached long-stream checks have now completed as recorded below. Combined v5 source qualification remains pending. Parent-only commands, from the checkout with the usual lane environment:

```bash
D=models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder
E="$D/endurance"
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_decode_native_norm_probe \
  --output "$E/v4_decode_native_norm_device.json"

# Run serially. Baseline is the identical frozen source, not an older timing graph.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot "$D/selected_fused_decoder_v4.py.snapshot" --layer 0 --length 256 --context 512 \
  --output "$E/v4_decode_native_norm_control.json"
for variant in heads_bf16 heads_fp32 rows_bf16 rows_fp32; do
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "$E/v4_decode_native_norm_${variant}.py.snapshot" --layer 0 --length 256 --context 512 \
    --output "$E/v4_decode_native_norm_${variant}_benchmark.json"
done

# Example for a faster short-PCC-passing candidate; repeat for seeds42 and154.
# Cache913 is stress_8192_0.pt; the other caches have _seed42/_seed154 suffixes.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot "$E/v4_decode_native_norm_rows_fp32.py.snapshot" --layer 0 --seed 913 \
  --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache "$D/oracle_cache/stress_8192_0.pt" \
  --output "$E/v4_decode_native_norm_rows_fp32_stress_seed913.json"
```

The example does not select rows-FP32 in advance. Device validation failures, model quality failures and actual complete-trace latency losses must be recorded separately. A faster passing candidate would still need normal frozen-source qualification before selection. No Stop Rule or completion claim follows from preparation.

## Completed isolated device measurement

[Raw report](v4_decode_native_norm_device.json), [run provenance](v4_decode_native_norm_device.json.provenance.json), and [coverage/source identities](v4_decode_native_norm_device_coverage.json) record the parent-owned device execution. The report SHA-256 is `b7a9b4c61a4cccca2a4449bf8c793c7ce13cbece0f711deb0fa45bdf26246e07`. The mesh closed successfully. All five adapters reproduced their own trace output bitwise. All 18 true-head padding cases (three fixture seeds, two output policies, and verified +17/-23/NaN fills) preserved the consumed output exactly. All 15 adapter cases produced finite BF16 outputs of logical shape `[1,1,4096]`.

| Complete adapter | Traced median ms | Change versus current adapter |
|---|---:|---:|
| baseline | 0.039624795 | +0.000% |
| heads_bf16 | 0.033986988 | -14.228% |
| heads_fp32 | 0.034664292 | -12.519% |
| rows_bf16 | 0.041854102 | +5.626% |
| rows_fp32 | 0.042258808 | +6.647% |

These timings include the actual normalization, gate, dtype/layout conversions and head packing, excluding recurrence and final output projection. They use seed 154 with seven samples of ten trace replays each after warmup. The true-head policies save about 5.64 µs (BF16) and 4.96 µs (FP32); compact head-as-token variants cost about 2.23–2.63 µs more in this isolated current-source measurement. The subsequent whole-decoder controls are recorded below; isolated adapter results do not select or reject a complete candidate by themselves.

Against the current adapter, BF16 native output differs in 1906–1925 of 4096 codes with diagnostic PCC 0.999992730–0.999993678; FP32 native output differs in 1694–1714 codes with PCC 0.999994063–0.999995089. Both geometries report the same code-count/PCC metrics for each corresponding output policy and fixture. These arithmetic differences are retained as measurements, not failed quality gates. The subsequent unchanged whole-decoder PCC and 8192-token stream results below now distinguish the policies. This isolated measurement itself makes no model-quality, production-selection, final qualification or Stop Rule claim.

## Completed current-source whole-decoder comparisons

[Model coverage and immutable report/provenance hashes](v4_decode_native_norm_model_coverage.json) bind these parent-owned results. All four short S256/context512 runs pass their unchanged PCC check and report deterministic eager/traced output. The current b163 control is shared with the adjacent cache-layout investigation; its recorded source hash is b163, not an older .9 ms graph.

| Source/result | Warmed prefill ms | Traced decode ms | Decode PCC | Decode change versus b163 |
|---|---:|---:|---:|---:|
| [v4_persistent_rm_cache_baseline_s256.json](v4_persistent_rm_cache_baseline_s256.json) | 107.493589 | 1.317830756 | 0.9999779065 | +0.000% |
| [v4_decode_native_norm_heads_bf16_s256.json](v4_decode_native_norm_heads_bf16_s256.json) | 107.332349 | 1.307491213 | 0.9999819054 | -0.785% |
| [v4_decode_native_norm_heads_fp32_s256.json](v4_decode_native_norm_heads_fp32_s256.json) | 108.187119 | 1.306965249 | 0.9999651753 | -0.824% |
| [v4_decode_native_norm_rows_bf16_s256.json](v4_decode_native_norm_rows_bf16_s256.json) | 107.704911 | 1.318766968 | 0.9999819054 | +0.071% |
| [v4_decode_native_norm_rows_fp32_s256.json](v4_decode_native_norm_rows_fp32_s256.json) | 108.679746 | 1.318953000 | 0.9999651753 | +0.085% |

Compact rows-BF16 and rows-FP32 fail to beat the measured current control by approximately 0.94 and 1.12 µs, respectively. These are small measured losses, not a claim of a large statistically established regression. They are rejected on the required complete-decode speed comparison; no separate long-stream quality result is claimed for them. True-head BF16 improves complete decode by 10.34 µs (0.785%); FP32 improves by 10.87 µs (0.824%) but fails the quality test below. Prefill implementation is unchanged across these norm-only sources, so the measured prefill variation is not attributed to a prefill optimization.

## Cached long-stream model gates

Each run uses the same canonical 128-token prefix, 8192 external decode tokens and 33-token continuation, with one repeat and cached CPU oracle. This is candidate qualification evidence, not the required final fresh multi-repeat validation.

| Policy / seed / raw result | Gate | Minimum decode output PCC | Minimum decode contribution PCC | Minimum continuation contribution PCC |
|---|---|---:|---:|---:|
| [v4_decode_native_norm_heads_bf16_stress_0_seed913.json](v4_decode_native_norm_heads_bf16_stress_0_seed913.json) | PASS | 0.996648264968 | 0.996207914184 | 0.999730376269 |
| [v4_decode_native_norm_heads_bf16_stress_0_seed42.json](v4_decode_native_norm_heads_bf16_stress_0_seed42.json) | PASS | 0.996991749917 | 0.995818587874 | 0.999386044685 |
| [v4_decode_native_norm_heads_bf16_stress_0_seed154.json](v4_decode_native_norm_heads_bf16_stress_0_seed154.json) | PASS | 0.997471575783 | 0.997026821929 | 0.999534070502 |
| [v4_decode_native_norm_heads_fp32_stress_0_seed154.json](v4_decode_native_norm_heads_fp32_stress_0_seed154.json) | FAIL | 0.994973244625 | 0.994330296640 | 0.999534070502 |

Heads-FP32 fails at seed 154 position 5991: output PCC 0.994973244625 and contribution PCC 0.994330296640, both below the unchanged 0.995 bar. Its FP32 output policy is therefore rejected by an actual model gate, not by intermediate code differences. The heads-BF16 source changes only this native output dtype relative to heads-FP32 and passes all three seeds. Rounding the native weighted-normalization-times-sigmoid result to BF16 before the external FP32 Z product is consequently material to this tested graph. The result does not by itself prove which downstream routing decision amplified the difference; no new tensor-level routing diagnosis is claimed. Earlier router-tie evidence at the same position is context, not a substitute for a fresh capture.

The measured heads-BF16 source is the passing norm policy retained for a combined candidate. This does not mark production selected or final stage qualification complete.

## Prepared cache-writer plus native-norm composition

[v5_cache_writers_native_norm.py.snapshot](v5_cache_writers_native_norm.py.snapshot), SHA-256 `1a593ec9a2522dd579194d5082ffe5e987a81fc87be287590bdec086e067d455`, combines the exact heads-BF16 tail with `v5_cache_writers.py.snapshot` SHA-256 `9d33bc14b5f349a9dabfbd4e6ce6344c31047fbcfcc2716f35de0133e2d8b7f7`. [Composition proof](v5_cache_writers_native_norm.provenance.json) and [patch](v5_cache_writers_native_norm.patch) establish that the norm tail is AST-identical to the tested donor and all other module AST is identical to the writer base. The old norm tail in the writer base was first checked against b163. Syntax compilation passes. Production is unchanged; the combination has no device result yet.

Parent-only paired command (fresh output names):

```bash
E=models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance
for candidate in v5_cache_writers v5_cache_writers_native_norm; do
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "$E/${candidate}.py.snapshot" --layer 0 --length 256 --context 512 \
    --output "$E/${candidate}_norm_composition_s256.json"
done
```

The combined source still requires its own long-stream checks and final frozen-source qualification before any completion claim.
