# Diagnostic: serial singleton-L1 projection inside prefill

Candidate `v3_prefill_serial_l1.py.snapshot` is based on `v3_combined_prefill_consumers.py.snapshot` (SHA256 `12469ebab843dd5194bae041f30fdea5f9f0839c87fc3b5a996a33fcad4a0761`). Candidate SHA256 is `b8f0bb72ca84d7b61582896cccab2ec501ebb9f15fa2ee4ac251a4a804d1f50a`. The comparison baseline is separately recorded as immutable selected-v3 SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`; the new combined base is not relabelled as selected v3. This diagnostic does not change production, frozen prior candidates, or core harnesses.

The multirow packed-projection branch slices each logical token and calls the existing singleton branch with explicit `projection_k=1024`, then concatenates outputs. Since the recursive input has one logical row, it takes the preceding singleton branch and returns without further recursion. The singleton branch is AST-identical to the combined base. Each internal row retains canonical PF K1024 arithmetic; this is projection subdivision only, without recurrent updates, additional public cache casts, or changed public prefill boundaries. The candidate removes only the now-unused776MiB expanded PF weight upload. Existing compact scan weight and all other setup remain unchanged.

Each completed singleton output is copied to DRAM before being retained for concat, whose output is also explicitly DRAM. This is necessary to keep L1 occupancy bounded by the current row: a logical FP32 row `[1,1,6208]` occupies794,624 bytes with physical tile-height32. Retaining128 such outputs would add101,711,872 bytes beside the current101,711,872-byte product-plus-prefix pair. This candidate includes the row copy in both execution and timing rather than assuming retained singleton outputs are free. Products and prefixes remain L1, with the already-corrected explicit logical-H1 FastReduce output; source/native arithmetic is the same as the validated singleton adapter.

This differs from the rejected DRAM microbatch4/16/32 variants: only one token's product and prefix are live in L1, no full product/prefix passes through DRAM, and it uses the compact32-index singleton gather. It pays repeated input-row slicing, layout conversions, host dispatch, output copies, and final concat. S128 times cannot be inferred by multiplying the isolated singleton latency by128. Nonaligned input-row slices can entail extra untilization; the unmodified complete-method timings intentionally include that cost. There is no new native kernel or arithmetic association change.

Concat is source-legal for the many logical-H1 / padded-H32 operands. The top-level codegen router caps N-way inputs at8 (`concat/codegen/concat_codegen_supported.hpp:23`), so S32/128 use native fallback. `concat/concat.cpp:96–143` detects padding on the concat axis, untilizes each operand with unpadding, performs row-major concat, and retilizes the result. Rank3 is handled by its unsqueeze/squeeze adapter. `concat/device/concat_device_operation.cpp:411–483` automatically batches interleaved concat above47 inputs and recursively concatenates those intermediate results; a model-side bounded concat tree is therefore unnecessary for legality. S128 has three native first-level groups plus their final concat. All per-row unpadding, native batching and final retilization occur within the measured `self.linear` invocation. This is a substantial possible performance cost, not a free view or an omitted setup step.

`tests/v3_prefill_serial_l1_probe.py` is a separate diagnostic; the previous PF probe is unchanged. Its default comparison baseline is selected v3, with an explicit `--baseline-snapshot` argument and separately hash-bound candidate base. It runs the actual snapshot methods on actual layer0 checkpoint weights and clearly labelled synthetic normalized BF16 rows. It compares all6208 FP32/BF16 outputs for S2/32/33/128/129 and singleton K512/K1024. Any mismatch skips that case's timing. It records warmed complete **eager** adapter latency, including Python dispatch and synchronized completion, as well as warmed complete traced adapter latency. Every input slice, singleton call, DRAM row copy and concat is inside those timings. First/last traced outputs and a post-eager output are checked against the baseline. The parent's full decoder benchmark remains the separate whole-model test.

The source-only scope check confirms only `from_state_dict` and `linear` differ, with the singleton branch unchanged. `v3_prefill_serial_l1_cpu.json` passes selected-column FP32 partial/final mapping checks for33 rows and microbatch1. As with the prior probe, cumsum's positive-zero initial accumulator versus FastReduce's copied first product has not been proven bit-identical for every possible signed-zero input; actual output comparisons permit no exception. The candidate/probe/CPU artifact hashes are in the provenance sidecar. No device execution was performed by the preparer.

Run serially on the parent's device queue:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_prefill_serial_l1_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_prefill_serial_l1_probe.json
```

Default settings are physicalchip0, trace32MiB, L1-small32768, five eager samples and fifteen traced samples of five replays. The probe synchronizes/releases traces and closes the device in `finally`. Hardware allocation, exactness and performance remain pending. Neither the source legality nor the prior singleton speed establishes a PF improvement.

## Parent device result

`v3_prefill_serial_l1_probe.json` subsequently passed all seven cases, all6208 columns at both FP32/BF16 boundaries, first/last traced replay checks and post-eager checks. It reports `status=exact`, `device_exact=true`, and `device_closed=true`. The artifact binds the frozen candidate and probe hashes; the provenance sidecar now binds the device artifact hash. These are parent-owned measurements; the preparer did not execute hardware.

| Rows | Traced selected-v3 ms | Traced serial-L1 ms | Eager selected-v3 ms | Eager serial-L1 ms |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.992217 | 0.518634 | 1.070067 | 0.639994 |
| 32 | 11.069683 | 8.102387 | 11.264039 | 8.261387 |
| 33 | 14.438258 | 8.587507 | 14.531767 | 9.017333 |
| 128 | 48.473242 | 32.641130 | 48.663172 | 32.946660 |
| 129 | 49.241445 | 34.804903 | 49.827570 | 35.104987 |

Both columns include all input-row slicing, per-row dispatch where applicable, complete singleton projection, DRAM row copies, native concat batching, untilize/unpadding and final retilization. Eager timing is synchronized end to end. Thus host dispatch and concatenation did not erase this adapter's measured gain on these sizes. Whole-decoder S256/S4097 benchmarks and full acceptance remain separate parent work; this artifact alone does not select production.

## Legacy setup allocations are now unused

In this frozen serial candidate, AST inspection finds exactly one reference to each of `self.reduce_attention_weight` and `self.reduce_prefill_attention_weight`, both assignment targets in setup at lines109/110; there are zero loads. Singleton K512 and K1024 both reshape `scan_attention_weight`, and every multirow packed projection calls that singleton branch with K1024. The intermediate CPU `padded` value is used only by those two dead uploads.

Repository searches found no outside production caller of these attributes. The still-qualified live v3 implementation and historical snapshots use their own old attributes and must remain unchanged. Diagnostic fixtures assign the names for old baselines; `linear_frontend_stream_probe.py` conditionally creates them only when its inspected linear source names them. `summarize_fused_roofline.py` references them in an AST fallback for a projection product inheriting DRAM placement; it is not a runtime tensor consumer. A future selected-source roofline still needs the normal geometry update, independently of removing these dead assignments.

Removing both candidate-only uploads and their now-dead `padded` computation would reclaim two29,360,128-byte BF16 device buffers, totaling58,720,256 bytes (56MiB), without changing any runtime tensor read or arithmetic. The snapshot and probe remain unchanged here, as requested. That removal should be made as a separately hash-bound selection edit; old probe fixtures can keep their baseline allocations for historical reproducibility.
