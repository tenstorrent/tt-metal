# Isolated rank-5 head broadcast candidate

Frozen base: `../selected_fused_decoder.py.snapshot`, SHA256 `61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178`. Candidate: `exact_next_rank5_heads.py.snapshot`; adjacent `.patch` and `.provenance.json` identify the precise change. Production source and existing core tests were not edited.

Only `_recurrent_step` changes. Replace contiguous head duplication with an explicit group/member axis: state `[1,32,128,128]` views as `[1,16,2,128,128]`; Q/K `[1,16,128,1]` view as `[1,16,1,128,1]`. Broadcasting repeats each Q/K head across the two member heads. Delta views as `[1,16,2,1,128]`. All three product results view back to `[1,32,128,128]` before their original consumers. Index identity is `head = 2*group + member`; no data permutation occurs in these views.

The standalone FP32 query multiplication by `128**-0.5` remains a separate operation, on the sixteen distinct heads before the broadcast. This candidate does not include the independent query-scale activation fusion. State BF16 round/widen activation, decay arithmetic, delta subtraction/multiply, state add, all memory placements, and the ordered sixteen-then-eight recurrence contraction remain unchanged. The separately inlined prefill state correction remains unchanged.

Native source contracts:

- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:450`: permits singleton broadcasting through rank six. The new member-axis broadcast is rank five.
- `binary_ng_program_factory.cpp:45`: explicitly extracts rank-five outer dimensions from padded shapes; the member axis is an outer dimension rather than a padded spatial axis.
- `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:757`: unchanged last two dimensions and placement permit metadata-only views. Every new view preserves H/W (128/128 for states/products, 128/1 for Q/K, 1/128 for delta), including their tile padding. This does not introduce an H=16 padded reshape.

`tests/rank5_recurrence_probe.py` compares the frozen and candidate methods directly. It captures the decayed state, memory product, update product, output product, both reductions, final updated state, and returned output. Device cases cover saved real positions 128, 130, 7703, both cache-rounding settings, and both EXP(g) and explicit decay branches. The explicit-decay fixture is shared CPU EXP data; it is not a device EXP accuracy claim. Temporary hooks are restored in `finally`; host conversions happen only after the complete recurrence call. Comparisons are strict FP32 code equality, including signed zero. Device open uses physical chip 0, reserves 32 KiB small L1, and closes in `finally`. Provenance includes both source hashes and capture hashes.

CPU-only validation completed: `rank5_recurrence_cpu.json`, 48/48 comparisons exact on all three real captures and both state-rounding settings. This proves the indexing and arithmetic transformation in the CPU model, not the native rank-five implementation or its latency. Native dispatch, broadcasting correctness, and performance still require the parent's serialized device run; no hardware was executed here.

Run the device discriminator before measuring the snapshot:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.rank5_recurrence_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/rank5_recurrence_device.json
```

For the CPU-only proof, add `--oracle-only` and use a separate output path. A passing primitive probe is diagnostic evidence, not a substitute for real-weight warmed latency and fresh-HF long-stream qualification.
