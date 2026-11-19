# v3 carry and router data movement candidates

Status: root device diagnostic passed exact outputs for all ten cases. Full-path timing and acceptance remain pending. Production remains v3 SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`. These are independent graph rewrites, not arithmetic changes.

| Snapshot | Change | SHA256 |
| --- | --- | --- |
| [v3_movement_baseline.py.snapshot](v3_movement_baseline.py.snapshot) | Frozen v3 | `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba` |
| [v3_conv_rm_carry.py.snapshot](v3_conv_rm_carry.py.snapshot) | Reuse existing RM history for unaligned carry slice | `05196f487dcca8c9f60a5d046971aa59a2cbc92ea13b2b5b82776658bb0310e6` |
| [v3_router_rm_scores.py.snapshot](v3_router_rm_scores.py.snapshot) | Keep scores RM until final routing shape; remove identity casts | `606e104375015930cf28b600e5d103d6d7e9bcc7b6ffe5bdbe786f29b8c00b0a` |
| [v3_rm_carry_router.py.snapshot](v3_rm_carry_router.py.snapshot) | Both rewrites | `d3a704dcbb2575835a2273594e8abc2223a6ca96d7a052ca526825246164f4ca` |

## Convolution carry

The actual v3 convolution already materializes BF16 RM history once for taps 1–3. The final `history[:, S:S+3, :]` still slices the original TILE tensor. Native `slice.cpp:224–235` chooses RM for a TILE begin not aligned to tile height; line 382 converts the whole input to RM. Reusing existing `history_rm` for `S % 32 != 0`, then tilizing only three carry rows, removes that repeated conversion. Tile-aligned S keeps the original branch. BF16 history, FP32 product/addcmul order, activation boundaries, and TILE state allocation are unchanged.

This differs from the rejected FP32 shared-history proposal: no wider history is created. It also preserves the selected v3 BF16 RM tap sharing.

## Router score and id adapters

`generalized_moe_gate_device_operation.cpp:61–65` requires BF16 scores and UINT16 ids. The current decode branch immediately tilizes eight scores as `[1,8]`, then later performs a BF16 identity cast and reshapes to `[1,8,1,1]`. The candidate keeps those scores RM, reshapes in RM, and tilizes only the final shape. It also removes the UINT16 identity cast on native ids. Both consumer shapes, layouts and dtypes remain fixed; PF routing is unchanged.

The prior [v2 decode operation sequence](../profile_v2/tracy/layer_0/decode_op_sequence.csv) attributes 1.953 us to initial score tilize, 0.847 us to the ids identity cast, 0.896 us to the score identity cast, and 2.093 us to the later tiled routing reshape (steps 160, 163, 171, 172). These support testing dispatch/layout elimination; they are not a v3 speedup prediction. The final tilize remains required, and the native gate, sparse matmuls, weighted BF16 multiply and reduction remain unchanged.

## Exact complete-adapter probe

[tests/v3_data_movement_probe.py](../../../tests/v3_data_movement_probe.py) extracts the actual convolution block through carry copy and the actual four post-gate adapter assignments from each snapshot. It records snapshot and extracted-AST hashes, verifies individual/combined extracted blocks agree, and compares baseline, individual and combined outputs. It does not reconstruct a supposedly equivalent implementation outside the snapshots.

Convolution cases use checkpoint layer-0 taps and seeded BF16 history at S=1/32/33/127/128/129/256. They check FP32 preactivation, final BF16 output and exact next-carry codes. Complete convolution timing includes history concat, all slices, all layout/cast adapters, products, activation and carry copy. Four initial calls stabilize repeated S1 history before tracing, and traced outputs/carry are checked again.

Router cases use seeds 42/154/913 with checkpoint gate weights and CPU BF16 linear logits from synthetic hidden rows. The device then runs the actual native gate to produce operands for the adapters. These are checkpoint-derived inputs, not captured model inputs. Every post-gate score/id adapter remains inside trace timing; final shape/layout/dtype and exact score/id values are checked. The gate itself is common and excluded from adapter timing. Full decoder benchmarking below includes it and every other runtime operation. Router is decode-only; the seven lengths exercise convolution carry, not a nonexistent multirow native-gate path.

[CPU setup evidence](v3_data_movement_probe_cpu.json) confirms checkpoint loading, AST extraction, independent/combined block identity and seven carry-index mappings. It does not claim native operations succeed. The old measured v2 probe is unchanged.

Root-only device command after the active queue finishes:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_data_movement_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_data_movement_probe.json
```

Run paired full-path controls, retaining all conversions and ten prefill/sixty decode samples:

```bash
for candidate in movement_baseline conv_rm_carry router_rm_scores rm_carry_router; do
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_${candidate}.py.snapshot" \
    --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
    --output "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_${candidate}_s256.json"
done
```

Selection remains pending exact device adapter results, measured full-path benefit and required endurance coverage. The router candidate also affects the full-attention decoder's MoE and needs its layer-3 full-path control if retained.

## Root device results and identity-only follow-up

[v3_data_movement_probe.json](v3_data_movement_probe.json) reports `device_exact=true` across all ten cases, including native preactivation/output/carry or router ids/scores and first/last traced outputs. The measured probe is retained byte-for-byte as [v3_data_movement_probe.original.py.snapshot](v3_data_movement_probe.original.py.snapshot), SHA256 `5ee948b7c79e66e2c8d5a1fa36dfca02630327de54880fcfa7156703b1cce82b`, matching the report. Existing reports and candidate snapshots are unchanged.

| Complete convolution adapter | Baseline ms | RM carry ms |
| --- | ---: | ---: |
| s1 | 0.172696123 | 0.169093302 |
| s32 | 0.215287600 | 0.215495890 |
| s33 | 0.305199390 | 0.293377880 |
| s127 | 0.654364889 | 0.613798387 |
| s128 | 0.609939033 | 0.609841803 |
| s129 | 0.768549228 | 0.727436598 |
| s256 | 0.963664195 | 0.964075886 |

Tile-aligned lengths retain the original branch and differ only by timing noise. Unaligned carry reuse improves the complete adapter; root is running full-path controls.

The router RM-layout rewrite is exact but slower at all three seeds: seed42 0.017491356 → 0.018553482 ms, seed154 0.017512403 → 0.018515391 ms, seed913 0.017497409 → 0.018528523 ms. Its combined snapshot has the same adapter and similarly regresses. Full-path losing-control results remain pending.

Source explains why counting removed dispatches was insufficient. The old early tilize pads one `[1,8]` plane to one 32×32 tile. Keeping scores RM until `[1,8,1,1]` instead asks the tilizer to produce eight independently padded 32×32 planes. `core/to_layout/to_layout_op.cpp:102–105,222` computes the requested padded shape and calls `tilize_with_val_padding`. The old later TILE reshape uses the direct `prim::reshape_view` path (`data_movement/reshape_view/reshape.cpp:500–560`), with no default extra padding-fill for BF16. Both yield the same final logical tensor, but the candidate moves a larger batch/padding geometry through the tilizer. This is a source-supported explanation consistent with the measured total slowdown, not an uncollected per-op latency attribution.

A new [v3_router_identity_casts.py.snapshot](v3_router_identity_casts.py.snapshot), SHA256 `8f652a540ba6d5c6aa7aa4ee18f1fea568e14b00e5e9979782cf5ca269ce23f9`, removes only the UINT16→UINT16 ids cast and BF16→BF16 score cast. The original score tilize and tiled reshape geometry remain unchanged. Native gate dtype validation already proves these casts are identities. This is an isolated v3 candidate, not combined with carry or norm changes.

The probe now accepts `--router-snapshot`; an override compares only baseline and the alternate router, rather than mislabeling the old combined RM-layout snapshot as an identity-only control. The default still reproduces the original three-variant test. [CPU setup evidence](v3_router_identity_casts_probe_cpu.json) passes all three checkpoint-derived router cases with the alternate snapshot; device results remain pending.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_data_movement_probe \
  --kind router \
  --router-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_router_identity_casts.py.snapshot \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_router_identity_casts_probe.json

python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_router_identity_casts.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_router_identity_casts_s256.json
```
