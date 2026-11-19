# Same-logit native routing discriminator

The actual combined-accurate-residual capture at position 7944 selects native expert 74 where HF selects 202. CPU canonical routing on **those same TT logits** selects the HF set including 202 and 64. This is a local routing discrepancy at a three-way tied boundary; remaining upstream differences are not necessary to produce this particular membership discrepancy. Equal scores do not make the different expert outputs interchangeable, and no quality threshold is relaxed.

[CPU setup evidence](v5_router_tie_cpu.json) confirms this separately for direct BF16 topk, direct FP32 topk and FP32 softmax→topk. All three CPU variants choose the HF set on the actual TT 7944 logits. Position 181 is deliberately retained as a contrasting case: its actual TT logits choose 52 rather than HF 184 in all three CPU variants, so a device router matching CPU on that input would still not recover HF membership there.

## Parent-owned device probe

[New probe](../../../tests/v5_router_tie_probe.py) consumes the actual HF and TT logits from [combined181](v5_combined_layer34_stream_probe.json.pos181.pt) and [accurate-residual7944](v5_accurate_residual_layer34_worst_probe.json.pos7944.pt), four fixtures total. Despite the latter filename, its executed source is combined-accurate-residual `e3aea1c50aaf54b0ffa69949f8fdcc6bbe7232a64fac3a9fed8d3eaf052f2e6c`, not selected-V5-only `7cb0`. Each capture's source/hash is explicitly retained in the report.

The probe executes AST-extracted actual decode router bodies and actual consumer adapters from four snapshots:

| Variant | Decode producer | Snapshot SHA256 |
| --- | --- | --- |
| Native baseline | Generalized gate with its actual input packing, sharding and output slices | `e3aea1c50aaf54b0ffa69949f8fdcc6bbe7232a64fac3a9fed8d3eaf052f2e6c` |
| [BF16 topk](v5_router_bf16_topk.py.snapshot) | Direct BF16 topk; FP32 softmax over selected logits; BF16 scores | `297217d317236a23ccdd1fef6c12ee4f55d59838f8896ba9d6d484ba3bf11add` |
| [FP32 topk](v5_router_fp32_topk.py.snapshot) | Widen BF16 logits, FP32 topk, selected FP32 softmax, BF16 scores | `748aa8f76390654e829297dd8ab3a852dd1cb560b2fa34e320c8df1f9383fb83` |
| [FP32 softmax/topk](v5_router_fp32_softmax_topk.py.snapshot) | FP32 softmax over all 256 logits, topk8, FP32 sum/divide renormalization, BF16 scores | `626288ecba392e5719fa2f27a2921e871cdc25ca6b785adb4baaac2861c6fdb4` |

The three candidates change only the `s == 1` router producer inside `_moe`, based on e3aea. Their sibling `.py.provenance.json` files prove that restoring that branch gives the entire baseline module AST. Prefill, logits projection, expert arithmetic and existing routing consumers are unchanged. FP32 topk produces 32-bit IDs; the candidates explicitly cast to UINT16 before the existing sparse-expert consumer, whose valid IDs are 0–255. No host runtime fallback, device source change or additional persistent allocation is introduced. Setup native gate buffers remain allocated in the diagnostic snapshots.

Every eager/traced timing includes the complete router plus output adapters through BF16 TILE routing `(1,8,1,1)` and UINT16 ROW_MAJOR sparse IDs `(1,1,1,8)`: all casts, packing, sharding, topk/softmax, renormalization and output reshapes/layout changes. Logits projection is excluded equally. The probe verifies output shapes/dtypes, finite scores, unique IDs, input immutability and exact replay against each variant's own eager outputs. It reports membership against both same-input CPU routing and captured HF routing, with scores compared by expert ID. A tie mismatch is evidence, not a reason to silently skip or alter quality gates.

Native [topk validation](../../../../../../ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp) accepts BF16 and FP32 on Blackhole; FP32 uses 32-bit indices. Native [softmax validation](../../../../../../ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_device_operation.cpp) accepts FP32. The explicit compute configuration is HiFi4, nonapproximate, FP32 destination. Existing topk wrappers handle the singleton padded TILE height as in the model's existing prefill route. The completed device probe established installed-binding/runtime compatibility for all four variants and fixtures.

`stable=True` is not used: its documented lowest-index tie rule is not the observed CPU topk8 rule. No candidate is assumed to reproduce Torch's unstable tie behavior on all inputs. Even a same-logit match at these two positions requires actual model stream qualification before selection.

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v5_router_tie_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_router_tie_device.json
```

CPU setup and AST extraction passed without importing TTNN, and Black/isort/pre-commit passed. The probe opens only physical chip 0 and closes it in `finally`; it has not been executed on hardware by this agent. The parent-owned component probe has now completed as recorded below. Full model controls subsequently failed as recorded below; no variant is selected.

## Completed device component comparison

The [device report](v5_router_tie_device.json) completed with 16 fixture/variant combinations. Every variant's trace output matched its own eager output and preserved the input logits; the device closed cleanly. [Decision summary and raw hashes](v5_router_tie.decision.json) preserve the observations. **None of the tested alternatives preserves HF/CPU tied membership across these fixtures.**

At 7944, both the actual HF logits and actual TT logits yield the same mismatch:

| Device router | Disputed selected experts | CPU/HF selection | Traced complete adapter on TT logits, ms |
| --- | --- | --- | ---: |
| Native gate |64,74 |202,64 |0.016241544 |
| BF16 direct topk |74,202 |202,64 |0.049955398 |
| FP32 direct topk |74,202 |202,64 |0.062211952 |
| FP32 softmax→topk |74,202 |202,64 |0.088270684 |

At 181 on **HF logits**, native gate and both direct-topk variants choose HF 184, but FP32 softmax→topk chooses 52. On actual TT 181 logits all four select 52, matching the same-input CPU selection and differing from HF's original input case. This is why same-input CPU membership and captured HF membership are recorded separately. Merely widening logits or taking topk after FP32 softmax does not establish HF tie equivalence.

All three alternative complete adapters are slower than native in every measured fixture. Those costs include selection, score arithmetic and output adapters, not logits projection or expert execution. Common-expert score differences are recorded separately; matching a tied score does not compensate for selecting another expert. The parent subsequently completed full 8192 model streams for all three variants, reported below. The full-model rejection rests on those completed results, not an inference from the component test. No threshold or tie-based exception is proposed.

## Completed full-stream model controls

All three parent-owned layer 34 seed 154 controls completed the 128-prefix +8192-decode +33-continuation trajectory, one repeat, using the same cached oracle. **Every alternative fails the unchanged PCC gate and is rejected.**

| Variant | Failing decode rows | Minimum contribution PCC | Minimum position | Position 7944 contribution PCC |
| --- | ---: | ---: | ---: | ---: |
| bf16_topk | 237 | 0.937916597415 | 4307 | 0.986243086999 |
| fp32_topk | 237 | 0.937916597415 | 4307 | 0.986243086999 |
| fp32_softmax_topk | 245 | 0.937925540455 | 4307 | 0.986260197403 |

The alternative selection improves the formerly severe 7944 error but does not pass that token or the full stream. Other tied or near-tied routing decisions still fail; no single cause is assigned to every remaining row. All candidates remain unselected. [Updated decision](v5_router_tie.decision.json) and each candidate provenance preserve exact model JSON/provenance/log hashes, executed source, failure rows and diagnostic scope. The CPU-topk exact-tie source investigation is separate work; it is not a completed repair.

- [bf16_topk full-stream report](v5_router_bf16_topk_layer34_seed154.json), SHA256 `bddaf020109aaf5b6bfdefccc16777a3e8ae2511bac3a14eae060497fdd09b79`.
- [fp32_topk full-stream report](v5_router_fp32_topk_layer34_seed154.json), SHA256 `4528c578de72feda946d0366d4cd0b303fce974bac204633e5f73195ca99cff9`.
- [fp32_softmax_topk full-stream report](v5_router_fp32_softmax_topk_layer34_seed154.json), SHA256 `3cc991fd6476a245517ef38b56d7223903b21d0e6485e2c8a24b409cdfb12a2d`.
