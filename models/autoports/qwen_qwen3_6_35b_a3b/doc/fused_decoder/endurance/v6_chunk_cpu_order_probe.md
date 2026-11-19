# CPU accumulation order required by the actual chunk64 prefix

The [CPU probe](../../../tests/v6_chunk_cpu_order_probe.py) recreates actual layer34/seed154 prefix operands and compares installed Torch matrix products against independently compiled scalar C++. A **single FP32 fused multiply-add accumulator traversing K in increasing order** reproduces all numerical values of all six tested matrix products, across all32 heads and both64-token chunks. Five products are bit-exact; `k_cumdecay @ state` has52 differences confined to signed-zero bits. No nonzero or BF16 numerical differences occur in that full-head FMA comparison.

The [JSON](v6_chunk_cpu_order_probe.json) records full-head comparisons, source/prefix/compiler hashes, Torch build identity, and sample operand/output hashes. Competing algorithms use the first eight batch matrices: four heads × two chunks for the first three products, or eight heads for state updates. The scalar helper explicitly uses `std::fma`; `-ffp-contract=off` prevents accidental fusion in the separate multiply/add control.

| Product | Sample output elements | Sequential FMA FP32 differences | Sequential rounded multiply/add FP32 differences | Products then Torch sum FP32 differences |
| --- | ---: | ---: | ---: | ---: |
| k_beta @ key.T | 32,768 | 0 | 0 | 27,316 |
| A @ v_beta | 65,536 | 0 | 6,325 | 14,216 |
| A @ (k_beta × exp(cumulative_g)) | 65,536 | 0 | 15,430 | 22,246 |
| scaled_key.T @ v_new, chunk0 | 131,072 | 0 | 58,195 | 69,080 |
| scaled_key.T @ v_new, chunk1 | 131,072 | 0 | 63,146 | 74,540 |
| k_cumdecay @ incoming_state | 65,536 | 0 | 42,201 | 51,479 |

Separate multiplication followed by addition creates7 and11 BF16 differences in the two sampled final state updates. Thus matching only contraction topology or using a generic product/reduction graph is insufficient to claim exact state parity. Independent2/4/8/16 accumulators with linear or tree combination also fail to reproduce these FP32 products.

Two further ordering requirements were measured:

- **Cumulative log decay:** the canonical FP32 `cumsum` is bit-exact to FP64 cumulative accumulation followed by FP32 rounding on all4,096 entries. Sequential FP32 accumulation differs at2,372 entries, max absolute difference6.103515625e-5. No BF16 differences occur at that immediate boundary, but these FP32 values feed exponentials and state updates.
- **Triangular correction row reduction:** row7 matches a simple sequential FP32 sum. Rows31 and63 do not: simple sequential sums differ at1,100 and2,578 output elements respectively. Tested2/4/8 accumulator arrangements, including combining accumulators before a scalar remainder, do not exactly match those larger rows. The best tested interleaved4-accumulator linear combination still differs at509/1,803 elements. These tests found no exact general row-reduction mapping; they do not establish that one is impossible.

For a device graph, sequential-K accumulation remains the measured matrix-product target. However, ordinary TTNN `addcmul` is **not proven equivalent to IEEE single-rounding FMA**. The repository's `tests/ttnn/unit_tests/operations/eltwise/test_ternary_category1_float32.py:54–69` explicitly states that FP32 `mac`/`addcmul` do not carry the full product into the add. Its exactness guarantee applies to BF16-grid operands, not arbitrary FP32 operands from the transformed chunk matrices. An ordered `addcmul` implementation is consequently an approximation experiment until checked on these actual operands. An exact candidate may require compensated products/sums or another model-local kernel strategy, together with measured cumsum and triangular-reduction parity.

Command, CPU only:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk_cpu_order_probe
```

No hardware, TTNN import, production decoder edit, native/shared source edit, or acceptance-gate change was performed.
