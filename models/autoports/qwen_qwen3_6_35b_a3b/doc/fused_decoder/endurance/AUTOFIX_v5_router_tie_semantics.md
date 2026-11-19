# Layer34 routing: CPU top-k tie semantics versus existing native configurations

**No inspected native configuration guarantees the installed Torch CPU `topk(8)` tied membership for a256-entry row.** This is a statement about the exposed contracts and algorithms, not a claim that an exact device implementation is impossible or that a tested alternative cannot satisfy the unchanged model gates. The parent is testing native alternatives. No hardware, production or native edits were performed here; no reference or threshold change is proposed.

## Actual worst-row evidence

The root's [accurate-residual capture](v5_accurate_residual_layer34_worst_probe.json), position7944, records three identical BF16 router logits at expert64,74,202: **1.1484375**. Six experts have larger logits, so only two of these three fit in top8.

| Consumer of the same captured TT logits | Last two expert IDs |
| --- | --- |
| Canonical CPU FP32 softmax, then Torch topk8 | 202,64 |
| Actual native generalized gate | 64,74 |
| CPU stable descending sort by original index | 64,74 |

The actual HF router also chooses202,64. The actual native score/result capture is already a device observation made by the root, not a predicted result from this investigation. Original capture tensor SHA256 is `a675662c1c894b1c6c51778b622a9ddc033e6c7002d4d5c0e2311c1690e493e8`. This distinction is **membership**, not merely a permutation of the same selected experts. Even with identical logits in this comparison, tie selection remains a separate discrepancy from frontend numerical error.

Selected-score arithmetic is also a separate remaining boundary: on these same TT logits the canonical selected tied score is0.1044921875, while native records0.10400390625. This memo does not attribute the full output error exclusively to either membership or scores; the parent owns integrated discriminating controls.

## Installed Torch implementation and why fixed tie priorities do not reproduce it

Local primary source is `python_env/lib/python3.12/site-packages/torch/include/ATen/native/TopKImpl.h`:

- Lines31–49 create `(value, original_index)` pairs in original row order.
- Line45 chooses partial sort only when `k*64 <= dim_size`. For k8/N256,512>256, so that branch is not selected.
- Lines67–75 call `std::nth_element(begin, begin+k-1, end, comparator)` and, when sorted, `std::sort(begin, begin+k-1, comparator)`.
- Both comparators compare **only values**, with explicit NaN handling; they do not compare the index member when values tie.

The installed Torch build reports2.11.0+cpu, GCC13.3 and C++17. The available GCC13 standard-library header `/usr/include/c++/13/bits/stl_algo.h:1957–1978` implements `__introselect` with pivot partitioning, a depth-limited heap fallback and a small-range insertion sort. This supports the value-dependent permutation explanation; the installed Torch header and direct CPU observations are the primary evidence, rather than assuming every detail of the compiler's build headers matches the local system copy.

The installed `torch.topk.__doc__` explicitly does not guarantee stable indices for tied elements. That API fact does not excuse the observed model PCC failure: this project's frozen CPU oracle has specific tied membership that still affects the decoder output.

[Saved permutation proof](v5_torch_topk_tie_permutation_proof.json), SHA256 **`38dec7837bc98e85de12a8eeff49625740e1a85a30d5ed6514cf1859a57326de`**, contains all three256-element BF16 input-code vectors, per-vector hashes, source hashes, Torch build identity and captured-source linkage. The proof keeps the three tied entries and all six strictly higher entries exactly fixed. It permutes only the247 strictly lower entries, preserving their multiset. It uses a CPU generator with seed0, then canonical FP32 softmax and topk8:

| Lower-logit arrangement | Torch selected tied subset |
| --- | --- |
| Original saved row | 64,202 |
| First seeded permutation | 64,74 |
| Second seeded permutation | 74,202 |

All three possible tied subsets occur. Thus a single static expert-index priority (low index, high index or a fixed bias that breaks the tie) cannot emulate the observed Torch selection for these three inputs. This proof does not rule out a dynamic device algorithm that reproduces the CPU partition behavior.

Recheck the saved proof without regenerating the model stream:

```python
proof = json.load(open("v5_torch_topk_tie_permutation_proof.json"))
for case in proof["cases"]:
    logits = torch.tensor(case["input_bf16_codes"], dtype=torch.int16).view(torch.bfloat16)
    assert logits.float().softmax(-1).topk(8).indices.tolist() == case["top8"]
```

## Existing native contracts

Paths in this section are repository-relative.

| Native option | Exact source contract | Consequence for this request |
| --- | --- | --- |
| `ttnn.topk(..., stable=True)` | `ttnn/cpp/ttnn/operations/reduction/topk/topk_nanobind.cpp:59–60`: exact-value ties use lowest original index; user-supplied indices cannot be combined with stable mode. `device/kernels/compute/topk_common_funcs.hpp:11–23,76–82` selects a stable bitonic comparator or packed value/index sort key. | Guaranteed low-index membership is a different guarantee from Torch nth-element membership. On this fixture it gives64/74, not202/64. |
| `ttnn.topk(..., stable=False)` | Same binding: default false. `topk_common_funcs.hpp:49–82` transposes tiles and invokes `topk_local_sort`; `:114–161` performs bitonic rebuild. | Existing network order, not ATen's CPU partition algorithm. A coincident match on a fixture or finite gate suite is useful evidence but not an all-row Torch-tie guarantee. |
| Top-k layout/core/output-index variants | `topk.cpp:433–438,659–729` forwards sorted/stable/index/layout choices to the device operation. `topk_nanobind.cpp:59` defines supplied indices as output labels. | Labels, layout and core selection do not expose an ATen nth-element comparator or partition mode. Their actual finite-suite behavior can still be measured by the parent's alternative probe. |
| `generalized_moe_gate` | `experimental/deepseek/moe/generalized_moe_gate/generalized_moe_gate_nanobind.cpp:64–112` exposes bias, input index labels, sigmoid, topk, selected-score normalization and grouped mode; no tie-policy argument. `device/unified_kernels/generalized_moe_gate.hpp:220–241` selects the single256-expert gate. | No switch requests Torch CPU selection. Bias changes score selection; grouped mode changes the routing algorithm. Neither is an exact CPU tie-compatibility contract. |
| Generalized gate's native selection | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h:30–56` loads bias/score keys and tracks index/score payload through swaps; `:96–160,220–276` implements SFPSWAP-based bitonic phases. `tt_metal/hw/inc/api/compute/experimental/generalized_moe_gate.h:181–235` merges the256-expert runs. | Value-key sorting network differs structurally from CPU nth-element. The observed64/74 does not establish that this gate universally implements lowest-index stable sorting; only its result for this saved row is claimed. |
| Large-index top-k | `ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:296–300` explicitly describes its tie order as deterministic but unspecified and excludes it from stable requests. | No CPU nth-element guarantee. |
| GPT router fusion | `experimental/topk_router_gpt/device/topk_router_gpt_device_operation.cpp:23–28` requires128 experts; `device/kernels/compute.cpp:183–193` uses top-k sort/merge/rebuild. | Not a drop-in256-expert adapter, and no CPU tie-mode contract. |

A canonical guarantee would require the same dynamic tied-selection behavior as the frozen CPU implementation, or another independently proven equivalent algorithm. None of the inspected APIs exposes that behavior. This is the precise existing-configuration gap. It is not a blanket impossibility statement, does not pre-judge the parent's three native alternatives, and does not authorize changing the oracle or functional PCC threshold.
