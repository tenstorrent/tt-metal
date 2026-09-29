# DeepSeek-V4.1-Flash reference: CPU-runnable

This directory holds a vendored copy of the DeepSeek-V4.1-Flash reference inference code. The upstream tilelang/CUDA kernels are replaced by torch ports, so the model runs on CPU with `world_size=1` and without initializing `torch.distributed`.

## Provenance

- Upstream: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>, revision
  `dba1be0a40aa45a94ad051997016db3960a90277`, directory `inference/`.
- License: MIT (upstream `LICENSE`: "Copyright (c) 2023 DeepSeek"). The vendored files carry that notice as an SPDX header.

| File | Origin |
|---|---|
| `model.py` | upstream `inference/model.py` (deviations below) |
| `engram.py` | upstream `inference/engram.py` (provenance comment only) |
| `vision.py` | upstream `inference/vision.py` (provenance comment only) |
| `image_processor.py` | upstream `inference/image_processor.py` (provenance comment only) |
| `config.json` | upstream `inference/config.json`, unchanged (the released model config) |
| `kernel_cpu.py` | new (Tenstorrent, Apache-2.0): torch ports of upstream `inference/kernel.py` |
| `testing.py` | new (Tenstorrent, Apache-2.0): small config, stub tokenizer, seeded weight init, prefill helper |
| `oracle.py` | new (Tenstorrent, Apache-2.0): cached expected results (block captures, shared and final state, chunk contract) for real-dims layer subsets, synthetic or checkpoint weights |
| `__init__.py` | new, empty |

Upstream files that are not vendored: `kernel.py` (tilelang, replaced by `kernel_cpu.py`), `generate.py`, `convert.py`, `run.sh`, `requirements.txt`, and `examples/`.

## Deviations from upstream (complete list)

1. Each vendored `.py` file begins with a 2-line SPDX header naming the upstream copyright and MIT license, then a provenance comment block (3 lines, plus a blank line) that points to this README. `model.py` also carries `# isort: skip_file` and `# fmt: off` (9 header lines in total), because the repository's black and isort pre-commit hooks would otherwise reflow upstream code (assert wrapping, import order). The other vendored files are already stable under those hooks.
2. `model.py`: four import statements are changed. Everything else is byte-identical to upstream. Check with `diff <(tail -n +10 model.py) <upstream>/inference/model.py`; for the other files, use `tail -n +7`.
   - `from engram import ...` becomes `from .engram import ...`. The same change applies to `image_processor` and `vision`. Rationale: package-relative imports.
   - `from kernel import (...)` becomes `from .kernel_cpu import (...)`. Rationale: CPU kernels. The imported names and signatures are the same.

No other change was needed to run on CPU. `Transformer.__init__` already sets `world_size = 1` when `torch.distributed` is not initialized, and every `dist.*` call is guarded by `world_size > 1`.

## Running it

Follow upstream `generate.py`: construct the model and run it under `torch.set_default_dtype(torch.bfloat16)`, or use the `set_dtype(torch.bfloat16)` context manager from `model.py`. Two contracts depend on this:
- `fp8_gemm` and `fp4_gemm` return the default dtype.
- `act_quant` and `fp4_act_quant` take bf16 input only, like the tilelang kernels (`in_dtype=BF16`).

`testing.py` wraps this in the following helpers:

```python
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.testing import build_small_model, prefill
model = build_small_model(seed=0)       # 6 backbone layers (all six block types) + 1 DSpark layer
logits, main_hidden = prefill(model, input_ids)  # start_pos 0 forward + DSpark window-cache seeding
```

`SmallScheduleConfig` puts one layer of each block type on the schedule, in this order: SWA_ONLY, KV_INDEX_SOURCE (ratio 2), CONSUMER_RATIO2, CANDIDATE_SOURCE (ratio 1), CONSUMER_RATIO1, CANDIDATE_INDEX_SOURCE. The block types are the canonical ones from `DeepSeekV41FlashConfig.block_type`. Engram runs on layers 0 (SWA) and 1 (KV source).

Weights are stored in checkpoint dtypes:
- dense weights: FP8 e4m3 with E8M0 32x32 block scales
- routed experts: FP4 `float4_e2m1fn_x2` with E8M0 per-32 scales
- Engram tables: FP8 with E8M0 per-32 row scales

The Engram hash needs a compressed token map. `StubTokenizer` supplies one without a real tokenizer.

The upstream code has module-level state: `world_size`, `default_dtype`, and the `shared_attn` runtime. So, as upstream says, run one model at a time per process. Layers run in order, and each prefill at start_pos 0 rewrites every shared slot before it is read.

## Kernel port fidelity (`kernel_cpu.py`)

| Function | Fidelity to tilelang | Notes |
|---|---|---|
| `act_quant` | bit-exact | per-block amax floored at 1e-4. With `scale_fmt`, the scale is `2^ceil(log2(amax * fp32(1/448)))` via the same exponent-bit trick; otherwise it is `amax * fp32(1/448)`. Values are clamped to ±448, then cast with RNE to e4m3. Inplace mode does QDQ back to bf16 |
| `fp4_act_quant` | bit-exact | E8M0 path: per-32 groups, amax floored at 6·2^-126, power-of-2 scale. E4M3 path: groups of 16, amax floored at 6·2^-9, scale is `e4m3(amax/6)`. E2M1 cast is RNE with ties to the even code. Packing puts the even element in the low nibble, matching upstream `convert.py` |
| `fp8_gemm`, `fp4_gemm` | exact block products; summation order within a block differs | Each K block (32) is summed in fp32, scaled by `a_s * b_s`, then accumulated in K order, as the kernel does. Only the order within a block differs from the tensor-core MMA, so rare 1 bf16-ulp output differences are possible |
| `sparse_attn` | same algorithm; GEMM order and `exp` differ | Online softmax over blocks of 64 indices. The running max starts at -1e30, so rows where every index is -1 give zeros. Index -1 gets logit -inf and a zeroed row. The fp32 sum of probabilities feeds the denominator; probabilities are cast to bf16 before PV. The sink is added unscaled. Output is bf16. Expected difference: about 1 bf16 ulp of the output |
| `hc_split_sinkhorn` | same fp32 operation order | Only `exp`/`sigmoid` may differ in the last fp32 ulp |

Tests: `models/demos/deepseek_v3_d_p/tests/v41/torch/test_v41_kernels_cpu.py` checks the kernels against hand cases and independent formulations. `test_v41_reference_cpu.py` runs the small-model prefill smoke test.

## Upstream behaviors worth knowing (not changed)

- `Indexer.forward` calls `topk(..., sorted=False)`. How it breaks ties among equal scores is implementation-defined, so CPU and CUDA can choose different positions when scores tie exactly (for example, all-zero relu scores).
- `image_processor.prepare_vl_inputs` imports `encoding` lazily. That module is not in upstream `inference/`, so tests build `ImageInput` directly.
- The `__main__` self-test in `model.py` targets CUDA and is not used here.
