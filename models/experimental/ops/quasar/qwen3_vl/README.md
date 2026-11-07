# Qwen3-VL (Quasar)

> Quasar-specific fork of `models/demos/qwen3_vl`, kept separate so it can be adapted to
> the emulator's compute-grid and SRAM constraints. All module paths are rooted at
> `models.experimental.ops.quasar.qwen3_vl`. Do not run pytest over both this tree and
> `models/demos/qwen3_vl` in a single session: their `demo/conftest.py` files register the
> same command-line options and pytest will error on the duplicates.

## Introduction
This codebase includes the Qwen3 family of models and currently supports the model variants:
- Qwen3-VL-32B: [Qwen/Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Install additional python dependencies:

```
pip install -r models/experimental/ops/quasar/qwen3_vl/requirements.txt
```

## How to Run
For a single user example:
```
MESH_DEVICE=<device_name> HF_MODEL=<model_name> pytest models/experimental/ops/quasar/qwen3_vl/demo/demo.py -k 'batch-1'
```

**Notes:**
- `<model_name>` is the HuggingFace model repo string, e.g. `Qwen/Qwen3-VL-3B-Instruct`
- `<device_name>` is the TT device string, e.g. `N150`, `N300`, `T3K`
- `-k` is the pytest filter; to run a specific test, use `-k <test_name>`; additional test names are listed in `models/experimental/ops/quasar/qwen3_vl/demo/demo.py`
- different model variants are supported on different devices:

| Model Variant      | `<model_name>` (HF_MODEL)                   | `<device_name>` (MESH_DEVICE) |
|--------------------|---------------------------------------------|-------------------------------|
| Qwen3-VL-32B     | Qwen/Qwen3-VL-32B-Instruct                | `T3K`                         |

### Full model on Quasar (craq-sim qsr3a2)
On Quasar the demo defaults to 2 vision blocks and 2 text layers, because the 32-core build has only 2 x 1 GiB of DRAM. The full model (24 vision blocks, 36 text layers, ~8.9 GB of bf16 weights) fits the three-die qsr3a2 package: 8x12 workers and 12 x 1 GiB of DRAM. Build craq-sim's `release_qsr3a2` target and put `tt_metal/soc_descriptors/quasar_qsr3a2_arch.yaml` next to its `libttsim.so` as `soc_descriptor.yaml`. Then:
```
export TT_METAL_SIMULATOR=<dir>/libttsim.so MESH_DEVICE=N150 HF_MODEL=Qwen/Qwen3-VL-4B-Instruct
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
export TTSIM_QSR3A2_MCAST_CROSS_DIE=1   # matmul multicasts span dies (tests/e2e/QUASAR_GAPS.md Q5)
pytest models/experimental/ops/quasar/qwen3_vl/demo/demo.py -sv -k batch-1 --vision_depth 24 --num_layers 36
```
- `TTSIM_QSR3A2_MCAST_CROSS_DIE=1` lets craq-sim deliver multicasts that cross a die boundary, which it otherwise aborts because their acknowledgements are not reliable on silicon. It is a simulator-only escape hatch; the matmul gap stays open.
- Matmuls with a sharded input run from DRAM on Quasar (QUASAR_GAPS Q4, workarounds in `tests/e2e/op_overrides.py`).
- At this depth the demo uses the model's deepstack taps 5/11/17 (#64363).
- 2026-11-05, craq-sim 450de768, prompt "Describe this image." (256x256 image), `--max_generated_tokens 27`: `This is a heartwarming, serene image of a woman and her` (14 tokens, before the run was stopped). HF on the CPU gives "This is a heartwarming, sun-drenched photograph capturing a tender moment between a woman and her dog on a beach at sunset." (27 tokens). Simulated time: vision 19.5 min, first token ~93 min, each decode step ~25 min, so the 27-token sentence takes ~13 h. The default 2 tokens (`This is a`) take 2 h 48 min.
- `--vision_depth 2 --num_layers 2` runs the same code paths in ~18 min.

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.
