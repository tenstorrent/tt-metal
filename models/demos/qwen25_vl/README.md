# Qwen2.5-VL

## Introduction
This codebase includes the Qwen2.5 family of models and currently supports the model variants:
- Qwen2.5-VL-3B: [Qwen/Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Qwen2.5-VL-7B: [Qwen/Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Qwen2.5-VL-32B: [Qwen/Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
- Qwen2.5-VL-72B: [Qwen/Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Install additional python dependencies:

```
pip install -r models/demos/qwen25_vl/requirements.txt
```

## How to Run
For a single user example:
```
MESH_DEVICE=<device_name> HF_MODEL=<model_name> pytest models/demos/qwen25_vl/demo/demo.py -k 'batch-1'
```

**Notes:**
- `<model_name>` is the HuggingFace model repo string, e.g. `Qwen/Qwen2.5-VL-3B-Instruct`
- `<device_name>` is the TT device string, e.g. `N150`, `N300`, `T3K`
- `-k` is the pytest filter; to run a specific test, use `-k <test_name>`; additional test names are listed in `models/demos/qwen25_vl/demo/demo.py`
- different model variants are supported on different devices:

| Model Variant      | `<model_name>` (HF_MODEL)                   | `<device_name>` (MESH_DEVICE)      |
|--------------------|---------------------------------------------|------------------------------------|
| Qwen2.5-VL-3B      | Qwen/Qwen2.5-VL-3B-Instruct                 | `N150`, `N300`, `T3K`, `TG`        |
| Qwen2.5-VL-7B      | Qwen/Qwen2.5-VL-7B-Instruct                 | `N300`, `T3K`, `TG`                |
| Qwen2.5-VL-32B     | Qwen/Qwen2.5-VL-32B-Instruct                | `T3K`, `TG`                        |
| Qwen2.5-VL-72B     | Qwen/Qwen2.5-VL-72B-Instruct                | `T3K`, `TG`                        |

### Galaxy (32 chips)
On a Wormhole Galaxy (`MESH_DEVICE=TG`) every variant runs data-parallel: the 32 chips are opened as a
4x8 mesh and split into four 1x8 lanes, each lane serving an independent batch of users with the same
tensor-parallel layout as a T3K (`GALAXY_DEFAULT_DATA_PARALLEL` in `tt/model_config.py`). The demo's
`batch-32` test therefore runs 128 users (4 lanes x 32):
```
MESH_DEVICE=TG HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct pytest models/demos/qwen25_vl/demo/demo.py -k 'batch-32'
```
`TT_DATA_PARALLEL=<n>` overrides the number of lanes (`1` = one 1x8 submesh, tensor-parallel only).
Models whose head counts do not divide across 8 chips (7B: 28 query / 4 KV heads; 3B: 16 / 2) are
padded to 32 / 8 heads per lane (`PAD_HEADS_FOR_TP_MODELS` in `models/tt_transformers/tt/model_config.py`).

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.
