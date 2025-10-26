# Qwen2.5-VL

## Introduction
This codebase includes the Qwen2.5 family of models and currently supports the model variants:
- Qwen2.5-VL-3B: [Qwen/Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
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

| Model Variant      | `<model_name>` (HF_MODEL)                   | `<device_name>` (MESH_DEVICE) |
|--------------------|---------------------------------------------|-------------------------------|
| Qwen2.5-VL-3B      | Qwen/Qwen2.5-VL-3B-Instruct                 | either `N150` or `N300`       |
| Qwen2.5-VL-32B     | Qwen/Qwen2.5-VL-32B-Instruct                | `T3K`                         |
| Qwen2.5-VL-72B     | Qwen/Qwen2.5-VL-72B-Instruct                | `T3K`                         |

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.
