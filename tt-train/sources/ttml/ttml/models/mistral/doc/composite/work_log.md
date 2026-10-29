# Composite bring-up work log

Model `mistralai/Mistral-Small-24B-Base-2501` @ `b0a2e4e`, recipe LoRA r16/a32 on all block linears, seq 1024, batch 1, AdamW; one Blackhole chip (chip 0 of the QB2 box, host `e0118d75da98`, 110 cores, 31.83 GiB). Branch `agobeljicTT/mistral-bringup`. All commands from `tt-train/` with `TT_METAL_HOME=TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal`, `../python_env/bin/python`. Evidence under `doc/composite/evidence/` (`E` below); raw profiler dumps and the 66 MB LoRA-init `.npz` stay on disk there, gitignored.

## 2026-10-29

- Commits (local, not pushed):
  - `0dee658b57e` linear/binary backward skip the gradient of an operand that does not require it (port of `umales/qwen38`; rebuilt `_ttml` with `ninja -C build_Release`).
  - `5dd586625de` llama: `head_dim` decoupled, loader root-name aware and reads only the shards in `model.safetensors.index.json`, `Llama._embed` hook; `tests/python/test_llama.py` + loader tests: 133 passed.
  - `604ee6f05a1` `ttml.models.mistral` (config, model, frozen embedding, FLOPs, loader), `sources/examples/mistral_train/train.py`, `tests/python/test_mistral.py`.
  - (this commit) `train.py --mem-probe`, `torch_reference.py`, `analyze_tracy.py`, docs and evidence.
- PCC harness: `pytest tests/python/test_mistral.py -k "not huggingface"` (`E/pcc_harness_run1.log`, rmsnorm rerun after a test-code fix in `E/pcc_harness_run2_rmsnorm.log`), checkpoint test `-k huggingface` (`E/pcc_harness_run3_checkpoint.log`). Tables: `E/pcc_tables.txt`, summarised in `README.md`.
- 8-layer LoRA run, real weights: `train.py --layers 8 --steps 10 --warmup-steps 2 --loss-log E/loss_8layers.json --data-log E/data_8layers.json --dump-lora-init E/lora_init_8layers.npz` (`E/train_8layers.log`): 434.0 ms/step, MFU 33.32 % (8 of 40 layers). `--track-memory` printed zeros (the tracker needs the mesh opened by `ttml.open_device_mesh`; replaced by `--mem-probe`, which reads the allocator).
- Torch fp64 reference on the same windows and LoRA init: `torch_reference.py --layers 8 --data-log ... --lora-init ... --out E/loss_torch_8layers.json --threads 12` (`E/torch_reference_8layers.log`, ~75 s per step on CPU). Comparison `E/loss_comparison.txt`.
- Device chain `E/device_chain.sh` (`E/device_chain.log`), serial on chip 0:
  - memory probe at 2/4/8 layers (`E/mem_probe_*layers.log`): retained activations 778 MiB/layer, weights+state 1.067 GiB/layer + 3.0 GiB; recorded in `doc/plan.md` §4 calibration.
  - Tracy at 1/2/4/8 layers: `python -m tracy -r -p --op-support-count 40000 -o E/tracy_{L}layers train.py --layers L --steps 4 --warmup-steps 2 --random-init --random-data` (`E/tracy_*layers.log`, compact per-op CSVs `E/tracy_*layers_ops.csv`, full reports on disk). AICLK 800 MHz (0x320) and the ASIC temperature were sampled after each run with `tt-smi -s`.
  - watcher: `TT_METAL_WATCHER=1 train.py --layers 1 --steps 2 --random-init --random-data` (`E/watcher_1layer.log`; watcher log 113211 lines, 0 errors, excerpt `E/watcher_1layer_watcher_excerpt.txt`).
  - determinism: the 8-layer real-weight run repeated with the same seed (`E/train_8layers_run2.log`, `E/loss_8layers_run2.json`): all 10 losses bitwise identical; 453.4 ms/step on a warm chip straight after the profiler runs (first run 434.0 on a cool chip).
- Non-profiled wall-clock at 1/2/4 layers for the host-gap fit: `train.py --layers L --steps 6 --warmup-steps 2 --random-init --random-data` (`E/wall_*layers.log`).
- Analysis: `analyze_tracy.py --warmup 2 --steps 2 --full-depth 40 1:... 2:... 4:... 8:...` -> `E/tracy_analysis.txt` / `.json`; one-step op sequence at 1 layer `E/op_sequence_1layer.txt`. Results in `doc/perf_analysis.md`.
- Clock probe: `train.py --layers 4 --steps 40` with `tt-smi -s` every 10 s (`E/aiclk_probe_4layers.log`, `E/aiclk_probe_4layers_tt-smi.txt`): 1350 MHz under load, one 1162 MHz sample at 72 C / 168 W, 800 MHz idle. Perf runs on this box want a cool chip (15 min idle between heavy runs, per the Qwen3.8 experience) and a clean `/dev/shm`.
