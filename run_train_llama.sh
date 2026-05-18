echo "Python path: $(which python3)"

export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export TT_VISIBLE_DEVICES=0

env -u TT_METAL_DPRINT_CORES \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \
python3 -m tracy -r -v -p tt-train/sources/examples/lora_llama/train_lora_llama_sft.py \
  --model_config /home/pglusac/tt-metal/tt-train/configs/model_configs/llama3_1_8B.yaml \
  --pretrained meta-llama/Llama-3.1-8B \
  --mesh_shape 1,1 \
  --batch 1 \
  --steps 2 \
  --profile_warmup_steps 1
