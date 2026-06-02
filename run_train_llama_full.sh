echo "Python path: $(which python3)"

export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export TT_MESH_GRAPH_DESC_PATH="/home/ttuser/pglusac/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"

export TT_VISIBLE_DEVICES=1

env -u TT_METAL_DPRINT_CORES \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \
python3 -m tracy -r -v -p tt-train/sources/examples/lora_llama/train_lora_llama_sft.py \
  --model_config /home/ttuser/pglusac/tt-metal/tt-train/configs/model_configs/llama3_2_1B_hf.yaml \
  --pretrained meta-llama/Llama-3.2-1B \
  --mesh_shape 1,1 \
  --batch 1 \
  --steps 11 \
  --profile_warmup_steps 10 \
  --no_lora
