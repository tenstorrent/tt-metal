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
python3 -m tracy -r -v -p tt-train/sources/examples/qwen3/train.py \
  --model_path Qwen/Qwen3-1.7B \
  --max_seq_len 1024 \
  --mesh_shape 1 1 \
  --batch_size 1 \
  --steps 2 \
  --warmup_steps 0 \
  --eval_every 0 \
  --gen_every 0 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_targets q_proj v_proj \
  --profile_warmup_steps 1
