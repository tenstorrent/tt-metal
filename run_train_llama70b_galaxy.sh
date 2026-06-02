#!/usr/bin/env bash
# =============================================================================
# Llama-3.1-70B LoRA fine-tune on a single Galaxy (32 Blackhole chips).
#
# Sharding strategy: 2D mesh — DP=4 × TP=8 = 32 chips.
#   mesh_shape  [8, 4]   (MGD physical topology is [8, 4]; mesh dims must match)
#   tp_axis     0        (size 8 → TP=8 along the 8-dim)
#   dp_axis     1        (size 4 → DP=4 along the 4-dim)
#   model       llama3_1_70B_hf.yaml  (HF-matched: 64Q heads, 8KV heads,
#                                       vocab 128256, intermediate 28672)
#
# WHY NOT TP=32?
#   HF Llama-3.1-70B uses Grouped-Query Attention with 8 KV heads. Tensor
#   parallelism requires tp_size to divide both `num_attention_heads` (64)
#   and `num_key_value_heads` (8) — so max TP = 8 when loading HF weights.
#   The existing `llama70b_tp32.yaml` works around this by setting
#   `num_groups: 32` (replicating KV heads 4×), but the existing safetensors
#   loader does not perform that replication automatically. With HF weights,
#   TP=8 is the clean ceiling; we use DP=4 to fill the remaining Galaxy
#   bandwidth.
#
# Notes / caveats:
#   1. Uses a custom MGD (single_galaxy_partial_mesh_graph_descriptor.textproto)
#      because this Galaxy lives in a quad-pod: some eth links are routed
#      to other galaxies in the pod, so adjacent chips have 2 within-galaxy
#      eth links instead of 4. The stock single_galaxy MGD would fail
#      fabric validation. If you move to a standalone single Galaxy with
#      full 4-link wiring, swap to single_galaxy_mesh_graph_descriptor.textproto.
#   2. For Shakespeare-from-scratch training with TP=32, see
#      tt-train/configs/training_configs/training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml
#      — that's a different setup and uses the 32-group yaml.
#   3. For multi-galaxy + pipeline parallelism see
#      tt-train/sources/examples/python/multihost/pipeline_parallel_training/
#      and tt-train/docs/DISTRIBUTED_TRAINING.md.
#   4. Global batch in DP=4 mode is `--batch * 4` per optimizer step.
# =============================================================================

echo "Python path: $(which python3)"

export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export TT_MESH_GRAPH_DESC_PATH="/home/ttuser/pglusac/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
export TT_METAL_LOGGER_TYPES=Op
export TT_METAL_LOGGER_LEVEL=Debug
export LOGGER_LEVEL=DEBUG


python3 tt-train/sources/examples/lora_llama/train_lora_llama_sft.py \
  --model_config /home/ttuser/pglusac/tt-metal/tt-train/configs/model_configs/llama3_1_70B_hf_1L.yaml \
  --pretrained meta-llama/Llama-3.1-70B \
  --mesh_shape 1,4 \
  --tp_axis 1 \
  --dp_axis 0 \
  --batch 1 \
  --steps 2 \
  --profile_warmup_steps 1

# env -u TT_METAL_DPRINT_CORES \
# TT_METAL_WATCHER_NOINLINE=1 \
# TT_METAL_WATCHER_DEBUG_DELAY=10 \
# TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
# TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
# TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
# TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
# TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \
# python3 -m tracy -r -v -p tt-train/sources/examples/lora_llama/train_lora_llama_sft.py \
#   --model_config /home/ttuser/pglusac/tt-metal/tt-train/configs/model_configs/llama3_1_70B_hf_1L.yaml \
#   --pretrained meta-llama/Llama-3.1-70B \
#   --mesh_shape 1,4 \
#   --tp_axis 1 \
#   --dp_axis 0 \
#   --batch 1 \
#   --steps 2 \
#   --profile_warmup_steps 1
