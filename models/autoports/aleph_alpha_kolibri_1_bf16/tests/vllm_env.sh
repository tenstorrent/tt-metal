# Source from the tt-metal root, after python_env/bin/activate.
source /home/vkovacevic/kolibri/lane-env.sh
export VLLM_TT_PLUGIN_ROOT=/home/vkovacevic/kolibri/vllm-tt-plugin
export TT_MODEL_CLASS_OVERRIDES='TTKolibri1ForCausalLM=models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator_vllm:KolibriForCausalLM'
export KOLIBRI_CHECKPOINT_DIR=/home/vkovacevic/kolibri/checkpoints/7a8f290e7858825c3cf5e4c447ba68345de9f1d3
export TT_METAL_OPERATION_TIMEOUT_SECONDS=60
export OMP_NUM_THREADS=8
export KOLIBRI_VLLM_EVENTS="$PWD/$MODEL_DIR/readiness_vllm/events.jsonl"
