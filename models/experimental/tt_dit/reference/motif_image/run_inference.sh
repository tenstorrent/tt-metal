# Motif Image 6B Preview - Inference helper
# Usage:
#   uv run bash run_inference.sh
# Notes:
#   - Expects configs/motif_image.json and checkpoints/motif_image_preview.bin
#   - Adjust TORCH_INDEX_URL when installing to match your CUDA/ROCm/CPU environment
python inference.py \
 --model-config configs/motif_image.json \
 --model-ckpt checkpoints/motif_image_preview.bin \
 --seed 0 \
 --steps 50 \
 --resolution 1024 \
 --prompt-file prompts/sample_prompts.txt \
 --guidance-scales 5.0 \
 --output-dir outputs \
 --batch-size 1 \
 --zero-masking \
 --use-linear-quadratic-schedule \
 --linear-quadratic-emulating-steps 100 \
 --zero-embedding-for-cfg \
 --negative-strategy-switch-t 0.85