"""Constants for SAM2 TTNN implementation.
Pinned Hugging Face model revision and version requirements."""

# Pinned HF model revision for facebook/sam2-hiera-tiny
# Verified: 2025-08-15
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"
SAM2_MODEL_REVISION = "7c218beaf0bb87874785f32b582f640134fc1c09"

# Minimum transformers version required
# HF Sam2Model was introduced in transformers 4.56.0.dev0
TRANSFORMERS_MIN_VERSION = "4.56.0.dev0"

# Image mode parameters
IMAGE_SIZE = 1024
INPUT_CHANNELS = 3
NUM_FEATURE_LEVELS = 3
BACKBONE_FEATURE_SIZES = [[256, 256], [128, 128], [64, 64]]

# PCC validation thresholds
PCC_THRESHOLD_STAGE1 = 0.99
PCC_THRESHOLD_STAGE2 = 0.99
PCC_THRESHOLD_STAGE3 = 0.99
