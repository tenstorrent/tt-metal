# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Constants and model loading utilities for RF-DETR Medium (detection only).
RF-DETR Medium: 576×576, DINOv2-ViT-S backbone, 33,687,458 params, COCO AP50:95 = 54.7

Component params:
  DINOv2: 22,097,664
  Projector: 1,444,864
  Decoder (4 layers): 6,084,992
  Total: 33,687,458
"""


import torch

RFDETR_MEDIUM_L1_SMALL_SIZE = 32768

# --- Model architecture constants (from RFDETRMediumConfig) ---
RESOLUTION = 576
PATCH_SIZE = 16
NUM_PATCHES_PER_SIDE = RESOLUTION // PATCH_SIZE  # 36
NUM_PATCHES = NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE  # 1296
NUM_WINDOWS = 2  # 2×2 = 4 windows
NUM_WINDOWS_SQUARED = NUM_WINDOWS * NUM_WINDOWS  # 4
PATCHES_PER_WINDOW_SIDE = NUM_PATCHES_PER_SIDE // NUM_WINDOWS  # 18
PATCHES_PER_WINDOW = PATCHES_PER_WINDOW_SIDE * PATCHES_PER_WINDOW_SIDE  # 324

# Token counts (no register tokens for dinov2_windowed_small)
TOKENS_PER_WINDOW = PATCHES_PER_WINDOW + 1  # 325 (1 CLS + 324 patches)
FULL_ATTN_SEQ_LEN = TOKENS_PER_WINDOW * NUM_WINDOWS_SQUARED  # 1300

# DINOv2-ViT-S backbone (dinov2_windowed_small: no registers)
VIT_HIDDEN_SIZE = 384
VIT_NUM_HEADS = 6
VIT_NUM_LAYERS = 12
VIT_MLP_RATIO = 4
VIT_INTERMEDIATE_SIZE = VIT_HIDDEN_SIZE * VIT_MLP_RATIO  # 1536
VIT_HEAD_SIZE = VIT_HIDDEN_SIZE // VIT_NUM_HEADS  # 64
NUM_REGISTER_TOKENS = 0  # "dinov2_windowed_small" does NOT use registers
OUT_FEATURE_INDEXES = [3, 6, 9, 12]  # 1-indexed stages; features from after layers 2,5,8,11

# Windowed vs full attention layers (0-indexed)
# window_block_indexes = {0,1,2,4,5,7,8,10,11} (windowed)
# Full attention at layers {3, 6, 9} (non-windowed)
# Pattern: 3W → 1F → 2W → 1F → 2W → 1F → 2W
WINDOW_BLOCK_INDEXES = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FULL_ATTN_LAYER_INDEXES = [3, 6, 9]

# Projector
PROJECTOR_SCALE = ["P4"]
HIDDEN_DIM = 256  # transformer/decoder hidden dim

# Decoder
DEC_LAYERS = 4
SA_NHEADS = 8  # self-attention heads
CA_NHEADS = 16  # cross-attention (deformable) heads
DEC_N_POINTS = 2  # deformable attention sampling points
NUM_QUERIES = 300
NUM_SELECT = 300
DIM_FEEDFORWARD = 2048
GROUP_DETR = 13  # groups for training; inference uses 1
POSITIONAL_ENCODING_SIZE = 36

# Detection
NUM_CLASSES = 90  # COCO (model outputs 91 = 90 + 1 background)
BBOX_REPARAM = True
LITE_REFPOINT_REFINE = True
TWO_STAGE = True

# Pretrained weights
PRETRAIN_WEIGHTS_FILENAME = "rf-detr-medium.pth"


def load_torch_model(model_location_generator=None, device="cpu"):
    """
    Load the RF-DETR Medium PyTorch model with pretrained COCO weights.

    Returns:
        model: PyTorch LWDETR model in eval mode
    """
    try:
        from rfdetr import RFDETRMedium
    except ImportError:
        raise ImportError("rfdetr package not installed. Install with: pip install rfdetr")

    rfdetr_model = RFDETRMedium()
    return rfdetr_model.model.model.eval()


def load_torch_model_from_checkpoint(checkpoint_path, device="cpu"):
    """
    Load RF-DETR Medium from a local checkpoint file.

    Args:
        checkpoint_path: path to .pth checkpoint
        device: device to load onto

    Returns:
        model: PyTorch LWDETR model in eval mode, state_dict
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    return state_dict
