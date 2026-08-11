# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


# Default LoRA weights (UNet-only adapter).
TEST_LORA_REPO_ID = "artificialguybr/ColoringBookRedmond-V2"
TEST_LORA_FILENAME = "ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"

# Text-encoder-impacting LoRA: trains both CLIP text encoders *and* the UNet,
# is not DoRA, and has alpha != rank (so it also exercises scale application).
# Used to cover the text-encoder fuse/rollback path, which the default UNet-only
# adapter above does not touch.
TE_TEST_LORA_REPO_ID = "RalFinger/alien-style-lora-sdxl"
TE_TEST_LORA_FILENAME = "alienzkin-sdxl.safetensors"
# Directory under the CIv2 large-file cache's huggingface prefix where this
# adapter is staged (CIv2 runners have no HF egress, so hf_hub_download cannot
# fetch it there). Staged 2026-07-21 alongside the other SDXL LoRA fixtures.
TE_TEST_LORA_CI_CACHE_DIR = "alien-style-lora"
