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
