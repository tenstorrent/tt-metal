# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


# Both default adapters below are third-party Hugging Face repos, so they are
# pinned to an explicit commit. Without a revision the uploader can replace the
# weights in place: the download still succeeds, but the PCC references these
# tests assert against no longer describe the file, so the suite goes red on a
# change nobody here made. The pin makes that a visible 404 instead.

# Default LoRA weights (UNet-only adapter).
TEST_LORA_REPO_ID = "artificialguybr/ColoringBookRedmond-V2"
TEST_LORA_FILENAME = "ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"
TEST_LORA_REVISION = "0e67e0de2b603db085e525e7f6194b24dc60033d"

# Text-encoder-impacting LoRA: trains both CLIP text encoders *and* the UNet,
# is not DoRA, and has alpha != rank (so it also exercises scale application).
# Used to cover the text-encoder fuse/rollback path, which the default UNet-only
# adapter above does not touch.
TE_TEST_LORA_REPO_ID = "RalFinger/alien-style-lora-sdxl"
TE_TEST_LORA_FILENAME = "alienzkin-sdxl.safetensors"
TE_TEST_LORA_REVISION = "68113675e36623e483d6342548b4c031e552fb7f"
# Directory under the CIv2 large-file cache's huggingface prefix where this
# adapter is staged (CIv2 runners have no HF egress, so hf_hub_download cannot
# fetch it there). Staged 2026-07-21 alongside the other SDXL LoRA fixtures.
TE_TEST_LORA_CI_CACHE_DIR = "alien-style-lora"
