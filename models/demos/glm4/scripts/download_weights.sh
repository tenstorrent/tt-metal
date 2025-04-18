#!/bin/bash

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# This script downloads the GLM-4 9B Chat model files from Hugging Face.

# Ensure git-lfs is installed
# On Ubuntu/Debian: sudo apt-get update && sudo apt-get install git-lfs
# On Fedora/CentOS: sudo yum install git-lfs
# On macOS (using Homebrew): brew install git-lfs

# Set the Hugging Face model repository ID
MODEL_ID="THUDM/glm-4-9b-chat"
# Set the target directory where the model should be saved
# Default: current directory (.) - change if needed
TARGET_DIR="."

# Clone the repository without LFS files initially for faster clone
# The target directory will be named after the model (e.g., glm-4-9b-chat)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/$MODEL_ID $TARGET_DIR/$MODEL_ID

# Navigate into the repository directory
cd $TARGET_DIR/$MODEL_ID

# Pull the LFS files (the actual model weights)
echo "Downloading LFS files (model weights)... This may take a while."
git lfs pull

cd ..

echo "GLM-4 model download complete in directory: $TARGET_DIR/$MODEL_ID"

# Optional: Make the script executable
# chmod +x download_weights.sh
