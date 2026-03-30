#!/bin/bash
# Use this on the compute server when home is full. Source it *before* pip install.
# Makes pip and Hugging Face use project disk; prevents pip from using ~/.local.

export HF_HOME=/proj_sw/user_dev/dchrysostomou/hf_cache
export PIP_CACHE_DIR=/proj_sw/user_dev/dchrysostomou/.pip_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
# Prevent Python from using or writing to ~/.local/lib (avoids "No space" when home is full)
export PYTHONNOUSERSITE=1

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
echo "HF_HOME=$HF_HOME PIP_CACHE_DIR=$PIP_CACHE_DIR PYTHONNOUSERSITE=1"
