#!/bin/bash
# Setup script for OpenVLA model weights

set -e

echo "=== OpenVLA Model Setup ==="
echo ""

# Check if HF token is already configured
if python_env/bin/python -c "import huggingface_hub; print(huggingface_hub.get_token())" | grep -q "None"; then
    echo "Step 1: Setting up Hugging Face authentication"
    echo "You will need to:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'Read' permissions"
    echo "3. Request access to these gated models:"
    echo "   - https://huggingface.co/meta-llama/Llama-2-7b-hf (REQUIRES APPROVAL - you mentioned this is pending)"
    echo "   - https://huggingface.co/openvla/openvla-7b (publicly accessible ✓)"
    echo ""
    read -p "Do you have your Hugging Face token ready? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python_env/bin/python -c "from huggingface_hub import login; login()"
    else
        echo "Please get your token and run this script again."
        exit 1
    fi
else
    echo "✓ Hugging Face authentication already configured"
fi

echo ""
echo "Step 2: Setting environment variables"
export HF_MODEL=meta-llama/Llama-2-7b-hf
echo "export HF_MODEL=meta-llama/Llama-2-7b-hf"

echo ""
echo "Step 3: Testing model access"
echo "Testing access to meta-llama/Llama-2-7b-hf..."
if python_env/bin/python -c "
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', token=True)
    print('✓ Successfully accessed Llama-2-7b-hf')
except Exception as e:
    print('✗ Failed to access Llama-2-7b-hf:', str(e))
    exit(1)
"; then
    echo "✓ Model access test passed"
else
    echo "✗ Model access test failed"
    exit 1
fi

echo ""
echo "Step 4: Downloading OpenVLA weights"
echo "This will download ~14GB of model weights..."
read -p "Continue with download? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create download directory
    DOWNLOAD_DIR="$HOME/openvla_weights"
    mkdir -p "$DOWNLOAD_DIR"

    echo "Downloading to: $DOWNLOAD_DIR"

    # Download OpenVLA weights
    python_env/bin/python -c "
import os
from huggingface_hub import hf_hub_download

files_to_download = [
    'model.safetensors.index.json',
    'model-00001-of-00003.safetensors',
    'model-00002-of-00003.safetensors',
    'model-00003-of-00003.safetensors'
]

for file in files_to_download:
    print(f'Downloading {file}...')
    hf_hub_download('openvla/openvla-7b', file, local_dir='$DOWNLOAD_DIR')
    print(f'✓ Downloaded {file}')
"

    echo ""
    echo "Step 5: Setting OPENVLA_WEIGHTS environment variable"
    export OPENVLA_WEIGHTS="$DOWNLOAD_DIR/"
    echo "export OPENVLA_WEIGHTS=\"$DOWNLOAD_DIR/\""

    echo ""
    echo "Step 6: Testing OpenVLA test"
    echo "Running a quick test to verify everything works..."
    cd /home/ubuntu/work/openvla/tt-metal
    HF_MODEL=meta-llama/Llama-2-7b-hf OPENVLA_WEIGHTS="$DOWNLOAD_DIR/" python_env/bin/python -m pytest models/tt_transformers/tt/multimodal/open_vla.py::test_openvla_model -x --tb=short -q

else
    echo "Download cancelled. You can run the download later with:"
    echo "huggingface-cli download openvla/openvla-7b model.safetensors.index.json"
    echo "huggingface-cli download openvla/openvla-7b model-00001-of-00003.safetensors"
    echo "huggingface-cli download openvla/openvla-7b model-00002-of-00003.safetensors"
    echo "huggingface-cli download openvla/openvla-7b model-00003-of-00003.safetensors"
fi

echo ""
echo "=== Setup Complete ==="
echo "To run tests with real weights, use:"
echo "export HF_MODEL=meta-llama/Llama-2-7b-hf"
echo "export OPENVLA_WEIGHTS=$DOWNLOAD_DIR/"
echo "python_env/bin/python -m pytest models/tt_transformers/tt/multimodal/open_vla.py -v"
