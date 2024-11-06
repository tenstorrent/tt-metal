#!/bin/bash

# Define model directories from environment variables with fallbacks
LLAMA_DIRS=(
    # "${LLAMA_32_1B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-1B-Instruct}"
#    "${LLAMA_32_3B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-3B-Instruct}"
#    "${LLAMA_31_8B_DIR:-/proj_sw/user_dev/llama31-8b-data/Meta-Llama-3.1-8B-Instruct}"
    "${LLAMA_32_11B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-11B-Vision-Instruct}"
#    "${LLAMA_31_70B_DIR:-/proj_sw/llama3_1-weights/Meta-Llama-3.1-70B-Instruct/repacked}"
)

# Create reference_outputs directory if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/reference_outputs"
mkdir -p "$OUTPUT_DIR"

# Function to get model size from directory path
get_model_size() {
    if [[ $1 == *"-1B"* ]]; then
        echo "1b"
    elif [[ $1 == *"-3B"* ]]; then
        echo "3b"
    elif [[ $1 == *"-8B"* ]]; then
        echo "8b"
    elif [[ $1 == *"-11B"* ]]; then
        echo "11b"
    elif [[ $1 == *"-70B"* ]]; then
        echo "70b"
    else
        echo "unknown"
    fi
}

# Loop through each LLAMA directory
for DIR in "${LLAMA_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "Warning: Directory $DIR does not exist, skipping..."
        continue
    fi

    # Get model size for output filename
    MODEL_SIZE=$(get_model_size "$DIR")
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_SIZE}.refpt"

    echo "Generating reference outputs for ${MODEL_SIZE} model..."
    echo "Using weights from: ${DIR}"
    echo "Output will be saved to: ${OUTPUT_FILE}"

    # Set LLAMA_DIR environment variable and run the Python script
    LLAMA_DIR="$DIR" python3 "${SCRIPT_DIR}/generate_reference_outputs.py" \
        --total_length 1024 \
        --output_file "$OUTPUT_FILE"
done

echo "All reference outputs have been generated!"
