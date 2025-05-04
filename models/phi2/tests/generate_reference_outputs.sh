#!/bin/bash

# Parse command line arguments
TOTAL_LENGTH=1024  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --total-length)
            TOTAL_LENGTH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Generate reference outputs for Llama models"
            echo
            echo "Options:"
            echo "  --total-length N    Set the total sequence length (default: 1024)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Define model directories from environment variables with fallbacks
LLAMA_DIRS=(
    "${LLAMA_32_1B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-1B-Instruct}"
    "${LLAMA_32_3B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-3B-Instruct}"
    "${LLAMA_31_8B_DIR:-/proj_sw/user_dev/llama31-8b-data/Meta-Llama-3.1-8B-Instruct}"
    "${LLAMA_32_11B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-11B-Vision-Instruct}"
    "${LLAMA_31_70B_DIR:-/proj_sw/llama3_1-weights/Meta-Llama-3.1-70B-Instruct/repacked}"
    "${LLAMA_32_90B_DIR:-/proj_sw/user_dev/llama32-data/Llama3.2-90B-Vision-Instruct}"
    "${QWEN_25_7B_DIR:-/proj_sw/user_dev/Qwen/Qwen2.5-7B-Instruct}"
    "${QWEN_25_72B_DIR:-/proj_sw/user_dev/Qwen/Qwen2.5-72B-Instruct}"
)

# Create reference_outputs directory if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/reference_outputs"
mkdir -p "$OUTPUT_DIR"

# Function to get model name from directory path
get_model_name() {
    local dir_name=$(basename "$1")
    # If the path ends in /repacked, use the parent directory name instead
    if [ "$dir_name" = "repacked" ]; then
        dir_name=$(basename "$(dirname "$1")")
    fi
    echo "$dir_name"
}

# Loop through each LLAMA directory
for DIR in "${LLAMA_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "Warning: Directory $DIR does not exist, skipping..."
        continue
    fi

    # Get model size for output filename
    MODEL_NAME=$(get_model_name "$DIR")
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_full.refpt"

    echo "Generating reference outputs for ${MODEL_SIZE} model..."
    echo "Using weights from: ${DIR}"
    echo "Output will be saved to: ${OUTPUT_FILE}"

    # Set LLAMA_DIR environment variable and run the Python script
    LLAMA_DIR="$DIR" python3 "${SCRIPT_DIR}/generate_reference_outputs.py" \
        --total_length "$TOTAL_LENGTH" \
        --output_file "$OUTPUT_FILE"
done

echo "All reference outputs have been generated!"
