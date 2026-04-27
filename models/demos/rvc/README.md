# Retrieval-based Voice Conversion (RVC) Demo

This directory contains a demo implementation of Retrieval-based Voice Conversion (RVC) using Tenstorrent's TTNN APIs.

## Overview

Retrieval-based Voice Conversion (RVC) is a technique for converting voice characteristics from one speaker to another. This implementation demonstrates how to use TTNN operators to perform voice conversion on Tenstorrent hardware.

## Files

- `main.py`: Main RVC demo script with model implementation
- `README.md`: This file

## Usage

```bash
cd models/demos/rvc
python main.py --help
```

## Implementation Details

The RVC implementation includes:

1. **Encoder**: Extracts features from input voice
2. **Decoder**: Converts the encoded features to target voice
3. **TTNN Integration**: Uses TTNN operators for neural network operations

## TTNN Operators Used

- `matmul`: Matrix multiplication operations
- `add`: Element-wise addition
- `relu`: ReLU activation function
- Various tensor manipulation operations

## Hardware Requirements

This implementation requires Tenstorrent hardware with TTNN support.