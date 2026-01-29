# LLVC Voice Conversion on Tenstorrent Hardware

This directory contains the implementation of LLVC (Low-Latency Low-Resource Voice Conversion) model optimized for Tenstorrent hardware using TTNN APIs.

## Overview

LLVC is a real-time voice conversion model designed for low latency and CPU efficiency. This implementation brings LLVC to Tenstorrent hardware for ultra-high-throughput, ultra-low-latency voice conversion.

## Features

- **Real-time voice conversion**: Optimized for streaming inference with minimal delay
- **TTNN implementation**: Full model implementation using Tenstorrent's TTNN APIs
- **Streaming support**: True streaming inference with chunked processing and state management
- **Hardware acceleration**: Leverages TT hardware for high-throughput processing

## Architecture

The LLVC model consists of:

1. **Lightweight Encoder**: Dilated causal convolution-based encoder for efficient feature extraction
2. **Cached Convolution Pre-net** (optional): Additional convolutional layers with state caching
3. **Causal Transformer Decoder**: Transformer decoder with causal attention for sequence generation
4. **Streaming State Management**: Efficient handling of convolutional and attention states across chunks

## Files

- `ttnn_llvc.py`: Main TTNN implementation of the LLVC model
- `demo.py`: Demo script for running voice conversion inference

## Usage

### Prerequisites

- Tenstorrent hardware (N150 or N300)
- TT-Metal environment set up
- LLVC model checkpoint and config files

### Running the Demo

```bash
cd models/demos/llvc

# Basic inference
python demo.py -i input_audio.wav -o output_audio.wav

# Streaming inference
python demo.py -i input_audio.wav -o output_audio.wav -s -n 2

# With custom checkpoint and config
python demo.py -p /path/to/checkpoint.pth -c /path/to/config.json -i input.wav -o output.wav
```

### Command Line Options

- `-p, --checkpoint_path`: Path to LLVC checkpoint file (default: llvc_models/models/checkpoints/llvc/G_500000.pth)
- `-c, --config_path`: Path to LLVC config file (default: experiments/llvc/config.json)
- `-i, --input_file`: Path to input audio file (default: test_wavs/example.wav)
- `-o, --output_file`: Path to output audio file (default: converted_output.wav)
- `-s, --streaming`: Use streaming inference
- `-n, --chunk_factor`: Chunk factor for streaming inference (default: 1)

## Model Configuration

The model supports various configuration options through the config file:

- `enc_dim`: Encoder dimension (default: 512)
- `num_enc_layers`: Number of encoder layers (default: 10)
- `dec_dim`: Decoder dimension (default: 256)
- `num_dec_layers`: Number of decoder layers (default: 2)
- `dec_chunk_size`: Decoder chunk size (default: 72)
- `L`: Stride factor (default: 8)
- `use_pos_enc`: Use positional encoding (default: True)
- `lookahead`: Use lookahead context (default: True)

## Performance Targets

- **Throughput**: At least 50 tokens/second for decoder generation
- **Real-time Factor**: < 0.3 for streaming mode
- **Latency**: < 100ms for streaming chunks
- **Accuracy**: Speaker similarity > 70%, WER < 3.0, token accuracy > 95%

## Implementation Notes

### Stage 1: Bring-Up (Current)
- ✅ Basic TTNN implementation of LLVC components
- ✅ Encoder with dilated causal convolutions
- ✅ Decoder with causal transformer layers
- ✅ Streaming inference support
- ✅ Buffer management for state caching

### Stage 2: Basic Optimizations (Next)
- Memory configuration optimization
- Sharding strategies for convolutions
- Efficient state management
- Layer fusion where possible

### Stage 3: Advanced Optimizations (Future)
- Maximum core utilization
- Flash Attention integration
- Ultra-low latency optimizations
- Multi-stream batching

## Dependencies

- `ttnn`: Tenstorrent Neural Network library
- `torch`: PyTorch for tensor operations
- `torchaudio`: Audio processing library
- `numpy`: Numerical computations

## References

- [LLVC Paper](https://arxiv.org/abs/2311.00873)
- [LLVC Repository](https://github.com/KoeAI/LLVC)
- [TTNN Documentation](https://docs.tenstorrent.com/ttnn/)
- [TT-Metal Streaming Models](https://github.com/tenstorrent/tt-metal/tree/main/models)</content>
<parameter name="filePath">/home/mahmudsudo/tt-metal/models/demos/llvc/README.md