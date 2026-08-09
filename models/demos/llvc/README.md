LLVC (Low-Latency Low-Resource Voice Conversion) on TTNN
This directory contains the production-grade TTNN implementation of the LLVC (Koe AI) voice conversion model for Tenstorrent Wormhole/Blackhole hardware.

Architecture Overview
LLVC is optimized for ultra-low latency (<50ms) real-time voice conversion.

Prenet: 12 Causal Convolutional Blocks mapping raw PCM 16kHz audio to 512-dim features.
Encoder: 8 Depthwise-Separable Dilated Residual Blocks (dilations=[1, 2, 4, 8, 16, 32, 1, 2]).
Decoder: 13-Frame Causal Cross-Attention & Dimensionality Reduction (512 -> 256).
Vocoder: Causal ConvTranspose1d Waveform Synthesizer.
Memory Management: Interleaved L1 and DRAM ttnn.MemoryConfig for intermediate activations.
Streaming State Cache: 4 state buffers for zero-latency chunked inference.
Quick Start & Usage
1. Non-Streaming Mode (Full Context)
To run full-context conversion on an entire audio file: pytest models/demos/llvc/tests/test_llvc.py::test_real_llvc_accuracy

2. Streaming Mode (Real-Time Chunk Processing)
To run low-latency streaming chunk inference (chunk size 50ms / 100ms): python3 models/demos/llvc/demo/perf_llvc.py --mode streaming --chunk-ms 50

Verification & PCC Accuracy
Run the accuracy test against the PyTorch reference model: pytest models/demos/llvc/tests/test_llvc.py

Target accuracy: PCC >= 0.99 (Pearson Correlation Coefficient).
