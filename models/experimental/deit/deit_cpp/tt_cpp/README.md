# DeiT C++ TTNN Demo

This directory contains a C++ TTNN port of the optimized DeiT demo flow, aligned against the optimized Python reference files:

- `models/demos/deit_tiny/tt/ttnn_optimized_sharded_deit_wh.py`
- `models/demos/wormhole/deit_tiny/demo/deit_test_infra.py`
- `models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py`

The goal of this port is to reproduce the optimized Wormhole DeiT inference path with C++ TTNN APIs, including model preprocessing, sharded execution, and 2CQ trace-based performance measurement.

Unlike the earlier TorchScript-based prototype, the current flow does **not** depend on LibTorch or OpenCV. Model assets are exported as a `manifest.json` file plus raw tensor blobs under `weights/`, and the C++ demo reads those native artifacts directly.

## Files

- `deit_inference.h` / `deit_inference.cpp`
  - DeiT model definition in C++
  - patch embedding, embeddings, layernorm, attention, feedforward, encoder, classifier path
  - program config construction for Wormhole sharded execution

- `deit_test_infra.hpp` / `deit_test_infra.cpp`
  - manifest-driven model loading and parameter preprocessing
  - input setup for L1 sharded / DRAM sharded execution
  - C++ equivalent of the Python test infra wrapper

- `deit_model/bin2pt.py`
  - exports Hugging Face DeiT teacher weights to `manifest.json` + `weights/*.bin`
  - reshapes and fuses tensors into the exact parameter names/layouts consumed by `deit_inference.cpp`

- `demo_deit_ttnn_inference_perf_e2e_2cq_trace.cpp`
  - end-to-end performance demo
  - compile pass, cache pass, trace capture, warmup, and measurement loop

## Current Status

This C++ demo has been validated to:

- build successfully through the repo-root build
- initialize device / mesh successfully
- run compile, cache, trace capture, warmup, and measurement successfully
- execute end-to-end inference on the provided DeiT teacher manifest

The current implementation already includes the important fixes needed to match the optimized Python path more closely, including:

- correct QKV interleaved weight preprocessing
- wrapper-based `split_query_key_value_and_split_heads` / `concatenate_heads` usage
- rank normalization where the C++ API path produces `[B, 1, S, H]` tensors but downstream wrappers expect `[B, S, H]`
- valid matmul multicast `out_block_h` / `out_block_w` settings
- 2CQ trace demo flow with host-to-device copies and replay-time `reshard_out` reuse

## Export model assets

Generate the native DeiT manifest and tensor blobs from the repository root using the repo Python environment:

```bash
python models/experimental/deit/deit_cpp/deit_model/bin2pt.py
```

This writes:

```text
models/experimental/deit/deit_cpp/deit_model/manifest.json
models/experimental/deit/deit_cpp/deit_model/weights/
```

## Build

Build from the repository root:

```bash
./build_metal.sh --build-deit-cpp
```

Expected binary:

```text
build_Release/bin/deit_cpp_demo
```



## Run

Command format from repo root:

```bash
./build_Release/bin/deit_cpp_demo <batch_size> <manifest_path>
```

- `<batch_size>`: first CLI argument, for example `1`
- `<manifest_path>`: optional path to the exported `manifest.json`

Example run from repo root:

```bash
./build_Release/bin/deit_cpp_demo 1 models/experimental/deit/deit_cpp/deit_model/manifest.json
```

Here, `1` is the batch size passed as the first CLI argument. If omitted, the demo defaults to batch size `1`.

If no manifest path is provided, the demo defaults to:

```text
models/experimental/deit/deit_cpp/deit_model/manifest.json
```

## Expected Runtime Flow

On a successful run, the demo should progress through:

- device initialization
- compile pass
- cache pass
- trace capture
- warmup
- performance measurement

Typical output includes timing lines similar to:

```text
Compile done in ... 0.472932s
Cache pass done in ... 0.00934929s
ttnn_deit_base_batch_size1 inference time (avg): ... 0.00292072s
ttnn_deit_base_batch_size1 compile time: ... 0.47644s
Samples per second(FPS): ... 342.624
```
