# Wan2.2-T2V-A14B

## Introduction

[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) is a leading text-to-video generative model.

This model is implemented in the TT-DiT library to enable inference on Wormhole LoudBox and Galaxy systems.

## Upcoming Features
- optimized performance on Wormhole systems
- functional and optimized performance on Blackhole systems

## Performance

This section will be updated as performance work progresses.
There are many items we're making progress on to improve performance:
- increased matmul utilization
- increased SDPA utilization
- overlapped AllGather-Matmul and Matmul-ReduceScatter
- fused binary ops
- overlapped weight AllGather with compute

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Generate a video with the pipeline test. Same comment here, use 2x4 on 8-chip systems and 4x8 on 32-chip systems.
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "4x8"
```

## Limitations

While output videos look good, we have many items of work in progress to improve correctness.
As of now, running on systems smaller than a 2x4 Wormhole mesh is not well supported. The model is large and requires 8-chips worth of memory to run.
Performance optimization is in progress.
