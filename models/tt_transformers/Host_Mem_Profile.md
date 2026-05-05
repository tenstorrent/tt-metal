# Model memory on host for performance and accuracy

Host memory usage results in MB are collected from host_mem_profiler.py by running the target demo under memory_profiler at
100 ms sampling intervals. You can generate these results by running the script with --mesh-device, --hf-model, and
--test arguments in [demo/simple_text_demo.py](demo/simple_text_demo.py)); it produces a timestamped PNG plot under profiling_results/ and prints peak and baseline RSS to the
console.

Note that all measurements include child processes spawned by the TT device runtime, giving a realistic view of total
host-side footprint. Peak memory is defined as the maximum RSS observed across the full test lifetime — including
model weight loading, KV-cache allocation, and decode iterations. Baseline is the RSS at process start before any
model work begins.

All figures below were captured with a maximum generation of 200 tokens (200 decode iterations) to match the
conditions used in PERF.md.

## Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:
* Qwen-2.5-7B, which uses bfp8 MLP and bfp16 attention weights in all decoder layers
* Llama-3.1-8B which uses bfp8 MLP in only the 32nd decoder layer and bfp4 MLP elsewhere

| Model             | Device      | batch-1 (MB)  | batch-32 (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|---------------|---------------|-----------|
| Llama-3.2-1B      | T3K         | 3532          | 2712          | 119.8         | 32        |
| Llama-3.2-3B      | T3K         | 7846          | 5388          | 68.5          | 52        |
| Llama-3.1-8B      | T3K         | 13683         | 6922          | 64.3          | 53        |
| Llama-3.2-11B     | T3K         | 13885         | 7254          | 62.7          | 47        |
| Llama-3.3-70B     | T3K         | 34752         | 36082         | 16.6          | 164       |
| Llama-3.2-90B     | T3K         | 34427         | 34842         | 6             | 5535      |
| Qwen2.5-7B        | N300        | 16597         | 17754         | 24.6          | 92        |
| Qwen2.5-72B       | T3K         | 147106        | 145432        | 15.2          | 225       |
| Qwen2.5-32B       | T3K         | 53434         | 15594         | 22.4          | 190       |
| Qwen3-32B         | T3K         | 56033         | 20422         | 22.9          | 123       |
| QwQ-32B           | T3K         | 69900         | 67186         | 20.7          | 105       |

## Accuracy

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | batch-1 (MB)  | batch-32 (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|---------------|---------------|-----------|
| Llama-3.2-1B      | T3K         | 4654          | 2707          | 120.5         | 28        |
| Llama-3.2-3B      | T3K         | 10374         | 5270          | 67.9          | 69        |
| Llama-3.1-8B      | T3K         | 20406         | 7704          | 60.8          | 81        |
| Llama-3.2-11B     | T3K         | 20265         | 7051          | 61.4          | 53        |
| Llama-3.1-70B     | T3K         | 129181        | 34250         | 16.5          | 168       |
| Llama-3.2-90B     | T3K         | 155139        | 34997         | 6             | 5600      |
| Qwen2.5-7B        | N300        | 20189         | 17956         | 24.6          | 92        |
| Qwen2.5-72B       | T3K         | 146204        | 147032        | 15.1          | 216       |
| Qwen2.5-32B       | T3K         | 74162         | 14811         | 19.7          | 183       |
| Qwen3-32B         | T3K         | 76288         | 20785         | 19.6          | 119       |
| QwQ-32B           | T3K         | 72627         | 67278         | 18.3          | 120       |


## Multimodal Models
The results are collected using this script [demo/simple_vision_demo.py](demo/simple_vision_demo.py)); with host memory profiler and evaluated on both languange and vision branches.
The memory is reported in MB.

| Model             | Device      | batch-1 (MB)  | batch-32 (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|---------------|---------------|-----------|
| Llama-3.2-11B     | T3K         | 14309         | 8610          | 61.4          | 53        |
| Llama-3.2-90B     | T3K         | 75451         | -             | 6             | 5600      |
| Qwen2.5-VL-3B     | N300        | 9319          | 9068          | 24.6          | 92        |
| Qwen2.5-VL-7B     | N300        | 18745         | 10946         | 24.6          | 92        |
| Qwen2.5-VL-72B    | T3K         | 160437        | 147925        | 15.1          | 216       |
| Qwen2.5-VL-32B    | T3K         | 71960         | 68464         | 19.7          | 183       |
| Qwen3-VL-32B      | T3K         | 76297         | 21514         | 19.6          | 119       |
