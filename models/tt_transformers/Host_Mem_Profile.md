# Model memory on host for performance and accuracy

[Host memory usage results](../sample_data/host_mem_profiling/) at peak memory (in MB) are collected from host_mem_profiler.py by running the target demo under memory_profiler at
100 ms sampling intervals. You can generate these results by running the script with --mesh-device, --hf-model, and
--test arguments in [demo/simple_text_demo.py](demo/simple_text_demo.py); it produces a timestamped PNG plot under profiling_results/ and prints peak and baseline RSS to the
console.

Note that all measurements include child processes spawned by the TT device runtime, giving a realistic view of total
host-side footprint. Peak memory is defined as the maximum RSS observed across the full test lifetime — including
model weight loading, KV-cache allocation, and decode iterations. Baseline is the RSS at process start before any
model work begins.

Host memory (RSS) was profiled across three configurations — [Performance](#performance), [Accuracy](#accuracy), and [Multimodal](#multimodal-models) — using up to 200
decode iterations on T3K (8-chip) and N300 (2-chip) devices.

All [numbers](../sample_data/host_mem_profiling/) below were captured with a maximum generation of 200 tokens (200 decode iterations) to match the
conditions used in [performance table](PERF.md).

## Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:
* Qwen-2.5-7B, which uses bfp8 MLP and bfp16 attention weights in all decoder layers
* Llama-3.1-8B which uses bfp8 MLP in only the 32nd decoder layer and bfp4 MLP elsewhere

| Model             | Device      | batch-1 max mem (MB)  | batch-32 max mem (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------------------|-----------------------|---------------|-----------|
| Llama-3.2-1B      | T3K         |         3532          |         2712          | 119.8         | 32        |
| Llama-3.2-3B      | T3K         |         7846          |         5388          | 68.5          | 52        |
| Llama-3.1-8B      | T3K         |         13683         |         6922          | 64.3          | 53        |
| Llama-3.2-11B     | T3K         |         13885         |         7254          | 62.7          | 47        |
| Llama-3.1-70B     | T3K         |         35820         |         35187         | 16.6          | 164       |
| Llama-3.3-70B     | T3K         |         101640        |         36176         | 16.6          | 164       |
| Llama-3.2-90B     | T3K         |         34427         |         34842         | 6             | 5535      |
| Qwen2.5-7B        | N300        |         16597         |         17754         | 24.6          | 92        |
| Qwen2.5-72B       | T3K         |         147106        |         145432        | 15.2          | 225       |
| Qwen2.5-32B       | T3K         |         53434         |         15594         | 22.4          | 190       |
| Qwen3-32B         | T3K         |         56033         |         20422         | 22.9          | 123       |
| QwQ-32B           | T3K         |         69900         |         67186         | 20.7          | 105       |

## Accuracy

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | batch-1 max mem (MB)  | batch-32 max mem (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|----------------------|------------------------|---------------|-----------|
| Llama-3.2-1B      | T3K         |        4654          |          2707          | 120.5         | 28        |
| Llama-3.2-3B      | T3K         |        10374         |          5270          | 67.9          | 69        |
| Llama-3.1-8B      | T3K         |        20406         |          7704          | 60.8          | 81        |
| Llama-3.2-11B     | T3K         |        20265         |          7051          | 61.4          | 53        |
| Llama-3.1-70B     | T3K         |        154122        |          37370         | 16.5          | 168       |
| Llama-3.3-70B     | T3K         |        100112        |          35026         | 16.5          | 168       |
| Llama-3.2-90B     | T3K         |        155139        |          34997         | 6             | 5600      |
| Qwen2.5-7B        | N300        |        20189         |          17956         | 24.6          | 92        |
| Qwen2.5-72B       | T3K         |        146204        |          147032        | 15.1          | 216       |
| Qwen2.5-32B       | T3K         |        74162         |          14811         | 19.7          | 183       |
| Qwen3-32B         | T3K         |        76288         |          20785         | 19.6          | 119       |
| QwQ-32B           | T3K         |        72627         |          67278         | 18.3          | 120       |


## Multimodal Models
The results are collected using this script [demo/simple_vision_demo.py](demo/simple_vision_demo.py)); with host memory profiler and evaluated on both language and vision branches.
The memory is reported in MB.

| Model             | Device      | batch-1 max mem (MB)  | batch-32 max mem (MB) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------------------|-----------------------|---------------|-----------|
| Llama-3.2-11B     | T3K         |         14309         |         8610          | 61.4          | 53        |
| Llama-3.2-90B     | T3K         |         75451         |         -             | 6             | 5600      |
| Qwen2.5-VL-3B     | N300        |         9319          |         9068          | 24.6          | 92        |
| Qwen2.5-VL-7B     | N300        |         18745         |         10946         | 24.6          | 92        |
| Qwen2.5-VL-72B    | T3K         |         160437        |         147925        | 15.1          | 216       |
| Qwen2.5-VL-32B    | T3K         |         71960         |         68464         | 19.7          | 183       |
| Qwen3-VL-32B      | T3K         |         76297         |         21514         | 19.6          | 119       |


[Performance](#performance) vs. [Accuracy](#accuracy) modes trade memory for precision: accuracy mode uses BF16 attention weights versus bfp8/bfp4
in performance mode, and typically raises peak memory. The increase is clearest at batch-1, where most models grow noticeably Llama-3.1-8B rises from 13.7 GB to 20.4 GB and Qwen2.5-32B from 53 GB to 74 GB — while 70B+ models show a smaller relative increase since attention weight precision is already reduced in both modes. At batch-32 the picture is mixed: KV-cache and activations dominate, and several models (Llama-3.2-1B/3B/11B, Llama-3.3-70B, Qwen2.5-32B) report slightly lower peaks in accuracy mode rather than higher.

Memory scales roughly with model size, ranging from ~3–4 GB for 1B models up to ~147–155 GB for 72–90B models. The
largest models (Qwen2.5-72B, Llama-3.2-90B) show little difference between batch-1 and batch-32, suggesting KV-cache
and weight loading dominate over activation memory. Mid-size models (32B class) exhibit a larger batch-1 to batch-32
gap, likely due to KV-cache growth with sequence length at batch-1.

Throughput and TTFT follow expected trends: smaller models deliver higher tokens/s/user (up to 120 t/s/u for 1B) and
low TTFT (~30 ms), while 90B models drop to 6 t/s/u with TTFT exceeding 5.5 seconds.

[Multimodal](#multimodal-models) models carry a modest overhead versus their text-only counterparts — Llama-3.2-11B adds ~400 MB at batch-1
though Qwen2.5-VL-72B reaches the highest overall footprint at ~160 GB.
