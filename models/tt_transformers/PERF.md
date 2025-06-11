# Model performance and accuracy

Performance collected from [demo/simple_text_demo.py](demo/simple_text_demo.py) and accuracy collected from [tests/test_accuracy.py](tests/test_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table` or `pareto`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_accuracy.py` parses the below to determine expected values +- 0.5. In May 2025 we switched the default to measuring the accuracy by prefilling 512 tokens and generating another 511, rather than generating 128 tokens in earlier versions. This caused overall accuracy values to drop slightly.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

## Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:
* Qwen-2.5-7B, which uses bfp8 MLP and bfp16 attention weights in all decoder layers
* Llama-3.1-8B which uses bfp8 MLP in only the 32nd decoder layer and bfp4 MLP elsewhere

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| Llama3.2-1B       | N150        | 78        | 96        | 87.8          | 26        |
| Llama3.2-1B       | N300        | 78        | 96        | 105.9         | 22        |
| Llama3.2-1B       | T3K         | 78        | 96        | 119.8         | 32        |
| Llama3.2-1B       | TG          | 78        | 96        | 51.0          |           |
| Llama3.2-3B       | N150        | 87        | 97        | 54.0          | 55        |
| Llama3.2-3B       | N300        | 87        | 97        | 68.0          | 39        |
| Llama3.2-3B       | T3K         | 87        | 97        | 68.5          | 52        |
| Llama3.2-3B       | TG          | 87        | 97        | 33.5          |           |
| Llama3.1-8B       | N150        | 88        | 97        | 28.3          | 104       |
| Llama3.1-8B       | N300        | 88        | 97        | 44.2          | 67        |
| Llama3.1-8B       | T3K         | 88        | 97        | 64.3          | 53        |
| Llama3.1-8B       | T3K  (DP=4) |           |           | 39.6          | 58        |
| Llama3.1-8B       | T3K  (DP=8) |           |           | 24.9          | 86        |
| Llama3.1-8B       | TG          | 88        | 97        | 29.5          |           |
| Llama3.2-11B      | N300        | 87        | 97        | 44.1          | 67        |
| Llama3.2-11B      | T3K         | 87        | 97        | 62.7          | 47        |
| Llama3.2-11B      | TG          | 87        | 97        | 29.5          |           |
| Llama3.1-70B      | T3K         | 96        | 100       | 16.6          | 164       |
| Llama3.1-70B      | TG          | 95        | 100       | 12.7          |           |
| Llama3.1-70B      | TG   (DP=4) |           |           | 14.8          | 189       |
| Llama3.2-90B      | T3K         | 87        | 99        | 6             | 5535      |
| Qwen2.5-7B        | N300        | 84        | 96        | 24.6          | 92        |
| Qwen2.5-72B       | T3K         | 99        | 100       | 15.2          | 225       |
| Qwen3-32B         | T3K         | 89        | 97        | 22.9          | 123       |
| Phi3.5-mini       | N150        |           |           | 43.2          | 98        |
| Phi3.5-mini       | N300        |           |           | 57.8          | 62        |
| Phi3.5-mini       | T3K         |           |           | 48.8          | 51        |
| Mistral-7B        | N150        | 95        | 99        | 29.75         | 100.24    |
| Mistral-7B        | N300        | 95        | 99        | 47.01         | 65.95     |
| Mistral-7B        | T3K         | 95        | 99        | 67.82         | 53.93     |


## Accuracy

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| Llama3.2-1B       | N150        | 85        | 98        | 84.7          | 29        |
| Llama3.2-1B       | N300        | 85        | 98        | 102.8         | 21        |
| Llama3.2-1B       | T3K         | 85        | 98        | 120.5         | 28        |
| Llama3.2-1B       | TG          | 85        | 98        | 48.4          |           |
| Llama3.2-3B       | N150        | 92        | 99        | 47.6          | 63        |
| Llama3.2-3B       | N300        | 92        | 99        | 63.5          | 41        |
| Llama3.2-3B       | T3K         | 92        | 99        | 67.9          | 69        |
| Llama3.2-3B       | TG          | 92        | 99        | 33.6          |           |
| Llama3.1-8B       | N150        | 95        | 100       | 25.2          | 138       |
| Llama3.1-8B       | N300        | 95        | 100       | 38.8          | 79        |
| Llama3.1-8B       | T3K         | 95        | 100       | 60.8          | 81        |
| Llama3.1-8B       | TG          | 95        | 100       | 29.5          |           |
| Llama3.2-11B      | N300        | 94        | 100       | 38.3          | 78        |
| Llama3.2-11B      | T3K         | 94        | 100       | 61.4          | 53        |
| Llama3.2-11B      | TG          | 94        | 100       | 29.5          |           |
| Llama3.1-70B      | T3K         | 95        | 100       | 16.5          | 168       |
| Llama3.1-70B      | TG          | 95        | 100       | 12.7          |           |
| Llama3.2-90B      | T3K         | 96        | 100       | 6             | 5600      |
| Qwen2.5-7B        | N300        | 84        | 96        | 24.6          | 92        |
| Qwen2.5-72B       | T3K         | 99        | 100       | 15.1          | 216       |
| Qwen3-32B         | T3K         | 95        | 100       | 19.6          | 119       |
| Phi3.5-mini       | N150        |           |           | 38.8          | 92        |
| Phi3.5-mini       | N300        |           |           | 53.9          | 63        |
| Phi3.5-mini       | T3K         |           |           | 48.6          | 53        |
| Mistral-7B        | N150        | 95        | 99        | 29.75         | 100.24    |
| Mistral-7B        | N300        | 95        | 99        | 47.01         | 65.95     |
| Mistral-7B        | T3K         | 95        | 99        | 67.82         | 53.93     |

##  Long-context (64K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 64k tokens.**

| Model          | Device | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|---------------|-----------|
| Llama3.2-1B    | N150   | 53.0          | 20066     |
| Llama3.2-1B    | N300   | 65.2          | 10949     |
| Llama3.2-1B    | T3K    | 73.7          | 5271      |
| Llama3.2-1B    | TG     |               |           |
| Llama3.2-3B    | N150   | 25.3          | 46743     |
| Llama3.2-3B    | N300   | 34.8          | 22921     |
| Llama3.2-3B    | T3K    | 41.0          | 10677     |
| Llama3.2-3B    | TG     |               |           |
| Llama3.1-8B    | N150   | 16.9          | 64385     |
| Llama3.1-8B    | N300   | 26.1          | 36229     |
| Llama3.1-8B    | T3K    | 38.1          | 16165     |
| Llama3.1-8B    | TG     |               |           |
| Llama3.2-11B   | N300   | 26.1          | 36247     |
| Llama3.2-11B   | T3K    | 38.4          | 16167     |
| Llama3.2-11B   | TG     |               |           |
| Llama3.1-70B   | T3K    | 11.9          | 74363     |
| Llama3.1-70B   | TG     |               |           |
| Qwen2.5-7B     | N300   |               |           |
| Qwen2.5-72B    | T3K    |               |           |

## Short-Context, Batch-32

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model          | Device | Speed (t/s/u) | avg TTFT (ms) |
|----------------|--------|---------------|---------------|
| Llama3.2-1B    | N150   | 54.7          | 38            |
| Llama3.2-1B    | N300   | 64.2          | 34            |
| Llama3.2-1B    | T3K    | 69.9          | 42            |
| Llama3.2-1B    | TG     |               |               |
| Llama3.2-3B    | N150   | 36.5          | 69            |
| Llama3.2-3B    | N300   | 45.8          | 51            |
| Llama3.2-3B    | T3K    | 47.8          | 63            |
| Llama3.2-3B    | TG     |               |               |
| Llama3.1-8B    | N150   | 22.3          | 119           |
| Llama3.1-8B    | N300   | 33.5          | 80            |
| Llama3.1-8B    | T3K    | 45.6          | 64            |
| Llama3.1-8B    | TG     |               |               |
| Llama3.2-11B   | N300   | 33.4          | 79            |
| Llama3.2-11B   | T3K    | 45.1          | 64            |
| Llama3.2-11B   | TG     |               |               |
| Llama3.1-70B   | T3K    | 14.8          | 192           |
| Llama3.1-70B   | TG     |               |               |
| Qwen2.5-7B     | N300   |               |               |
| Qwen2.5-72B    | T3K    |               |               |

# Llama 3 model precision and math fidelity

## precision_cfg = {ff1_3: bfp4, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 85        | 98        | 100.3         | 69        |

## precision_cfg = {ff1_3: bfp4, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 88        | 98        | 100.3         | 55        |

## precision_cfg = {ff1_3: bfp4, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 87        | 98        | 96.8          | 51        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 87        | 98        | 98.5          | 50        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 91        | 98        | 99.0          | 60        |

## precision_cfg = {ff1_3: bfp8, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 89        | 99        | 95.2          | 49        |

## precision_cfg = {ff1_3: bf16, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 89        | 98        | 95.2          | 53        |

## precision_cfg = {ff1_3: bf16, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 91        | 98        | 94.4          | 57        |

## precision_cfg = {ff1_3: bf16, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 90        | 98        | 91.2          | 60        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: bf16}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 88        | 98        | 98.2          | 45        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: bfp8}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 90        | 98        | 101.0         | 57        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N300   | 90        | 98        | 99.4          | 73        |
