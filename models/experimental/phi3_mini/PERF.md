# Llama 3 model performance and accuracy

Performance collected from [demo/simple_text_demo.py](demo/simple_text_demo.py) and accuracy collected from [tests/test_accuracy.py](tests/test_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table` or `pareto`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_accuracy.py` parses the below to determine expected values +- 0.5.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

## Performance

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 (per DP group) and prefill_length is 128 tokens.**

| Model                    | Device     | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------------------|------------|-----------|-----------|---------------|-----------|
| Llama3.2-1B              | N150       | 88        | 98        | 84.5          | 26        |
| Llama3.2-1B              | N300       | 88        | 98        | 100.5         | 24        |
| Llama3.2-1B              | T3K        | 89        | 98        | 113.8         | 34        |
| Llama3.2-1B              | T3K  (DP=4)|           |           | 83.5          | 19        |
| Llama3.2-1B              | T3K  (DP=8)|           |           | 63.9          | 19        |
| Llama3.2-1B              | TG         | 88        | 99        | 51.0          |           |
| Llama3.2-3B              | N150       | 92        | 95        | 52.4          | 57        |
| Llama3.2-3B              | N300       | 92        | 97        | 65.3          | 39        |
| Llama3.2-3B              | T3K        | 91        | 97        | 65.4          | 51        |
| Llama3.2-3B              | TG         | 90        | 97        | 33.5          |           |
| Llama3.1-8B              | N150       | 88        | 100       | 27.8          | 107       |
| Llama3.1-8B              | N300       | 88        | 98        | 43.3          | 68        |
| Llama3.1-8B              | T3K        | 88        | 100       | 62.3          | 52        |
| Llama3.1-8B              | T3K  (DP=4)|           |           | 39.6          | 58        |
| Llama3.1-8B              | T3K  (DP=8)|           |           | 24.9          | 86        |
| Llama3.1-8B              | TG         | 86        | 98        | 29.5          |           |
| Llama3.2-11B             | N300       | 88        | 99        | 42.8          | 67        |
| Llama3.2-11B             | T3K        | 87        | 99        | 61.2          | 68        |
| Llama3.2-11B             | TG         | 86        | 98        | 29.5          |           |
| Llama3.1-70B             | T3K        | 97        | 100       | 16.3          | 182       |
| Llama3.1-70B             | TG         | 95        | 100       | 12.7          |           |
| Llama3.1-70B             | TG   (DP=4)|           |           | 14.8          | 189       |
| Llama3.2-90B             | T3K        | 87        | 99        | 6             | 5535      |
| Qwen2.5-7B               | N300       | 80        | 96        | 37.9          |           |
| Qwen2.5-72B              | T3K        | 98        | 100       | 12.8          |           |
| Phi3.5-mini              | N150       |           |           | 43.2          | 98        |
| Phi3.5-mini              | N300       |           |           | 57.8          | 62        |
| Phi3.5-mini              | T3K        |           |           | 48.8          | 51        |
| Phi-3-mini-128k-instruct | N150       | 91        | 99        | 43.4          | 78        |
| Phi-3-mini-128k-instruct | N300       | 91        | 99        | 56.9          | 60        |

## Accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model and the Qwen-2.5-72B model. **Batch_size=1 and prefill_length is 128 tokens.**

| Model                    | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B              | N150   | 91        | 98        | 82.0          | 29        |
| Llama3.2-1B              | N300   | 91        | 98        | 98.6          | 25        |
| Llama3.2-1B              | T3K    | 88        | 98        | 114.1         | 42        |
| Llama3.2-1B              | TG     | 87        | 98        | 51.3          |           |
| Llama3.2-3B              | N150   | 94        | 99        | 47.0          | 65        |
| Llama3.2-3B              | N300   | 90        | 98        | 61.1          | 47        |
| Llama3.2-3B              | T3K    | 92        | 98        | 65.2          | 53        |
| Llama3.2-3B              | TG     | 91        | 98        | 33.6          |           |
| Llama3.1-8B              | N150   | 93        | 100       | 24.8          | 127       |
| Llama3.1-8B              | N300   | 94        | 100       | 37.8          | 82        |
| Llama3.1-8B              | T3K    | 94        | 100       | 59.8          | 56        |
| Llama3.1-8B              | TG     | 88        | 100       | 29.5          |           |
| Llama3.2-11B             | N300   | 92        | 100       | 37.5          | 83        |
| Llama3.2-11B             | T3K    | 95        | 100       | 59.2          | 58        |
| Llama3.2-11B             | TG     | 88        | 100       | 29.5          |           |
| Llama3.1-70B             | T3K    | 98        | 100       | 14.1          | 210       |
| Llama3.1-70B             | TG     | 95        | 100       | 12.7          |           |
| Llama3.2-90B             | T3K    | 97        | 100       | 6             | 5600      |
| Qwen2.5-7B               | N300   | 80        | 96        | 33.4          |           |
| Qwen2.5-72B              | T3K    | 99        | 100       | 12.8          |           |
| Phi3.5-mini              | N150   |           |           | 38.8          | 92        |
| Phi3.5-mini              | N300   |           |           | 53.9          | 63        |
| Phi3.5-mini              | T3K    |           |           | 48.6          | 53        |
| Phi-3-mini-128k-instruct | N150   | 95        | 99        | 39.1          | 87        |
| Phi-3-mini-128k-instruct | N300   | 95        | 99        | 53.5          | 66        |

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

##  Long-context (32K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 32k tokens.**

| Model                     | Device | Speed (t/s/u) | TTFT (ms) |
|---------------------------|--------|---------------|-----------|
| Phi-3-mini-128k-instruct  | N300   | 24.4          | 11259.6  |

## Short-Context, Batch-32

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model                     | Device | Speed (t/s/u) | avg TTFT (ms) |
|---------------------------|--------|---------------|---------------|
| Llama3.2-1B               | N150   | 54.7          | 38            |
| Llama3.2-1B               | N300   | 64.2          | 34            |
| Llama3.2-1B               | T3K    | 69.9          | 42            |
| Llama3.2-1B               | TG     |               |               |
| Llama3.2-3B               | N150   | 36.5          | 69            |
| Llama3.2-3B               | N300   | 45.8          | 51            |
| Llama3.2-3B               | T3K    | 47.8          | 63            |
| Llama3.2-3B               | TG     |               |               |
| Llama3.1-8B               | N150   | 22.3          | 119           |
| Llama3.1-8B               | N300   | 33.5          | 80            |
| Llama3.1-8B               | T3K    | 45.6          | 64            |
| Llama3.1-8B               | TG     |               |               |
| Llama3.2-11B              | N300   | 33.4          | 79            |
| Llama3.2-11B              | T3K    | 45.1          | 64            |
| Llama3.2-11B              | TG     |               |               |
| Llama3.1-70B              | T3K    | 14.8          | 192           |
| Llama3.1-70B              | TG     |               |               |
| Qwen2.5-7B                | N300   |               |               |
| Qwen2.5-72B               | T3K    |               |               |
| Phi-3-mini-128k-instruct  | 150    | 25.4          | 192.5         |
| Phi-3-mini-128k-instruct  | N300   | 36.8          | 173.1         |


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
