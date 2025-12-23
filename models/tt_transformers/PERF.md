# Model performance and accuracy

Performance and token accuracy using teacher forcing is collected from [demo/simple_text_demo.py](demo/simple_text_demo.py) with the `ci-token-matching` test case. You can generate this table by running these tests with the `lt` tool (tell it to run `table` or `pareto`) and pressing `m` whilst in the results section to export to markdown.

Note that token accuracy parses the below to determine expected values +- 0.5. In May 2025 we switched the default to measuring the accuracy by prefilling 512 tokens and generating another 511, rather than generating 128 tokens in earlier versions. This caused overall accuracy values to drop slightly.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

Performance only tests were done using [demo/simple_text_demo.py](demo/simple_text_demo.py) script with the `ci-1` and `ci-32` test cases using both `performance` and `accuracy` settings.

## Performance-ci-token-matching

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:
* Qwen-2.5-7B, which uses bfp8 MLP and bfp16 attention weights in all decoder layers
* Llama-3.1-8B which uses bfp8 MLP in only the 32nd decoder layer and bfp4 MLP elsewhere

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B      | N150        | 79        | 97        | 17.9          | 56        |
| Llama-3.2-1B      | N300        | 79        | 97        | 19.7          | 52        |
| Llama-3.2-1B      | T3K         | 80        | 97        | 24.3          | 33        |
| Llama-3.2-3B      | N150        | 89        | 98        | 14.4          | 143       |
| Llama-3.2-3B      | N300        | 89        | 98        | 13.6          | 133       |
| Llama-3.2-3B      | T3K         | 91        | 99        | 14.4          | 78        |
| Llama-3.1-8B      | N150        | 90        | 97        | 12.7          | 295       |
| Llama-3.1-8B      | N300        | 90        | 97        | 12.3          | 224       |
| Llama-3.1-8B      | T3K         | 90        | 98        | 13.6          | 116       |
| Llama-3.1-70B     | T3K         | 96        | 100       | 5.7           | 593       |
| Qwen2.5-7B        | N300        | 84        | 96        | 12.7          | 308       |
| Qwen2.5-72B       | T3K         | 99        | 100       | 5.0           | 603       |
| Qwen2.5-Coder-32B | T3K         | 96        | 99        | 6.7           | 325       |
| Qwen3-32B         | T3K         | 89        | 97        | -             | -         |
| Phi3.5-mini       | N150        | -         | -         | 20.0          | 211       |
| Phi3.5-mini       | N300        | -         | -         | 14.0          | 166       |
| Phi3.5-mini       | T3K         | -         | -         | 13.7          | 89        |
| Mistral-7B        | N150        | 95        | 99        | 21.3          | 287       |
| Mistral-7B        | N300        | 95        | 100       | 14.5          | 233       |
| Mistral-7B        | T3K         | 95        | 100       | 14.3          | 114       |
| Phi-3-mini-128k-instruct | N150 | 89        | 99        | 20.4          | 190       |
| Phi-3-mini-128k-instruct | N300 | 89        | 99        | 14.0          | 156       |
| Phi-4             | N300        | 97        | 100       | 11.8          | 429       |
| Mixtral-8x7B-v0.1 | T3K         | 95        | 100       | 12.7          | 313       |


## Accuracy-ci-token-matching

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B      | N150        | 79        | 97        | 17.9          | 60        |
| Llama-3.2-1B      | N300        | 79        | 97        | 19.7          | 53        |
| Llama-3.2-1B      | T3K         | 80        | 97        | 24.4          | 33        |
| Llama-3.2-3B      | N150        | 89        | 98        | 14.4          | 167       |
| Llama-3.2-3B      | N300        | 89        | 98        | 13.6          | 143       |
| Llama-3.2-3B      | T3K         | 91        | 99        | 14.5          | 78        |
| Llama-3.1-8B      | N150        | 90        | 97        | 12.7          | 353       |
| Llama-3.1-8B      | N300        | 90        | 97        | 12.2          | 253       |
| Llama-3.1-8B      | T3K         | 90        | 98        | 13.5          | 123       |
| Llama-3.1-70B     | T3K         | 96        | 100       | 5.7           | 589       |
| Qwen2.5-7B        | N300        | 84        | 96        | 12.7          | 312       |
| Qwen2.5-72B       | T3K         | 99        | 100       | 5.0           | 604       |
| Qwen2.5-Coder-32B | T3K         | 96        | 99        | 6.7           | 360       |
| Qwen3-32B         | T3K         | 89        | 97        | -             | -         |
| Phi3.5-mini       | N300        | -         | -         | 14.0          | 183       |
| Phi3.5-mini       | T3K         | -         | -         | 13.6          | 94        |
| Mistral-7B        | N150        | 95        | 99        | 21.3          | 349       |
| Mistral-7B        | N300        | 95        | 100       | 14.7          | 249       |
| Mistral-7B        | T3K         | 95        | 100       | 14.3          | 116       |
| Phi-3-mini-128k-instruct | N150 | 89        | 99        | 20.2          | 194       |
| Phi-3-mini-128k-instruct | N300 | 89        | 99        | 14.0          | 163       |
| Phi-4             | N300        | 97        | 100       | 11.8          | 454       |
| Mixtral-8x7B-v0.1 | T3K         | 95        | 100       | 12.6          | 314       |

## Performance-ci-1

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| Llama-3.2-1B      | N150        | 24.3          | 27        |
| Llama-3.2-1B      | N300        | 39.9          | 18        |
| Llama-3.2-1B      | T3K         | 85.2          | 13        |
| Llama-3.2-3B      | N150        | 20.4          | 51        |
| Llama-3.2-3B      | N300        | 32.3          | 41        |
| Llama-3.2-3B      | T3K         | 55.6          | 27        |
| Llama-3.1-8B      | N150        | 15.1          | 105       |
| Llama-3.1-8B      | N300        | 25.7          | 74        |
| Llama-3.1-8B      | T3K         | 50.6          | 39        |
| Llama-3.1-70B     | T3K         | 15.4          | 163       |
| Qwen2.5-7B        | N300        | 28.9          | 74        |
| Qwen2.5-72B       | T3K         | 14.2          | 177       |
| Qwen2.5-Coder-32B | T3K         | 21.5          | 92        |
| Qwen3-32B         | T3K         | 22.5          | 98        |
| Phi3.5-mini       | N300        | 44.5          | 48        |
| Phi3.5-mini       | T3K         | 58.8          | 30        |
| Mistral-7B        | N150        | 17.6          | 292       |
| Mistral-7B        | N300        | 44.0          | 66        |
| Mistral-7B        | T3K         | 59.9          | 35        |
| Phi-3-mini-128k-instruct | N150 | 31.0          | 68        |
| Phi-3-mini-128k-instruct | N300 | 44.6          | 53        |
| Phi-4             | N300        | 25.9          | 121       |
| Mixtral-8x7B-v0.1 | T3K         | 23.4          | 121       |

## Accuracy-ci-1

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| Llama-3.2-1B      | N150        | 24.1          | 25        |
| Llama-3.2-1B      | N300        | 39.7          | 20        |
| Llama-3.2-1B      | T3K         | 85.1          | 15        |
| Llama-3.2-3B      | N150        | 19.5          | 63        |
| Llama-3.2-3B      | N300        | 31.3          | 43        |
| Llama-3.2-3B      | T3K         | 55.5          | 27        |
| Llama-3.1-8B      | N150        | 14.2          | 135       |
| Llama-3.1-8B      | N300        | 23.6          | 79        |
| Llama-3.1-8B      | T3K         | 39.7          | 47        |
| Llama-3.1-70B     | T3K         | 15.4          | 166       |
| Qwen2.5-7B        | N300        | 28.6          | 75        |
| Qwen2.5-72B       | T3K         | 14.2          | 177       |
| Qwen2.5-Coder-32B | T3K         | 19.0          | 107       |
| Qwen3-32B         | T3K         | 19.3          | 116       |
| Phi3.5-mini       | N300        | 36.8          | 58        |
| Phi3.5-mini       | T3K         | 55.9          | 32        |
| Mistral-7B        | N150        | 17.0          | 348       |
| Mistral-7B        | N300        | 38.0          | 76        |
| Mistral-7B        | T3K         | 57.6          | 38        |
| Phi-3-mini-128k-instruct | N150 | 28.7          | 77        |
| Phi-3-mini-128k-instruct | N300 | 42.4          | 50        |
| Phi-4             | N300        | 21.6          | 139       |
| Mixtral-8x7B-v0.1 | T3K         | 23.2          | 120       |

## Performance-ci-32

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| Llama-3.2-1B      | N150        | 23.6          | 21        |
| Llama-3.2-1B      | N300        | 38.9          | 15        |
| Llama-3.2-1B      | T3K         | 85.0          | 11        |
| Llama-3.2-3B      | N150        | 18.5          | 52        |
| Llama-3.2-3B      | N300        | 30.0          | 38        |
| Llama-3.2-3B      | T3K         | 54.5          | 22        |
| Llama-3.1-8B      | N150        | 13.9          | 104       |
| Llama-3.1-8B      | N300        | 23.9          | 66        |
| Llama-3.1-8B      | T3K         | 49.5          | 34        |
| Llama-3.1-70B     | T3K         | 15.3          | 161       |
| Qwen2.5-7B        | N300        | 25.7          | 72        |
| Qwen2.5-72B       | T3K         | 14.0          | 173       |
| Qwen2.5-Coder-32B | T3K         | 21.1          | 87        |
| Phi3.5-mini       | N300        | 27.5          | 47        |
| Phi3.5-mini       | T3K         | 48.0          | 26        |
| Mistral-7B        | N150        | 17.6          | 292       |
| Mistral-7B        | N300        | 44.0          | 66        |
| Mistral-7B        | T3K         | 59.9          | 35        |
| Phi-3-mini-128k-instruct | N300 | 27.6          | 43        |
| Mixtral-8x7B-v0.1 | T3K         | 23.1          | 118       |

## Accuracy-ci-32

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| Llama-3.2-1B      | N150        | 23.4          | 26        |
| Llama-3.2-1B      | N300        | 38.6          | 17        |
| Llama-3.2-1B      | T3K         | 85.0          | 11        |
| Llama-3.2-3B      | N150        | 17.8          | 58        |
| Llama-3.2-3B      | N300        | 29.2          | 42        |
| Llama-3.2-3B      | T3K         | 55.0          | 23        |
| Llama-3.1-8B      | N300        | 22.1          | 78        |
| Llama-3.1-8B      | T3K         | 48.2          | 36        |
| Llama-3.1-70B     | T3K         | 15.3          | 161       |
| Qwen2.5-7B        | N300        | 25.3          | 72        |
| Qwen2.5-72B       | T3K         | 14.1          | 175       |
| Qwen2.5-Coder-32B | T3K         | 18.1          | 105       |
| Phi3.5-mini       | T3K         | 39.6          | 27        |
| Mistral-7B        | N300        | 34.2          | 75        |
| Mistral-7B        | T3K         | 55.2          | 36        |
| Phi-3-mini-128k-instruct | N300 | 26.7          | 48        |
| Mixtral-8x7B-v0.1 | T3K         | 22.5          | 118       |

##  Long-context (64K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 64k tokens.**

| Model          | Device | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|---------------|-----------|
| Llama-3.2-1B   | N150   | 53.0          | 20066     |
| Llama-3.2-1B   | N300   | 65.2          | 10949     |
| Llama-3.2-1B   | T3K    | 73.7          | 5271      |
| Llama-3.2-1B   | TG     |               |           |
| Llama-3.2-3B   | N150   | 25.3          | 46743     |
| Llama-3.2-3B   | N300   | 34.8          | 22921     |
| Llama-3.2-3B   | T3K    | 41.0          | 10677     |
| Llama-3.2-3B   | TG     |               |           |
| Llama-3.1-8B   | N150   | 16.9          | 64385     |
| Llama-3.1-8B   | N300   | 26.1          | 36229     |
| Llama-3.1-8B   | T3K    | 38.1          | 16165     |
| Llama-3.1-8B   | TG     |               |           |
| Llama-3.2-11B  | N300   | 26.1          | 36247     |
| Llama-3.2-11B  | T3K    | 38.4          | 16167     |
| Llama-3.2-11B  | TG     |               |           |
| Llama-3.1-70B  | T3K    | 11.9          | 74363     |
| Llama-3.1-70B  | TG     |               |           |
| Qwen2.5-7B     | N300   |               |           |
| Qwen2.5-72B    | T3K    |               |           |

##  Long-context (32K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 32k tokens.**

| Model                     | Device | Speed (t/s/u) | TTFT (ms) |
|---------------------------|--------|---------------|-----------|
| Phi-3-mini-128k-instruct  | N300   | 26.1          | 10072     |

## Short-Context, Batch-32

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model          | Device | Speed (t/s/u) | avg TTFT (ms) |
|----------------|--------|---------------|---------------|
| Llama-3.2-1B   | N150   | 54.7          | 38            |
| Llama-3.2-1B   | N300   | 64.2          | 34            |
| Llama-3.2-1B   | T3K    | 69.9          | 42            |
| Llama-3.2-1B   | TG     |               |               |
| Llama-3.2-3B   | N150   | 36.5          | 69            |
| Llama-3.2-3B   | N300   | 45.8          | 51            |
| Llama-3.2-3B   | T3K    | 47.8          | 63            |
| Llama-3.2-3B   | TG     |               |               |
| Llama-3.1-8B   | N150   | 22.3          | 119           |
| Llama-3.1-8B   | N300   | 33.5          | 80            |
| Llama-3.1-8B   | T3K    | 45.6          | 64            |
| Llama-3.1-8B   | TG     |               |               |
| Llama-3.2-11B  | N300   | 33.4          | 79            |
| Llama-3.2-11B  | T3K    | 45.1          | 64            |
| Llama-3.2-11B  | TG     |               |               |
| Llama-3.1-70B  | T3K    | 14.8          | 192           |
| Llama-3.1-70B  | TG     |               |               |
| Qwen2.5-7B     | N300   |               |               |
| Qwen2.5-72B    | T3K    |               |               |
| Phi-3-mini-128k-instruct  | 150    | 25.66         | 68.58         |
| Phi-3-mini-128k-instruct  | N300   | 39.4          | 85.99         |

# Llama 3 model precision and math fidelity

## precision_cfg = {ff1_3: bfp4, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 85        | 98        | 100.3         | 69        |

## precision_cfg = {ff1_3: bfp4, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 88        | 98        | 100.3         | 55        |

## precision_cfg = {ff1_3: bfp4, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: lofi, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 87        | 98        | 96.8          | 51        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 87        | 98        | 98.5          | 50        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 91        | 98        | 99.0          | 60        |

## precision_cfg = {ff1_3: bfp8, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 89        | 99        | 95.2          | 49        |

## precision_cfg = {ff1_3: bf16, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 89        | 98        | 95.2          | 53        |

## precision_cfg = {ff1_3: bf16, ff2: bfp8, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 91        | 98        | 94.4          | 57        |

## precision_cfg = {ff1_3: bf16, ff2: bf16, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi4, li_ff2: hifi4, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 90        | 98        | 91.2          | 60        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: bf16}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 88        | 98        | 98.2          | 45        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: bfp8}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 90        | 98        | 101.0         | 57        |

## precision_cfg = {ff1_3: bfp8, ff2: bfp8, wqkv: bfp8, wo: bfp4, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: hifi2, li_qkv_decode: hifi2, li_o_decode: lofi, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: lofi, sdpa_prefill: hifi4}

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama-3.2-1B   | N300   | 90        | 98        | 99.4          | 73        |
