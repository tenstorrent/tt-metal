# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values +- 0.5.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

## Performance

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 128 tokens.**

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N150   | 88        | 98        | 95.7          | 41        |
| Llama3.2-1B    | N300   | 90        | 98        | 113.3         | 48        |
| Llama3.2-1B    | T3K    | 89        | 98        | 108.4         | 45        |
| Llama3.2-1B    | TG     | 88        | 99        | 51.0          |           |
| Llama3.2-3B    | N150   | 91        | 98        | 52.5          | 70        |
| Llama3.2-3B    | N300   | 91        | 98        | 64.8          | 57        |
| Llama3.2-3B    | T3K    | 89        | 98        | 64.0          | 54        |
| Llama3.2-3B    | TG     | 90        | 97        | 33.5          |           |
| Llama3.1-8B    | N150   | 85        | 98        | 30.1          | 120       |
| Llama3.1-8B    | N300   | 86        | 98        | 43.1          | 84        |
| Llama3.1-8B    | T3K    | 86        | 98        | 61.8          | 64        |
| Llama3.1-8B    | TG     | 86        | 98        | 29.5          |           |
| Llama3.2-11B   | N300   | 87        | 99        | 43.3          | 94        |
| Llama3.2-11B   | T3K    | 86        | 99        | 58.8          | 71        |
| Llama3.2-11B   | TG     | 86        | 98        | 29.5          |           |
| Llama3.1-70B   | T3K    | 95        | 99        | 16.2          | 180       |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |           |
| Qwen2.5-7B     | N300   | 80        | 96        | 37.9          |           |
| Qwen2.5-72B    | T3K    | 98        | 100       | 12.8          |           |


## Accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model and the Qwen-2.5-72B model. **Batch_size=1 and prefill_length is 128 tokens.**

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N150   | 88        | 99        | 91.3          | 43        |
| Llama3.2-1B    | N300   | 89        | 100       | 109.5         | 52        |
| Llama3.2-1B    | T3K    | 91        | 99        | 108.9         | 50        |
| Llama3.2-1B    | TG     | 87        | 98        | 51.3          |           |
| Llama3.2-3B    | N150   | 91        | 99        | 47.0          | 78        |
| Llama3.2-3B    | N300   | 92        | 100       | 61.4          | 72        |
| Llama3.2-3B    | T3K    | 92        | 100       | 63.6          | 58        |
| Llama3.2-3B    | TG     | 91        | 98        | 33.6          |           |
| Llama3.1-8B    | N150   | 91        | 99        | 24.6          | 140       |
| Llama3.1-8B    | N300   | 89        | 98        | 37.8          | 95        |
| Llama3.1-8B    | T3K    | 91        | 99        | 58.2          | 60        |
| Llama3.1-8B    | TG     | 88        | 100       | 29.5          |           |
| Llama3.2-11B   | N300   | 91        | 99        | 37.7          | 95        |
| Llama3.2-11B   | T3K    | 89        | 98        | 58.0          | 63        |
| Llama3.2-11B   | TG     | 88        | 100       | 29.5          |           |
| Llama3.1-70B   | T3K    | 95        | 100       | 16.2          | 183       |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |           |
| Qwen2.5-7B     | N300   | 80        | 96        | 33.4          |           |
| Qwen2.5-72B    | T3K    | 99        | 100       | 12.8          |           |

##  Long-context (64K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 64k tokens.**

| Model          | Device | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|---------------|-----------|
| Llama3.2-1B    | N150   | 59.2          | 20284     |
| Llama3.2-1B    | N300   | 73.8          | 11049     |
| Llama3.2-1B    | T3K    | 75.3          | 5350      |
| Llama3.2-1B    | TG     |               |           |
| Llama3.2-3B    | N150   | -             | -         |
| Llama3.2-3B    | N300   | 36.0          | 23249     |
| Llama3.2-3B    | T3K    | 42.3          | 10917     |
| Llama3.2-3B    | TG     |               |           |
| Llama3.1-8B    | N150   | -             | -         |
| Llama3.1-8B    | N300   | 26.7          | 36612     |
| Llama3.1-8B    | T3K    | 39.7          | 16552     |
| Llama3.1-8B    | TG     |               |           |
| Llama3.2-11B   | N300   | 26.7          | 36626     |
| Llama3.2-11B   | T3K    | 39.8          | 16541     |
| Llama3.2-11B   | TG     |               |           |
| Llama3.1-70B   | T3K    | 12.0          | 75290     |
| Llama3.1-70B   | TG     |               |           |
| Qwen2.5-7B     | N300   |               |           |
| Qwen2.5-72B    | T3K    |               |           |

## Short-Context, Batch-32

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model          | Device | Speed (t/s/u) | avg TTFT (ms) |
|----------------|--------|---------------|---------------|
| Llama3.2-1B    | N150   | 62.0          | 57            |
| Llama3.2-1B    | N300   | 72.2          | 51            |
| Llama3.2-1B    | T3K    | 73.6          | 55            |
| Llama3.2-1B    | TG     |               |               |
| Llama3.2-3B    | N150   | 37.9          | 87            |
| Llama3.2-3B    | N300   | 47.4          | 72            |
| Llama3.2-3B    | T3K    | 49.3          | 75            |
| Llama3.2-3B    | TG     |               |               |
| Llama3.1-8B    | N150   | 24.3          | 137           |
| Llama3.1-8B    | N300   | 34.3          | 105           |
| Llama3.1-8B    | T3K    | 47.9          | 74            |
| Llama3.1-8B    | TG     |               |               |
| Llama3.2-11B   | N300   | 34.5          | 98            |
| Llama3.2-11B   | T3K    | 47.6          | 80            |
| Llama3.2-11B   | TG     |               |               |
| Llama3.1-70B   | T3K    | 14.9          | 203           |
| Llama3.1-70B   | TG     |               |               |
| Qwen2.5-7B     | N300   |               |               |
| Qwen2.5-72B    | T3K    |               |               |
