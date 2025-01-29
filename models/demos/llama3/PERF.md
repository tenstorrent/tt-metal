# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values +- 0.5.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

## LlamaOptimizations.performance

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 128 tokens.**

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------|--------|-----------|-----------|---------------|-----------|
| 1b    | N150   | 88        | 98        | 95.7          | 41        |
| 1b    | N300   | 90        | 98        | 113.3         | 48        |
| 1b    | T3K    | 89        | 98        | 108.4         | 45        |
| 1b    | TG     | 88        | 99        | 51.0          |           |
| 3b    | N150   | 91        | 98        | 52.5          | 70        |
| 3b    | N300   | 91        | 98        | 64.8          | 57        |
| 3b    | T3K    | 89        | 98        | 64.0          | 54        |
| 3b    | TG     | 90        | 97        | 33.5          |           |
| 8b    | N150   | 85        | 98        | 30.1          | 120       |
| 8b    | N300   | 86        | 98        | 43.1          | 84        |
| 8b    | T3K    | 86        | 98        | 61.8          | 64        |
| 8b    | TG     | 86        | 98        | 29.5          |           |
| 11b   | N300   | 87        | 99        | 43.3          | 94        |
| 11b   | T3K    | 86        | 99        | 58.8          | 71        |
| 11b   | TG     | 86        | 98        | 29.5          |           |
| 70b   | T3K    | 95        | 99        | 16.2          | 180       |
| 70b   | TG     | 95        | 100       | 12.7          |           |


## LlamaOptimizations.accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model. **Batch_size=1 and prefill_length is 128 tokens.**

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------|--------|-----------|-----------|---------------|-----------|
| 1b    | N150   | 88        | 99        | 91.3          | 43        |
| 1b    | N300   | 89        | 100       | 109.5         | 52        |
| 1b    | T3K    | 91        | 99        | 108.9         | 50        |
| 1b    | TG     | 87        | 98        | 51.3          |           |
| 3b    | N150   | 91        | 99        | 47.0          | 78        |
| 3b    | N300   | 92        | 100       | 61.4          | 72        |
| 3b    | T3K    | 92        | 100       | 63.6          | 58        |
| 3b    | TG     | 91        | 98        | 33.6          |           |
| 8b    | N150   | 91        | 99        | 24.6          | 140       |
| 8b    | N300   | 89        | 98        | 37.8          | 95        |
| 8b    | T3K    | 91        | 99        | 58.2          | 60        |
| 8b    | TG     | 88        | 100       | 29.5          |           |
| 11b   | N300   | 91        | 99        | 37.7          | 95        |
| 11b   | T3K    | 89        | 98        | 58.0          | 63        |
| 11b   | TG     | 88        | 100       | 29.5          |           |
| 70b   | T3K    | 95        | 100       | 16.2          | 183       |
| 70b   | TG     | 95        | 100       | 12.7          |           |

## Llama long-context

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 64k tokens.**

| Model | Device | Speed (t/s/u) | TTFT (ms) |
|-------|--------|---------------|-----------|
| 1b    | N150   | 59.2          | 20284     |
| 1b    | N300   | 73.8          | 11049     |
| 1b    | T3K    | 75.3          | 5350      |
| 1b    | TG     |               |           |
| 3b    | N150   | -             | -         |
| 3b    | N300   | 36.0          | 23249     |
| 3b    | T3K    | 42.3          | 10917     |
| 3b    | TG     |               |           |
| 8b    | N150   | -             | -         |
| 8b    | N300   | 26.7          | 36612     |
| 8b    | T3K    | 39.7          | 16552     |
| 8b    | TG     |               |           |
| 11b   | N300   | 26.7          | 36626     |
| 11b   | T3K    | 39.8          | 16541     |
| 11b   | TG     |               |           |
| 70b   | T3K    | 12.0          | 75290     |
| 70b   | TG     |               |           |

## Llama 32-users

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model | Device | Speed (t/s/u) | avg TTFT (ms) |
|-------|--------|---------------|---------------|
| 1b    | N150   | 62.0          | 57            |
| 1b    | N300   | 72.2          | 51            |
| 1b    | T3K    | 73.6          | 55            |
| 1b    | TG     |               |               |
| 3b    | N150   | 37.9          | 87            |
| 3b    | N300   | 47.4          | 72            |
| 3b    | T3K    | 49.3          | 75            |
| 3b    | TG     |               |               |
| 8b    | N150   | 24.3          | 137           |
| 8b    | N300   | 34.3          | 105           |
| 8b    | T3K    | 47.9          | 74            |
| 8b    | TG     |               |               |
| 11b   | N300   | 34.5          | 98            |
| 11b   | T3K    | 47.6          | 80            |
| 11b   | TG     |               |               |
| 70b   | T3K    | 14.9          | 203           |
| 70b   | TG     |               |               |
