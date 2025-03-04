# Llama 3 model performance and accuracy

Performance collected from [demo/simple_text_demo.py](demo/simple_text_demo.py) and accuracy collected from [tests/test_accuracy.py](tests/test_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_accuracy.py` parses the below to determine expected values +- 0.5.

Also note that all the performance metrics below were taken for a maximum generation of 200 tokens, i.e., 200 decode iterations.

## Performance

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 128 tokens.**

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N150   | 88        | 98        | 84.5          | 58        |
| Llama3.2-1B    | N300   | 91        | 98        | 100.5         | 54        |
| Llama3.2-1B    | T3K    | 89        | 98        | 113.8         | 41        |
| Llama3.2-1B    | TG     | 88        | 99        | 51.0          |           |
| Llama3.2-3B    | N150   | 92        | 95        | 52.4          | 76        |
| Llama3.2-3B    | N300   | 92        | 97        | 65.3          | 56        |
| Llama3.2-3B    | T3K    | 91        | 97        | 65.4          | 64        |
| Llama3.2-3B    | TG     | 90        | 97        | 33.5          |           |
| Llama3.1-8B    | N150   | 88        | 100       | 27.8          | 121       |
| Llama3.1-8B    | N300   | 88        | 100       | 43.3          | 85        |
| Llama3.1-8B    | T3K    | 88        | 100       | 62.3          | 69        |
| Llama3.1-8B    | TG     | 86        | 98        | 29.5          |           |
| Llama3.2-11B   | N300   | 90        | 99        | 42.8          | 84        |
| Llama3.2-11B   | T3K    | 87        | 99        | 61.2          | 75        |
| Llama3.2-11B   | TG     | 86        | 98        | 29.5          |           |
| Llama3.1-70B   | T3K    | 97        | 100       | 16.3          | 182       |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |           |
| Qwen2.5-7B     | N300   | 80        | 96        | 37.9          |           |
| Qwen2.5-72B    | T3K    | 98        | 100       | 12.8          |           |
| Phi3.5-mini    | N150   |           |           | 43.2          | 98        |
| Phi3.5-mini    | N300   |           |           | 57.8          | 62        |
| Phi3.5-mini    | T3K    |           |           | 48.8          | 51        |


## Accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model and the Qwen-2.5-72B model. **Batch_size=1 and prefill_length is 128 tokens.**

| Model          | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|-----------|-----------|---------------|-----------|
| Llama3.2-1B    | N150   | 91        | 98        | 82.0          | 55        |
| Llama3.2-1B    | N300   | 91        | 98        | 98.6          | 59        |
| Llama3.2-1B    | T3K    | 88        | 98        | 114.1         | 42        |
| Llama3.2-1B    | TG     | 87        | 98        | 51.3          |           |
| Llama3.2-3B    | N150   | 94        | 99        | 47.0          | 83        |
| Llama3.2-3B    | N300   | 90        | 98        | 61.1          | 64        |
| Llama3.2-3B    | T3K    | 92        | 98        | 65.2          | 63        |
| Llama3.2-3B    | TG     | 91        | 98        | 33.6          |           |
| Llama3.1-8B    | N150   | 93        | 100       | 24.8          | 160       |
| Llama3.1-8B    | N300   | 94        | 100       | 37.8          | 100       |
| Llama3.1-8B    | T3K    | 94        | 100       | 59.8          | 79        |
| Llama3.1-8B    | TG     | 88        | 100       | 29.5          |           |
| Llama3.2-11B   | N300   | 92        | 100       | 37.5          | 97        |
| Llama3.2-11B   | T3K    | 95        | 100       | 59.2          | 64        |
| Llama3.2-11B   | TG     | 88        | 100       | 29.5          |           |
| Llama3.1-70B   | T3K    | 98        | 100       | 14.1          | 210       |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |           |
| Qwen2.5-7B     | N300   | 80        | 96        | 33.4          |           |
| Qwen2.5-72B    | T3K    | 99        | 100       | 12.8          |           |
| Phi3.5-mini    | N150   |           |           | 38.8          | 92        |
| Phi3.5-mini    | N300   |           |           | 53.9          | 63        |
| Phi3.5-mini    | T3K    |           |           | 48.6          | 53        |

##  Long-context (64K Tokens)

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=1 and prefill_length is 64k tokens.**

| Model          | Device | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|---------------|-----------|
| Llama3.2-1B    | N150   | 53.0          | 20191     |
| Llama3.2-1B    | N300   | 65.2          | 10973     |
| Llama3.2-1B    | T3K    | 73.7          | 5271      |
| Llama3.2-1B    | TG     |               |           |
| Llama3.2-3B    | N150   | 25.3          | 46936     |
| Llama3.2-3B    | N300   | 34.8          | 23115     |
| Llama3.2-3B    | T3K    | 41.0          | 10727     |
| Llama3.2-3B    | TG     |               |           |
| Llama3.1-8B    | N150   | 16.9          | 65083     |
| Llama3.1-8B    | N300   | 26.1          | 36422     |
| Llama3.1-8B    | T3K    | 38.1          | 16287     |
| Llama3.1-8B    | TG     |               |           |
| Llama3.2-11B   | N300   | 26.1          | 36422     |
| Llama3.2-11B   | T3K    | 38.4          | 16288     |
| Llama3.2-11B   | TG     |               |           |
| Llama3.1-70B   | T3K    | 11.9          | 74363     |
| Llama3.1-70B   | TG     |               |           |
| Qwen2.5-7B     | N300   |               |           |
| Qwen2.5-72B    | T3K    |               |           |

## Short-Context, Batch-32

This configuration uses bfp4 MLP FF1+FF3 for all models. **Batch_size=32 and prefill_length is 128 tokens.**

| Model          | Device | Speed (t/s/u) | avg TTFT (ms) |
|----------------|--------|---------------|---------------|
| Llama3.2-1B    | N150   | 54.7          | 55            |
| Llama3.2-1B    | N300   | 64.2          | 48            |
| Llama3.2-1B    | T3K    | 69.9          | 57            |
| Llama3.2-1B    | TG     |               |               |
| Llama3.2-3B    | N150   | 36.5          | 84            |
| Llama3.2-3B    | N300   | 45.8          | 68            |
| Llama3.2-3B    | T3K    | 47.8          | 71            |
| Llama3.2-3B    | TG     |               |               |
| Llama3.1-8B    | N150   | 22.3          | 134           |
| Llama3.1-8B    | N300   | 33.5          | 93            |
| Llama3.1-8B    | T3K    | 45.6          | 79            |
| Llama3.1-8B    | TG     |               |               |
| Llama3.2-11B   | N300   | 33.4          | 100           |
| Llama3.2-11B   | T3K    | 45.1          | 76            |
| Llama3.2-11B   | TG     |               |               |
| Llama3.1-70B   | T3K    | 14.8          | 192           |
| Llama3.1-70B   | TG     |               |               |
| Qwen2.5-7B     | N300   |               |               |
| Qwen2.5-72B    | T3K    |               |               |
