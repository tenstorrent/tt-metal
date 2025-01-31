# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values +- 0.5.

## Performance

This configuration uses bfp4 MLP FF1+FF3 for all models.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| Llama3.2-1B    | N150   | 90        | 98        | 91.0          |
| Llama3.2-1B    | N300   | 90        | 98        | 98.8          |
| Llama3.2-1B    | T3K    | 88        | 98        | 97.8          |
| Llama3.2-1B    | TG     | 88        | 99        | 51.0          |
| Llama3.2-3B    | N150   | 91        | 98        | 49.2          |
| Llama3.2-3B    | N300   | 90        | 98        | 56.8          |
| Llama3.2-3B    | T3K    | 91        | 98        | 54.5          |
| Llama3.2-3B    | TG     | 90        | 97        | 33.5          |
| Llama3.1-8B    | N150   | 88        | 99        | 28.6          |
| Llama3.1-8B    | N300   | 85        | 98        | 38.9          |
| Llama3.1-8B    | T3K    | 84        | 97        | 53.7          |
| Llama3.1-8B    | TG     | 86        | 98        | 29.5          |
| Llama3.2-11B   | N300   | 87        | 98        | 38.6          |
| Llama3.2-11B   | T3K    | 88        | 98        | 52.6          |
| Llama3.2-11B   | TG     | 86        | 98        | 29.5          |
| Llama3.1-70B   | T3K    | 95        | 99        | 14.7          |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |
| Qwen2.5-7B     | N300   | 81        | 96        | 37.9          |
| Qwen2.5-72B    | T3K    | 99        | 100       | 12.8          |


## LlamaOptimizations.accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the Llama-3.1-70B model and the Qwen-2.5-72B model.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| Llama3.2-1B    | N150   | 89        | 98        | 86.8          |
| Llama3.2-1B    | N300   | 88        | 100       | 98.1          |
| Llama3.2-1B    | T3K    | 90        | 99        | 97.5          |
| Llama3.2-1B    | TG     | 87        | 98        | 51.3          |
| Llama3.2-3B    | N150   | 92        | 100       | 44.2          |
| Llama3.2-3B    | N300   | 92        | 99        | 54.2          |
| Llama3.2-3B    | T3K    | 90        | 99        | 55.6          |
| Llama3.2-3B    | TG     | 91        | 98        | 33.6          |
| Llama3.1-8B    | N150   | 91        | 99        | 23.6          |
| Llama3.1-8B    | N300   | 92        | 99        | 34.5          |
| Llama3.1-8B    | T3K    | 91        | 99        | 49.8          |
| Llama3.1-8B    | TG     | 88        | 100       | 29.5          |
| Llama3.2-11B   | N300   | 91        | 99        | 33.8          |
| Llama3.2-11B   | T3K    | 91        | 99        | 52.6          |
| Llama3.2-11B   | TG     | 88        | 100       | 29.5          |
| Llama3.1-70B   | T3K    | 95        | 99        | 14.7          |
| Llama3.1-70B   | TG     | 95        | 100       | 12.7          |
| Qwen2.5-7B     | N300   | 81        | 96        | 33.4          |
| Qwen2.5-72B    | T3K    | 99        | 100       | 12.8          |
