# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `accuracy,demo`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values.

## LlamaOptimizations.performance

This configuration uses bfp4 MLP FF1+FF3 for all models.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b | N150 | 79 | 98 | 90.5 |
| 1b | N300 | 81 | 98 | 101.7 |
| 1b | T3K | 81 | 98 | 97.5 |
| 3b | N150 | 85 | 96 | 49.0 |
| 3b | N300 | 88 | 97 | 56.9 |
| 3b | T3K | 88 | 97 | 54.5 |
| 8b | N150 | 86 | 98 | 28.4 |
| 8b | N300 | 84 | 98 | 38.6 |
| 8b | T3K | 84 | 98 | 52.6 |
| 11b | N300 | 86 | 97 | 38.6 |
| 11b | T3K | 84 | 98 | 52.6 |
| 70b | T3K | 95 | 100 | 14.3 |

## LlamaOptimizations.accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b | N150 | 77 | 96 | 85.8 |
| 1b | N300 | 80 | 98 | 98.6 |
| 1b | T3K | 78 | 98 | 97.2 |
| 3b | N150 | 88 | 98 | 44.1 |
| 3b | N300 | 88 | 98 | 53.9 |
| 3b | T3K | 88 | 98 | 54.8 |
| 8b | N150 | 89 | 98 | 23.5 |
| 8b | N300 | 90 | 98 | 34.1 |
| 8b | T3K | 88 | 97 | 49.9 |
| 11b | N300 | 90 | 97 | 33.8 |
| 11b | T3K | 88 | 97 | 52.6 |
| 70b | T3K | 95 | 100 | 14.5 |
