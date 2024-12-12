# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values +- 0.5.

## LlamaOptimizations.performance

This configuration uses bfp4 MLP FF1+FF3 for all models.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b | N150 | 88 | 98 | 85.6 |
| 1b | N300 | 88 | 98 | 93.6 |
| 1b | T3K | 88 | 98 | 90.5 |
| 3b | N150 | 89 | 98 | 46.3 |
| 3b | N300 | 91 | 98 | 52.8 |
| 3b | T3K | 89 | 98 | 52.0 |
| 8b | N150 | 87 | 98 | 27.5 |
| 8b | N300 | 86 | 98 | 36.5 |
| 8b | T3K | 84 | 97 | 46.7 |
| 11b | N300 | 88 | 98 | 36.4 |
| 11b | T3K | 87 | 98 | 46.8 |
| 70b | T3K | 94 | 100 | 13.9 |

## LlamaOptimizations.accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b | N150 | 88 | 98 | 81.7 |
| 1b | N300 | 88 | 98 | 91.5 |
| 1b | T3K | 88 | 98 | 87.8 |
| 3b | N150 | 89 | 98 | 41.9 |
| 3b | N300 | 91 | 98 | 50.4 |
| 3b | T3K | 89 | 98 | 51.4 |
| 8b | N150 | 87 | 98 | 22.9 |
| 8b | N300 | 86 | 98 | 32.8 |
| 8b | T3K | 84 | 97 | 46.0 |
| 11b | N300 | 88 | 98 | 32.4 |
| 11b | T3K | 87 | 98 | 44.1 |
| 70b | T3K | 94 | 100 | 13.9 |
