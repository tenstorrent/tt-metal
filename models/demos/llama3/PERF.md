# Llama 3 model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) and accuracy collected from [tests/test_llama_accuracy.py](tests/test_llama_accuracy.py). You can generate this table by running these tests with the `lt` tool (tell it to run `table` or `precision`) and pressing `m` whilst in the results section to export to markdown.

Note that `test_llama_accuracy.py` parses the below to determine expected values +- 0.5.

## LlamaOptimizations.performance

This configuration uses bfp4 MLP FF1+FF3 for all models.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 87        | 98        | 91.0          |
| 1b    | N300   | 87        | 98        | 98.8          |
| 1b    | T3K    | 87        | 98        | 97.8          |
| 1b    | TG     | 88        | 99        | 51.0          |
| 3b    | N150   | 90        | 98        | 49.2          |
| 3b    | N300   | 90        | 98        | 56.8          |
| 3b    | T3K    | 88        | 98        | 54.5          |
| 3b    | TG     | 90        | 97        | 33.5          |
| 8b    | N150   | 86        | 99        | 28.6          |
| 8b    | N300   | 85        | 98        | 38.9          |
| 8b    | T3K    | 84        | 97        | 53.7          |
| 8b    | TG     | 86        | 98        | 29.5          |
| 11b   | N300   | 87        | 98        | 38.6          |
| 11b   | T3K    | 88        | 98        | 52.6          |
| 11b   | TG     | 86        | 98        | 29.5          |
| 70b   | T3K    | 95        | 99        | 14.7          |
| 70b   | TG     | 95        | 100       | 12.7          |


## LlamaOptimizations.accuracy

This configuration uses bfp4 MLP FF1+FF3 only for the 3.1-70B model.

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 89        | 98        | 86.8          |
| 1b    | N300   | 88        | 99        | 98.1          |
| 1b    | T3K    | 86        | 99        | 97.5          |
| 1b    | TG     | 87        | 98        | 51.3          |
| 3b    | N150   | 92        | 100       | 44.2          |
| 3b    | N300   | 92        | 99        | 54.2          |
| 3b    | T3K    | 91        | 98        | 55.6          |
| 3b    | TG     | 91        | 98        | 33.6          |
| 8b    | N150   | 91        | 99        | 23.6          |
| 8b    | N300   | 91        | 99        | 34.5          |
| 8b    | T3K    | 90        | 99        | 49.8          |
| 8b    | TG     | 88        | 100       | 29.5          |
| 11b   | N300   | 91        | 99        | 33.8          |
| 11b   | T3K    | 91        | 99        | 52.6          |
| 11b   | TG     | 88        | 100       | 29.5          |
| 70b   | T3K    | 95        | 99        | 14.7          |
| 70b   | TG     | 95        | 100       | 12.7          |

## LlamaOptimizations.ff1_bfp4_lofi_ff2_bfp4_lofi_ff3_bfp4_lofi

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 84        | 98        | 58.1          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_bfp4_lofi_ff2_bfp8_ff3_bfp4_hifi2

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 87        | 98        | 55.8          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bfp4_lofi_ff2_bf16_hifi4

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 84        | 98        | 53.0          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bfp8_hifi2_ff2_bfp4_lofi

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 86        | 98        | 56.9          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bfp8_hifi2_ff2_bfp8_hifi2

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 86        | 99        | 56.3          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bfp8_hifi2_ff2_bf16_hifi4

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 89        | 99        | 53.7          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bf16_hifi4_ff2_bfp4_lofi

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 85        | 98        | 51.6          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bf16_hifi4_ff2_bfp8_hifi2

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 88        | 98        | 53.0          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |

## LlamaOptimizations.ff1_3_bf16_hifi4_ff2_bf16_hifi4

| Model | Device | Top-1 (%) | Top-5 (%) | Speed (t/s/u) |
|-------|--------|-----------|-----------|---------------|
| 1b    | N150   | 0         | 0         | 0             |
| 1b    | N300   | 90        | 99        | 50.7          |
| 1b    | T3K    | N/A       | N/A       | N/A           |
| 1b    | TG     | N/A       | N/A       | N/A           |
| 3b    | N150   | N/A       | N/A       | N/A           |
| 3b    | N300   | N/A       | N/A       | N/A           |
| 3b    | T3K    | N/A       | N/A       | N/A           |
| 3b    | TG     | N/A       | N/A       | N/A           |
| 8b    | N150   | N/A       | N/A       | N/A           |
| 8b    | N300   | N/A       | N/A       | N/A           |
| 8b    | T3K    | N/A       | N/A       | N/A           |
| 8b    | TG     | N/A       | N/A       | N/A           |
| 11b   | N300   | N/A       | N/A       | N/A           |
| 11b   | T3K    | N/A       | N/A       | N/A           |
| 11b   | TG     | N/A       | N/A       | N/A           |
| 70b   | T3K    | N/A       | N/A       | N/A           |
| 70b   | TG     | N/A       | N/A       | N/A           |
