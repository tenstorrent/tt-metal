# Model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) with the `batch-1` test case.

## Text Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models.

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen25_vl-3b     | N150         | 31.85         | 752.11    |
| qwen25_vl-3b     | N300         | 39.16         | 532.78    |
| qwen25_vl-32b    | T3K          | 21.91         | 1445.89   |
| qwen25_vl-72b    | T3K          | 13.66         | 2723.91   |

## Vision Performance

Note: Vision performance is still undergoing optimization.

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen25_vl-3b     | N150         | 204.11        | 17520     |
| qwen25_vl-3b     | N300         | 200.08        | 17880     |
| qwen25_vl-32b    | T3K          | 205.84        | 17380     |
| qwen25_vl-72b    | T3K          | 198.59        | 18010     |
