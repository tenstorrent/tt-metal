# Model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) with the `batch-1` test case.

## Text Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models.

| Model            | Device       | Speed (t/s/u) | TTFT (ms) |
|------------------|--------------|---------------|-----------|
| qwen25_vl-3b     | N150         | 31.85         | 752.11    |
| qwen25_vl-3b     | N300         | 39.16         | 532.78    |
| qwen25_vl-7b     | N300         | 33.16         | 910.55    |
| qwen25_vl-32b    | T3K          | 21.91         | 1445.89   |
| qwen25_vl-72b    | T3K          | 13.66         | 2723.91   |

## Vision Performance

Note: Vision performance is still undergoing optimization.

| Model            | Device       | Speed (t/s/u) | TTFT (ms) |
|------------------|--------------|---------------|-----------|
| qwen25_vl-3b     | N150         | 760.67        | 4700      |
| qwen25_vl-3b     | N300         | 1151.94       | 3110      |
| qwen25_vl-7b     | N300         | 1150.39       | 3110      |
| qwen25_vl-32b    | T3K          | 1137.21       | 3150      |
| qwen25_vl-72b    | T3K          | 818.57        | 4370      |
