# Model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) with the `batch-1` test case.

## Text Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models.

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen3_vl-32b    | T3K          | 18.72         | 1828.41   |

## Vision Performance

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen25_vl-32b    | T3K          | 1148.41        | 2400     |
