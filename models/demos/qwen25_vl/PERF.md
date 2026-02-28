# Model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) with the `batch-1` test case.

## Configuration

| Parameter | Text Performance | Vision Performance |
|-----------|------------------|-------------------|
| Batch size | 1 | 1 |
| Prompt file | `demo/sample_prompts/text_only.json` | `demo/sample_prompts/demo.json` |
| Image resolution | N/A | 2048×1365 pixels (demo.jpeg) |
| Vision tokens | N/A | ~3,577 tokens (98×146 patches, 14×14 patch size, 4×4 merge) |
| Input sequence length (ISL) | Short text prompt | ~3,577 vision tokens + text prompt |
| Output sequence length (OSL) | Up to 200 tokens | Up to 200 tokens |
| Max sequence length | 4096 | 4096 |
| Precision | bfp4 MLP, bfp8 attention | bfp4 MLP, bfp8 attention |

## Text Performance

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
