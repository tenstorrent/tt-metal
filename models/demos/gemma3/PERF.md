# Model performance and accuracy

Performance collected from [demo/text_demo.py](demo/text_demo.py)/[demo/vision_demo.py](demo/vision_demo.py) and accuracy collected from [tests/text_demo.py](tests/text_demo.py) with `-k token-matching` flag.

Note: Accuracy and perf numbers have to be gathered in separate runs. For accuracy measurements we have to disable tracing. Perf measurements are obtained with tracing enabled.
Note: these are provisional numbers only for base functional bring up of Gemma-3. No effort has been made yet to improve performance for this model.

## Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| gemma-3-4b-it     | N150        | 85        | 96        | 34            | 64        |
| gemma-3-4b-it     | N300        | 85        | 96        | 36.02         | 89.95     |
| gemma-3-27b-it    | T3K         | 91        | 99        | 16.91         | 466.7     |

## Vision Performance

This configuration uses bfp4 MLP and bfp8 attention weights for all models except:

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| gemma-3-4b-it     | N150        | N/A       | N/A       | 25.84         | 920.04    |
| gemma-3-4b-it     | N300        | N/A       | N/A       | 30.49         | 670.83    |
| gemma-3-27b-it    | T3K         | N/A       | N/A       | 14.54         | 975.86    |



## Accuracy

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| gemma-3-4b-it     | N150        | 91        | 99        | 30            | 76        |
| gemma-3-4b-it     | N300        | 90        | 99        | 33.5          | 110.75    |
| gemma-3-27b-it    | T3K         | 94        | 99        | 15.88         | 493.36    |

## Vision Accuracy

This configuration uses bfp8 MLP and BF16 attention weights (70B+ models use bfp8 attention and bfp4 MLP).
Llama 3 models test as insensitive to attention precision and so we use bfp8 attention and kv-cache for them even in accuracy mode.

| Model             | Device      | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|-----------|-----------|---------------|-----------|
| gemma-3-4b-it     | N150        | N/A       | N/A       | 24.39         | 948.9     |
| gemma-3-4b-it     | N300        |  N/A      |  N/A      | 29.55         | 678.13    |
| gemma-3-27b-it    | T3K         | N/A       |  N/A      | 13.84         | 995.06    |
