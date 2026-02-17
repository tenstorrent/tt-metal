# tinyllama (not relevant, for testing)
### tinyllama-char (before fix on embeddings)
1. Memory efficient runner, single device
```
  Forward Pass        :     300.38 ms
  Backward Pass       :     642.71 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      82.58 ms
  Other               :       0.00 ms
  TOTAL               :    1025.67 ms
```

2. Default runner, single device
```
  Forward Pass        :     300.87 ms
  Backward Pass       :     339.97 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      82.63 ms
  Other               :       0.00 ms
  TOTAL               :     723.48 ms
```

### tiny-llama 32k

1. Memory efficient runner, single device
```
  Forward Pass        :     310.49 ms
  Backward Pass       :     655.59 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      92.57 ms
  Other               :       0.00 ms
  TOTAL               :    1058.65 ms

```

2. Default runner, single device
```
  Forward Pass        :     308.22 ms
  Backward Pass       :     351.65 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      92.77 ms
  Other               :       0.00 ms
  TOTAL               :     752.64 ms
```

# Llama 8b
## Experiment 1.1-1.2 Single device forward, backward, optimizer step
### 2 blocks default runner
Number of parameters: 616583168
```
  Forward Pass        :      56.07 ms
  Backward Pass       :      75.36 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      49.11 ms
  Other               :       0.00 ms
  TOTAL               :     180.54 ms
```
==> fwd_2 = 56.1 ms
==> bwd_2 = 75.3 ms
### 4 blocks default runner
Number of parameters: 971018240
```
  Forward Pass        :     101.41 ms
  Backward Pass       :     135.54 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :      73.91 ms
  Other               :       0.00 ms
  TOTAL               :     310.86 ms
```

==> fwd_4 = 101.4 ms
==> bwd_4 = 135.5 ms
==> opt4  = 73.91 ms

==> fwd_per_block = (fwd_4 - fwd_2) / 2 = 22.65
==> bwd_per_block = (bwd_4 - bwd_2) / 2 = 30.10

overhead_fwd = fwd_2 - 2 * fwd_per_block = 10.8 ms
overhead_bwd = bwd_2 - 2 * bwd_per_block = 15.1 ms

==> fwd_time_s = overhead_fwd + 32 * fwd_per_block = 735.6 ms
==> bwd_time_s = overhead_bwd + 32 * bwd_per_block = 978.3 ms

==> opt_time_s = opt_4 * (params_32blocks / params_4blocks) = 608.93 ms

### 8 blocks default runner
Number of parameters: 1679888384
```
  Forward Pass        :     190.70 ms
  Backward Pass       :     255.94 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :     132.98 ms
  Other               :       0.00 ms
  TOTAL               :     579.62 ms

```

error_fwd = abs(fwd_8 - (overhead_fwd + 8 * fwd_per_block)) / fwd_8 = 0.007341373885684351
error_bwd = 0.00015628662967879988
error_opt = abs(opt_8 - (opt_4 * params8 / params4)) / opt_8 = 0.038454338244338596


## Experiment 1.2 - 1.3 Memory / Performance Profiling vs Batch Size
Let's take smaller model with 8 blocks + memory_efficient (it fits in memory)

| Batch Size | Step time (ms) | Peak Memory (GB) | Memory Estimated | Delta memory |
| ---------- | -------------- | ---------------- | ---------------- | ------------ |
| 1          | 751.9          | 11.8             | 11.40            |              |
| 2          | 1345.1         | 12.7             | 12.29            |              |
| 4          | 2403.5         | 14.7             | 14.06            |              |
| 8          | 4681.8         | 18.5             | 17.60            |              |
| 16         | 10500.3        | 26.2             | 24.69            |              |

## Experiment 1.4 Gradient checkpoint overhead
no checkpoint:
```
  Forward Pass        :     190.70 ms
  Backward Pass       :     255.94 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :     132.98 ms
  Other               :       0.00 ms
  TOTAL               :     579.62 ms

```
with checkpoint:
```
  Forward Pass        :     181.34 ms
  Backward Pass       :     423.68 ms
  Gradient Sync       :       0.00 ms
  Optimizer Step      :     112.52 ms
  Other               :       0.00 ms
  TOTAL               :     717.54 ms
```


## Experiment 3. DDP all-reduce
### Llama 8b with 8 blocks with DDP=8, memory efficient
```
  Forward Pass        :     192.25 ms
  Backward Pass       :     440.59 ms
  Gradient Sync       :     365.24 ms
  Optimizer Step      :     132.90 ms
  Other               :       0.00 ms
  TOTAL               :    1130.99 ms
```

==> grad_rd_8 = 365.2 ms

ar_bytes_sent_per_gpu= (2 * (N - 1) / N) / num_links * M
N = number of devices
M = gradients size =  1,679,888,384 param * sizeof(bfloat16) = 3.129 GB
num_links = 2 if ring, 1 if linear topology.

**Effective link BW = (2 * (N - 1) / N) / num_links * M / t = 1.75M / t = 16.1 GB/s == 32% utilization???**

### Llama 8b with 8 blocks with DDP=32, memory efficient
```

```
