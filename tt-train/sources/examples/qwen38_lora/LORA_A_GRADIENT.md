# Column-parallel LoRA: `lora_A` gradient was never summed across TP

## The problem

`LoraColumnParallelLinear` (`ttml/modules/lora.py`) wraps a column-parallel base layer:

```
y_local = x @ W_localᵀ + scaling · (x @ Aᵀ) @ B_localᵀ
```

- `x` is replicated on every TP device (after `broadcast`, or `sp_gather` under sequence parallelism).
- `lora_A` `[r, in]` is **replicated**: every device holds a full copy.
- `lora_B` `[out, r]` is **column-sharded**: device `d` holds the rows for its output columns `out_d`.

In backward, with `h = x @ Aᵀ` (replicated):

```
dB_d   = dY_dᵀ @ h          correct: it is the gradient of device d's own shard
dh_d   = dY_d @ B_d          partial: only device d's output columns contribute
dA_d   = dh_dᵀ @ x           partial
```

The true gradient is `dA = Σ_d dA_d`. Nothing summed it: `sync_gradients` only reduces over `dp`, and its
docstring assumes that replicated parameters "see identical inputs on every TP rank so their grads match".
That holds for the input, but not for the output gradient that reaches `A` through a sharded `B`.

Consequences:
- Each device updated `lora_A` with roughly 1/tp of its real gradient, a different part on each device.
- After the first optimizer step the 4 copies of `lora_A` differ. From then on every device computes `h`
  with a different `A`, so the model being trained isn't one consistent model.
- The LoRA input gradient (`dh_d @ A`) is still summed by the outer `broadcast`/`sp_gather` backward. It was
  only consistent while the `A` copies were still identical.

Throughput is unaffected (MFU numbers stand). With `--random-init` the effect was invisible. It matters for
any real fine-tuning run.

Not affected:
- `LoraRowParallelLinear`: `A` is row-sharded and `B` replicated. `h` is all-reduced, so `dh` (and therefore
  `dB` and each shard's `dA`) is already complete on every device.
- Plain `LoraLinear` (no TP).

## The fix

`_SumGradAcrossTP` is an autograd function that is the identity in forward and all-reduces the gradient
(sum) over the TP axis in backward. It wraps `lora_A` before the `x @ Aᵀ` matmul:

```python
lora_A = _SumGradAcrossTP.apply(self.lora_A.tensor, self.cluster_axis)
h = ttml.ops.linear.linear(lora_input, lora_A, None)
```

Only the gradient flowing into the parameter is summed. `dx` still uses the partial `dh_d` and is summed
once by the surrounding broadcast/gather, so nothing is double-counted. The cost is one `[r, in]`
all-reduce per column-parallel LoRA layer per step (rank 16 × 5120 bf16 = 160 KB), which is negligible.

## Check

`finetune/finetune_govreport.py --check-lora-a` reads every column-parallel `lora_A` from each of the
4 TP devices after each optimizer step and asserts the copies are bit-identical. Read the copies device
by device (`ttnn.get_device_tensors`): a mesh composer returns one copy for a replicated tensor, so a
check built on it always reports 0.

Measured on the real Qwen3.8-27B checkpoint (4-layer truncation, GovReport, seq 8192, TP=4 + SP,
2 examples per step, lr 1e-4):

| | step 1 | step 2 | steps 3–4 |
|---|---|---|---|
| Without the fix | 0 (`lora_B` starts at 0, so `dA` = 0) | **1.5e-4** max abs difference between copies | (run stops at the assert) |
| With the fix | 0 | 0 | 0 |

Without the fix the copies drift by roughly one AdamW step (~lr) per update from step 2 on. With the
fix they stay identical.
