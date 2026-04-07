# GPT OSS Token Accuracy Testing

## Why this change

GPT OSS previously measured token accuracy using a trivial prompt — `"What are the prime factors of 1?"` — with on-the-fly CPU reference generation inside `tests/accuracy/test_model.py`. This had several problems:

1. **Weak signal.** A short, simple prompt exercises a narrow slice of the model. High accuracy on it does not guarantee the model is faithful across harder, longer text.
2. **No cross-repo comparability.** `tt_transformers` (and `tt-xla`) use a standardised benchmark: a 128-token window from *A Tale of Two Cities* encoded into `.refpt` files that store both the reference token sequence and the CPU-generated top-5 logits. Without the same benchmark, GPT OSS accuracy numbers could not be compared 1:1 against other model implementations.

The goal was to bring GPT OSS in line with the established `tt_transformers` methodology: pre-generated `.refpt` reference files, teacher-forced decoding on a hard literary prompt, and top-1 / top-5 accuracy reported in the same way so results are directly comparable.

## What is teacher forcing

During normal autoregressive generation the model feeds its own predicted token back as input for the next step. Errors compound — one wrong token shifts the entire continuation, making accuracy hard to measure.

Teacher forcing breaks this feedback loop. After each decode step the *reference* (ground-truth) token is substituted in place of the model's prediction before the next forward pass. Every position therefore sees the correct history, and each token prediction is an independent measurement of model fidelity.

## How the `.refpt` reference files work

Each `.refpt` file is a PyTorch checkpoint (`torch.save`) produced by running a CPU reference model over the *Tale of Two Cities* text (stored compressed as `tale-of-two-cities.txt.bz2` in `tt_transformers`). It contains:

| Key                | Shape          | Description |
|--------------------|----------------|-------------|
| `reference_tokens` | `[1, 128]`    | Full token sequence (first half = prefill prompt, second half = decode ground truth) |
| `top5_tokens`      | `[127, 5]`    | Top-5 predicted token IDs at each position (from CPU logits) |

At load time the 128-token sequence is split at the midpoint: the first 64 tokens become the prefill input and the remaining 64 become the decode reference. `top5_tokens` is sliced correspondingly so index 0 aligns with the first decode position.

The same `.refpt` format and generation script (`models/tt_transformers/tests/generate_reference_outputs.py`) is shared across `tt_transformers`, `tt-xla`, and now GPT OSS.

## What was changed

### `models/demos/gpt_oss/demo/text_demo.py`

- **`TokenAccuracy` class** — loads a `.refpt` file by model name, splits reference tokens into prefill/decode halves, implements `collect_predicted_tokens()` (returns ground-truth token for teacher forcing) and `compute_accuracy()` (top-1 and top-5).
- **`"token_accuracy"` parametrize entry** added to `test_gpt_oss_demo` with the correct constraints: `batch_size=1`, `max_generated_tokens=64`, `temperature=0`, `enable_decode_trace=False`, `stop_at_eos=False`.
- **Teacher forcing hook** in the decode loop — before `decode_forward`, `token_acc.collect_predicted_tokens(out_tok[0].item())` records the prediction and overwrites `out_tok[0]` with the reference token.
- **Accuracy reporting** after the decode loop prints top-1 and top-5 percentages.

### `models/demos/gpt_oss/tests/reference_outputs/`

- **`gpt-oss-20b.refpt`** — pre-generated reference file for the 20B model.
- **`gpt-oss-120b.refpt`** — pre-generated reference file for the 120B model.

Both were generated using CPU weights and the same *Tale of Two Cities* window used by `tt_transformers`.

### `models/demos/gpt_oss/tests/accuracy/test_model.py` (pre-existing, unchanged)

The original accuracy test using `"What are the prime factors of 1?"` remains in place. It still serves as a quick smoke test with on-the-fly CPU reference generation. The new `.refpt`-based test is a separate, more rigorous benchmark that runs through the main demo entry point.

## How to run

```bash
# GPT OSS 20B — 1x8 mesh
pytest models/demos/gpt_oss/demo/text_demo.py -k "token_accuracy and mesh_1x8" -v

# GPT OSS 120B — 4x8 mesh
pytest models/demos/gpt_oss/demo/text_demo.py -k "token_accuracy and mesh_4x8" -v
```

## Key constraints

- **Trace must be off** (`enable_decode_trace=False`) — teacher forcing replaces the input token dynamically each step, which is incompatible with traced execution.
- **Batch size = 1** — teacher forcing only applies to user 0.
- **Greedy sampling** (`temperature=0`) — deterministic output is required for meaningful accuracy measurement.
- **`stop_at_eos=False`** — the test must run all 64 decode tokens regardless of EOS.

## TODO

- [ ] Run first test and record baseline top-1/top-5 numbers for 20B and 120B
- [ ] Set accuracy thresholds and enable `run_in_ci=True`
- [ ] Add a GPT OSS-specific `generate_reference_outputs.py` (to regenerate `.refpt` when model weights change)
