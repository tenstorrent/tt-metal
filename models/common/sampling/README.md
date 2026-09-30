# Sampling Module Overview

The `models.common.sampling` package bundles everything needed to run on-device
sampling (top-k / top-p / temperature/ seed) plus presence/frequency/repetition
penalties with optional trace capture.

## Key Components
- `SamplingGenerator`: high-level class that owns both `TTSampling` and
  `TTPenalties`, exposes helper methods to reset sampling parameters, penalties,
  prompt/output state, and to run sampling with or without trace capture.
- `format_sampling_params`: utility that pads/clamps sampling parameters to the
  hardware-friendly layout expected by `TTSampling`.
- `LogProbsCalculator`: computes per-token log-probabilities across a sharded
  vocabulary using numerically stable log-softmax (global max / sum-exp
  reduction across devices).

## Quick Start
```python
from models.common.sampling import SamplingGenerator, format_sampling_params

sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl)

params = format_sampling_params(user_params, max_batch_size=32)
sampling.reset_sampling_params(params)

sampling.reset_seed(seed)

sampling.reset_prompt_tokens(prompt_tokens)   # torch tensor shaped [B, S]
sampling.reset_output_state(output_tokens)

tt_tokens = sampling.sample(
    tt_logits,
    tt_out_tok=tt_out_buffer,
)
```

`SamplingGenerator.sample()` accepts `enable_trace=True` to record/replay
sampling traces.

## Trace output lifetime

Construct `SamplingGenerator` before capturing any model or sampling trace. It
allocates token output buffers up front, alongside the existing log-probability
buffers. Compatible sampling trace keys and decode buckets reuse these buffers;
argmax and regular sampling keep their existing, different output shapes.

`precompile()` and `capture_trace()` use the same explicit token output. An
optional output changes the program-cache key, so an enclosing model trace must
also supply the token output used for its sampling warmup. Caller-provided
`tt_out_tok` buffers must likewise exist before related traces are captured.

Traced device results are **borrowed**, not snapshots. Read or copy tokens before
another trace or warmup writes the same token buffer. Log-probabilities are reused
by both traced and eager sampling. For retention, complete a host read or copy
into separate device storage allocated before capture. An asynchronous read must
be ordered before the next write; merely retaining a Python tensor reference does
not preserve its contents.

`reset_trace()` releases sampling traces but retains the sampler-owned token
buffers, because model traces can remain live. Reinitialize the sampler and its
related traces to change the configured sampling batch. Inputs must already be
padded to that batch; trace capture never lazily allocates a new output shape.

The eager `sample(enable_trace=False)` path still allocates its token output when
`tt_out_tok` is omitted. Direct `TTSampling` calls retain their existing behavior.

## File Map

| File | Purpose |
|---|---|
| `generator.py` | `SamplingGenerator` orchestrator; `SamplingParams`; `format_sampling_params`; `broadcast_sampling_params`; `chunk_sampling_params`; `SeedManager` |
| `tt_sampling.py` | `TTSampling` — on-device top-k/top-p/temp with multi-device all-gather |
| `tt_penalties.py` | `TTPenalties` — presence / frequency / repetition penalties |
| `tt_log_probs.py` | `LogProbsCalculator` — log-softmax across sharded vocabulary |
| `_utils.py` | Shared helpers: `clamp`, `is_default_value`, `filter_none`, `split_list` |

## Required `args` Attributes

```python
vocab_size: int           # actual vocabulary size (unpadded)
cluster_shape: tuple      # (rows, cols) of the device mesh, e.g. (4, 8)
```

Optional (with defaults):

```python
padded_vocab_size: int    # tile-aligned total vocab; defaults to vocab_size
max_batch_size: int       # per sampling row; default 32
max_top_k: int            # default 32
sampling_dp: int          # >1 for multi-row DP; default 1
sub_core_grids            # CoreRangeSet or None
model_config: dict        # keys: GALAXY_NUM_LINKS, DECODE_SAMPLING_INPUT_MEMCFG, SAMPLING_AG_CONFIG
```

## `data_parallel` vs `sampling_dp`

These are different concepts and should not be mixed:

- **`data_parallel`** lives above this package. It means multiple TT model
  instances / submeshes process different requests in parallel.
- **`sampling_dp`** lives inside this package. It means one TT model instance
  has multiple independent sampling groups, usually one per mesh row.

For `sampling_dp > 1`:
- logits are still computed per sampling group
- but sampling params, seeds, and penalty state are flattened to
  `max_batch_size * sampling_dp`
- those flattened host tensors are then row-sharded onto the device

Decode already follows this contract by using `chunk_sampling_params(...)`
plus `apply_decode_state(...)`.

## Param Distribution API

**`SamplingParams`**: Canonical dataclass for sampling parameters (temp, top_k, top_p, penalties, seed, log_probs). Import from `models.common.sampling`. vLLM has its own duck-type-compatible `TTSamplingParams`.

**`broadcast_sampling_params(params, idx, slot_len=32)`**: Expand a single user's params to fill `slot_len` slots. Used during prefill.

**`chunk_sampling_params(params, sampling_dp)`**: Split a SamplingParams into `sampling_dp` pieces. List fields split evenly; scalars replicated. Works with duck-typed objects (vLLM).

**`SamplingGenerator.apply_prefill_state(...)`**: Reset params, seeds, prompt tokens, and output state for a prefill request.

**`SamplingGenerator.apply_decode_state(chunks, ...)`**: Format/merge params and apply for one model instance. Handles both simple (1 chunk) and row-sharded (multiple chunks) cases. Does NOT advance seeds — callers manage `seed_manager.get_new_values()` separately.

## Pitfalls

**`padded_vocab_size` vs `vocab_size`**: TTSampling device offsets for global token IDs must use the padded vocab size to match how the LM head shards logits across devices. Using unpadded `vocab_size` for offsets shifts token IDs from devices 1+ and produces garbled output.

**Padded vocab logits**: If the LM head pads output weights beyond the real tokenizer vocabulary, the sampler must mask those padded token IDs before force-argmax or local top-k. Zero-padded LM-head weights are useful for legal sharded matmul shapes, but they are not a sampling mask.

**`sampling_dp`**: When >1, k/p/temp tensors must have length `max_batch_size * sampling_dp` and are row-sharded via `ShardTensor2dMesh(dims=(0, None))`. Use `chunk_sampling_params` + `apply_decode_state` to distribute params across mesh rows.

**Batched prefill + on-device sampling**: This path is only valid when the
runtime prefill compute layout matches the sampling-group layout. If a model
uses `sampling_dp > 1` but does not expose a row-sharded batched-prefill input
contract, batched prefill must fall back to sequential prefill for correctness.

**Trace invalidation**: Changing `force_argmax_sampling` state invalidates captured traces. Force-argmax is triggered when callers pass k=1, p=1.0, temp=1.0 (note: p=1.0 means "no top-p filtering", distinct from the internal initialization default of p=0). `SamplingGenerator.reset_sampling_params` handles this.

## Future Work

- Consolidate DeepSeek's minimal `SamplingParams` (in `models/demos/deepseek_v3/tt/generator.py`) to use the common one
