# Sampling Module

On-device top-k / top-p / temperature sampling with penalties and trace
capture for Tenstorrent models.

## File Map

| File | Purpose |
|---|---|
| `generator.py` | `SamplingGenerator` -- high-level orchestrator (owns TTSampling + TTPenalties, trace capture); also `format_sampling_params`, `SeedManager` |
| `tt_sampling.py` | `TTSampling` -- low-level on-device sampling with multi-device all-gather |
| `tt_penalties.py` | `TTPenalties` -- presence / frequency / repetition penalties |
| `tt_log_probs.py` | `LogProbsCalculator` -- numerically stable log-softmax across sharded vocabulary |
| `_utils.py` | Shared helpers: `clamp`, `is_default_value`, `filter_none` |

## Integration Steps

### 1. Prepare the args object

Your model's args object must expose `vocab_size` and `cluster_shape` at
minimum:

```python
class ModelArgs:
    vocab_size = 128256
    padded_vocab_size = 128256  # must be divisible by TP device count
    cluster_shape = (4, 8)      # (rows, cols) of the device mesh
    max_batch_size = 32         # per sampling row
    max_top_k = 32
    sub_core_grids = None       # or a CoreRangeSet
```

Optional args attributes: `sampling_all_gather_axis` (default 0),
`sub_core_grid_topk`, `start_core`, `sampling_dp` (>1 for multi-row DP),
`model_config` (dict with keys `GALAXY_NUM_LINKS`,
`DECODE_SAMPLING_INPUT_MEMCFG`, `SAMPLING_AG_CONFIG`).

### 2. Create the SamplingGenerator

```python
from models.common.sampling import SamplingGenerator, format_sampling_params

sampling_gen = SamplingGenerator(
    args=args,
    mesh_device=mesh_device,
    tt_ccl=tt_ccl,
    enable_internal_trace=True,
)
```

### 3. Format and set sampling parameters

`SamplingParams` is defined in `models/tt_transformers/tt/generator.py`.
Pad/clamp before passing to the generator:

```python
from models.tt_transformers.tt.generator import SamplingParams

raw = SamplingParams(temperature=0.6, top_k=10, top_p=0.9)
formatted = format_sampling_params(raw, max_batch_size=args.max_batch_size)
sampling_gen.reset_sampling_params(formatted)
```

### 4. Run sampling in decode loop

```python
tt_tokens, tt_log_probs = sampling_gen.sample(
    tt_logits,
    enable_trace=True,
    tt_out_tok=tt_output_buffer,  # optional pre-allocated output
)
```

### 5. Seeds and penalties (optional)

```python
# Seeds
sampling_gen.seed_manager.reset_seed(seeds=[42], user_ids=[0])
sampling_gen.seed_manager.get_new_values(empty_slots=range(32))

# Penalties (activate automatically when non-default values are set)
sampling_gen.reset_prompt_tokens(prompt_tokens)   # torch [B, S]
sampling_gen.reset_output_state(output_tokens)     # torch [B, S] or None
```

## Multi-Row Data Parallel Sampling

For models using multiple mesh rows (e.g., GPT-OSS on 4x8 Galaxy with 4
independent rows of 32 users each = 128 total):

1. Set `args.sampling_dp = 4`
2. Provide k/p/temp tensors of size `max_batch_size * sampling_dp` (e.g., 128)
3. The module row-shards parameters with `ShardTensor2dMesh(dims=(0, None))`
