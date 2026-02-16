# Sampling Module

On-device top-k / top-p / temperature sampling with penalties and trace capture for Tenstorrent models.

## File Map

| File | Purpose |
|---|---|
| `generator.py` | `SamplingGenerator` orchestrator; `SamplingParams`; `format_sampling_params`; `broadcast_sampling_params`; `chunk_sampling_params`; `SeedManager` |
| `tt_sampling.py` | `TTSampling` — on-device top-k/top-p/temp with multi-device all-gather |
| `tt_penalties.py` | `TTPenalties` — presence / frequency / repetition penalties |
| `tt_log_probs.py` | `LogProbsCalculator` — log-softmax across sharded vocabulary |
| `_utils.py` | Shared helpers: `clamp`, `is_default_value`, `filter_none`, `split_list` |

See [README.md](README.md) for quick-start code.

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

## Param Distribution API

**`SamplingParams`**: Canonical dataclass for sampling parameters (temp, top_k, top_p, penalties, seed, log_probs). Import from `models.common.sampling`. vLLM has its own duck-type-compatible `TTSamplingParams`.

**`broadcast_sampling_params(params, idx, slot_len=32)`**: Expand a single user's params to fill `slot_len` slots. Used during prefill.

**`chunk_sampling_params(params, sampling_dp)`**: Split a SamplingParams into `sampling_dp` pieces. List fields split evenly; scalars replicated. Works with duck-typed objects (vLLM).

**`SamplingGenerator.apply_prefill_state(...)`**: Reset params, seeds, prompt tokens, and output state for a prefill request.

**`SamplingGenerator.apply_decode_state(chunks, ...)`**: Format/merge params and apply for one model instance. Handles both simple (1 chunk) and row-sharded (multiple chunks) cases. Does NOT advance seeds — callers manage `seed_manager.get_new_values()` separately.

## Pitfalls

**`padded_vocab_size` vs `vocab_size`**: TTPenalties requires per-device vocab width to be tile-aligned (multiple of 32). Models must set `padded_vocab_size` accordingly and pad logits to match. However, TTSampling device offsets for global token IDs use the *unpadded* `vocab_size` — using padded offsets shifts token IDs from devices 1+ and produces garbled output. The split is handled internally; just set both attributes correctly on args.

**`sampling_dp`**: When >1, k/p/temp tensors must have length `max_batch_size * sampling_dp` and are row-sharded via `ShardTensor2dMesh(dims=(0, None))`. Use `chunk_sampling_params` + `apply_decode_state` to distribute params across mesh rows.

**Trace invalidation**: Changing `force_argmax_sampling` state invalidates captured traces. Force-argmax is triggered when callers pass k=1, p=1.0, temp=1.0 (note: p=1.0 means "no top-p filtering", distinct from the internal initialization default of p=0). `SamplingGenerator.reset_sampling_params` handles this.

## Future Work

- Consolidate DeepSeek's minimal `SamplingParams` (in `models/demos/deepseek_v3/tt/generator.py`) to use the common one
