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

**`SamplingGenerator.apply_decode_state(chunks, ...)`**: Execute the sampling
half of vLLM's explicit decode update contract. `reload_sampling_params=True`
formats/merges and uploads parameters; `reset_sampling_state=True` rebuilds
prompt/output penalty state. The flags are independent. The method does NOT
advance seeds — callers apply slot remaps first, reset/align seeds when state is
reset, and call `seed_manager.get_new_values()` exactly once per sampled token.
Both command flags are required at every decode call.

This contract includes the unconditional first-decode reseed for `seed=None`
also addressed by
[tt-metal#51556](https://github.com/tenstorrent/tt-metal/pull/51556).
It does not depend on that PR.
`reset_sampling_state=True` calls `reset_seed_from_slots(...)` rather than the
conditional helper, ensuring decode-only sampling actually initializes and
uploads fresh device seeds when both the requested and cached seed are `None`.

## vLLM Decode Update Contract

Refactored vLLM model adapters advertise
`decode_input_update_contract = 1`. The vLLM TT plugin sends these adapters
four boolean commands on every decode:

- `reload_inputs`: copy every forward trace input.
- `reload_page_table`: copy only page-table inputs while preserving
  device-produced token/position state.
- `reload_sampling_params`: upload sampling configuration.
- `reset_sampling_state`: rebuild mutable penalty/RNG state for the layout.

The plugin also sends `slot_remap` as layout data on every version-1 decode,
including host-sampling steps. `slot_remap[i] = j` means every persistent state
owned by new slot `i` must take the continuing request state from old slot `j`
before the forward reads it. This is broader than sampler state: recurrent or
convolution state indexed by decode slot must be remapped too. Stateless
adapters accept and may ignore the value.

Generators execute these commands without adding page-table comparisons,
sampling-mode checks, or model-specific forced reloads. The corresponding
vLLM plugin falls back to the legacy `reset_batch` interface for adapters that
do not advertise the contract, preserving their existing reload and overlap
behavior. vLLM warns that correctness is not guaranteed on that compatibility
path. This lets vLLM land first and adapters opt in as they are refactored. The
marker is negotiation metadata on vLLM-facing adapters only; all refactored
generator APIs require direct callers, including demos and warmup code, to
provide all four commands. No model-side fallback heuristics are restored. Any
demo-side decision to retain traced inputs is made at the call site.

`model_capabilities["supports_async_decode"]` is separate from contract
versioning. It certifies that a vLLM wrapper supports split async readback and
device-resident sampled-token feedback; wrappers without it receive explicit
full-input reload commands instead.

### Requirements for `supports_async_decode=True`

A model wrapper may opt in only if all of the following hold:

- `decode_forward(..., read_from_device=False)` and
  `read_decode_output(..., async_read=True)` split submission from observational
  readback.
- Device sampling writes the selected token into the persistent token input
  consumed by the next decode.
- Decode forward advances the persistent position exactly once; sampling and
  readback never advance it.
- Page tables can be refreshed without copying or rebinding token, position, or
  RoPE trace inputs.
- All four reload commands are honored independently, without model-local
  heuristics escalating page-table-only refresh into a full reload.
- Slot remap applies before the forward to every persistent slot-indexed model
  state in both sampling modes. For device sampling, parameter upload,
  penalty/RNG reset, and seed advancement follow in that order, with one seed
  advance per sampled token.
- Persistent buffers remain valid through deferred readback and until an
  explicit reload replaces them.

If any item is unsupported, leave the capability absent or `False`. vLLM will
disable async scheduling and issue a full input reload for every decode. The
authoritative mode definitions, transition matrix, and correctness invariant
live in the paired vLLM plugin document
`plugins/vllm-tt-plugin/docs/decode-reload-contract.md`.

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
