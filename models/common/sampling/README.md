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
from models.common.sampling import (
    SamplingGenerator,
    chunk_sampling_params,
    format_sampling_params,
)

sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl)

# Prefill: one call sets params, seeds and penalty history for these slots.
sampling.apply_prefill_state(
    sampling_params=format_sampling_params(user_params, max_batch_size=32),
    prompt_tokens=prompt_tokens,    # torch tensor shaped [B, S]
    empty_slots=[0, 1, 2],
)

# Decode: apply_decode_update owns the ordering the contract requires. Do not
# drive reset_sampling_params / the seed manager directly from a generator.
sampling.apply_decode_update(
    chunk_sampling_params(user_params, sampling_dp),
    reload_sampling_params=True,
    reset_sampling_state=True,
    seeds=format_sampling_params(user_params, sampling.tt_sampling.max_batch_size).seed,
    active_slots=[0, 1, 2],
)

tt_tokens = sampling.sample(
    tt_logits,
    tt_out_tok=tt_out_buffer,
)
```

`reset_seed` and `reset_seed_from_slots` live on `SeedManager`, not on
`SamplingGenerator`, and differ in whether their `seeds` argument is in request
order or device-slot order. Reach for them only from inside the sampling module.

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
plus `apply_decode_update(...)`, which chunks through to `apply_decode_state`.

## Param Distribution API

**`SamplingParams`**: Canonical dataclass for sampling parameters (temp, top_k, top_p, penalties, seed, log_probs). Import from `models.common.sampling`. vLLM has its own duck-type-compatible `TTSamplingParams`.

**`broadcast_sampling_params(params, idx, slot_len=32)`**: Expand a single user's params to fill `slot_len` slots. Used during prefill.

**`chunk_sampling_params(params, sampling_dp)`**: Split a SamplingParams into `sampling_dp` pieces. List fields split evenly; scalars replicated. Works with duck-typed objects (vLLM).

**`SamplingGenerator.apply_prefill_state(...)`**: Reset params, seeds, prompt tokens, and output state for a prefill request.

**`SamplingGenerator.apply_decode_update(chunks, ...)`**: The entry point a
generator calls for one decode's sampling update. It owns the order the contract
requires: parameter upload, then penalty state, then the seed reset, then at most
one seed advance. Per-family code supplies the slot layout and nothing else.

- `reload_sampling_params` / `reset_sampling_state`: the two commands, passed
  through as received.
- `seeds`: slot-indexed seed values padded to the sampler's batch size. Required
  whenever either command is set and `active_slots` is non-empty.
- `active_slots`: device slots sampling this step. `None` means every slot; an
  empty list means none, which skips the reset entirely (a reset marks the
  manager dirty even when it moves nothing, forcing a full-batch upload next
  step).
- `positions`: slot-indexed absolute decode positions, consulted **only** on a
  state reset. Host positions are authoritative only on a reloading step, and
  vLLM guarantees a state reset always accompanies one. Aligning on a
  params-only update would tie the RNG stream to a position that lags the device.
- `advance_seeds`: `False` for generators that advance inside their sampling
  call, which is the only way to keep one advance per sampled token when state
  application and sampling are separate calls.

Do not drive `reset_seed_from_slots` / `align_seed_counters_to_positions`
directly: a per-family copy of the ordering silently changes reproducibility, and
a test asserts no generator does.

Pass **raw, unformatted** chunks. `format_sampling_params` inverts temperature and
is not idempotent, so pre-formatting inverts it back. Pass `None` only when the
caller uploaded the parameters itself (Galaxy does, because its inactive-slot fill
needs params already padded to the device batch).

**`SamplingGenerator.apply_decode_state(chunks, ...)`**: The parameter and penalty
half, called by `apply_decode_update`. `reload_sampling_params=True`
formats/merges and uploads parameters; `reset_sampling_state=True` rebuilds
prompt/output penalty state. It does not touch seeds. A state reset without
`prompt_tokens` keeps the existing prompt mask, since penalties cannot rebuild it
without the real tokens; the output counters are cleared either way.

## vLLM Decode Update Contract

The command semantics live in
[`models/common/decode_contract.py`](../decode_contract.py), next to the helper
that adapters use to reject commands they cannot execute. The mode definitions,
transition matrix, and negotiation table are owned by the paired vLLM plugin
document
[`plugins/vllm-tt-plugin/docs/decode-reload-contract.md`](https://github.com/tenstorrent/vllm/blob/dev/plugins/vllm-tt-plugin/docs/decode-reload-contract.md)
in the tenstorrent/vllm repository. This section
records only what a tt-metal generator author has to honor.

Adapters advertise support with `decode_input_update_contract = 1`, declared on
the base generator that implements the commands. The plugin then sends the four
commands plus `slot_remap` on every decode, including host-sampling steps.

`slot_remap[i] = j` means every persistent state owned by new slot `i` must take
the continuing request state from old slot `j` before the forward reads it. This
is broader than sampler state: recurrent, convolution, and RoPE state indexed by
decode slot must be remapped too. Stateless adapters accept and may ignore the
value.

Every slot-owning subsystem consumes the remap exactly once on every accepted
version-1 decode. State read by the forward is remapped before that read. A
dormant sampler may consume it after successful decode/readback, which preserves
retry safety because slot remaps are non-idempotent. An authoritative rebuild may
replace the remap for that subsystem; inactivity may not. Version-0 adapters
retain their historical remap behavior unchanged.

**One exemption**, and only one: state that is not addressable by vLLM slot cannot
be remapped, only reset. The known case is unseeded on-device RNG, whose state is
a per-core hardware PRNG register (`PRNG_SEED_Seed_Val`) with no move primitive.
`SeedManager.apply_slot_remap` returns early on an all-unseeded batch for exactly
that reason, which also skips its host-side per-slot bookkeeping, since that
bookkeeping only describes seeds nothing has registered. An adapter with any other
unmovable slot state must say so here, next to the code that skips the remap.
Inconvenience is not an exemption.

**Host input authority** (vLLM requirement 7, the one adapter authors most often
break): `tokens` and `start_pos` are authoritative only when `reload_inputs` is
true. When it is false they are deliberately one step behind, and nothing may be
derived from them, forward inputs or sampling state. Deriving an RNG counter from
`start_pos` on a steady step makes the sampled stream depend on when the async
readback landed, so the same request and seed stop reproducing. `positions` is
therefore consulted only on a state reset, which vLLM guarantees implies a reload.

Generators execute these commands without adding page-table comparisons,
sampling-mode checks, or model-specific forced reloads. An adapter that cannot
execute part of the contract rejects those combinations loudly
(`require_full_input_reload`) rather than silently degrading; advertising version 1
while quietly ignoring a command is an adapter bug. Rejecting is conformant only
while the adapter leaves `supports_async_decode` off, which is what stops vLLM
from planning the combination.

### Version-0 holdouts

This list is authoritative for which tt-metal generator stacks still take the
legacy path, and the paired vLLM document points here for it.

- [`models/common/models/executor.py`](../models/executor.py), driving
  `Llama3Generator` and its eleven `models/common/models/*/generator.py` siblings.
  It still infers reloads from its own call history and from a `torch.equal`
  page-table comparison, and declares no marker, so vLLM keeps it on the legacy
  `reset_batch` path. Converting it needs host tests for that executor stack which
  do not exist yet. A test asserts none of these files declares the marker, since
  doing so without converting the executor makes the first vLLM decode a bare
  `TypeError`.

### Driving these adapters from vLLM

The plugin falls back to the legacy `reset_batch` interface for adapters that do
not advertise the contract, preserving their existing reload and overlap
behavior, and logs that correctness is not guaranteed on that path. This lets
vLLM land first and adapters opt in as they are refactored.

**A vLLM predating [tenstorrent/vllm#458](https://github.com/tenstorrent/vllm/pull/458)
cannot drive a contract adapter.** What it does depends on the tier:

- The model-tier generators (`models/tt_transformers/tt/generator.py`,
  `models/demos/llama3_70b_galaxy/tt/generator.py`) reject `reset_batch` by name
  with a `ValueError` naming the fix. They cannot simply accept the call: their
  commands carry host-authoritative defaults, so an old vLLM's layout changes
  would be silently reinterpreted as a full reload on every step.
- The three wrapper-tier adapters that declare the commands as required
  keyword-only arguments (DeepSeek, T3000 Llama, Mllama) raise `TypeError` for the
  missing arguments.
- The remaining wrappers, including Qwen2.5-VL and Qwen3-VL, absorb `**kwargs` and
  forward to a model-tier generator, so they fail the same way it does: by name on
  `reset_batch`, or not at all on a host-sampling step that sends neither
  `reset_batch` nor the commands. Such a step takes the host-authoritative
  defaults, which is the conservative shape, so it is accepted rather than
  rejected.

Update the vLLM pin along with tt-metal. In the plugin the pin lives in the
deployment's vLLM checkout, not in this repo.

vLLM sends all four commands explicitly on every contract decode. A direct caller
(demo, warmup, accuracy test) may omit one, in which case it takes the
host-authoritative default: `reload_inputs=True` and the other three `False`. That
is the only permitted default, and a test enforces it across every adapter. No
model-side fallback heuristics are restored, and any decision to retain traced
inputs is made at the call site.

`model_capabilities["supports_async_decode"]` is a separate key from contract
versioning. It certifies that a vLLM wrapper supports split async readback and
device-resident sampled-token feedback; wrappers without it receive explicit
full-input reload commands instead. The two are not fully orthogonal: leaving the
capability off is what makes a rejecting adapter safe.

### Requirements

The normative requirement list is owned by the paired vLLM document, which splits
it into obligations of any version-1 adapter (5, 6, 8, 9) and the additional ones
needed to advertise `supports_async_decode` (1 to 4, and 7). Do not restate it
here: a second copy in a second repo loses clauses, and the clause it lost last
time was requirement 7.

What this repo adds on top, all sampling-side ordering:

- Parameter upload, then penalty state, then the seed reset, then at most one seed
  advance per sampled token. `apply_decode_update` owns that order; supply the
  slot layout and let it run.
- Slot remaps are applied before `apply_decode_update`, because they are not
  idempotent and the layer that owns the device-slot mapping owns the timing.
- A global `slot_remap` is rebased onto rank-local slots before it reaches
  per-rank state; use `rank_local_slot_remap`, which also checks that the plugin
  and the model agree on the per-rank stride.

If any requirement is unsupported, leave the capability absent or `False`. vLLM
will disable async scheduling and issue a full input reload for every decode.
Record the reason next to the capability: it is a deliberate throughput trade, and
without a stated reason someone will flip it back.

## Pitfalls

**`padded_vocab_size` vs `vocab_size`**: TTSampling device offsets for global token IDs must use the padded vocab size to match how the LM head shards logits across devices. Using unpadded `vocab_size` for offsets shifts token IDs from devices 1+ and produces garbled output.

**Padded vocab logits**: If the LM head pads output weights beyond the real tokenizer vocabulary, the sampler must mask those padded token IDs before force-argmax or local top-k. Zero-padded LM-head weights are useful for legal sharded matmul shapes, but they are not a sampling mask.

**`sampling_dp`**: When >1, k/p/temp tensors must have length `max_batch_size * sampling_dp` and are row-sharded via `ShardTensor2dMesh(dims=(0, None))`. Use `chunk_sampling_params` + `apply_decode_update` to distribute params across mesh rows.

**Batched prefill + on-device sampling**: This path is only valid when the
runtime prefill compute layout matches the sampling-group layout. If a model
uses `sampling_dp > 1` but does not expose a row-sharded batched-prefill input
contract, batched prefill must fall back to sequential prefill for correctness.

**Trace invalidation**: Changing `force_argmax_sampling` state invalidates captured traces. Force-argmax is triggered when callers pass k=1, p=1.0, temp=1.0 (note: p=1.0 means "no top-p filtering", distinct from the internal initialization default of p=0). `SamplingGenerator.reset_sampling_params` handles this.

## Future Work

- Consolidate DeepSeek's minimal `SamplingParams` (in `models/demos/deepseek_v3/tt/generator.py`) to use the common one
