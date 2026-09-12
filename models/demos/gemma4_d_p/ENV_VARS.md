# Gemma4 environment variables

Variables read directly by `gemma4_d_p`, plus the external offline setting used
in its README. Defaults below reflect the current code.

## Model and cache paths

| Variable | Default | Purpose |
|---|---|---|
| `HF_MODEL` | Unset | Hugging Face model ID or local checkpoint directory. Set this for the demo; `create_tt_model(model_path=...)` can override it. |
| `TT_CACHE_PATH` | Local checkpoint directory, or `$HF_HOME/tt_cache/<model-id-with-slashes-replaced-by-double-hyphens>` | Root for TT weight caches. |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache location; also supplies the fallback TT-cache root. |
| `HOME` | Process environment | Used indirectly by home-directory expansion for the default Hugging Face cache. |
| `HF_HUB_OFFLINE` | Unset | External Hugging Face setting used in the README: `1` requests offline operation with cached model artifacts. Not read directly by Gemma code. |

Cache directories must already exist; path resolution does not create or mirror them.

Sources: `tt/common.py`, `tt/model_config.py`, `demo/text_demo_prefill.py`.

## Prefill and demo

| Variable | Default | Purpose |
|---|---|---|
| `GEMMA4_MAX_SEQ_LEN` | Test context length | Overrides the model's KV-cache context capacity in the demo. |
| `GEMMA4_PREFILL_TRACE_REGION_SIZE` | `256_000_000` bytes | Device trace-memory reservation for demo fixtures; read at module import. |
| `GEMMA4_PREFILL_LOAD_FULL_WEIGHTS` | `0` | Force full checkpoint/state-dict loading instead of the cache-first path. |
| `GEMMA4_PREFILL_L1_ACT` | `0` | Use L1 instead of DRAM for short-lived attention activations selected by `prefill_short_lived_memcfg()`. |

Sources: `demo/text_demo_prefill.py`, `tt/attention/operations.py`.

## Communication

| Variable | Default | Purpose |
|---|---|---|
| `GEMMA4_CCL_TOPOLOGY` | Ring | `linear`, `line`, or `l` selects Linear for TP collectives. Ring attention independently uses Linear. |
| `GEMMA4_CCL_ASYNC` | `0` | Use asynchronous reduce-scatter + all-gather for TP all-reduce. |
| `GEMMA4_CCL_PERSISTENT_BUF` | `1` | Reuse async collective destination buffers. Synchronous all-reduce ignores this. |
| `GEMMA4_CCL_CHUNKS_PER_SYNC` | `10` | Async collective packet grouping; clamped to at least 1. |
| `GEMMA4_CCL_NUM_WORKERS` | `2` | Async collective workers per link; clamped to at least 1. |
| `GEMMA4_CCL_NUM_BUFFERS` | `2` | Async collective buffers per channel; clamped to at least 1. |

Source: `tt/ccl.py`. Link count defaults to 2 on Blackhole and 1 otherwise.
Explicit constructor arguments override the manager defaults.
Boolean enable flags accept `1`, `true`, or `yes` (case-insensitive).
`GEMMA4_CCL_PERSISTENT_BUF` is instead disabled by `0`, `false`, or `no`.
