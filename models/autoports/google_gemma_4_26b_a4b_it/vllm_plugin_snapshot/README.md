# vLLM TT Plugin

> [!CAUTION]
>
> ## This in-tree copy of the plugin is deprecated
>
> The plugin is developed and released from its own repository, against
> **upstream vLLM**:
> [github.com/tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin).
> The copy here is frozen and will be archived with the rest of this fork. Use
> the standalone repository for installation, issues and pull requests.

Tenstorrent backend plugin for vLLM.

`vllm-tt-plugin` integrates Tenstorrent hardware into vLLM using the standard
plugin mechanism. Install it alongside vLLM and, when `ttnn` is importable,
TT hardware is automatically available as a vLLM platform.

The plugin is self-contained: model registration, platform detection, request
validation, scheduling, worker execution, model loading, async decode, single-
process standard and multi-lane execution, gathered data-parallel execution
and `tt-run` / MPI launch orchestration all live here. Nothing TT-specific
needs to touch vLLM
core.

## Package Layout

```text
plugins/vllm-tt-plugin/
+-- src/vllm_tt_plugin/
|   +-- entrypoints.py       # vLLM plugin entry points
|   +-- platform.py          # TTPlatform and config validation
|   +-- model_registry.py    # TT model architecture registration
|   +-- worker.py            # TT worker implementation
|   +-- model_runner.py      # TT model execution bridge
|   +-- scheduler.py         # TT scheduling policy
|   +-- lane_scheduler.py    # Single-process multi-lane (lane-DP) coordinator
|   +-- engine.py            # TT engine core and DP engine processes
|   +-- launcher.py          # tt-run / MPI launch integration
|   +-- loader.py            # TT model loader
|   +-- input_batch.py       # TT input-batch representation
|   +-- async_decode.py      # Decode overlap helpers
|   +-- config.py            # TT plugin config access
+-- docs/                    # TT runtime notes
+-- examples/                # Offline and OpenAI-server examples
+-- tests/tt/                # Server-facing TT plugin tests
```

## Requirements

If testing a specific model, check the
[TT-Metal LLMs table](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms)
for the appropriate tt-metal and vLLM commits.

vLLM requires Python `>=3.10,<3.14`. Python 3.10.12 is the default `python3` on
Ubuntu 22.04.

## Environment Setup

Install tt-metal first by following
[INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
If installing tt-metal from source, build it, create the virtual environment,
and set the environment variables needed for tt-metal tests.

Activate the environment where tt-metal is installed, then install vLLM and the
TT plugin from the vLLM repository root:

```bash
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
```

The script installs base vLLM with `VLLM_TARGET_DEVICE=empty` because `tt` is
provided by this plugin at runtime, not by the base vLLM build. It then installs
the plugin with a few dependencies. Most dependencies come from the
active tt-metal env.

To install or refresh only the plugin package:

```bash
uv pip install -e plugins/vllm-tt-plugin
```

To run the offline Qwen-VL example, also install its extra:

```bash
uv pip install -e "plugins/vllm-tt-plugin[examples]"
```

After the first setup, activate the same environment before running vLLM:

```bash
source "$PYTHON_ENV_DIR/bin/activate"
```

`VLLM_TARGET_DEVICE` is a build-time variable only and does not need to be set at
runtime. The TT platform is detected automatically when `ttnn` is importable.

Optionally install pre-commit hooks for local development:

```bash
pre-commit install
```

## Verify Plugin Discovery

The editable install registers two vLLM entry points:

| Entry point group | Name | Target |
| --- | --- | --- |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |

`platform_plugin()` returns `vllm_tt_plugin.platform.TTPlatform` only when
`ttnn` is importable. This keeps ordinary vLLM environments from accidentally
selecting the TT platform.

Quick checks:

```bash
python -c "import vllm_tt_plugin; print(vllm_tt_plugin.__file__)"
python -c "import ttnn; print('ttnn available')"
```

If `VLLM_PLUGINS` is set, it must allow both TT entry point names:

```bash
export VLLM_PLUGINS=tt,tt_model_registry
```

## Hugging Face Access

To run Meta Llama 3.1 or 3.2 models, request access on Hugging Face:

- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama 3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

After access is approved, create an access token in Hugging Face settings and
log in from Python:

```python
from huggingface_hub import login

login()
```

## Preparing TT-Metal Models

For the target model, follow any setup instructions in the corresponding
tt-metal demo. For Llama 3.1, Llama 3.2, and Qwen 2.5 models, follow the
[tt-transformers demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
for weights and environment variables.

## Running The Offline Inference Example

Run offline generation with the default Llama 3.1 70B model:

```bash
MESH_DEVICE=T3K python plugins/vllm-tt-plugin/examples/offline_inference_tt.py
```

Measure offline performance for one batch of prompts:

```bash
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py --measure_perf
```

To run a different text model, set `MESH_DEVICE` to `N150`, `N300`, `T3K`, `TG`,
or a mesh shape such as `"(4,8)"`, then pass `--model`:

- Llama 3.1 8B: `--model "meta-llama/Llama-3.1-8B"`
- Llama 3.2 1B: `--model "meta-llama/Llama-3.2-1B"`
- Llama 3.2 3B: `--model "meta-llama/Llama-3.2-3B"`
- Qwen 2.5 7B: `--model "Qwen/Qwen2.5-7B"`
- Qwen 2.5 72B: `--model "Qwen/Qwen2.5-72B"`
- DeepSeek R1 Distill Llama 70B: `--model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`
- GPT-OSS 20B: `--model "openai/gpt-oss-20b"`
- GPT-OSS 120B: `--model "openai/gpt-oss-120b"`

For Llama 3.1 8B on N150, set `--max_model_len 32768`; see the tt-metal model
demo for context-length details.

To run Llama 70B on Wormhole Galaxy:

```bash
MESH_DEVICE=TG \
LLAMA_DIR=<path-to-weights> \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "worker_l1_size": 1344544, "trace_region_size": 216580672}}'
```

To run GPT-OSS 20B on Wormhole Galaxy:

```bash
MESH_DEVICE="(4,8)" \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model "openai/gpt-oss-20b" \
  --max_seqs_in_batch 1
```

To run Qwen3-32B on Blackhole Galaxy:

```bash
MESH_DEVICE=TG \
TT_QWEN3_TEXT_VER=qwen3_32b_galaxy \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model "Qwen/Qwen3-32B" \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "worker_l1_size": 1345000, "trace_region_size": 184915840}}'
```

Wormhole Galaxy defaults to `FABRIC_1D_RING` and Blackhole Galaxy defaults to
`FABRIC_2D_TORUS_XY`, so those recipes do not need an explicit `fabric_config`.

Run Llama 3.2 Vision on N300:

```bash
MESH_DEVICE=N300 \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model "meta-llama/Llama-3.2-11B-Vision-Instruct" \
  --multi_modal \
  --max_seqs_in_batch 16 \
  --num_repeat_prompts 8
```

Useful vision-model variants:

- Llama 3.2 11B Vision on QuietBox: set `MESH_DEVICE=T3K` and `--max_seqs_in_batch 32`.
- Llama 3.2 90B Vision: set `MESH_DEVICE=T3K`, `--model "meta-llama/Llama-3.2-90B-Vision-Instruct"`, and `--max_seqs_in_batch 4`.
- Qwen 2.5-VL 32B: set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-32B"`, and `--max_seqs_in_batch 32`.
- Qwen 2.5-VL 72B: set `MESH_DEVICE=T3K`, `--model "Qwen/Qwen2.5-VL-72B"`, `--max_seqs_in_batch 32`, `--max_model_len 2048`, and `--additional-config '{"tt": {"trace_region_size": 28467200}}'`.
- Gemma 3 27B: set `MESH_DEVICE=T3K`, `--model "google/gemma-3-27b-it"`, `--max_seqs_in_batch 32`, `--additional-config '{"tt": {"l1_small_size": 768, "fabric_config": "FABRIC_1D"}}'`, `--multi_modal`, `--multi_image`, and `--mm_processor_kwargs '{"use_fast": true, "do_convert_rgb": true}'`.

For debugging V1, set `VLLM_ENABLE_V1_MULTIPROCESSING=0` to disable
multiprocessing. This is useful for stepping through code or making scheduling
deterministic, but it is not compatible with DP models.

## Running The Server Example

Start the OpenAI-compatible server:

```bash
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/server_example_tt.py
```

Send a completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 32,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
  }'
```

Sampling parameters beyond `temperature`, `top_k`, and `top_p` require
compatibility sampling mode. The compatibility sampling pathway is selected per
batch when any request in the batch requires it.

For vision models, start the server with the correct `--model`, then send a chat
completion request with image content. Qwen 2.5-VL models can use either a
base64 `data:image/...` URL or a real URL such as
`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`.

## Configuration

TT options are passed through vLLM's generic additional config namespace:

```bash
--additional-config '{"tt": {"sample_on_device_mode": "all"}}'
```

Plugin code reads this through `vllm_tt_plugin.config.get_tt_config()`, which
returns `vllm_config.additional_config["tt"]`.

Common options:

| Key | Purpose |
| --- | --- |
| `sample_on_device_mode` | Select on-device sampling mode, currently `all` or `decode_only` when supported by the model. |
| `trace_mode` | Control TT tracing: `all`, `decode_only`, or `none`. Default: `all`. |
| `enable_model_warmup` | Warm up the model before the server reports healthy. Default: `true`. |
| `trace_region_size` | Trace region size for TT runtime tracing. |
| `worker_l1_size` | Worker L1 size override. |
| `l1_small_size` | Small L1 size override. |
| `fabric_config` | Fabric config such as `DISABLED`, `FABRIC_1D`, `FABRIC_2D`, `FABRIC_1D_RING`, `FABRIC_2D_TORUS_XY`, or `CUSTOM`. Any `ttnn.FabricConfig` name is accepted. Defaults: Wormhole Galaxy `FABRIC_1D_RING`, Blackhole Galaxy `FABRIC_2D_TORUS_XY`, other multi-device `FABRIC_1D`. |
| `fabric_reliability_mode` | Fabric reliability mode, such as `STRICT_INIT` or `RELAXED_INIT`. |
| `dispatch_core_axis` | Dispatch core axis, `row` or `col`. |
| `always_compat_sampling` | Use vLLM's LogitProcessor and sampler path even when not required by the batch. Default: `false`. |
| `input_queue_batching_delay` | Short idle delay in seconds to allow more requests to coalesce before TT execution. Default: `0.002`. |
| `optimizations` | Select model/runtime optimization profile, such as `accuracy` or `performance`. |
| `register_test_models` | Register non-production TT test models for infrastructure tests. Default: `false`. |
| `rank_binding` | Rank-binding YAML used for `tt-run` / MPI launches. |
| `mpi_args` | MPI launch arguments, for example host and rankfile settings. |
| `extra_ttrun_args` | Additional raw arguments passed to `tt-run`. |
| `config_pkl_dir` | Shared directory used to pass launch config to remote hosts. |
| `env_passthrough` | Environment variable names or glob patterns propagated to remote hosts. |

## Runtime Architecture

`TTPlatform.check_and_update_config()` is the main handoff from vLLM into the TT
runtime. It validates configuration, registers TT model architectures, and
selects the TT-owned runtime classes through vLLM's extension points:

| vLLM config field | TT implementation |
| --- | --- |
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `parallel_config.engine_core_cls` | `vllm_tt_plugin.engine.TTEngineCore` |
| `parallel_config.engine_core_proc_cls` | `vllm_tt_plugin.engine.TTEngineCoreProc` |
| `parallel_config.dp_engine_core_proc_cls` | `vllm_tt_plugin.engine.TTDPEngineCoreProc` |
| `parallel_config.engine_core_launcher_cls` | `vllm_tt_plugin.launcher.TTCoreEngineLauncher` |
| `scheduler_config.scheduler_cls` | `vllm_tt_plugin.scheduler.TTScheduler` or `vllm_tt_plugin.lane_scheduler.TTLaneCoordinator` |

The execution model matches TT hardware characteristics:

- A TT step is either prefill-only or decode-only.
- Chunked prefill is not used.
- Async scheduling overlaps decode submission with host-side scheduling when
  the model declares support.
- For Galaxy-generator models (Llama3 70B, Qwen3-32B), `--data_parallel_size N`
  runs as `N` in-process TT lanes scheduled by `TTLaneCoordinator` (one engine,
  one device mesh); see [Single-Process Galaxy Serving](#single-process-galaxy-serving).
- For other models, `--data_parallel_size N` uses gathered multi-process DP:
  local rank inputs are collected, executed across the TT mesh, and outputs
  scattered back to the participating ranks.
- Multi-host execution uses `tt-run` / MPI while vLLM sees a normal
  engine-client handshake.

For a deeper walk-through of the scheduling and execution model, read
`docs/SCHEDULING.md`.

## Single-Process Galaxy Serving

Galaxy text models served by the single-execute Galaxy generator
(Llama 3.3 70B via `TT_LLAMA_TEXT_VER=llama3_70b_galaxy`, Qwen3-32B via
`TT_QWEN3_TEXT_VER=qwen3_32b_galaxy`) run on a single Galaxy device mesh, so
they use single-process TT lanes: one vLLM engine process with internal TT
lanes.

Serve them with the familiar `--data_parallel_size N --max_num_seqs M` flags;
the TT backend transparently maps them to `N` in-process lanes. No config
changes are needed. Wormhole Galaxy example (fabric defaults to `FABRIC_1D_RING`):

```bash
MESH_DEVICE=TG \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
VLLM_RPC_TIMEOUT=900000 \
python plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --data_parallel_size 4 \
  --max_num_seqs 8 \
  --async-scheduling \
  --additional-config '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "worker_l1_size": 1344544, "trace_region_size": 220000000}}'
```

`--data_parallel_size 4 --max_num_seqs 8` runs `4` TT lanes of `8` requests
each (`32` concurrent total); `--max_num_seqs` is the per-lane capacity. For
these single-execute Galaxy models this replaces the gathered DP=4 setup they
used historically, so there is nothing to migrate. This conversion is specific
to the Galaxy generators; other model families still run `--data_parallel_size`
as gathered multi-process DP (see [Runtime Architecture](#runtime-architecture)).
At startup the backend logs that it is running single-process lane-DP.

## Supported Model Families

The plugin registers TT-prefixed model architectures backed by tt-metal model
implementations. Current families:

- Llama 3.1 / 3.2 / 3.3 text models (`TTLlamaForCausalLM`)
- Llama 3.2 vision models (`TTMllamaForConditionalGeneration`)
- Qwen 2.5 and Qwen 3 text models (`TTQwen2ForCausalLM`, `TTQwen3ForCausalLM`)
- Qwen 2.5-VL and Qwen 3-VL vision-language models
- Mistral and Mistral 3 multimodal models
- Gemma 3 multimodal models
- DeepSeek V3 (`TTDeepseekV3ForCausalLM`)
- GPT-OSS 20B / 120B (`TTGptOssForCausalLM`)

Model availability, supported device shapes, max sequence limits, and required
environment variables are documented in the corresponding tt-metal model demos.

### Registering models dynamically (`EXTRA_MODELS_DIR`)

Instead of adding a hard-coded line to `platform.py`, a model can be registered at
startup by dropping a **bundle folder** under a directory named by the
`EXTRA_MODELS_DIR` environment variable. Each subfolder holds a `vllm_metadata.json`
and the adapter class (plus its dependencies):

```text
$EXTRA_MODELS_DIR/
  my-model/
    vllm_metadata.json      # {"arch": "<HFArch>", "main_class": "module:Class", ...}
    <adapter class + deps>
```

At import time the plugin scans `EXTRA_MODELS_DIR`, appends each folder to `sys.path`
(so an installed package of the same name is never shadowed), and registers `arch`
under the plugin's `TT`-prefixed convention (`TT<HFArch>`) pointing at `main_class`.
This lets a distribution tool (e.g. `tt-kernel`) deliver a ready-to-serve model with no
source edit to the plugin. The built-in map above stays enabled by default; set
`TT_VLLM_BUILTIN_MODELS=0` to rely solely on `EXTRA_MODELS_DIR`.

## Operational Constraints

`TTPlatform` rejects or adjusts unsupported feature combinations early, giving a
clear error before anything reaches the device:

- Tensor parallel and pipeline parallel execution are not supported.
- Speculative decoding is not currently supported.
- LoRA is not currently supported.
- Chunked prefill is disabled.
- Prompt logprobs are rejected at request validation time.
- Prefix caching is enabled only for models that declare TT support for it.
- Async decode overlap is enabled only for models that declare the capability.

These are TT runtime characteristics, not vLLM plugin API limitations.

## Benchmarking

Offline benchmarking is done by passing `--measure_perf` to
`offline_inference_tt.py`:

```bash
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --measure_perf
```

Client-server benchmarking can be done with `vllm bench serve` after starting
the server:

```bash
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 32 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el
```

For prefix-cache experiments, use prompts with shared prefixes:

```bash
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --prompts_json plugins/vllm-tt-plugin/examples/prompts_overlapping.json
```

You can also pass `--random-prefix-len <N>` to `vllm bench serve`.

## Testing

The plugin ships server-facing tests under `tests/tt`. Start a vLLM server with
a TT model, then run:

```bash
pytest plugins/vllm-tt-plugin/tests/tt -v \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

Tests cover request isolation, sampling behavior, penalties, logprobs,
host-only parameter handling, and TT utility helpers.

Plugin-local unit tests that do not require a running server live directly
under `plugins/vllm-tt-plugin/tests/`, for example:

```bash
pytest plugins/vllm-tt-plugin/tests/test_lane_scheduler.py
```

## Running On Multi-Host Systems

For multi-host offline inference or serving, launch vLLM from the host that has
MPI rank 0, as determined from the rankfile. Under the hood, the plugin uses
`tt-run` from tt-metal to spawn MPI processes on each host.

Example offline inference on two Wormhole Galaxy hosts with DP=2:

```bash
MESH_DEVICE="(8,8)" \
python -u plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model <MODEL_NAME> \
  --data_parallel_size 2 \
  --async_engine \
  --additional-config '{
    "tt": {
      "rank_binding": "<TT_METAL>/tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
      "extra_ttrun_args": "--tcp-interface cnx1",
      "mpi_args": "--host <HOST1>,<HOST2> --map-by rankfile:file=/etc/mpirun/rankfile",
      "config_pkl_dir": "<SHARED_TMP_DIR>",
      "fabric_config": "FABRIC_1D",
      "fabric_reliability_mode": "RELAXED_INIT",
      "env_passthrough": ["VLLM_*", "MESH_DEVICE"]
    }
  }'
```

Notes:

- The `rank_binding` YAML needs an absolute path for `mesh_graph_desc_path`.
- `config_pkl_dir` must be shared by all hosts.
- Environment variables can be propagated through `env_passthrough` in
  `--additional-config` or through `global_env` in the rank-binding file.
- `extra_ttrun_args` passes raw flags to `tt-run`, such as
  `"--tcp-interface cnx1"`, `"--bare"`, or `"--debug-gdbserver"`.

## Hybrid Attention Models

Hybrid attention models have mixed sliding-window and full-attention layers
such as Gemma 3, Gemma 4, and GPT-OSS. They opt in to upstream vLLM's hybrid KV
cache manager through a per-model spec hook on the registered TT model class.

The hybrid manager packs sliding and full layers into separate
`KVCacheGroupSpec`s, sized by upstream's
[Hybrid KV Cache Manager design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).
Sliding-window layers then occupy only `sliding_window` worth of KV state per
request instead of `max_seq_len`. On Gemma 4 31B at 256k context this is roughly
a 6x reduction in KV cache memory.

To enable hybrid KV cache support for a TT model:

1. Inherit from `models.tt_transformers.tt.generator_vllm.HybridAttentionForCausalLM`
   instead of `Generator`. The base class provides a default
   `get_kv_cache_spec` classmethod that builds per-layer specs from
   `hf_config.text_config.layer_types`.
2. Implement `prefill_forward` and `decode_forward` to consume the
   `page_tables_per_group` kwarg and route each layer to the right group's page
   table.
3. Implement `allocate_kv_cache_per_layer(per_layer_specs)`. The base class
   default delegates to `allocate_vllm_kv_cache_per_layer`.

Models that do not opt in stay on the legacy `Generator` path: uniform
single-group KV cache, one page table, and no behavioral change. The plugin only
sends `page_tables_per_group` to model classes that expose `get_kv_cache_spec`.

Hybrid models are not yet supported with `data_parallel_size > 1`; the DP
merged-input gather path collapses to group 0 only. Use DP=1 with hybrid models
until per-group DP gather lands.

## Status

This plugin currently lives alongside a vLLM fork that adds a small set of
generic plugin mechanism extensions not yet present in upstream vLLM. Once those
extensions are part of upstream, this plugin will be fully self-contained and
will work against stock vLLM without any fork.

The extensions needed are:

- `ParallelConfig.engine_core_cls`: lets a platform plugin select an
  alternative `EngineCore` implementation.
- `ParallelConfig.engine_core_proc_cls`: lets a platform plugin select an
  alternative `EngineCoreProc` implementation.
- `ParallelConfig.dp_engine_core_proc_cls`: lets a platform plugin select an
  alternative data-parallel engine process implementation.
- `ParallelConfig.engine_core_launcher_cls`: lets a platform plugin own the
  engine launch and handshake topology.

## Development Notes

- Normal Python changes under `src/vllm_tt_plugin/` take effect after restarting
  the Python or vLLM process.
- Reinstall the plugin when package metadata or entry points change, such as
  edits to `pyproject.toml`.
- Model capability declarations (`model_capabilities` dict on the model class)
  are the preferred way to gate features like async decode and prefix caching,
  rather than hard-coded model-name checks.
