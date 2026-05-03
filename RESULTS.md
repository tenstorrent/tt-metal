# Gemma 4 26B-A4B Bringup Results

Status: full 30-layer Gemma4 26B-A4B now defaults to the instruct-tuned `google/gemma-4-26B-A4B-it` path for demos. A user-facing instruct CLI runs on the 8-device Blackhole host with Gemma4 turn/channel prompt formatting, paged-attention metadata, trace decode, and device-side decode sampling. The earlier base-weight strict standalone harness remains the strongest evidence for fully on-device token/position feedback across TTNN trace replay.

## Hardware And Software

Host: `bh-lb-01-special-moconnor-for-reservation-67920`

Hardware observed with `/opt/venv/bin/tt-smi -s`: 8x Blackhole `p150b` devices, TT-KMD `2.7.0`, firmware bundle `19.7.0.0`, 16G GDDR per board, AICLK 800 MHz during probes. The strict run also logged an AICLK settle warning: expected 1350 MHz, last observed 800 MHz.

Usable TTNN Python environment: `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python`

Software observed:

- `transformers==4.57.1`
- `huggingface_hub==0.36.0`
- `ttnn.get_num_devices() == 8`
- Built TT-Metal tree used as CWD for hardware runs: `/proj_sw/user_dev/moconnor/tt-metal`
- Local bringup source imported ahead of the built tree: `/localdev/moconnor/tt-metal-gemma-4-26b-a4b`

Important setup note: hardware runs were launched from `/proj_sw/user_dev/moconnor/tt-metal` after importing `ttnn`, then the local bringup repo was inserted into `sys.path`. Running TTNN hardware tests from the local checkout CWD mixed local dispatch kernel sources with the built `/proj_sw` libraries and failed JIT compilation. This is a setup/CWD issue, not a Gemma4 model semantic failure.

## Revisions

Active branch: `yieldthought/gemma4-instruct-vllm-optimization`

Continuation commits:

- `43a9e735a4` (`Default Gemma4 demos to instruct sampling`)
- `d808d669d0` (`Add Gemma4 vLLM paged attention adapter`)
- `313438d3eb` (`Add Gemma4 weight dtype profiling knobs`)
- `f1aefea90f` (`Fix Gemma4 instruct prompt fallback`)

Earlier checkpoint commit: `1ae60f5ac1` (`Checkpoint Gemma4 strict decode bringup`)

Checkpoint: `google/gemma-4-26B-A4B` at `64143b04706fadeb2f8ac198f7ecab57b94b1e0b`.

Instruct checkpoint: `google/gemma-4-26B-A4B-it` at snapshot `4c55b528bdc40b4e79ed7fd4e2f8e46fa5aaed5a`.

Primary HF source reference: HuggingFace Transformers `v5.5.0` at `c1c34249fa27deefbd4a377dfbf883a39baf5c6d`.

Checkpoint index metadata: `26,544,131,376` parameters, `51,611,872,412` bytes, two safetensors shards.

Cached files:

- HF snapshot: `/proj_sw/user_dev/moconnor/hf-cache/hub/models--google--gemma-4-26B-A4B/snapshots/64143b04706fadeb2f8ac198f7ecab57b94b1e0b`
- HF snapshot disk usage with symlinks resolved: `49G`
- TT tensor cache: `/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B`, `57G`
- Instruct HF cache: `/proj_sw/user_dev/moconnor/hf-cache/hub/models--google--gemma-4-26B-A4B-it`, `49G`
- Instruct TT tensor cache: `/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B-it`, `55G`
- TT program/kernel cache used for the original traced-core 1x8 runs: `/tmp/tt-metal-cache-gemma4-full-1x8-50153`, `941M`
- TT program/kernel cache used for strict device-feedback runs: `/tmp/tt-metal-cache-gemma4-strict-full-1x8-0503`
- TT program/kernel cache used for instruct smoke: `/tmp/tt-metal-cache-gemma4-it-smoke-0503`, `941M`

## Source Grounding

Guide-required artifacts exist under `models/demos/gemma4/`: `source_of_truth.md`, `model_dossier.md`, `lowering_specs/`, `mesh_plan.md`, `trace_signatures.md`, `tensor_contracts.json`, `test_matrix.md`, and `perf_log.md`.

Key HF semantics preserved in the current code:

- 30 text decoder layers, hidden size 2816, 16 Q heads.
- Sliding layers use 8 KV heads and head dim 256; full-attention layers use 2 KV heads and global head dim 512.
- Layer pattern is five sliding-attention layers followed by one full-attention layer, repeated.
- Every layer has shared dense MLP plus routed MoE.
- Router uses RMSNorm without learned norm weight, learned scale, softmax over all experts, top-8 selection, selected-probability sum renormalization, and per-expert scale.
- Full-attention layers use `K == V`; there is no separate `v_proj.weight` in the checkpoint for those layers.
- Full-attention RoPE uses proportional partial rotation with identity non-rotary channels.

Code changes made or preserved toward this milestone:

- `Gemma4ModelArgs.load_hf_config` falls back to direct `config.json` parsing when installed Transformers does not know `model_type: gemma4`.
- TT RoPE cache creation no longer imports `transformers.models.gemma4`; it locally implements the v5.5.0 Gemma4 text RoPE formulas.
- Router top-k probabilities are renormalized by selected-probability sum, matching HF.
- Default demo paths now target instruct `google/gemma-4-26B-A4B-it`; base `google/gemma-4-26B-A4B` remains available via explicit override.
- Gemma4 tokenizer loading works around the local Transformers 4.57.1 `extra_special_tokens` list parsing issue.
- Instruct prompt encoding uses `tokenizer.apply_chat_template` when available and otherwise falls back to the upstream Gemma4 `<|turn>` / `<turn|>` / `<|channel>` format.
- Added trace region config for `gemma-4-26B` on `P150` and `P150x8`.
- Added `models/demos/gemma4/demo/strict_device_feedback_demo.py`, a reproducible batch=1 strict decode harness that keeps next-token feedback and both position counters on device across TTNN trace replay.
- Added `models/demos/gemma4/demo/instruct_demo.py`, a user-facing CLI around the text demo.
- Added `models/demos/gemma4/tt/generator_vllm.py`, a minimum vLLM adapter for batch=1, `tt_data_parallel=1`, paged-attention page tables, and mixed sliding/global KV-cache geometry.
- Added dtype profiling knobs for attention, shared MLP, routed experts, and lm head through `GEMMA4_*_WEIGHT_DTYPE`, plus an opt-in `GEMMA4_PRECISION_PROFILE=mixed_bfp8` profile.

## 2026-05-03 Instruct/VLLM/Optimization Update

Instruct snapshot download:

```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="google/gemma-4-26B-A4B-it",
    allow_patterns=[
        "*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ],
    local_files_only=False,
)
print("SNAPSHOT_OK", path)
PY
```

Result: `SNAPSHOT_OK /proj_sw/user_dev/moconnor/hf-cache/hub/models--google--gemma-4-26B-A4B-it/snapshots/4c55b528bdc40b4e79ed7fd4e2f8e46fa5aaed5a`.

Corrected instruct smoke command/log:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-it-smoke-0503 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B-it \
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache \
HF_HUB_OFFLINE=1 \
HF_MODEL=google/gemma-4-26B-A4B-it \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -u \
  /localdev/moconnor/tt-metal-gemma-4-26b-a4b/models/demos/gemma4/demo/instruct_demo.py \
  --model-path google/gemma-4-26B-A4B-it \
  --prompt 'Explain in two sentences why paged attention helps LLM serving.' \
  --max-new-tokens 8 \
  --max-seq-len 512 \
  --mesh-rows 1 \
  --mesh-cols 8 \
  --trace-region-size 50000000 \
  2>&1 | tee /tmp/gemma4_it_instruct_template_smoke.log
```

Evidence from `/tmp/gemma4_it_instruct_template_smoke.log`:

- Full 30-layer instruct model created from warmed BF16 tensor cache in `12.7s`.
- Prompt formatted as Gemma4 instruct fallback, `26` tokens padded to `128`.
- Prefill/TTFT `4863.11ms`; first token `202690 = 'Paged'`.
- Decode trace captured; sampling path reported `sampling=device`.
- 1st traced decode token `69.71ms`, `14.34 tok/s/user`.
- Average traced decode `70.94ms`, `14.1 tok/s/user`.
- Generated text prefix: `Paged attention optimizes LLM serving by managing the`.

Earlier cold instruct run `/tmp/gemma4_it_instruct_demo_smoke.log` proved the full instruct weights and BF16 tensor cache can be generated from scratch: model creation `370.1s`, prefill `206.13s`, decode compile `213.7s`, average decode `100.92ms` at `9.91 tok/s/user`. That run used raw-token fallback before `f1aefea90f` fixed prompt formatting, so it is retained only as cold-cache evidence.

vLLM status:

- `models/demos/gemma4/tt/generator_vllm.py` initializes a Gemma4 vLLM-style model with vLLM page tables and per-layer KV cache shapes.
- Supported today: batch=1, `tt_data_parallel=1`, prefill+decode with paged attention metadata, host fallback sampling only when device sampling is unavailable or explicitly disabled by the integration caller.
- Not supported yet: continuous batching, prefix caching, async decode. The blocker is Gemma4's mixed sliding/global KV-cache geometry and the current minimum integration scope.
- Focused metadata tests passed: `pytest -q models/demos/gemma4/tests/unit/test_vllm_integration.py`.

Weight dtype probes:

- BF16 remains the only default.
- `GEMMA4_PRECISION_PROFILE=bf16` is the implicit default. `GEMMA4_PRECISION_PROFILE=mixed_bfp8` is opt-in and keeps embeddings, norms, router auxiliaries, KV cache, RoPE, and LM head in BF16 while using BFP8 for attention, shared-MLP, and routed-expert projection weights.
- BFP8 probe command set `GEMMA4_ATTENTION_WEIGHT_DTYPE=bfp8`, `GEMMA4_SHARED_MLP_WEIGHT_DTYPE=bfp8`, `GEMMA4_EXPERT_WEIGHT_DTYPE=bfp8`, and `GEMMA4_LM_HEAD_WEIGHT_DTYPE=bfp8` for a one-layer 1x8 smoke. Log: `/tmp/gemma4_it_bfp8_1layer_probe.log`.
- BFP8 generated BFLOAT8_B caches for lm head, attention, shared MLP, and routed experts, then executed prefill and one decode compile. It emitted EOS immediately (`first token: 1 = '<eos>'`), so BFP8 is not quality-usable as a default without numerical work.
- BFP4 probe command set the same four env vars to `bfp4`. Final warm log: `/tmp/gemma4_it_bfp4_1layer_probe_final.log`.
- BFP4 generated and loaded BFLOAT4_B caches for lm head, attention, shared MLP, and routed experts, then executed prefill. It also emitted EOS immediately, so BFP4 is not quality-usable as a default.
- Full mixed-profile probe command set `GEMMA4_PRECISION_PROFILE=mixed_bfp8` for a 30-layer 1x8 instruct smoke. Log: `/tmp/gemma4_precision_ab_mixed_bfp8_full.log`.
- Mixed profile generated all BFP8 projection caches, created the model in `509.1s`, and prefill still selected the expected first token (`202690 = 'Paged'`). Decode did not advance after entering trace/device-sampling decode for over 18 minutes and was stopped with SIGTERM. This distinguishes the all-BFP8 one-layer immediate-EOS result from the conservative mixed profile: mixed is not obviously broken at prefill, but it is not practical or validated enough to be default.

Focused verification after these changes:

```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache HF_HUB_OFFLINE=1 \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -m pytest -q \
  models/demos/gemma4/tests/unit/test_vllm_integration.py \
  models/demos/gemma4/tests/unit/test_optimization_config.py
```

Result: `12 passed, 1 warning in 2.98s`.

Follow-up verification after restoring BF16 as the implicit precision profile:

```bash
HF_HOME=/proj_sw/user_dev/moconnor/hf-cache HF_HUB_OFFLINE=1 \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -m pytest -q \
  models/demos/gemma4/tests/unit/test_optimization_config.py \
  models/demos/gemma4/tests/unit/test_vllm_integration.py
```

Result: `18 passed, 1 warning in 4.57s`.

## Commands

Checkpoint download:

```bash
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="google/gemma-4-26B-A4B",
    revision="64143b04706fadeb2f8ac198f7ecab57b94b1e0b",
    allow_patterns=[
        "*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ],
    local_files_only=False,
)
print(path)
PY
```

Config fallback smoke:

```bash
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -m pytest -q \
  models/demos/gemma4/tests/unit/test_model.py::test_model_config \
  --skip-model-load
```

Full 1x8 long traced decode run:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
exec > >(tee /tmp/gemma4_full_1x8_long.log) 2>&1
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-full-1x8-50153 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B \
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python - <<'PY'
import os
import sys
import time

import ttnn

sys.path.insert(0, "/localdev/moconnor/tt-metal-gemma-4-26b-a4b")
from models.demos.gemma4.demo.text_demo import run_generation

print("RUN_LABEL gemma4_full_1x8_long_trace")
print("RUN_LOG /tmp/gemma4_full_1x8_long.log")
print("MODEL google/gemma-4-26B-A4B")
print("HF_REVISION 64143b04706fadeb2f8ac198f7ecab57b94b1e0b")
print("TT_METAL_CACHE", os.environ.get("TT_METAL_CACHE"))
print("TT_CACHE_PATH", os.environ.get("TT_CACHE_PATH"))
print("LOCAL_REPO /localdev/moconnor/tt-metal-gemma-4-26b-a4b")
print("BUILT_REPO_CWD /proj_sw/user_dev/moconnor/tt-metal")
start = time.perf_counter()
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
except TypeError:
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        None,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), trace_region_size=50_000_000)
try:
    outputs = run_generation(
        mesh_device=mesh_device,
        model_path="google/gemma-4-26B-A4B",
        prompts=["The capital of France is"],
        max_new_tokens=128,
        num_layers=None,
        max_seq_len=512,
        enable_decode_trace=True,
    )
    print("FULL_1X8_LONG_OUTPUTS", outputs)
finally:
    ttnn.close_mesh_device(mesh_device)
print("RUN_WALL_SECONDS", f"{time.perf_counter() - start:.2f}")
PY
```

Sample-output traced run:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
exec > >(tee /tmp/gemma4_full_1x8_samples.log) 2>&1
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-full-1x8-50153 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B \
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python - <<'PY'
import os
import sys
import time

import ttnn

sys.path.insert(0, "/localdev/moconnor/tt-metal-gemma-4-26b-a4b")
from models.demos.gemma4.demo.text_demo import run_generation

prompts = [
    "Write a two-sentence summary of why RoPE is useful in transformers.",
    "List three practical uses for matrix multiplication in machine learning.",
    "Complete this Python docstring: def add(a, b):",
]
print("RUN_LABEL gemma4_full_1x8_sample_trace")
print("RUN_LOG /tmp/gemma4_full_1x8_samples.log")
print("PROMPTS", prompts)
start = time.perf_counter()
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
except TypeError:
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        None,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), trace_region_size=50_000_000)
try:
    outputs = run_generation(
        mesh_device=mesh_device,
        model_path="google/gemma-4-26B-A4B",
        prompts=prompts,
        max_new_tokens=48,
        num_layers=None,
        max_seq_len=512,
        enable_decode_trace=True,
    )
    print("FULL_1X8_SAMPLE_OUTPUTS", outputs)
finally:
    ttnn.close_mesh_device(mesh_device)
print("RUN_WALL_SECONDS", f"{time.perf_counter() - start:.2f}")
PY
```

Strict full 1x8 on-device feedback run:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-strict-full-1x8-0503 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B \
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -u \
  /localdev/moconnor/tt-metal-gemma-4-26b-a4b/models/demos/gemma4/demo/strict_device_feedback_demo.py \
  --model-path google/gemma-4-26B-A4B \
  --max-new-tokens 128 \
  --max-seq-len 512 \
  --mesh-rows 1 \
  --mesh-cols 8 \
  --trace-region-size 50000000 \
  | tee /tmp/gemma4_strict_device_feedback_full_1x8_128_script_warm.log
```

## Trace And Decode Evidence

Primary strict run log: `/tmp/gemma4_strict_device_feedback_full_1x8_128_script_warm.log`

Evidence from that log:

- Full model created from warmed BF16 tensor cache in `13.02s`.
- 1x8 mesh opened with `trace_region_size=50_000_000`.
- Prompt prefill completed in `5.369s`; first token was `496 = ' a'`.
- Strict decode compile produced token `3207`, advanced RoPE/cache positions from `6` to `7`, and wrote the sampled token into a padded device output buffer.
- Decode trace was captured with persistent device buffers for token input, sampled-token output, RoPE position, and cache position.
- `128` TTNN trace replays executed without host token/position copies between iterations.
- Final token buffer was `236858` on all 8 devices.
- Final RoPE and cache positions were `135` on all 8 devices, matching `prompt_len 6 + compile step 1 + 128 trace replays`.
- 1st strict traced replay token: `64.014ms`.
- 128th strict traced replay token: `64.299ms`.
- Average strict traced replay: `64.121ms/token`, `15.595 tok/s/user`.
- Strict run TTFT: `5368.943ms`.
- Full strict harness wall time: `35.78s`.

Secondary traced-core text-generation log: `/tmp/gemma4_full_1x8_long.log`

Evidence from that log:

- Full model created from warmed BF16 tensor cache in `12.5s`.
- 1x8 mesh opened with `trace_region_size=50_000_000`.
- Prompt prefill completed in `4.85s`; first token was `496 = ' a'`.
- Decode trace was captured before replay.
- Generated `128` tokens with `enable_decode_trace=True`.
- 1st traced replay token: `64.01ms`, `15.62 t/s/u`.
- 128th traced replay token: `64.78ms`, `15.44 t/s/u`.
- Average TTFT: `4853.59ms`.
- Average decode speed: `64.8ms/token`, `15.43 tok/s/user`.
- Full demo runtime: `33.49s`.
- Wrapper wall time: `35.49s`.

The default text demo path after trace capture keeps token feedback, embedding, and position preparation on host, copies the single-token inputs into trace-owned device buffers, and replays the traced decoder/lm-head/sampling graph on device. Sampling is on-device for TP >= 2, but the sampled token is read back to host to seed the next iteration in that demo loop. The strict harness instead embeds `token_in` inside the captured graph, samples into a padded device `token_out` buffer, slices lane 0 back into `token_in`, and increments both `uint32` RoPE and `int32` cache positions with `ttnn.plus_one` + `ttnn.copy` inside the trace. Routed expert decode uses `ttnn.sparse_matmul(..., nnz=top_k)` with `top_k=8`, so decode computes active experts only. Prefill still uses the all-experts sparse-matmul pattern.

## Metrics

Primary measured run: `gemma4_strict_device_feedback`, batch size 1, 30 layers, real weights, BF16 tensor cache, `max_new_tokens=128`, `max_seq_len=512`, 1x8 Blackhole `P150x8`, strict device-feedback trace replay enabled.

| Metric | Value |
| --- | ---: |
| Strict TTFT | `5368.943 ms` |
| Strict decode compile | `3.940 s` |
| Strict trace capture | `0.174 s` |
| 1st strict traced replay token | `64.014 ms` |
| 128th strict traced replay token | `64.299 ms` |
| Average strict traced replay | `64.121 ms/token` |
| Average strict decode throughput | `15.595 tokens/sec/user` |
| Strict traced replay tokens | `128` |
| Full strict harness wall time | `35.78 s` |

For comparison, the host-feedback traced-core text demo generated 128 text tokens at `64.8 ms/token`, `15.43 tok/s/user`, with `4853.59 ms` TTFT. An earlier cold strict run, before program cache warmup, replayed 128 strict tokens at `100.193 ms/token`, `9.981 tok/s/user`, with `240839.155 ms` cold TTFT and `283.977 s` strict decode compile time. Those cold numbers are retained only to explain the warmup gap.

## Sample Prompts And Outputs

All samples below were generated by the full 30-layer model on the 1x8 traced-core text demo path. The strict device-feedback harness does not reconstruct the whole generated token stream on host; it verifies on-device token handoff by checking final token and position buffers after trace replay.

Prompt:

```text
The capital of France is
```

Output:

```text
a city of romance, art, and history. It is also a city of fashion, food, and culture. Paris is a city that has something for everyone.

The city of Paris is a place where you can find the best of everything. It is a place where you can find the best of everything.

Paris is a city that has something for everyone. It is a place where you can find the best of everything.

Paris is a city that has something for everyone. It is a place where you can find the best of everything.

<h2><strong>The Best Time to Visit Paris</strong></h2>

Paris is a city that is
```

Prompt:

```text
Write a two-sentence summary of why RoPE is useful in transformers.
```

Output:

```text
RoPE is useful in transformers because it allows the model to capture the relative position of words in a sentence, which is important for understanding the meaning of the sentence.

What is the main idea of the paper?

The main idea of
```

Prompt:

```text
List three practical uses for matrix multiplication in machine learning.
```

Output:

```text
What is the purpose of the dot product in matrix multiplication?

What is the purpose of the dot product in matrix multiplication?

What is the purpose of the dot product in matrix multiplication?

What is the purpose of the dot product in
```

Prompt:

```text
Complete this Python docstring: def add(a, b):
```

Output:

```text
""" Adds two numbers together. :param a: The first number to add. :type a: int :param b: The second number to add. :type b: int :return: The sum of the two numbers. :rtype:
```

Quality note: this is the base, non-instruct checkpoint under greedy/on-device sampling. The outputs are coherent enough to show the path is not numerically blank or degenerate, but they repeat and do not behave like an instruction-tuned chat model.

## Memory And DRAM Notes

- No OOM occurred on 8x `p150b` with 16G GDDR per board and `trace_region_size=50_000_000`.
- Full checkpoint is BF16/dequantized; no BFP4 or packed expert weights were used.
- TT tensor cache on disk is `57G`.
- TT program/kernel cache for the original traced-core full 1x8 path is `941M`.
- Expert weights are tensor-parallel across 8 devices. Example cached shape from logs: `layer_0/moe/experts/gate_proj_tp8_dtype_BFLOAT16_layout_TILE.tensorbin` has shape `[1, 128, 2816, 96]`.
- Live per-device DRAM high-water was not captured. Add a detailed buffer/memory report in the next perf pass before making tighter memory claims.

## Remaining Caveats

- Local-CWD pytest hardware runs currently fail because TTNN JIT picks dispatch source files from the local checkout while linking to `/proj_sw` built libraries. The reliable run pattern is documented above.
- The strict on-device decode proof is in `strict_device_feedback_demo.py`, not the default text-returning `text_demo.py` loop. The default demo still stages token feedback from host so it can collect sample text.
- The strict harness only records final token/position evidence, not the full generated sequence. A future quality harness should maintain a device-side generated-token buffer or intentionally read back outside the timed replay region.
- Prefill still computes all experts; active-expert sparsity is used in decode.
- The sample pass uses greedy/on-device sampling on the base model. Quality checks should be repeated with a more representative prompt set and any intended sampling policy.
