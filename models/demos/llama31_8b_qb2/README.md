# Llama 3.1-8B-Instruct on Blackhole QuietBox 2

A tensor-parallel implementation of all 32 Llama 3.1-8B decoder layers on a
four-device Blackhole mesh (`P300x2`, two P300c cards). The model uses TTNN for
prefill, decode, the language-model head, and device sampling. vLLM supplies the
scheduler and OpenAI API through `tenstorrent/vllm-tt-plugin`.

The implementation comes from an agentic research bring-up. Its selected
[precision policy](config/precision.json) is fixed: BFP8 QKV, output, down, and
head weights; BFP4 packed gate/up weights; BF16 activations; BFP8 K/V cache; and
HiFi2 head matmuls. RMSNorm affine weights are folded into the following linear
weights in float32 before rounding to BF16.

## Runtime

`tt/decoder.py` owns one decoder layer and its decode workspace. All layers reuse
the same four projection buffers. `tt/model.py` streams the checkpoint one layer
at a time and builds the full model. `tt/generator.py` owns the live traces,
resident token feedback, and sampling state. `tt/generator_vllm.py` implements
the plugin's versioned input-update contract: page-table growth can update the
mapping without overwriting queued token feedback.

The model supports batches up to 32, 128-token cache pages, and a shared
131,072-token cache arena. That arena is shared across requests; it is not
131,072 tokens for each of 32 requests. Prefix caching and scheduler-driven
chunked prefill are disabled. The decoder handles prompt chunks internally.

Device sampling considers the top 32 candidates. The plugin routes penalties,
logprobs, and other unsupported sampling options through its host sampler.
Host and device sampling use different random streams, so seeded output is not
guaranteed to stay identical when a request changes sampling mode.

## Run

Build tt-metal and activate its Python environment. Install the current vLLM TT
plugin with its `docs/install-vllm-tt.sh` script. Cache the gated Hugging Face
checkpoint at revision `0e9e39f249a16976918f6564b8830bc894c89659`, or set
`LLAMA_MODEL_PATH` to that local snapshot. The model opens its checkpoint
offline. Compiled kernels are cached beneath `TT_METAL_CACHE`.

The plugin changes are in [vllm-tt-plugin #116](https://github.com/tenstorrent/vllm-tt-plugin/pull/116).
Until that dependency merges, both scheduled and manual workflow runs and the
registry command default to `yieldthought/llama31-qb2-serving`. The workflow's
plugin-ref input or `VLLM_TT_PLUGIN_REF` can select another branch or tag.

```bash
export TT_LLAMA_TEXT_VER=llama31_8b_qb2 MESH_DEVICE=P300x2
export LLAMA_MODEL_PATH=$(python -c 'from models.demos.llama31_8b_qb2.tt.model import checkpoint_path; print(checkpoint_path())')
python -m vllm.entrypoints.openai.api_server \
    --model "$LLAMA_MODEL_PATH" \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --block-size 128 --max-num-seqs 32 --max-model-len 131072 \
    --max-logprobs -1 --async-scheduling \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":268435456,"l1_small_size":16384}}'
```

The model declares its 1D ring fabric and 8192-byte router payload in
`model_capabilities["fabric_config"]`. The plugin applies these defaults before
opening the mesh; they do not need to be repeated in the launch command.

## Verification and weekly CI

The **Agentic Research Model Tests** workflow runs every Saturday at 07:00 UTC.
Select `llama3.1-8b-qb2` and `bh_quietbox_2` for a manual run. Its Tier 3 entry
has a 10-minute test budget, independent of the daily model pipelines.

The [CI entry](../../../tests/pipeline_reorg/agentic_research_model_tests.yaml)
contains all setup, server, test, and cleanup commands. To reproduce CI, run its
`cmd` block from the repository root on an exclusive QB2, with the checkpoint
cached and `HF_TOKEN` available for the gated evaluation dataset.

The critical decoder test compares real checkpoint weights against Hugging Face
at `(batch, prompt length)` values `(1, 129)`, `(9, 1025)`, and `(32, 128)`.
It requires PCC ≥ 0.99, exact replay after physical page remapping, and exact
replay through batch-family changes. It does not establish full-context
accuracy at 131,072 tokens.

The serving benchmark uses Meta's first 28 IFEval rows, in published order,
twice. Prompts and scoring instructions come from pinned dataset revisions;
inputs and outputs are saved alongside the summary. Each prompt is warmed with
one output token before timing. The same 56 serial streaming requests supply
both accuracy and throughput measurements, with greedy decoding, seed 42,
natural EOS, and a 1,280-token output limit.

The benchmark requires mean IFEval accuracy ≥ 75%, mean per-request decode
throughput ≥ 110 tokens/s/user, and identical text on both passes. IFEval
accuracy is the mean of its strict/loose prompt and instruction scores. Decode
throughput excludes the first token; aggregate throughput includes full request
latency. The central accuracy target is the research baseline of 80.47%, with
7% relative tolerance. This small fixed subset is a regression check, not a
full IFEval evaluation.

A prior QB2 source-build run of the decoder checks and serving benchmark completed
in 6 minutes 33 seconds:
83.47% mean IFEval accuracy, 131.24 decode tokens/s/user, and identical text across
both passes. The server's Python dependencies were already installed; the
scorer environment and model kernel cache were fresh. These are measurements of
this fixed workload, not guarantees for other request lengths or concurrency.

The weekly command now selects the whole model test directory, including the
device token-history wrap test and host regressions for generation readback and
hardware admission. The expanded command needs a fresh QB2 run; the prior timing
does not measure those added checks.

The scorer runs in a separate environment to preserve the serving dependency
set. `tests/report.py` converts its saved summary into the standard tt-metal
benchmark artifact, so results and centralized target validation use the
existing CI infrastructure.
