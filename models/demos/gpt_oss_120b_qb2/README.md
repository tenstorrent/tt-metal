# GPT-OSS 120B on four Blackhole chips

This implementation shards GPT-OSS 120B across a 1×4 Blackhole mesh. It uses
indexed sparse experts, chunked prefill, sliding-window KV rings and device
sampling. Adam Housman developed the original implementation.

**Draft integration:** the model source is imported from the measured
`1800a9402b5e976a48bbf6b3dfa05a7cbc803953` integration. Native operation tests,
host contracts and a complete 100-token single-prompt reference test passed at
that earlier source boundary with the transfer setting below. Those results do
not qualify this current-main tree. Serving measurements also used recorded
fixed-width/sequential-prefill overlays; they do not establish the default
variable-bucket path. Full quality, matched baseline/performance, weekly CI and
two-cycle reopen qualification remain unfinished. The source package's accuracy
and throughput numbers are not validation of this code.

The model's decode wrapper and vLLM adapter use current main's explicit input,
page-table and sampling-state commands. Source-method host regressions cover
async token feedback, page-only refresh, teacher forcing, trace recapture and
sampling-state reuse. These checks use CPU tensors and mock device operations;
they do not qualify execution on this tree. Integrate the separately reviewed
prerequisites and validate the combined tree before running the recipes below.

## Configuration

| Property | Selected configuration |
| --- | --- |
| Checkpoint | `openai/gpt-oss-120b`, revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| Decoder | All 36 layers; 1×4 tensor parallelism, ring fabric |
| Maximum context | 131,072 tokens, including output |
| Decode buckets | 1, 4, 8 and 32; requests 9–16 use the 32-row bucket |
| Serving capacity | At most 32 sequences; shared full-attention pool of 4,640 blocks of 64 tokens |
| Sliding attention | A 768-token ring per device slot; 128-token attention window |
| Weights | BF4 experts; BF8 attention and LM head; BF16 router, embedding and norms |
| Precision policy | [precision.json](precision.json), required at startup |

The pool reserves one output block per admitted sequence. At 32 sequences the
scheduler can hold 294,912 prompt/context tokens across users; it cannot hold 32
full-length contexts simultaneously. Check available disk before loading: the
checkpoint occupies about 61 GiB and the selected decoder tensor cache about
63 GiB, with additional space needed for terminal tensors and conversion.

## Prerequisites

Use matching tt-metal Python/native artifacts and its supported SFPI compiler.
The integration requires these native changes:

- [Chunked SDPA sinks and sliding-window masks, #58436](https://github.com/tenstorrent/tt-metal/pull/58436).
- [Blackhole router topology, #58472](https://github.com/tenstorrent/tt-metal/pull/58472).
- [Indexed sparse matmul bias, #58480](https://github.com/tenstorrent/tt-metal/pull/58480).

The shared-generator changes are reviewed separately:

- [Compact prefill request/page ownership, #58504](https://github.com/tenstorrent/tt-metal/pull/58504).
- [Decode output rows across changing buckets, #58542](https://github.com/tenstorrent/tt-metal/pull/58542).

Indexed sparse bias is merged. The other prerequisites above are not included
in this model-only import and must land through their normal review/check gates.

Serving uses the current official [vLLM TT plugin installation procedure](https://github.com/tenstorrent/vllm-tt-plugin#installation)
with merged [input/slot integration #152](https://github.com/tenstorrent/vllm-tt-plugin/pull/152)
and [host RNG ownership #154](https://github.com/tenstorrent/vllm-tt-plugin/pull/154).
The former submission-width proposal #151 was superseded by Metal #58542 and
is not a dependency. Keep serving and evaluation dependencies in separate
environments when their version requirements differ.

Cached weights require `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` on the tested
QB2 (KMD 2.10.0, IOMMU enabled). The default pinned path timed out uploading a
read-only tensor cache; the copy path completed. This is an explicit runtime
configuration, also used upstream for the weight-load regression tracked in
[#57763](https://github.com/tenstorrent/tt-metal/issues/57763). A shared root cause
for the timeout and that performance issue has not been established. Keep this
setting in both test and server environments until the pinned path is qualified.

## Standalone reference test

Download the pinned checkpoint through the normal Hugging Face client, excluding
`original/*` and `metal/*`. Set the paths below to your own snapshot, writable
cache and independent reference artifact. The reference SHA256 is checked by the
test; it is not an output produced by this implementation.

```bash
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export GPT_OSS_120B_SNAPSHOT=/path/to/pinned/snapshot
export TT_METAL_CACHE=/path/to/writable/gpt-oss-120b-cache
export GPT_OSS_120B_REFERENCE=/path/to/aime24_chat_100_top100.refpt
export GPT_OSS_120B_RESULTS=/path/to/results

python -m pytest models/demos/gpt_oss_120b_qb2/tests/test_full_model.py -v
```

The test checks all 100 reference positions through prefill and traced,
teacher-forced decode. It requires top-5 agreement ≥0.98 and top-100 agreement
1.0. It includes one warmup and three further decode runs, and checks that device
sampling does not read back full logits. These are one-prompt token-agreement
checks; full AIME evaluation is separate.

Layer state tests use independent Transformers 5.12.1 CPU references. Generate
these in a separate CPU Torch environment from the same verified checkpoint.
Each generation streams weights one layer at a time and saves a hash manifest;
the boundary reference executes all 36 HF layers. Outputs belong outside the
checkout.

```bash
python models/demos/gpt_oss_120b_qb2/tests/generate_reference.py \
  --snapshot "$GPT_OSS_120B_SNAPSHOT" --kind boundaries --output /path/to/references/boundaries
python models/demos/gpt_oss_120b_qb2/tests/generate_reference.py \
  --snapshot "$GPT_OSS_120B_SNAPSHOT" --kind batches --output /path/to/references/batches
```

Then return to the TTNN environment and run:

```bash
export GPT_OSS_120B_BOUNDARY_REFERENCE=/path/to/references/boundaries
export GPT_OSS_120B_BATCH_REFERENCE=/path/to/references/batches
python -m pytest models/demos/gpt_oss_120b_qb2/tests/test_layer_state.py -v
```

These tests check real-weight sliding/full layers, thirteen active batch sizes,
request permutations, page remapping, ring wrap and warm/cold chunk continuation.
The per-row PCC threshold is 0.99. Full-model serving isolation and accuracy
qualification are separate from these layer checks.

Host contracts can run without devices:

```bash
python -m pytest --noconftest \
  models/tt_transformers/tests/test_batched_prefill_slots.py \
  models/demos/gpt_oss_120b_qb2/tests/test_serving_host.py -q
```

## Serve with the current plugin

Run from the tt-metal checkout in the supported plugin environment. Use a cache
owned by this server and an exclusive four-chip allocation. These commands are
prepared for device validation. CLI parsing has been checked against vLLM 0.26.0;
successful server startup alone is not serving qualification.

```bash
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export MESH_DEVICE=P150x4
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export GPT_OSS_120B_FULL_MODEL_TENSOR_CACHE="$TT_METAL_CACHE"
export TT_MODEL_CLASS_OVERRIDES='TTGptOssForCausalLM=models.demos.gpt_oss_120b_qb2.tt.generator_vllm:TTGptOssForCausalLM'
export GPT_OSS_120B_KV_POOL_BLOCKS=4640
export GPT_OSS_120B_SLIDING_RING=1
export GPT_OSS_120B_PREFIX_CACHING=1
export GPT_OSS_120B_CHUNK_WARMUP=8192
export TORCHDYNAMO_DISABLE=1
export VLLM_SYSTEM_START_DATE=2026-08-31

python -m vllm.entrypoints.openai.api_server \
  --model "$GPT_OSS_120B_SNAPSHOT" \
  --served-model-name openai/gpt-oss-120b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --block-size 64 --max-num-seqs 32 \
  --max-model-len 131072 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
  --structured-outputs-config '{"reasoning_parser":"openai_gptoss","enable_in_reasoning":false}' \
  --additional-config '{"tt":{"l1_small_size":16384,"sample_on_device_mode":"all","trace_region_size":750000000,"decode_interleave_enabled":true,"decode_interleave_prefill_steps":1,"decode_interleave_decode_steps":8}}'
```

The tested serving configuration reserves 16,384 bytes of small L1. The default
zero-reserve path has a retained failure and is not qualified. The command above
still requires the separate prerequisites and combined-tree device validation.

The model declares its ring fabric requirement before the plugin opens the mesh.
For bulk throughput, set `GPT_OSS_120B_CHUNK_WARMUP=32768`, pass
`--max-num-batched-tokens 32768`, and set `decode_interleave_enabled` to `false`.
Measure both profiles on the same workload before comparing their latency.

Greedy sampling and supported top-k values up to 32 use the device route. Requests
with unsupported sampling parameters or logprobs use the plugin's host route;
verify its effect on concurrent requests. The supported `VLLM_SYSTEM_START_DATE` setting fixes Harmony prompt dates for
repeatable evaluation. Source-specific Harmony answer-reserve behavior is not
part of this integration. A response that spends
its entire output budget on reasoning can therefore have empty final content;
evaluation must record that as a failure.

## Qualification still required

Exercise partial batches, reordered/recycled slots, mixed host/device sampling,
long chunked prompts, prefix hits, ring wrap, cancellation and at least 128
completed requests of continuous churn. The source package reported prompt echo
and device-slot exhaustion under heavy long-context load; those reports remain
acceptance risks until reproduced and resolved against the current plugin.

Compare the maintained upstream GPT-OSS implementation using the same hardware,
checkpoint, precision/accuracy, prompt/output lengths, concurrency and timing
boundaries. Publish a measured benefit before treating this additional
implementation as ready for maintenance. Weekly CI targets and budgets must come
from the exact setup/test/serve/report/cleanup command measured on its target SKU.
