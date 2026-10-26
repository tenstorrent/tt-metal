# Qwen3.8-Flash-Next on Blackhole

Qwen3.8-Flash-Next (hybrid GDN / QSA attention, 512 routed experts, MTP layer, vision tower) on four Blackhole
devices as one 1x4 mesh. Routed experts run in BF4, dense weights in BF8, and the PLE n-gram table stays in the
checkpoint and is read by the host. The port is by Samuel Jett ([sjettTT](https://github.com/sjettTT)); see
[PROVENANCE.md](PROVENANCE.md). It runs on tt-metal main and calls only public operations.

Validated configuration: one request at a time with a 32,768-token allocation (32,704 usable tokens). The
standalone server also has MTP drafting, image input and concurrent lanes; their coverage is separate. The vLLM
adapter has one resident slot and no MTP.

## Hardware

| profile | hardware | status |
|---|---|---|
| `tt-quietbox` | QuietBox, 4x p150b, opened as a 1x4 line through `tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto` | validated |
| `p150-line` | 4x p150 in one host on an ethernet line; the route is derived at start | supported |
| `qb2` | QuietBox 2 (2x p300c), or a 4x p300 host with `--instance 0\|1` | untested |

## Build and checkpoint

Build this checkout with the standard [build instructions](../../../../README.md) using clang-20 and the default
SFPI toolchain, then install Transformers 5.16.1 (the checkpoint's `qwen4_exp` configuration).

```bash
./build_metal.sh --build-ttnn-tests
./create_venv.sh
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out "$CHECKPOINT"
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint "$CHECKPOINT"
```

The downloader pins ModelScope revision `2741eec155d03a8ce151b993ccce1a7b1e398d6b`, verifies each file's SHA256
and resumes partial downloads. Disk: 360 GB for the checkpoint, about 69 GB for the BF4 expert cache, about 23 GB
for the 32k component and model I/O caches, plus the build and the JIT cache. Keep the 104 GB PLE table in the
host page cache when timing.

Keep `CACHE_ROOT` dedicated to this model and unchanged while a server loads or runs: new cache files are published
by the converter's temporary-file and atomic-rename step before readers start; concurrent replacement, in-place
writes and truncation are unsupported. The model checks cache identity, checksums, headers, metadata and topology,
and releases loaded tensors when a check fails.

To verify a shared read-only checkpoint without network access or cache writes (the manifest's SHA256 is
`2618cf53db8e04d539d96bb03d3f5c6e08f8ad3dddd0a54252d5d1b739d84ded`; write the report outside the checkpoint):

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.verify_release \
  --checkpoint "$CHECKPOINT" --manifest "$CHECKPOINT_MANIFEST" \
  --output "$RESULTS/checkpoint-verification.json" --workers 2
```

## Standalone server

Run on an allocated four-device mesh:

```bash
export TT_METAL_TRACE_ALLOC_TRACKING=1
MODEL=models/demos/blackhole/qwen38_flash_next
bash "$MODEL/tools/run_qwen38_chat_server.sh" --profile tt-quietbox \
  --checkpoint "$CHECKPOINT" --cache-root "$CACHE_ROOT" --prepare-only
bash "$MODEL/tools/run_qwen38_chat_server.sh" --profile tt-quietbox \
  --checkpoint "$CHECKPOINT" --cache-root "$CACHE_ROOT" \
  --acceptance --require-json-96 --acceptance-only
```

The first command converts the missing BF4 expert layers and stops. The second replays the acceptance records,
writes the tokens and the runtime identity under `$CACHE_ROOT/runs/` and closes the mesh; drop `--acceptance-only`
to serve. The server speaks the OpenAI chat API (`/v1/chat/completions`, `/v1/models`, `/health`).

Main flags (`run_qwen38_chat_server.sh --help` lists all of them):

| flag | meaning |
|---|---|
| `--allocated-context N` | 32768 (default), 65536, 131072 or 262144 |
| `--acceptance`, `--require-json-96` | replay the CPU greedy records at start; refuse to serve unless `json` matches 96/96 |
| `--prepare-only [--bf4-stage-limit N]` | build the BF4 expert cache, at most N layers per run |
| `--mtp K` | MTP drafting with K = 3, 4 or 5 drafts (off by default) |
| `--prefill-slab ROWS` | prefill in slabs ahead of the 128-row chunks; where `moe_compute` refuses the slab in one call (as on main), each layer routes it in 128-token calls |
| `--no-sampling` | greedy requests only (the default server samples when a request asks for it) |
| `--stall-seconds N` | exit 1 after N seconds without device progress (default 300; 0 disables) |
| `--host ADDR --port N` | listen address; use `--host 127.0.0.1` for a local endpoint |

[docs/SERVER.md](docs/SERVER.md) covers the request rules, `/health`, follow-up turns, lanes, MTP and the container
form; [docs/PREFILL.md](docs/PREFILL.md) the prefill forms.

## vLLM

Use vllm-tt-plugin commit `1d87a00e7d91ec246582d07865ec4f8b0a8fb25c` with vLLM 0.26.0's empty target and point
`EXTRA_MODELS_DIR` at `tools/vllm_bundle`; the adapter declares its fabric configuration before the mesh opens.

```bash
export MODEL_WEIGHTS_DIR="$CHECKPOINT" QWEN38_CACHE_ROOT="$CACHE_ROOT"
export QWEN38_CACHE_LABEL=c32768-tt-quietbox TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_VISIBLE_DEVICES=0,1,2,3 MESH_DEVICE='(1, 4)'
export EXTRA_MODELS_DIR="$PWD/$MODEL/tools/vllm_bundle"
export TT_MESH_GRAPH_DESC_PATH="$PWD/$MODEL/tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto"  # QuietBox
python "$PLUGIN_ROOT/examples/server_example_tt.py" \
  --model "$CHECKPOINT" --served-model-name Qwen/Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port 8000 --shutdown-timeout 30 --max_num_seqs 1 --block_size 64 --max-model-len 32704 \
  --hf-overrides '{"architectures":["TTQwen4ExpForConditionalGeneration"]}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --additional-config '{"tt":{"l1_small_size":24576,"trace_region_size":0,"sample_on_device_mode":"decode_only"}}'
```

The adapter prefills in 128-row chunks, then 32-row chunks (`QWEN38_LONG_CHUNKS=0`: 32-row chunks only).
`QWEN38_PREFILL_SLAB=<rows>` (a multiple of 512 up to 2048) adds a prefill slab ahead of them; start-up is refused,
naming the variable, when the slab's resident weights do not fit.
`decode_only` samples on the CPU and returns the selected token; requests that need logprobs or host-only
processors return the full logits row to the plugin's sampler. Scheduler chunked prefill, prefix caching and
asynchronous decode are disabled. `tools/validate_vllm.py --output "$LIVE_RESULTS"` checks request isolation,
sampling, penalties, logprobs and rejections against a running server.

## Testing

```bash
QWEN38_CHECKPOINT="$CHECKPOINT" python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests
pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -v -s -k traced_128
```

`tests/` holds the no-device tests, the checkpoint-reading component tests (skipped without `QWEN38_CHECKPOINT`)
and the device tests (`*_device.py`, `*_tp.py`). `tests/test_gdn_public_device.py` checks the public GDN dispatch
against an independent recurrence; set `QWEN38_FUSED_DEVICE_TEST=1` on the held mesh. `tests/test_model_reuse_tp.py`
checks fixed generations, prefill-boundary A/B/A requests and program-cache counts. `demo/text_demo.py` is the
end-to-end test ([demo/README.md](demo/README.md)). No CI pipeline runs these device tests; run them manually on the
held mesh. [docs/TESTING.md](docs/TESTING.md) covers the acceptance gate,
the reference corpus and the regression harness.

`tools/benchmark.py` scores and times GSM8K through a live OpenAI endpoint (the official 1,319-item test file at
revision `3101c7d5072418e28b9008a6636bde82a006892c`, hash-checked; `--limit 64` is the fixed smoke subset).
`tools/report.py` with `CI=true` writes the standard tt-metal benchmark artifact; `--device-name` (`P150x4` or `QB2`)
must come from the allocation's topology record.

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.benchmark \
  --checkpoint "$CHECKPOINT" --dataset "$GSM8K_TEST_JSONL" --output-dir "$RESULTS"
CI=true python -m models.demos.blackhole.qwen38_flash_next.tools.report "$RESULTS/summary.json" \
  --device-name "$QUALIFIED_DEVICE_NAME"
```

## Validation

- The acceptance gate: the `json` record matches the CPU reference 96/96 (`--require-json-96`). Other greedy
  records leave the CPU stream after a near-tie; their first divergence index is recorded per run
  ([docs/NUMERICS.md](docs/NUMERICS.md)).
- Reference agreement, task quality (GSM8K) and runtime reuse are separate gates; record each with the model
  commit, tt-metal identity, compiler, topology, checkpoint revision, command and saved responses.
- The baselines under `tools/ci/baselines/` are proposals from the source fork, not results for this port. A
  standalone MTP rate is not vLLM throughput.
