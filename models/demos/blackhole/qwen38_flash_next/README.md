# Qwen3.8-Flash-Next on Blackhole

This model port originates from **Samuel Jett** ([sjettTT](https://github.com/sjettTT), sjett@tenstorrent.com),
source commit `cd9a11771107ea2c27da3303a0556ff7343e4af5`. See [PROVENANCE.md](PROVENANCE.md) for the source,
target base, original MoE prerequisites [#57448](https://github.com/tenstorrent/tt-metal/pull/57448) and
[#57564](https://github.com/tenstorrent/tt-metal/pull/57564), and the merged public GDN prerequisite.

The implementation uses four Blackhole devices as a 1x4 mesh, BF4 routed experts, BF8 dense weights, and a host
PLE table mapped from the checkpoint. The publication validation targets one request at a time and a 32,768-token
allocation (32,704 usable tokens). The standalone server also contains source MTP, vision and lane paths; coverage
for each must be read separately. The vLLM adapter has one resident slot and no MTP.

The historical measurements in `docs/` and proposal baselines under `tools/ci/` belong to the pinned source fork.
They are not performance or validation claims for this port. Ordinary greedy outputs can diverge from the CPU
reference after a near-tie; the JSON acceptance record, independent numerical controls and task accuracy are
separate gates. Never substitute a standalone MTP rate for vLLM throughput.

## Build and checkpoint

Build this tt-metal checkout using the standard [build instructions](../../../../README.md). The port requires the
native prerequisites described in PROVENANCE.md; a Python import alone does not validate that build. Use clang-20
and the supported default SFPI toolchain. Install the repository Python environment, then Transformers 5.16.1 for
this checkpoint's `qwen4_exp` configuration.

```bash
./build_metal.sh --build-ttnn-tests
./create_venv.sh
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out "$CHECKPOINT"
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint "$CHECKPOINT"
```

The downloader pins ModelScope revision `2741eec155d03a8ce151b993ccce1a7b1e398d6b`, verifies each file's SHA256,
and resumes incomplete downloads. The checkpoint occupies 360 GB. Allow approximately 69 GB for the BF4 cache,
23 GB for 32k component/model caches, and additional space for the build and JIT cache. The 104 GB PLE table should
remain in the host page cache for timing; the original source's smaller host-memory guidance does not guarantee
this residency.

Keep the entire cache namespace and artifact contents immutable during loading and through loaded host
tensor/view lifetimes and pending device transfers. Publish new files with the converter's temporary-file and
atomic-replace transaction before readers start. Concurrent replacement, in-place writes and truncation are
unsupported. The model passes canonical `.tensorbin` paths to the existing public `ttnn.load_tensor`; those
loads open files separately from the retained verification descriptors, so same-inode loading across concurrent
replacement is not guaranteed. Hash verification and stat checks do not create an immutable snapshot or detect
every writable-mmap modification. Cache identity, checksum, header, metadata and topology checks remain in the
model, and loaded tensors are released on validation failure. No shared descriptor-loader extension is required.

For a shared read-only checkpoint, verify every file without network or cache writes:

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.verify_release \
  --checkpoint "$CHECKPOINT" --manifest "$CHECKPOINT_MANIFEST" \
  --output "$RESULTS/checkpoint-verification.json" --workers 2
```

The manifest must have SHA256 `2618cf53db8e04d539d96bb03d3f5c6e08f8ad3dddd0a54252d5d1b739d84ded`;
write the report outside the checkpoint. Include this verification cost in the complete weekly recipe timing.

## Standalone validation and serving

Run only on an allocated four-device mesh. Keep `CACHE_ROOT` dedicated to this model. The QuietBox profile derives
and validates the current fabric order; physical device IDs are not shared across hosts.

```bash
export TT_METAL_TRACE_ALLOC_TRACKING=1
MODEL=models/demos/blackhole/qwen38_flash_next
bash "$MODEL/tools/run_qwen38_chat_server.sh" --profile tt-quietbox \
  --checkpoint "$CHECKPOINT" --cache-root "$CACHE_ROOT" --prepare-only
bash "$MODEL/tools/run_qwen38_chat_server.sh" --profile tt-quietbox \
  --checkpoint "$CHECKPOINT" --cache-root "$CACHE_ROOT" \
  --acceptance --require-json-96 --acceptance-only
```

The first command stages missing BF4 layers. The second writes acceptance tokens and runtime identity under
`$CACHE_ROOT/runs/` and closes the mesh. Chunk mode admits 12 short records; add `--prefill-slab 2048` to include the
13th, 2,228-token document record. MTP4 uses `--mtp 4` and its own source baseline. Remove `--acceptance-only` to
serve; set `--host 127.0.0.1` for a local endpoint. The source server's `--serve-seconds` limit includes startup.

The tiered component tests compare real checkpoint values with independent Torch references that are separately
checked against Transformers. `tests/test_model_reuse_tp.py` records fixed 128/50 generations and A/B/A requests at
prefill boundaries, including exact token IDs and program-cache counts. Both require the complete checkpoint and
prepared cache; skipped tests are not device coverage.

## vLLM

Use plugin commit `1d87a00e7d91ec246582d07865ec4f8b0a8fb25c` with vLLM 0.26.0's empty target. No shared plugin
registration change is needed: point `EXTRA_MODELS_DIR` at `tools/vllm_bundle`. The adapter declares its required
fabric configuration before mesh creation; explicit launch overrides retain the plugin's normal precedence.

Set `MODEL_WEIGHTS_DIR`, `QWEN38_CACHE_ROOT`, `QWEN38_CACHE_LABEL=c32768-tt-quietbox`,
`TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_VISIBLE_DEVICES=0,1,2,3`, and `MESH_DEVICE='(1, 4)'`.
On QuietBox also set `TT_MESH_GRAPH_DESC_PATH` to `tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto`.
Launch with `max_num_seqs=1`, `max_model_len=32704`, architecture override
`TTQwen4ExpForConditionalGeneration`, and TT configuration
`{"l1_small_size":24576,"trace_region_size":0,"sample_on_device_mode":"decode_only"}`.

```bash
export MODEL_WEIGHTS_DIR="$CHECKPOINT" QWEN38_CACHE_ROOT="$CACHE_ROOT"
export QWEN38_CACHE_LABEL=c32768-tt-quietbox TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_VISIBLE_DEVICES=0,1,2,3 MESH_DEVICE='(1, 4)'
export EXTRA_MODELS_DIR="$PWD/$MODEL/tools/vllm_bundle"
export TT_MESH_GRAPH_DESC_PATH="$PWD/$MODEL/tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto"
python "$PLUGIN_ROOT/examples/server_example_tt.py" \
  --model "$CHECKPOINT" --served-model-name Qwen/Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port 8000 --shutdown-timeout 30 --max_num_seqs 1 --block_size 64 --max-model-len 32704 \
  --hf-overrides '{"architectures":["TTQwen4ExpForConditionalGeneration"]}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --additional-config '{"tt":{"l1_small_size":24576,"trace_region_size":0,"sample_on_device_mode":"decode_only"}}'
```

Despite the plugin parameter's name, this adapter samples on the CPU: `decode_only` returns a selected token ID;
requests requiring logprobs or host-only processors return full logits to the plugin sampler. Run
`python -m models.demos.blackhole.qwen38_flash_next.tools.validate_vllm --output "$LIVE_RESULTS"` against the local
server to retain request isolation, sampling, penalties, logprobs and rejection checks. Penalties follow
vLLM's prompt/output history rules. Scheduler chunked prefill, prefix caching and asynchronous decode are disabled.
The model's internal chunk/slab traces are independent of scheduler chunking.

## Task quality and timing

`tools/benchmark.py` scores and times the same natural-EOS GSM8K responses through a live OpenAI endpoint. It
requires the exact official 1,319-item test file at revision `3101c7d5072418e28b9008a6636bde82a006892c`; its hash
is checked before requests. The default runs all items. `--limit 64` means the fixed first-64 smoke subset, not a
full benchmark. Responses, input token IDs, strict final-answer scoring, usage, TTFT and client timing are retained.
Use `tools/report.py` with `CI=true` to emit the standard tt-metal benchmark artifact from those measured results:

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.benchmark \
  --checkpoint "$CHECKPOINT" --dataset "$GSM8K_TEST_JSONL" --output-dir "$RESULTS"
CI=true python -m models.demos.blackhole.qwen38_flash_next.tools.report "$RESULTS/summary.json" \
  --device-name "$QUALIFIED_DEVICE_NAME"
```

Set `QUALIFIED_DEVICE_NAME` to `P150x4` or `QB2` from the physical allocation and topology qualification record;
four logical devices alone do not identify a QB2. The reporter requires this explicit identity.

The reporter emits `gsm8k_accuracy` in percent, `time_to_token` in seconds, and separate `tokens/s/user` and
aggregate `tokens/s` measurements. The shared target validator needs the corresponding GSM8K mapping before
weekly registration. Its legacy `save_partial_run_json` API writes a benchmark pickle under
`generated/benchmark_data`; this is the standard collector input.

## Qualification records

Attach measured qualification records to the publication review with the exact model commit, native library
identity, compiler, hardware topology, checkpoint revision, command, and saved responses. Keep source regression,
independent reference accuracy, task quality, and runtime reuse verdicts separate. Preserve failures and first
stderr alongside corrected runs. The public GDN prerequisite changes phase arithmetic; measurements made on the
older private-binding implementation do not qualify the new implementation automatically.

`tests/test_gdn_public_device.py` checks forced phased, forced fused and automatic public dispatch against an
independent recurrence, including masked MTP commits, replicas, input-state ownership, fresh input addresses,
program-cache reuse and trace replay. Set `QWEN38_FUSED_DEVICE_TEST=1` on the held mesh. Optional
`QWEN38_GDN_BASELINE_OUTPUT` points to saved old-arithmetic tensors for an additional exact/PCC diagnostic;
it does not replace the recurrence gate or establish full-model equivalence.
