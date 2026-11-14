# Qwen3.8-Flash-Next on Blackhole

Qwen3.8-Flash-Next (hybrid GDN / QSA attention, 512 routed experts, MTP layer, vision tower) on four Blackhole
devices as one 1x4 mesh. Routed experts run in BF4, dense weights in BF8, and the PLE n-gram table stays in the
checkpoint and is read by the host. The port is by Samuel Jett ([sjettTT](https://github.com/sjettTT)); see
[PROVENANCE.md](PROVENANCE.md). It runs on tt-metal main and calls only public operations.

Validated configuration: one request at a time with a 32,768-token allocation (32,704 usable tokens), served with
`--mtp 4` MTP drafting. The standalone server also has image input and concurrent lanes; their coverage is separate.
vLLM serving comes in a follow-up PR.

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
  --mtp 4 --acceptance --require-json-96 --acceptance-only
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
| `--mtp K` | MTP drafting with K = 3, 4 or 5 drafts; `--mtp 4` is the serving form (off by default) |
| `--prefill-slab ROWS` | prefill in slabs ahead of the 128-row chunks; where `moe_compute` refuses the slab in one call (as on main), each layer routes it in 128-token calls |
| `--no-sampling` | greedy requests only (the default server samples when a request asks for it) |
| `--stall-seconds N` | exit 1 after N seconds without device progress (default 300; 0 disables) |
| `--host ADDR --port N` | listen address; use `--host 127.0.0.1` for a local endpoint |

[docs/SERVER.md](docs/SERVER.md) covers the request rules, `/health`, follow-up turns, lanes and MTP;
[docs/PREFILL.md](docs/PREFILL.md) the prefill forms.

To chat with a running server from a terminal (`/reset` clears the history, `/quit` exits):

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://127.0.0.1:8000/v1
```

## Testing

```bash
QWEN38_CHECKPOINT="$CHECKPOINT" python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests
```

`tests/` holds the no-device tests, the checkpoint-reading component tests (skipped without `QWEN38_CHECKPOINT`)
and the device tests (`*_device.py`, `*_tp.py`). `tests/test_gdn_public_device.py` checks the public GDN dispatch
against an independent recurrence; set `QWEN38_FUSED_DEVICE_TEST=1` on the held mesh.
[docs/TESTING.md](docs/TESTING.md) covers the acceptance gate and the reference corpus.

`tools/benchmark.py` scores and times GSM8K through a live OpenAI endpoint (the official 1,319-item test file at
revision `3101c7d5072418e28b9008a6636bde82a006892c`, hash-checked; `--limit 64` takes its first 64 items).
`tools/report.py` with `CI=true` writes the standard tt-metal benchmark data; `--device-name` names the chips the run
used (`P150x4`, or `P300x2` for a QuietBox 2) and `--mtp-drafts` the server's `--mtp`.

```bash
python -m models.demos.blackhole.qwen38_flash_next.tools.benchmark \
  --checkpoint "$CHECKPOINT" --dataset "$GSM8K_TEST_JSONL" --output-dir "$RESULTS"
CI=true python -m models.demos.blackhole.qwen38_flash_next.tools.report "$RESULTS/summary.json" \
  --device-name P150x4 --mtp-drafts 4
```

### End-to-end demo

`demo/chat_demo.py` starts the server through the launcher with `--mtp 4 --acceptance --require-json-96`, so it
serves only after the `json` record matches 96/96. It then runs `tools/benchmark.py` on the first 64 GSM8K test
items through the server's endpoint, requires the benchmark's 0.80 floor, and stops the server. With `CI=true` it also
writes the benchmark data (`tools/report.py`).

```bash
MODEL_WEIGHTS_DIR="$CHECKPOINT" QWEN38_CACHE_ROOT="$CACHE_ROOT" QWEN38_DEVICE_NODES=4,5,6,7 \
  python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/demo/chat_demo.py
```

`QWEN38_PROFILE` picks the launcher profile (`p150-line` by default) and `QWEN38_DEVICE_NODES` the four chips of a
larger host. The BF4 expert cache must be in place (`--prepare-only`). On the 1x4 p150 line the demo took 13.5 min:
start-up 486 s with cold component and JIT caches, GSM8K 58/64 in 296 s. No CI pipeline runs the demo or the device
tests yet; run them on a held 1x4 mesh.

## Validation

- The acceptance gate: the `json` record matches the CPU reference 96/96 (`--require-json-96`). Other greedy
  records leave the CPU stream after a near-tie; their first divergence index is recorded per run
  ([docs/NUMERICS.md](docs/NUMERICS.md)).
- Reference agreement, task quality (GSM8K) and runtime reuse are separate gates; record each with the model
  commit, tt-metal identity, compiler, topology, checkpoint revision, command and saved responses.
