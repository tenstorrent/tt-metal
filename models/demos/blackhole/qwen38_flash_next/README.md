# Qwen3.8-Flash-Next on Blackhole Galaxy

This directory holds the Qwen3.8-Flash-Next port for Blackhole.
Each group of 8 chips runs the model at TP=8 as a 1x8 mesh (`MESH_DEVICE=P150x8`).
A Blackhole Galaxy has 32 chips, so serving runs DP=4: four vLLM servers, one per group, behind vllm-router.
Routed experts use BF4. Dense weights use BF8. The PLE table stays on the host and maps from the checkpoint.
Serving uses MTP speculative decode (8 lanes, k=3) and prefix caching.

## Credits

The port comes from the source of Samuel Jett ([sjettTT](https://github.com/sjettTT)):
[tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next), commit `cd9a11771107ea2c27da3303a0556ff7343e4af5`.
It relies on this upstream work:

- [#57564](https://github.com/tenstorrent/tt-metal/pull/57564) and [#57448](https://github.com/tenstorrent/tt-metal/pull/57448): Samuel Jett, merged.
- [#59945](https://github.com/tenstorrent/tt-metal/pull/59945): expert rows, merged in main.
- [#60469](https://github.com/tenstorrent/tt-metal/pull/60469): packed routing, Samuel Jett, open. This branch carries a port of its SingleCluster routing.
- [#57440](https://github.com/tenstorrent/tt-metal/pull/57440): public GDN, Izajasz Wrosz, merged.

The `gdn_source_chunk` kernels keep the exact kernel bytes of the source.

## Build and checkpoint

Build tt-metal with clang-20 and the default SFPI toolchain.
Then create the Python environment.
The repository environment (Transformers 5.12.1) reads the checkpoint's `qwen4_exp` configuration.

```bash
./build_metal.sh --build-ttnn-tests
./create_venv.sh
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out "$CHECKPOINT"
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint "$CHECKPOINT"
```

The downloader pins the ModelScope revision, checks the SHA256 of each file, and resumes a partial download.
Useful options are `--threads` (default 32) and `--listing-only`.
`prewarm_ple_table` reads the PLE table into the host page cache. `--report-only` prints the residency and reads nothing.
Keep the 104 GB PLE table in the page cache for timing runs.

To verify a shared read-only checkpoint without network access or cache writes:

```bash
python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.verify_release \
  --checkpoint "$CHECKPOINT" --manifest "$CHECKPOINT_MANIFEST" \
  --output "$RESULTS/checkpoint-verification.json" --workers 2
```

The manifest must have SHA256 `2618cf53db8e04d539d96bb03d3f5c6e08f8ad3dddd0a54252d5d1b739d84ded`.
Write the report outside the checkpoint.

Disk needs:

- Checkpoint: 360 GB.
- BF4 expert cache: about 65 GB per device order.
- Component cache: about 7 GB per context length and commit.
- JIT cache and build output: extra space.

Keep the cache directory unchanged while a model loads.
Every new commit writes a new component cache, so prune the directories of old commits.

## Demo (one 8-chip group)

`demo/text_demo.py` runs the model on one group at TP=8.
Set `TT_VISIBLE_DEVICES` to the 8 device nodes of one group (see the Serving section for the lists).

```bash
export PYTHONPATH=$PWD MESH_DEVICE=P150x8
export TT_VISIBLE_DEVICES=8,9,10,11,15,14,13,12  # group 1
export HF_MODEL=$CHECKPOINT                      # or Qwen/Qwen3.8-Flash-Next from the local Hugging Face cache
export TT_CACHE_PATH=$CACHE_ROOT                 # caches/<label>, caches/bf4-experts, demo-context
export TT_METAL_TRACE_ALLOC_TRACKING=1
unset TT_MESH_GRAPH_DESC_PATH
```

The demo picks the mesh graph descriptor from the hardware profile when `TT_MESH_GRAPH_DESC_PATH` is unset.
`QWEN38_RING=auto` (the default) selects the ring on a Galaxy.
`QWEN38_BF4_CORPUS` and `QWEN38_BF4_CORPUS_VERIFICATION` are needed only while `caches/bf4-experts` is empty.
The first run of each context builds its component cache and is slow.
The demo downloads War and Peace once into `$TT_CACHE_PATH/demo-context`.

Test functions:

| Function | Ids | What it checks |
|---|---|---|
| `test_demo_text` | `traced_128` to `traced_256k`, `determinism_128` | B=1, 128 greedy tokens. The pass key `58213` is in the output. `determinism_128` runs twice and the tokens match. |
| `test_demo_text_batched` | `batched_<isl>_b8`, `batched_<isl>_b32`, `batched_<isl>_bmax` | B users, each with its own prompt and pass key. The first 8 tokens equal the B=1 run. Each output holds its own key and no other key. |
| `test_demo_text_batched_mtp` | `batched_<isl>_b8_mtp3` | 8 lanes with 3 drafts. The first tokens match the B=1 run, each output holds its own key, and tokens per pass are above 1 and at most 4. |

Each prompt is a chat turn with thinking off.
The resident context is the smallest of 32768, 65536, 131072 and 262144 that holds the prompt.
The model gets `max_seq_len = context - 64`.
Set `CI=true` to write the benchmark JSON.

```bash
pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k "traced_128 and not traced_128k"
pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k traced_32k
pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k batched_32k_b8
pytest models/demos/blackhole/qwen38_flash_next/demo/text_demo.py -s -k batched_32k_b8_mtp3
```

## Performance

Measured on one 8-chip group of a Blackhole Galaxy (TP=8) with `demo/text_demo.py`, 128 greedy tokens per user.

| Test id | Users | ISL | TTFT | Decode |
|---|---|---|---|---|
| `traced_128` | 1 | 128 | 0.21 s | 89.6 t/s/u (11.17 ms/token) |
| `traced_32k` | 1 | 32k | 7.37 s | 79.2 t/s/u |
| `traced_128k` | 1 | 128k | 36.2 s | 74.5 t/s/u |
| `batched_128_b32` | 32 | 128 | 0.21 s/user | 25.4 t/s/u |
| `batched_32k_b8` | 8 | 32k | 7.37 s/user | 44.0 t/s/u |
| `batched_32k_b8_mtp3` | 8 | 32k | 9.2 s/user | 43.8 t/s/u, 2.41 tokens/pass |

Served through vLLM with MTP (one group, 8 concurrent users, GSM8K prompts), decode is about 63 tok/s/user.
The MTP gain depends on the text: the acceptance rate sets the tokens per pass.

## Serving (DP=4 with vLLM)

Run one server per group.
Each server needs the vLLM venv with the `vllm-tt-plugin` on `PYTHONPATH`.
Set `TT_VISIBLE_DEVICES` to the node list of the group, in this order:

| Group | `TT_VISIBLE_DEVICES` | Port |
|---|---|---|
| 0 | `0,1,2,3,7,6,5,4` | 8101 |
| 1 | `8,9,10,11,15,14,13,12` | 8102 |
| 2 | `16,17,18,19,23,22,21,20` | 8103 |
| 3 | `24,25,26,27,31,30,29,28` | 8104 |

These lists are the device order of the validated Galaxy.
The node numbers under `/dev/tenstorrent` can differ from the PCIe numbering.
On another Galaxy, check the node numbers and the chip order of each group, then edit the lists.
The position list `QWEN38_X8_ROUTE=0,4,5,6,7,3,2,1` indexes into `TT_VISIBLE_DEVICES`.
Do not run two jobs on the same group at the same time.

Environment of one server (shown for group 0):

```bash
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PLUGIN_ROOT/src:$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
M=$TT_METAL_HOME/models/demos/blackhole/qwen38_flash_next
export HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_SERVER_DEV_MODE=1 OMP_NUM_THREADS=4
export MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,7,6,5,4
export TT_MESH_GRAPH_DESC_PATH=$M/tools/p150_x8_1x8_ring_mesh_graph_descriptor.textproto
export QWEN38_X8_ROUTE=0,4,5,6,7,3,2,1 QWEN38_RING=1
export HF_MODEL=$CHECKPOINT TT_CACHE_PATH=$CACHE_ROOT TT_METAL_CACHE=$CACHE_ROOT/jit-cache
export EXTRA_MODELS_DIR=$M/tools/vllm_bundle
export QWEN38_VLLM_ALLOW_BF4_CONVERSION=1        # only until caches/bf4-experts exists
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_MODEL_CLASS_OVERRIDES="TTQwen4ExpForConditionalGeneration=models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp:Qwen38MTPForCausalLM"
export QWEN38_VLLM_MTP_K=3 QWEN38_MAMBA_SPEC=1
export QWEN38_MTP_CHAIN=1 VLLM_TT_SPEC_CHAIN=1   # chain mode
export QWEN38_MAX_TOKENS_ALL_USERS=1000000
export TMPDIR=/tmp/q38_g0                        # one directory per group
```

Command of one server:

```bash
python $PLUGIN_ROOT/examples/server_example_tt.py \
  --model $CHECKPOINT --served-model-name qwen38 Qwen/Qwen3.8-Flash-Next \
  --max_num_seqs 8 --max-model-len 262080 --port $PORT \
  --hf-overrides '{"architectures": ["TTQwen4ExpForConditionalGeneration"]}' \
  --block_size 2048 --enable-prefix-caching --mamba-cache-mode align --prefix-match-unit 2048 \
  --enable-chunked-prefill --max-num-batched-tokens 2080 --enable-prompt-tokens-details \
  --async-scheduling \
  --speculative_config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}' \
  --override-generation-config '{"temperature": 0.0}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --additional-config '{"tt": {"l1_small_size": 24576, "trace_region_size": 0, "prefill_chunk_alignment": 128, "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 32}}'
```

The setting is `BS=2048`, `MAXLEN=262080`, `DI_PREFILL=1`, `DI_DECODE=32`, and chain mode on.
Without prefix caching, use `--block_size 128 --no-enable-prefix-caching --no-enable-chunked-prefill`, drop `QWEN38_MAMBA_SPEC`, and set `--max-num-batched-tokens` to the model length.
The first start converts the BF4 experts and takes long.

Route the four servers with vllm-router 0.1.17:

```bash
vllm-router --host 127.0.0.1 --port 8100 \
  --worker-urls http://127.0.0.1:8101 http://127.0.0.1:8102 http://127.0.0.1:8103 http://127.0.0.1:8104 \
  --policy cache_aware --cache-threshold 0.5 --balance-abs-threshold 4 --balance-rel-threshold 1.5 \
  --request-timeout-secs 86400 --worker-startup-timeout-secs 7200 --worker-startup-check-interval 10 \
  --health-check-interval-secs 30 --health-check-timeout-secs 30 --health-failure-threshold 10 \
  --disable-retries --disable-circuit-breaker \
  --max-concurrent-requests 100000 --queue-size 1000 --queue-timeout-secs 86400 --prometheus-port 29000
```

Clients send requests to port 8100.
The served model names are `qwen38` and `Qwen/Qwen3.8-Flash-Next`.

## Tests

Run the tests that need no device from the repository root:

```bash
T=models/demos/blackhole/qwen38_flash_next/tests
HF_MODEL=$CHECKPOINT PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools python_env/bin/python -m pytest $T/*_static.py $T/*_no_device.py \
  --ignore=$T/test_vllm_mtp_no_device.py --ignore=$T/test_vllm_mtp_prefix_no_device.py --ignore=$T/test_mtp_lane_buckets_no_device.py
```

The three ignored files import the vLLM plugin.
Run them with the vLLM venv and the `vllm-tt-plugin` on `PYTHONPATH`.
Tests named `*_device.py` need a held 8-chip group and the environment of the Demo section.
Skipped tests are not device coverage.

The standalone chat server can replay greedy records on startup:

```bash
bash models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --acceptance --require-json-96 --acceptance-only ...
```

The records are in `tools/acceptance/greedy-prompts.jsonl`.
The `json` record must match the CPU reference 96/96, or the server refuses to start.

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `QWEN38_FUSED_OFF=<name,...\|all>` | unset | Turns off the named fused kernels and uses the composed op chains. An unknown name raises an error. |
| `QWEN38_FUSED=<name>` | unset | Turns on an opt-in fused kernel that is not in the default list. |
| `QWEN38_MTP_ROWS_TP8` | `1` | `0` turns off `gdn_rows_scan` at TP=8. |
| `QWEN38_ROW_REDUCE` | `1` | `0` uses the stock `reduce_scatter` instead of the `row_reduce` program. |
| `QWEN38_ROW_GATHER` | `1` | `0` uses the stock `all_gather` instead of the `row_gather` program. |
| `QWEN38_MOE_EXPERTS` | `1` | `0` turns off the one-program routed experts of the decode step. |
| `QWEN38_MOE_FUSE` | `1` | `0` runs the routed experts apart from the `moe_dense` program. |
| `QWEN38_MTP_CHAIN` | `0` | `1` serves the plugin chain mode. It needs `VLLM_TT_SPEC_CHAIN=1` and `--async-scheduling`. |
