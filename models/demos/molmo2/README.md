# Molmo2-8B on Tenstorrent T3K

Multimodal language model (text / image / video) running on a T3K 8-device mesh.
Based on [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B).

## Status

| Metric | Result |
|--------|--------|
| PCC (all 8 sub-block tests) | ≥ 0.998 |
| 105-video accuracy — direct demo | 98/100 (2026-04, not re-run on main) |
| 105-video accuracy — OpenAI API (vllm-tt-plugin, 3-4 Galaxy slices, 2026-10-14) | **97/100 = 97%** (`verification/run_suite.py`) |
| Prefill latency, server (median S≈2.4k tokens video) | ~1.7 s (34k-token/384-frame video: ~27 s) |
| Decode throughput (server, traced) | ~19 tok/s |
| Hardware | T3K (8 × Wormhole WH-B0) |

## Model Architecture

```
Input video/image
    │
    ▼
TtMolmo2ViTEncoder          ViT-L (25 of 27 blocks), hidden=1152
    │  capture layers 18, 24
    ▼
TtMolmo2ImagePooling2D      Cross-attention pooling, pool_dim=2304
    │
    ▼
TtMolmo2ImageProjector      SwiGLU projector [1152 → 12288 → 4096]
    │
    ▼  injected at image_patch_id (151938) positions
TtMolmo2Model               36 decoder layers, dim=4096, GQA (32Q/8KV heads)
    │
    ▼
Logits [batch, vocab=152064]
```

**T3K tensor-parallel layout:**
- Text QKV: column-parallel (ShardTensor2dMesh dims=(2,3)), AllGather after attention
- Text MLP: TP reduce_scatter + all_gather (CCL async, trace-safe)
- ViT: ShardTensorToMesh(dim=3) for QKV/W1/W3; AllReduce after each block
- Weights cached at `/tmp/molmo2_weight_cache/` (634 `.tensorbin` files, loads in ~18 s)

## Environment Setup

```bash
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

On a 32-chip Galaxy, run on one 1×8 slice (`SLICE` = 0..3) with the T3K mesh descriptor.
Opening a bare `MeshShape(1, 8)` on the full Galaxy without these can leave the fabric
links in a bad state (`Fabric Router Sync: Timeout`) until `tt-smi -glx_reset`.

```bash
SLICE=0
export TT_VISIBLE_DEVICES=$(seq -s, $((SLICE * 8)) $((SLICE * 8 + 7)))
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto
export MESH_DEVICE=T3K
```

### transformers version

tt-metal pins transformers 5.x, while the allenai/Molmo2-8B remote code targets
transformers==4.57.1. The TT model does not use the HF modeling code: `tt/hf_loading.py`
reads the weights directly from safetensors and loads the HF processor through a small
compatibility shim (outputs verified bit-exact against 4.57.1 for text / image / multi-image /
video). Only the HF reference scripts (`reference/test_functional.py`, which generates the
PCC goldens, and `verification/test_video_verification.py`) need a separate CPU-only
transformers==4.57.1 env — see the docstring of `reference/test_functional.py`.

## Demo

### Text / Image / Video generation

```bash
# Text-only (batch=1)
MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "text_only-batch1"

# Text-only (batch=2)
MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "text_only-batch2"

# Image + text
MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "image-batch1"

# Video + text
MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "video-batch1"

# Custom prompt file
MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py \
    --input_prompts path/to/prompts.json -v
```

**Prompt JSON format** (`demo/sample_inputs/`):
```json
[
  [{"role": "user", "content": "What is 2+2?"}],
  [{"role": "user", "content": [
      {"type": "video", "video": "path/to/video.mp4"},
      {"type": "text",  "text":  "Describe what you see."}
  ]}]
]
```

## Server (vllm-tt-plugin)

Current main serves TT models through the standalone `vllm-tt-plugin` on upstream vLLM 0.30.0 (which ships a native
Molmo2). Setup used for the numbers above:

1. Copy tt-metal's `python_env`, then in the copy install vLLM and the plugin (`UV_HTTP_TIMEOUT=600`):
   `VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-build-isolation --no-binary vllm vllm==0.30.0`
   (needs `setuptools-rust`, `setuptools-scm`) and `uv pip install -e .` from the plugin repo.
2. Apply `serve/vllm-tt-plugin-molmo2-multimodal.patch` to the plugin. By default the plugin only passes image
   `pixel_values` to the model and rejects video; the patch lets a model declare `supported_mm_modalities` /
   `mm_extra_input_fields` (Molmo2 needs the pooling indices and video tensors). Other models are unaffected.
3. `VLLM_ENV=<that venv> SLICES="1 2 3" serve/launch_4x_galaxy_t3k.sh` starts one server per 1x8 slice on ports
   8000+slice (`stop` to kill them). The adapter is registered with `TT_MODEL_CLASS_OVERRIDES`.
4. `python models/demos/molmo2/tests/server_requests_smoke.py --port 8001 --image img.jpg --video /abs/clip.mp4`
   (text, image URL / data-URL, two images, local and https video) and
   `python models/demos/molmo2/verification/run_suite.py --ports 8000 8001 8002 8003` (105-test suite).

The older docker / `run.py --docker-server` flow described below targets the retired fork-based image.

## Server (tt-inference-server, legacy docker flow)

Exposes an OpenAI-compatible HTTP API on port 8000.

### Local server (T3K or Galaxy 1×8 slice via `galaxy_t3k`)

```bash
cd tt-inference-server
export TT_METAL_HOME=/path/to/tt-metal
export JWT_SECRET=dummy
# For galaxy_t3k path, the t3k mesh descriptor must be set BEFORE the worker boots
# (the spec env_vars are applied after the metal env is initialised):
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto
export TT_FABRIC_CONFIG_OVERRIDE=FABRIC_1D

# Start server (loads model, warms up, then serves requests)
python run.py --model Molmo2-8B --workflow server --local-server \
    --tt-device galaxy_t3k --device-id 0,1,2,3,4,5,6,7 \
    --skip-system-sw-validation
# (use --tt-device t3k on a standalone T3K mesh; --device-id not required there)
```

### Docker server (validated on Galaxy 1×8 slice)

```bash
cd tt-inference-server
export TT_METAL_HOME=/path/to/tt-metal
export JWT_SECRET=dummy
export HF_TOKEN=...   # required by docker server

python run.py --model Molmo2-8B --workflow server --docker-server \
    --tt-device galaxy_t3k --device-id 0,1,2,3,4,5,6,7 \
    --skip-system-sw-validation --dev-mode \
    --override-docker-image \
      ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.12.0-454c9bf7002-7a07a97
```

The image is built once via:
```bash
python scripts/build_docker_images.py --build-metal-commit 454c9bf7002 \
    --ubuntu-version 22.04 --single-threaded
```
(~50 min wall-clock for the tt-metalium base + dev image overlay).

### 4× parallel galaxy_t3k servers on a single 32-chip Galaxy

Launch 4 local-server instances on devices 0–7, 8–15, 16–23, 24–31 with ports
8000–8003, staggered by ~30s so their tt-metal init phases don't collide:

```bash
for idx in 0 1 2 3; do
    start=$((idx * 8)); end=$((start + 7))
    TT_VISIBLE_DEVICES=$(seq -s, $start $end) \
    nohup python tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py \
        --model allenai/Molmo2-8B --tt-device galaxy_t3k \
        --port $((8000 + idx)) > /tmp/molmo2_inst${idx}.log 2>&1 &
    [ $idx -lt 3 ] && sleep 30
done
```

### Sending a request

The server requires a JWT bearer token (signed with `JWT_SECRET`):

```bash
TOKEN=$(python3 -c "import jwt,os; print(jwt.encode({'team_id':'tenstorrent','token_id':'debug-test'}, os.environ['JWT_SECRET'], algorithm='HS256'))")

# Video request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model": "allenai/Molmo2-8B",
    "messages": [{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"file:///abs/path/to/video.mp4"}},
        {"type":"text","text":"Describe what is happening in this video."}
    ]}],
    "max_tokens": 64,
    "temperature": 0
  }'

# Image request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model": "allenai/Molmo2-8B",
    "messages": [{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"https://example.com/img.jpg"}},
        {"type":"text","text":"What do you see?"}
    ]}],
    "max_tokens": 64,
    "temperature": 0
  }'

# Run 105-video benchmark suite against a running server.
# galaxy_t3k / t3k servers are batch=1 (max_concurrency=1) — use --concurrency 1.
# For 4-way chassis throughput run 4 servers on ports 8000-8003 and bench each
# with --concurrency 1 in parallel (see "4x parallel" section above).
python models/demos/molmo2/verification/run_dp4_concurrent.py \
    --server http://localhost:8000 \
    --tests models/demos/molmo2/verification/test.jsonl \
    --baseline models/demos/molmo2/verification/test_results.jsonl \
    --output /tmp/results.jsonl --concurrency 1
```

**Key server configuration** (`tt-inference-server/workflows/model_spec.py`):
- `video_backend: "molmo2"` — uses Molmo2VideoBackend (uniform_last_frame, max_fps=2, matches demo)
- `video_input_ids` in `_call_hf_processor` injects full HF token sequence including
  `<im_start>`/`<im_end>` frame markers, which is required for correct bidirectional attention

## Tests

### PCC sub-block tests (8 tests, ~2 min)

The goldens in `reference/golden/` are not checked in; generate them first with
`reference/test_functional.py` (transformers==4.57.1 env, CPU, ~2 min).

```bash
pytest models/demos/molmo2/tests/test_tt_text_decoder.py -v
```

Expected results (Galaxy 1×8 slice):

| Test | PCC |
|------|-----|
| test_prefill_mask | PASS |
| test_text_attention_pcc | 0.999838 |
| test_text_mlp_pcc | 0.999970 |
| test_decoder_block_pcc | 0.999926 |
| test_vit_encoder_pcc | 0.998176 |
| test_image_projector_pcc | 0.999777 |
| test_vision_adapter_pcc | 0.999948 |
| test_decoder_block_with_image_pcc | 0.999998 |

### TP MLP test

```bash
pytest models/demos/molmo2/tests/test_tp_mlp.py -v
```

### 105-video back-to-back suite (~10 min)

```bash
# Run all 105 tests (requires verification/ dir with test.jsonl + videos/)
MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_10_videos.py -v -s

# Run first N tests only
MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_10_videos.py -v -s --max_tests 10
```

Set `MOLMO2_VERIF_DIR` to point at a directory containing `test.jsonl`,
`test_results.jsonl`, and `videos/`.

## Profiling

### Profile all sub-blocks → Excel workbook

```bash
# Run one Tracy capture per block (no warmup needed — kernel duration is stable)
bash models/demos/molmo2/tests/run_block_profiles.sh \
    --seq-len 128 \
    --output molmo2_block_profile.xlsx

# Requires: pip install openpyxl
```

This produces `molmo2_block_profile.xlsx` with:
- **Summary** sheet: per-block totals (Matmul / AllBroadcast-CCL / SDPA / Other)
- One sheet per block: Op Code, Kernel [µs], Core Count, Parallelization, Input/Output
  shapes + layout + dtype + memory (colour-coded by op type)

### Profile a single block

```bash
python -m tracy -p -v -r -n vit_encoder \
    -o generated/profiler/block_reports/vit_encoder \
    models/demos/molmo2/tests/profile_single_block.py \
    --block vit_encoder --seq-len 128
```

Available blocks: `text_attention`, `text_mlp`, `decoder_block`,
`vit_encoder`, `image_pooling`, `image_projector`

### Build Excel from existing CSVs

```bash
python models/demos/molmo2/tests/make_profile_xlsx.py \
    --csv-dir generated/profiler/block_csvs \
    --output molmo2_block_profile.xlsx

# Or pass explicit CSVs
python models/demos/molmo2/tests/make_profile_xlsx.py \
    --csv vit_encoder_ops.csv text_mlp_ops.csv \
    --output partial.xlsx
```

### T3K profile summary (S=128, 1 crop)

| Block | Ops | Kernel [µs] | Bottleneck |
|-------|-----|-------------|------------|
| text_attention | 23 | 745 | SDPA (47%) |
| text_mlp | 13 | 724 | Matmul (85%) |
| decoder_block | 33 | 1,573 | Matmul + SDPA |
| vit_encoder (25 blocks) | 541 | 82,628 | **AllBroadcast CCL (68%)** |
| image_pooling | 30 | 14,867 | MorehSum + Concat |
| image_projector | 11 | 988 | Matmul (96%) |

**Primary optimization target**: ViT encoder AllBroadcast (all_reduce after each of 25 blocks).

## File Layout

```
models/demos/molmo2/
├── README.md                        # this file
├── ARCHITECTURE.md                  # detailed architecture notes
├── demo/
│   ├── demo.py                      # pytest-based demo (text / image / video)
│   └── sample_inputs/               # example prompt JSON files
├── reference/
│   └── functional.py                # pure-PyTorch reference implementation
├── tt/
│   ├── model_config.py              # Molmo2Config (dimensions, hyperparams)
│   ├── model.py                     # TtMolmo2Model (full assembly)
│   ├── attention.py                 # TtMolmo2TextAttention (GQA, QK-norm, RoPE)
│   ├── mlp.py                       # TtMolmo2TextMLP (TP SwiGLU)
│   ├── vision_block.py              # TtMolmo2ViTBlock (single ViT block)
│   ├── vision_encoder.py            # TtMolmo2ViTEncoder (25-block stack)
│   ├── image_pooling.py             # TtMolmo2ImagePooling2D (cross-attention)
│   ├── image_projector.py           # TtMolmo2ImageProjector (SwiGLU projector)
│   ├── prefill_mask.py              # build_molmo2_prefill_mask (causal + image-bidir)
│   ├── generator_vllm.py            # TTMolmo2ForConditionalGeneration (vLLM plugin)
│   └── trace_capture_utils.py       # Tracy trace capture helpers
├── tests/
│   ├── test_tt_text_decoder.py      # 8 PCC tests (attention / MLP / ViT / projector)
│   ├── test_tp_mlp.py               # TP MLP correctness test
│   ├── test_10_videos.py            # 105-video back-to-back suite
│   ├── profile_blocks.py            # all-blocks combined Tracy profiling
│   ├── profile_single_block.py      # single-block Tracy profiling
│   ├── run_block_profiles.sh        # orchestrates 6 per-block Tracy runs + Excel
│   └── make_profile_xlsx.py         # builds multi-sheet Excel from per-block CSVs
└── verification/                    # 105-video test suite assets
    ├── test.jsonl                   # test prompts
    ├── test_results.jsonl           # reference answers
    ├── videos/                      # video files (downloaded separately)
    └── run_video_tests.py           # HTTP client for server testing
```

## Known Limitations

0. **Slice 0 of the test Galaxy (tray 3 chip 4) hangs on long prefills** (34k-token video, 20-image prompt); slices 1-3 run
   everything, see BRINGUP_LOG Session 11. Decode is ~19 tok/s (earlier T3K reference ~35 tok/s):
   `verification/latency_comparison.md`. Request lengths up to the model's 36,864-token context work on healthy slices.

1. **Batch size = 1** — multi-user concurrent inference not yet implemented
   (`forward_decode_step` decodes KV-cache slot 0; the demo runs users back to back).
2. **Decode trace** — disabled in server path; trace captured at first request's S
   does not scale to requests with different sequence lengths (different SDPA
   program config). Each decode step uses `forward_decode_step` (TTNN, no trace).
3. **ViT AllBroadcast** — 68% of ViT device time is CCL (all_reduce after each of
   25 blocks). Target for async/pipelined CCL optimization.
4. **Very long multi-image prompts** — the 20-image stress test (S≈25K) starts
   correctly but degenerates into repetition after ~50 tokens (HF reference stays fluent).
5. **5 remaining failures** (idx 22, 26, 55, 56, 76 in the 105-video suite) —
   all have logit margins < 1.2; likely genuine model uncertainty on ambiguous
   video content.
