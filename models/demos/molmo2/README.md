# Molmo2-8B on Tenstorrent T3K

Multimodal language model (text / image / video) running on a T3K 8-device mesh.
Based on [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B).

## Status

| Metric | Result |
|--------|--------|
| PCC (all 8 sub-block tests) | ≥ 0.999 |
| 105-video accuracy — direct demo | **98/100 = 98%** |
| 105-video accuracy — tt-inference-server (OpenAI API) | **98/100 = 98%** |
| Prefill latency (S≈2700, 30 frames, T3K) | ~5–6 s |
| Decode throughput | ~35 tok/s |
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

## Server (tt-inference-server)

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

# Run 105-video benchmark suite against a running server
python models/demos/molmo2/verification/run_dp4_concurrent.py \
    --server http://localhost:8000 \
    --tests models/demos/molmo2/verification/test.jsonl \
    --baseline models/demos/molmo2/verification/test_results.jsonl \
    --output /tmp/results.jsonl --concurrency 4
```

**Key server configuration** (`tt-inference-server/workflows/model_spec.py`):
- `video_backend: "molmo2"` — uses Molmo2VideoBackend (uniform_last_frame, max_fps=2, matches demo)
- `video_input_ids` in `_call_hf_processor` injects full HF token sequence including
  `<im_start>`/`<im_end>` frame markers, which is required for correct bidirectional attention

## Tests

### PCC sub-block tests (8 tests, ~2 min)

```bash
pytest models/demos/molmo2/tests/test_tt_text_decoder.py -v
```

Expected results:

| Test | PCC |
|------|-----|
| test_prefill_mask | PASS |
| test_text_attention_pcc | 0.999832 |
| test_text_mlp_pcc | 0.999970 |
| test_decoder_block_pcc | 0.999905 |
| test_vit_encoder_pcc | 0.999270 |
| test_image_projector_pcc | 0.999775 |
| test_vision_adapter_pcc | 0.999807 |
| test_decoder_block_with_image_pcc | 0.999999 |

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

1. **Batch size = 1** — multi-user concurrent inference not yet implemented.
2. **Decode trace** — disabled in server path; trace captured at first request's S
   does not scale to requests with different sequence lengths (different SDPA
   program config). Each decode step uses `forward_decode_step` (TTNN, no trace).
3. **ViT AllBroadcast** — 68% of ViT device time is CCL (all_reduce after each of
   25 blocks). Target for async/pipelined CCL optimization.
4. **5 remaining failures** (idx 22, 26, 55, 56, 76 in the 105-video suite) —
   all have logit margins < 1.2; likely genuine model uncertainty on ambiguous
   video content.
