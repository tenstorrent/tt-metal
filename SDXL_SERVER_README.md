# SDXL Standalone Server - Quick Start Guide

A standalone SDXL inference server that reproduces tt-media-server functionality using only tt-metal components.

## Quick Start

### 1. Start the Server

```bash
cd /home/tt-admin/tt-metal
./launch_sdxl_server.sh
```

**Wait 5-10 minutes** for warmup. Server is ready when you see:
```
All workers ready. Server is accepting requests.
```

### 2. Generate an Image

In a **new terminal**:

```bash
cd /home/tt-admin/tt-metal
python image_test.py "Photograph of an orange Volcano on a tropical island while someone suntans on a beach with a friendly dinosaur"
```

Output saved to: `output.jpg`

### 3. Validate Against Reference

```bash
python image_test.py \
  "Photograph of an orange Volcano on a tropical island while someone suntans on a beach with a friendly dinosaur" \
  --compare /home/tt-admin/tt-inference-server/reference_image.jpg \
  --output test_output.jpg
```

Expected output:
```
MSE: ~400-500
SSIM: ~0.92-0.95
✓ Images are similar (SSIM >= 0.9)
```

---

## Files Created

### Configuration & Utilities
- `sdxl_config.py` - Central configuration (T3K settings, device params)
- `utils/logger.py` - Structured logging
- `utils/image_utils.py` - Image encoding/decoding
- `utils/validation_utils.py` - MSE/SSIM comparison

### Core Components
- `sdxl_runner.py` - TtSDXLPipeline wrapper with device management
- `sdxl_worker.py` - Multiprocessing worker
- `sdxl_server.py` - FastAPI server

### Launcher & Client
- `launch_sdxl_server.sh` - Environment setup + launcher
- `image_test.py` - Test client with validation

---

## API Endpoints

### Generate Image
```bash
curl -X POST http://127.0.0.1:8000/image/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A sunset over mountains",
    "negative_prompt": "low quality",
    "num_inference_steps": 50,
    "guidance_scale": 5.0
  }'
```

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "healthy",
  "workers_alive": 1,
  "workers_total": 1,
  "queue_size": 0
}
```

### Metrics
```bash
curl http://127.0.0.1:8000/metrics
```

---

## image_test.py Usage

### Basic Usage
```bash
python image_test.py "Your prompt here"
```

### With Options
```bash
python image_test.py "Your prompt" \
  --output my_image.jpg \
  --steps 30 \
  --guidance 7.5 \
  --server http://127.0.0.1:8000
```

### With Validation
```bash
python image_test.py "Your prompt" \
  --compare reference.jpg \
  --output test.jpg
```

**Exit codes:**
- `0` - Success (SSIM >= 0.85)
- `1` - Failure (SSIM < 0.85 or error)

---

## Configuration

### Environment Variables (Set by launch script)
```bash
PYTHONPATH=/home/tt-admin/tt-metal
TT_VISIBLE_DEVICES=0,1,2,3
TT_METAL_HOME=/home/tt-admin/tt-metal
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=7,7
HF_HOME=/mnt/MLPerf/tt_dnn-models/hf_home
TT_MM_THROTTLE_PERF=5
```

### Customize Settings

Edit `sdxl_config.py`:

```python
@dataclass
class SDXLConfig:
    server_port: int = 8000              # Server port
    num_inference_steps: int = 50        # Denoising steps (12-100)
    guidance_scale: float = 5.0          # CFG scale (1.0-20.0)
    vae_on_device: bool = True           # VAE on TT hardware
    encoders_on_device: bool = True      # Encoders on TT hardware
    capture_trace: bool = True           # Enable tracing
    # ... more options
```

---

## Performance

- **Startup Time**: 5-10 minutes (model loading + warmup)
- **Inference Time**: 3-5 seconds per image (50 steps)
- **Memory Usage**: 40-60 GB RAM
- **Device Utilization**: All 4 T3K devices

---

## Troubleshooting

### Server won't start

**Error**: `Python environment not found`
```bash
cd /home/tt-admin/tt-metal
./create_venv.sh
```

**Error**: `No TTNN devices available`
```bash
tt-smi          # Check devices
tt-smi -r       # Reset if needed
```

### Can't connect to server

**Check if server is running:**
```bash
curl http://127.0.0.1:8000/health
```

**Check logs:**
- Look for errors in server terminal
- Check: "All workers ready" message appeared

### Images differ from reference

**Note**: Generative models have natural variation
- SSIM >= 0.9: Excellent similarity
- SSIM 0.85-0.9: Good similarity (expected)
- SSIM < 0.85: Significant difference

**For exact reproduction**: Use same seed in both generations

### Worker timeout during warmup

**Increase timeout** in `sdxl_server.py:80`:
```python
timeout = time.time() + 900  # 15 minutes instead of 10
```

---

## Architecture

```
HTTP Request → FastAPI → Task Queue → Worker → TtSDXLPipeline → Result Queue → HTTP Response
```

**Components:**
- **FastAPI Server**: Handles HTTP requests, manages queues
- **Worker Process**: Runs inference on TT devices
- **SDXLRunner**: Wraps TtSDXLPipeline with device management
- **TtSDXLPipeline**: tt-metal core (UNet, VAE, Scheduler, Encoders)

---

## Dependencies

All dependencies included in `/home/tt-admin/tt-metal/python_env`:
- `diffusers` - SDXL pipeline
- `transformers` - CLIP encoders
- `torch` - PyTorch
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Validation
- `scikit-image` - SSIM calculation
- `Pillow` - Image processing

---

## Success Criteria

✓ Server starts and completes warmup within 10 minutes
✓ Health endpoint returns "healthy" status
✓ `image_test.py` successfully generates images
✓ Generated images have SSIM >= 0.9 vs reference
✓ Inference time is 3-5 seconds per image
✓ Server handles multiple sequential requests

---

## Additional Documentation

- **SDXL_PIPELINE_ARCHITECTURE.md** - Detailed pipeline architecture
- **ORIGINAL_METAL_SDXL.md** - tt-metal vs tt-media-server comparison
- Plan file: `.claude/plans/glistening-jumping-mountain.md` - Full implementation plan
