# TT-Metal SDXL Server Guide

Complete guide for running the SDXL inference server directly from tt-metal using Tenstorrent hardware acceleration.

## Quick Start

```bash
cd /home/tt-admin/tt-metal

# 1. Check if everything is ready
./check_sdxl_readiness.sh

# 2. Start the server
./start_sdxl_server.sh

# 3. Test in another terminal
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a beautiful landscape", "num_inference_steps": 20}'
```

## Overview

This server implementation:
- ✅ Runs directly from `/home/tt-admin/tt-metal/`
- ✅ Uses tt-metal's native SDXL implementation
- ✅ Leverages battle-tested code from `test_common.py`
- ✅ FastAPI with OpenAI-compatible endpoints
- ✅ Single-device focused (simple architecture)
- ✅ No external dependencies on tt-inference-server

## Files Created

| File | Purpose | Location |
|------|---------|----------|
| `sdxl_server.py` | Main FastAPI server | `/home/tt-admin/tt-metal/` |
| `start_sdxl_server.sh` | Startup script | `/home/tt-admin/tt-metal/` |
| `setup_sdxl_env.sh` | Environment setup | `/home/tt-admin/tt-metal/` |
| `check_sdxl_readiness.sh` | Diagnostic tool | `/home/tt-admin/tt-metal/` |
| `SDXL_SERVER_GUIDE.md` | This documentation | `/home/tt-admin/tt-metal/` |

## Architecture

### Design Philosophy

**Simplicity over complexity:**
- Single process, single device focus
- Direct imports from `models.experimental.stable_diffusion_xl_base`
- No abstraction layers or worker processes
- Uses tt-metal's own test infrastructure

**Native tt-metal integration:**
```python
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    create_tt_clip_text_encoders,
    batch_encode_prompt_on_device,
    run_tt_image_gen
)
```

**FastAPI-based:**
- Modern async Python web framework
- Automatic API documentation
- OpenAI-compatible request/response format
- Built-in validation with Pydantic

### Server Flow

```
Client Request
    ↓
FastAPI Endpoint (/image/generations)
    ↓
Validate & Authenticate
    ↓
SDXLPipelineState.generate_image()
    ↓
run_tt_image_gen() [from test_common.py]
    ↓
TT-Metal SDXL Pipeline (on-device inference)
    ↓
PIL Image → Base64 or File
    ↓
JSON Response to Client
```

## Prerequisites

### Required
- **tt-metal SDK**: Built and installed at `/home/tt-admin/tt-metal/`
- **Python environment**: `/home/tt-admin/tt-metal/python_env/` (created via `create_venv.sh`)
- **tt-metal build**: Compiled libraries in `build/lib/`
- **Tenstorrent device**: At least one device available

### System Dependencies
Automatically checked/installed by startup script:
- libgl1
- libsndfile1
- ffmpeg (optional, for audio)

### Python Dependencies
Automatically installed by startup script:
- fastapi
- uvicorn[standard]
- python-multipart
- pillow
- pydantic
- loguru

Already in tt-metal python_env:
- ttnn
- torch
- diffusers
- transformers

## Installation & Setup

### Option 1: Automatic Setup (Recommended)

The startup script handles everything:

```bash
cd /home/tt-admin/tt-metal
./start_sdxl_server.sh
```

First run will:
1. Check prerequisites
2. Install missing Python packages
3. Download SDXL model (~7GB) from HuggingFace
4. Initialize pipeline
5. Start server

### Option 2: Manual Setup

```bash
cd /home/tt-admin/tt-metal

# 1. Ensure tt-metal is built
./build_metal.sh --release

# 2. Create Python environment (if not exists)
./create_venv.sh

# 3. Run setup script
./setup_sdxl_env.sh

# 4. Verify readiness
./check_sdxl_readiness.sh

# 5. Start server
./start_sdxl_server.sh
```

## Configuration

### Environment Variables

All configuration via environment variables (set before running `start_sdxl_server.sh`):

#### Core Paths (Auto-configured)
```bash
export TT_METAL_HOME=/home/tt-admin/tt-metal
export PYTHONPATH=/home/tt-admin/tt-metal
export LD_LIBRARY_PATH=/home/tt-admin/tt-metal/build/lib
export PYTHON_ENV_DIR=/home/tt-admin/tt-metal/python_env
```

#### Device Configuration
```bash
export DEVICE_ID=0                    # Device to use (default: 0)
export TT_VISIBLE_DEVICES=0           # Visible devices
export TT_MM_THROTTLE_PERF=5          # Performance throttling (default: 5)
```

#### Server Settings
```bash
export PORT=8000                      # Server port (default: 8000)
export API_KEY=your-secret-key        # API authentication key
export OUTPUT_DIR=./generated_images  # Image output directory
```

#### Generation Defaults
```bash
export NUM_INFERENCE_STEPS=20         # Default steps (default: 20)
export GUIDANCE_SCALE=7.5             # CFG scale (default: 7.5)
export DEFAULT_WIDTH=1024             # Image width (default: 1024)
export DEFAULT_HEIGHT=1024            # Image height (default: 1024)
```

#### HuggingFace Configuration
```bash
export HF_HOME=/home/tt-admin/cache_root/huggingface  # Model cache
export HF_TOKEN=hf_xxxxx              # Token for gated models (optional)
export HF_MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0
```

#### Logging
```bash
export LOG_LEVEL=INFO                 # Python logging level
export LOGURU_LEVEL=INFO              # Loguru logging level
```

### Command-Line Options

```bash
./start_sdxl_server.sh [OPTIONS]

Options:
  --port PORT              : Server port (default: 8000)
  --device-id ID           : Device ID (default: 0)
  --api-key KEY            : API key for authentication
  --inference-steps N      : Default inference steps (default: 20)
  --help                   : Show help message
```

## Usage Examples

### Starting the Server

**Basic startup:**
```bash
./start_sdxl_server.sh
```

**Custom port:**
```bash
./start_sdxl_server.sh --port 8001
```

**With custom API key:**
```bash
./start_sdxl_server.sh --api-key "my-secret-key-123"
```

**Using environment variables:**
```bash
export API_KEY="production-key"
export NUM_INFERENCE_STEPS=50
export DEVICE_ID=1
./start_sdxl_server.sh
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "device_id": 0,
  "model_loaded": true,
  "uptime_seconds": 123.45
}
```

#### 2. Image Generation (OpenAI-compatible)

**Basic generation:**
```bash
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a beautiful landscape with mountains and a lake",
    "num_inference_steps": 20
  }'
```

**Advanced generation:**
```bash
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a serene japanese garden with cherry blossoms",
    "negative_prompt": "low quality, blurry, distorted",
    "num_inference_steps": 30,
    "guidance_scale": 8.0,
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "number_of_images": 2,
    "response_format": "b64_json"
  }'
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of desired image |
| `negative_prompt` | string | "low quality..." | What to avoid |
| `num_inference_steps` | int | 20 | Denoising steps (1-100) |
| `guidance_scale` | float | 7.5 | CFG scale (1.0-20.0) |
| `width` | int | 1024 | Image width (multiple of 8) |
| `height` | int | 1024 | Image height (multiple of 8) |
| `seed` | int | random | Random seed for reproducibility |
| `number_of_images` | int | 1 | Number of images (1-4) |
| `response_format` | string | "b64_json" | "b64_json" or "url" |

**Response (b64_json):**
```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

**Response (url):**
```json
{
  "created": 1700000000,
  "data": [
    {
      "url": "/generated_images/sdxl_a1b2c3d4.png"
    }
  ]
}
```

#### 3. Root Endpoint

```bash
curl http://localhost:8000/
```

Shows server info and available endpoints.

#### 4. Interactive Documentation

Visit in browser: **http://localhost:8000/docs**

Provides:
- Interactive API testing
- Full request/response schemas
- Try-it-out functionality
- Authentication testing

### Python Client Example

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Configuration
API_URL = "http://localhost:8000/image/generations"
API_KEY = "default-insecure-key"

# Request
response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "prompt": "a futuristic city at sunset",
        "negative_prompt": "low quality, blurry",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42,
        "response_format": "b64_json"
    }
)

# Parse response
if response.status_code == 200:
    data = response.json()
    b64_string = data["data"][0]["b64_json"]

    # Decode and save
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data))
    image.save("output.png")
    print("Image saved to output.png")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Troubleshooting

### Running Diagnostics

Always start with the diagnostic tool:

```bash
./check_sdxl_readiness.sh
```

This checks:
- ✓ Directory structure
- ✓ tt-metal build status
- ✓ Python environment
- ✓ Package installations
- ✓ SDXL model files
- ✓ Import functionality
- ✓ Device availability
- ✓ Disk space
- ✓ Port availability

### Common Issues

#### Issue: "tt-metal not built"
```bash
# Solution:
cd /home/tt-admin/tt-metal
./build_metal.sh --release
```

#### Issue: "Python environment not found"
```bash
# Solution:
cd /home/tt-admin/tt-metal
./create_venv.sh
```

#### Issue: "Module 'fastapi' not found"
```bash
# Solution:
./setup_sdxl_env.sh
# Or manually:
source python_env/bin/activate
pip install fastapi uvicorn[standard] pillow python-multipart
```

#### Issue: "Cannot import ttnn"
```bash
# Check PYTHONPATH:
export PYTHONPATH=/home/tt-admin/tt-metal
source python_env/bin/activate
python -c "import ttnn"
```

#### Issue: "No devices found"
```bash
# Check device status:
tt-smi

# Verify device permissions
ls -l /dev/tenstorrent/

# Check environment:
echo $TT_VISIBLE_DEVICES
```

#### Issue: "Model download fails"
```bash
# Check internet connection
ping huggingface.co

# Set HuggingFace token if needed:
export HF_TOKEN="hf_xxxxx"
./start_sdxl_server.sh

# Or pre-download:
python -c "from diffusers import StableDiffusionXLPipeline; \
           StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"
```

#### Issue: "Port 8000 already in use"
```bash
# Use different port:
./start_sdxl_server.sh --port 8001

# Or kill existing process:
lsof -ti:8000 | xargs kill -9
```

#### Issue: "Pipeline initialization fails"
```bash
# Check logs for specific error
./start_sdxl_server.sh

# Common fixes:
# 1. Ensure device is not being used:
tt-smi -r  # Reset devices

# 2. Check memory:
free -h

# 3. Verify model files:
ls -la ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/
```

#### Issue: "Generation is slow"
```bash
# 1. First run is slower (compilation)
# Subsequent runs will be faster

# 2. Reduce inference steps:
export NUM_INFERENCE_STEPS=10
./start_sdxl_server.sh

# 3. Check throttling:
export TT_MM_THROTTLE_PERF=1  # Less throttling
```

### Performance Tuning

**Faster generation (lower quality):**
```bash
export NUM_INFERENCE_STEPS=10
export GUIDANCE_SCALE=7.0
```

**Higher quality (slower):**
```bash
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=8.5
```

**Memory optimization:**
```bash
# Run VAE on CPU if device memory is limited
# (Requires code modification in sdxl_server.py)
```

## Production Deployment

### Security Considerations

**1. Change default API key:**
```bash
export API_KEY="$(openssl rand -base64 32)"
./start_sdxl_server.sh
```

**2. Use HTTPS:**
- Deploy behind nginx/traefik with SSL
- Or use uvicorn with SSL certificates:
```bash
uvicorn sdxl_server:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem
```

**3. Disable interactive docs in production:**
Edit `sdxl_server.py`:
```python
app = FastAPI(
    title="TT-Metal SDXL Server",
    docs_url=None,  # Disable /docs
    redoc_url=None  # Disable /redoc
)
```

### Systemd Service

Create `/etc/systemd/system/tt-sdxl-server.service`:

```ini
[Unit]
Description=TT-Metal SDXL Server
After=network.target

[Service]
Type=simple
User=tt-admin
WorkingDirectory=/home/tt-admin/tt-metal
Environment="TT_METAL_HOME=/home/tt-admin/tt-metal"
Environment="PYTHONPATH=/home/tt-admin/tt-metal"
Environment="LD_LIBRARY_PATH=/home/tt-admin/tt-metal/build/lib"
Environment="HF_HOME=/home/tt-admin/cache_root/huggingface"
Environment="API_KEY=your-production-key-here"
Environment="PORT=8000"
Environment="DEVICE_ID=0"
ExecStart=/home/tt-admin/tt-metal/start_sdxl_server.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tt-sdxl-server
sudo systemctl start tt-sdxl-server
sudo systemctl status tt-sdxl-server
```

View logs:
```bash
sudo journalctl -u tt-sdxl-server -f
```

### Monitoring

**Check health:**
```bash
curl http://localhost:8000/health
```

**Monitor with watch:**
```bash
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

**Check device utilization:**
```bash
watch -n 1 tt-smi
```

## Comparison with tt-media-server

| Feature | tt-metal SDXL Server | tt-media-server |
|---------|---------------------|-----------------|
| **Location** | `/home/tt-admin/tt-metal/` | `/home/tt-admin/tt-inference-server/` |
| **Architecture** | Single process, simple | Multi-process with workers |
| **Dependencies** | tt-metal only | Separate repository |
| **Code reuse** | Uses test_common.py | Custom wrappers |
| **Complexity** | Low (< 500 lines) | High (> 5000 lines) |
| **Setup** | One script | Docker or complex setup |
| **Maintenance** | Easier | More complex |
| **Multi-device** | Single device focus | Multi-device support |

**When to use tt-metal server:**
- ✅ Development and testing
- ✅ Single device deployment
- ✅ Simpler maintenance
- ✅ Direct tt-metal integration

**When to use tt-media-server:**
- ✅ Multi-device orchestration
- ✅ Production at scale
- ✅ Multiple model types (Whisper, etc.)
- ✅ Advanced queue management

## Advanced Topics

### Custom Model Configuration

Edit `sdxl_server.py` to change model configuration:

```python
# Use different optimization level
self.model_config = get_model_config(
    device=self.device,
    optimisations=ModelOptimisations.ACCURACY  # vs PERFORMANCE
)
```

### Adding Custom Endpoints

Add to `sdxl_server.py`:

```python
@app.get("/custom/endpoint")
async def custom_endpoint():
    return {"message": "Custom endpoint"}
```

### Modifying Generation Pipeline

The generation logic is in `SDXLPipelineState.generate_image()`. You can:
- Add preprocessing
- Modify prompt handling
- Implement different schedulers
- Add post-processing

### Logging Configuration

Adjust logging in `sdxl_server.py`:

```python
logger.add(
    "server.log",
    rotation="100 MB",
    level="DEBUG"
)
```

## Files Reference

### `/home/tt-admin/tt-metal/sdxl_server.py`
Main server implementation (Python). Contains:
- `ServerConfig`: Configuration class
- `SDXLPipelineState`: Pipeline management
- FastAPI routes and endpoints
- Request/response models

### `/home/tt-admin/tt-metal/start_sdxl_server.sh`
Startup script (Bash). Performs:
- Prerequisites checking
- Environment configuration
- Python activation
- Dependency installation
- Server launch

### `/home/tt-admin/tt-metal/setup_sdxl_env.sh`
Setup script (Bash). Handles:
- Directory creation
- Python package installation
- Import verification
- SDXL component checking

### `/home/tt-admin/tt-metal/check_sdxl_readiness.sh`
Diagnostic tool (Bash). Checks:
- All prerequisites
- Build status
- Package installations
- Device availability
- Configuration

## Support & Resources

**Official tt-metal documentation:**
- SDXL implementation: `models/experimental/stable_diffusion_xl_base/`
- Test utilities: `models/experimental/stable_diffusion_xl_base/tests/test_common.py`

**FastAPI documentation:**
- https://fastapi.tiangolo.com/

**Getting help:**
1. Run `./check_sdxl_readiness.sh` for diagnostics
2. Check server logs for errors
3. Review this guide's troubleshooting section

## Summary

**To get started:**
```bash
cd /home/tt-admin/tt-metal
./check_sdxl_readiness.sh
./start_sdxl_server.sh
```

**To test:**
```bash
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a serene mountain landscape"}'
```

**Key points:**
- Runs from tt-metal directory
- Uses tt-metal's Python environment
- Simple, single-process architecture
- OpenAI-compatible API
- Production-ready with proper configuration

Enjoy generating images with SDXL on Tenstorrent hardware! 🚀
