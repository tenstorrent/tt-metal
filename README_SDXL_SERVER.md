# SDXL Server - Quick Reference

**Location**: `/home/tt-admin/tt-metal/`

This directory now contains a complete, standalone SDXL inference server implementation for Tenstorrent hardware.

## 🚀 Quick Start (3 Steps)

```bash
cd /home/tt-admin/tt-metal

# Step 1: Check readiness
./check_sdxl_readiness.sh

# Step 2: Start server
./start_sdxl_server.sh

# Step 3: Test (in another terminal)
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a beautiful landscape"}'
```

## 📁 Files Overview

| File | Purpose | Size |
|------|---------|------|
| **`sdxl_server.py`** | Main FastAPI server implementation | 15K |
| **`start_sdxl_server.sh`** | Startup script (run this!) | 11K |
| **`setup_sdxl_env.sh`** | Environment setup only | 9.3K |
| **`check_sdxl_readiness.sh`** | Diagnostic tool | 15K |
| **`SDXL_SERVER_GUIDE.md`** | Complete documentation | 17K |
| **`README_SDXL_SERVER.md`** | This quick reference | - |

## ✨ Key Features

- ✅ **Native tt-metal integration** - Uses tt-metal's SDXL implementation directly
- ✅ **Simple architecture** - Single process, easy to understand and maintain
- ✅ **Battle-tested code** - Leverages `test_common.py` utilities
- ✅ **OpenAI-compatible API** - Standard REST API format
- ✅ **Self-contained** - No dependency on tt-inference-server
- ✅ **Production-ready** - Includes health checks, authentication, logging

## 🎯 What This Does

This implementation:

1. **Runs from tt-metal directory** - Uses `/home/tt-admin/tt-metal/python_env`
2. **Direct SDXL access** - Imports from `models.experimental.stable_diffusion_xl_base`
3. **FastAPI server** - Modern Python web framework
4. **Device management** - Initializes and manages Tenstorrent device
5. **Image generation** - Text-to-image via REST API

## 🔧 Common Commands

### Start the server
```bash
./start_sdxl_server.sh
```

### Start with custom settings
```bash
./start_sdxl_server.sh --port 8001 --device-id 0 --api-key "my-key"
```

### Run diagnostics
```bash
./check_sdxl_readiness.sh
```

### Setup environment only
```bash
./setup_sdxl_env.sh
```

### View documentation
```bash
cat SDXL_SERVER_GUIDE.md
# Or open in browser: http://localhost:8000/docs (when server is running)
```

## 📚 Documentation

**For complete details, see `SDXL_SERVER_GUIDE.md`:**
- Full installation instructions
- Configuration options
- API documentation
- Python client examples
- Troubleshooting guide
- Production deployment
- Performance tuning

## 🔍 Quick Diagnostics

If something isn't working:

```bash
# 1. Run the diagnostic tool
./check_sdxl_readiness.sh

# 2. Common fixes:
./build_metal.sh --release        # If tt-metal not built
./create_venv.sh                  # If Python env missing
./setup_sdxl_env.sh              # To install dependencies

# 3. Check device
tt-smi                           # View device status
```

## 🌐 API Endpoints

Once running, the server provides:

- **`GET /`** - Server info
- **`GET /health`** - Health check
- **`POST /image/generations`** - Generate images (OpenAI-compatible)
- **`GET /docs`** - Interactive API documentation
- **`GET /ping`** - Simple ping test

## 🎨 Generate Your First Image

```bash
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a serene japanese garden with cherry blossoms",
    "negative_prompt": "low quality, blurry",
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

Images are saved to `/home/tt-admin/tt-metal/generated_images/`

## ⚙️ Configuration

All configuration via environment variables:

```bash
export PORT=8000                      # Server port
export DEVICE_ID=0                    # Device to use
export API_KEY=your-secret-key        # Authentication
export NUM_INFERENCE_STEPS=20         # Default steps
export HF_HOME=/path/to/cache         # Model cache location
```

See `SDXL_SERVER_GUIDE.md` for complete configuration reference.

## 🆚 vs tt-media-server

| Feature | This (tt-metal) | tt-media-server |
|---------|-----------------|-----------------|
| Location | `/home/tt-admin/tt-metal/` | `/home/tt-admin/tt-inference-server/` |
| Complexity | Simple (< 500 lines) | Complex (> 5000 lines) |
| Architecture | Single process | Multi-process workers |
| Setup | One script | Docker or complex |
| Maintenance | Easy | More involved |
| Best for | Development, single device | Production, multi-device |

**This implementation is recommended for:**
- Local development
- Single device deployment
- Easier debugging and maintenance
- Direct access to tt-metal code

## 🐛 Troubleshooting

**Server won't start?**
```bash
./check_sdxl_readiness.sh  # Detailed diagnostics
```

**Missing dependencies?**
```bash
./setup_sdxl_env.sh
```

**Device issues?**
```bash
tt-smi                     # Check device status
tt-smi -r                  # Reset devices
```

**Port in use?**
```bash
./start_sdxl_server.sh --port 8001
```

## 📖 Next Steps

1. **Read the full guide**: `SDXL_SERVER_GUIDE.md`
2. **Test the server**: Use the curl commands above
3. **Explore the API**: Visit `http://localhost:8000/docs`
4. **Customize**: Edit environment variables or `sdxl_server.py`

## 🎉 Success!

You now have a fully functional SDXL server running on Tenstorrent hardware, integrated directly with tt-metal!

For questions or issues, refer to:
- **Complete guide**: `SDXL_SERVER_GUIDE.md`
- **Diagnostics**: `./check_sdxl_readiness.sh`
- **Server logs**: Run startup script to see detailed logs

Happy image generation! 🚀🎨
