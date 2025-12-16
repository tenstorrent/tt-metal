# Multi-Device Configuration for SDXL Server

## Why Multiple Devices?

SDXL requires significant memory. A single device doesn't have enough L1 memory to run the model. You need at least **2 devices** in TP2 (Tensor Parallel) mode.

## Your Hardware Options

You mentioned you have:
- **4x N300 devices**
- **1x T3K (8 devices)**

## Configuration Options

### Option 1: 2 Devices (TP2 - Recommended for Testing)

Use 2 N300 devices:

```bash
export MESH_ROWS=1
export MESH_COLS=2
export DEVICE_IDS="0,1"
./start_sdxl_server.sh
```

Or set in one command:
```bash
MESH_ROWS=1 MESH_COLS=2 DEVICE_IDS="0,1" ./start_sdxl_server.sh
```

### Option 2: 4 Devices (2x2 Mesh)

Use all 4 N300 devices:

```bash
export MESH_ROWS=2
export MESH_COLS=2
export DEVICE_IDS="0,1,2,3"
./start_sdxl_server.sh
```

### Option 3: 8 Devices (T3K)

Use the full T3K:

```bash
export MESH_ROWS=2
export MESH_COLS=4
export DEVICE_IDS="0,1,2,3,4,5,6,7"
./start_sdxl_server.sh
```

## Current Default

The server is now configured with:
- **MESH_ROWS=1**
- **MESH_COLS=2**
- **DEVICE_IDS="0,1"**

This uses 2 devices in a 1x2 mesh, which is the minimum for SDXL.

## Quick Test

Start the server with default 2-device config:
```bash
cd /home/tt-admin/tt-metal
./start_sdxl_server.sh
```

Then test:
```bash
python image_test.py "Beautiful purple volcano on a beach"
```

## Checking Configuration

Once the server starts, check the configuration:
```bash
curl http://localhost:8000/health
```

Response will show:
```json
{
  "status": "healthy",
  "mesh_shape": "1x2",
  "device_ids": "0,1",
  "pipeline_loaded": true,
  "uptime_seconds": 123.45
}
```

## Memory Requirements

| Configuration | Devices | Memory | Speed | Use Case |
|---------------|---------|--------|-------|----------|
| 1x1 (single)  | 1       | ❌ Too small | - | Won't work |
| 1x2 (TP2)     | 2       | ✅ Sufficient | Fast | Testing, single user |
| 2x2           | 4       | ✅✅ More | Faster | Multiple users |
| 2x4 (T3K)     | 8       | ✅✅✅ Most | Fastest | Production |

## Troubleshooting

### "Out of Memory" Error
- You're using only 1 device
- Solution: Use at least 2 devices with TP2 mode

### "Not enough devices for mesh"
- DEVICE_IDS doesn't have enough IDs for the mesh
- Example: MESH=2x2 needs 4 devices, but DEVICE_IDS="0,1" only has 2
- Solution: Match the number of devices to mesh size

### Device Not Found
- One of the device IDs doesn't exist
- Check available devices: `tt-smi`
- Update DEVICE_IDS to use valid device numbers

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| MESH_ROWS | 1 | Number of rows in mesh |
| MESH_COLS | 2 | Number of columns in mesh |
| DEVICE_IDS | "0,1" | Comma-separated device IDs |
| PORT | 8000 | Server port |
| NUM_INFERENCE_STEPS | 20 | Default inference steps |

## Examples

**Start with 2 devices (default):**
```bash
./start_sdxl_server.sh
```

**Start with 4 devices:**
```bash
DEVICE_IDS="0,1,2,3" MESH_ROWS=2 MESH_COLS=2 ./start_sdxl_server.sh
```

**Start with T3K (8 devices):**
```bash
DEVICE_IDS="0,1,2,3,4,5,6,7" MESH_ROWS=2 MESH_COLS=4 ./start_sdxl_server.sh
```

**Custom port with 2 devices:**
```bash
PORT=8001 DEVICE_IDS="2,3" ./start_sdxl_server.sh
```
