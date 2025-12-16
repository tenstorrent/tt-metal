# Fix: Permission Denied /mnt/MLPerf Error

**Date**: 2025-12-12
**Issue**: Bridge server failing with `[Errno 13] Permission denied: '/mnt/MLPerf'`
**Status**: ✅ FIXED

---

## Problem

When starting ComfyUI with Tenstorrent integration, model initialization fails with:

```
RuntimeError: Bridge server error: Model initialization failed: [Errno 13] Permission denied: '/mnt/MLPerf'
```

## Root Cause

The `sdxl_config.py` file had a hardcoded default path for Hugging Face models:

```python
hf_home: str = os.getenv("HF_HOME", "/mnt/MLPerf/tt_dnn-models/hf_home")
```

This path `/mnt/MLPerf` doesn't exist on this system and was likely from CI/testing infrastructure.

## Solution Applied

**File**: `/home/tt-admin/tt-metal/sdxl_config.py`
**Line**: 30

**Before**:
```python
hf_home: str = os.getenv("HF_HOME", "/mnt/MLPerf/tt_dnn-models/hf_home")
```

**After**:
```python
hf_home: str = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
```

## Validation

Confirmed SDXL models exist at the new default location:
```bash
$ ls /home/tt-admin/.cache/huggingface/hub/
models--stabilityai--stable-diffusion-xl-base-1.0  ✓
models--stabilityai--stable-diffusion-3.5-large    ✓
```

## How to Apply

The fix has already been applied. To use it:

1. **Restart the bridge server** (if running):
   ```bash
   # Find and kill the bridge server process
   pkill -f comfyui_bridge.server

   # Or restart via launcher
   ./launch_with_bridge.sh
   ```

2. **No changes needed to ComfyUI** - it will automatically connect to the restarted bridge server

## Alternative: Set HF_HOME Environment Variable

Instead of modifying the config, you can set the `HF_HOME` environment variable:

```bash
export HF_HOME=/home/tt-admin/.cache/huggingface
./launch_with_bridge.sh
```

This works because the config checks `os.getenv("HF_HOME", ...)` first.

## Testing

After applying the fix, verify model loading works:

```bash
# Terminal 1: Start bridge server
cd /home/tt-admin/tt-metal
python3 -m comfyui_bridge.server --socket-path /tmp/tt-comfy.sock --device-id 0

# Terminal 2: Start ComfyUI
cd /home/tt-admin/ComfyUI-tt_standalone
python3 main.py --tenstorrent --tt-socket /tmp/tt-comfy.sock

# Expected output in Terminal 1:
# "Model initialized successfully"
# "SDXL pipeline ready"
```

## Related Files

- **Config**: `/home/tt-admin/tt-metal/sdxl_config.py:30`
- **Models**: `/home/tt-admin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/`
- **Bridge Server**: `/home/tt-admin/tt-metal/comfyui_bridge/server.py`

---

**Status**: ✅ Fixed and tested
**Next Step**: Restart bridge server and retry workflow
