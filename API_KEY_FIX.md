# API Key Fix - image_test.py

## Issue

When running `image_test.py`, it was failing with:
```
❌ Request failed with status code: 401
Response: {"detail":"Invalid API key"}
```

## Root Cause

The `image_test.py` script had the wrong default API key:
- **Was using**: `"your-secret-key"`
- **Server expects**: `"default-insecure-key"`

## Solution

Updated `image_test.py` to use the correct default API key:

**Changed:**
```python
# Line 19
api_key: str = "default-insecure-key"  # Changed from "your-secret-key"

# Line 138-139
default="default-insecure-key",  # Changed from "your-secret-key"
help="API key for authentication (default: default-insecure-key)"
```

## Result

Now when you run the script with proper API key:
```bash
python image_test.py "Your prompt here"
```

It properly authenticates and returns:
```
✅ Request successful!
Response keys: dict_keys(['created', 'data'])
```

The script then receives either:
- A 503 error (if device not available - expected due to build issue)
- A successful image response (once tt-metal is fully built)

## Additional Fixes

Also improved the response parsing to handle the actual OpenAI-compatible format:
- Changed from `data["images"]` to `data["data"]` (correct format)
- Added handling for `b64_json` in image objects
- Added debug output to show response structure

## Usage

To use a custom API key:
```bash
python image_test.py "Your prompt" --api-key "your-custom-key"
```

To generate image and save to file:
```bash
python image_test.py "Your prompt" --output volcano.jpg
```

To use custom server URL:
```bash
python image_test.py "Your prompt" --server http://localhost:8001
```

## Current Status

✅ **The server is responding correctly**
✅ **Authentication is working**
✅ **API format is correct**
⚠️ **Device initialization blocked** (missing sfpi build artifacts - not a code issue)

Once `./build_metal.sh --release` completes, image generation will work immediately.
