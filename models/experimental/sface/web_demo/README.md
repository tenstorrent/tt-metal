# Face Matching Solution - Tenstorrent QuietBox

A production-ready face verification API running on Tenstorrent AI hardware.

## Features

- **Face Detection**: YuNet model optimized for TTNN
- **Face Recognition**: SFace (MobileFaceNet) with 0.95+ PCC accuracy
- **REST API**: FastAPI with JSON and Multipart support
- **Mock Database**: POC-ready reference image storage
- **Docker Support**: Containerized deployment for QuietBox

## Technical Specifications

| Specification | Value |
|--------------|-------|
| Input Resolution | 640x640 pixels |
| Minimum Face Size | 100x100 pixels |
| Supported Formats | JPEG, PNG, WEBP |
| Detection Latency | ~8ms |
| Recognition Latency | ~11ms |
| End-to-end Latency | ~20-25ms |
| Throughput | ~45 FPS |

## API Endpoints

### POST /api/v1/verify

Verify if a live selfie matches a reference identity.

#### Option 1: JSON with Base64 Images

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "live_selfie": "<base64_encoded_image>",
    "user_id": "user123"
  }'
```

Or with direct reference image:

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "live_selfie": "<base64_encoded_selfie>",
    "reference_image": "<base64_encoded_reference>"
  }'
```

#### Option 2: Multipart Form Data

```bash
curl -X POST "http://localhost:8000/api/v1/verify/multipart" \
  -F "live_selfie=@selfie.jpg" \
  -F "user_id=user123"
```

Or with direct reference image:

```bash
curl -X POST "http://localhost:8000/api/v1/verify/multipart" \
  -F "live_selfie=@selfie.jpg" \
  -F "reference_image=@id_photo.jpg"
```

#### Response Format

```json
{
  "match_status": true,
  "confidence_score": 0.87,
  "latency_ms": 24
}
```

| Field | Type | Description |
|-------|------|-------------|
| match_status | Boolean | Whether faces match (threshold: 0.75) |
| confidence_score | Float | Cosine similarity (0.00 - 1.00) |
| latency_ms | Integer | Processing time in milliseconds |

### Mock Database Endpoints (POC)

#### Register User

```bash
curl -X POST "http://localhost:8000/api/v1/mock-db/register" \
  -F "user_id=user123" \
  -F "reference_image=@id_photo.jpg"
```

#### List Users

```bash
curl "http://localhost:8000/api/v1/mock-db/users"
```

#### Delete User

```bash
curl -X DELETE "http://localhost:8000/api/v1/mock-db/users/user123"
```

## Deployment

### Option 1: Direct Run (Development)

```bash
# Activate tt-metal environment
source /path/to/tt-metal/python_env/bin/activate

# Start server
cd /path/to/tt-metal
./models/experimental/sface/web_demo/server/run_server.sh
```

### Option 2: Docker (Production)

```bash
# Build the container
cd models/experimental/sface/web_demo
docker build -t face-matching-api -f Dockerfile.quickbox .

# Run with tt-metal mounted
docker run -d \
  --name face-matching-api \
  -v /path/to/tt-metal:/tt-metal:ro \
  -v ./mock_database:/app/mock_database \
  --device /dev/tenstorrent \
  -p 8000:8000 \
  face-matching-api
```

### Option 3: Docker Compose

```bash
# Set tt-metal path
export TT_METAL_HOME=/path/to/tt-metal

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Directory Structure

```
web_demo/
├── server/
│   ├── fast_api_face_recognition.py  # Main API server
│   ├── requirements.txt
│   ├── run_server.sh
│   ├── registered_faces/             # Legacy face database
│   └── mock_database/                # POC mock database
│       └── <user_id>/
│           ├── embedding.npy
│           └── reference.jpg
├── client/
│   ├── face_recognition_streamlit.py # Demo UI
│   ├── requirements.txt
│   └── run_client.sh
├── Dockerfile.quickbox               # Production Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── README.md
```

## Error Handling

| HTTP Code | Error | Description |
|-----------|-------|-------------|
| 400 | Invalid image | Cannot decode Base64 or image format |
| 400 | No face detected | Face not found in image |
| 404 | User not found | user_id not in mock database |
| 500 | Server error | Internal processing error |

## Performance Tuning

- **Input Size**: Use 640x640 for best accuracy. 320x320 is faster but less accurate for small faces.
- **Threshold**: Default 0.75 balances false accepts/rejects. Increase for higher security.
- **Warmup**: First request takes longer (~4s) due to kernel compilation. Subsequent requests are fast.

## Integration Example (Python)

```python
import base64
import requests

def verify_face(selfie_path: str, user_id: str) -> dict:
    """Verify a selfie against a registered user."""

    # Read and encode selfie
    with open(selfie_path, "rb") as f:
        selfie_b64 = base64.b64encode(f.read()).decode()

    # Call API
    response = requests.post(
        "http://localhost:8000/api/v1/verify",
        json={
            "live_selfie": selfie_b64,
            "user_id": user_id
        }
    )

    return response.json()

# Usage
result = verify_face("selfie.jpg", "user123")
if result["match_status"]:
    print(f"Match! Confidence: {result['confidence_score']}")
else:
    print(f"No match. Confidence: {result['confidence_score']}")
```

## License

SPDX-License-Identifier: Apache-2.0
