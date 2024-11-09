#!/bin/bash
TT_BACKEND_TIMEOUT=0 /home/dvartanians/Metal/tt-metal/python_env/bin/uvicorn --host 0.0.0.0 --port 7000 fast_api_yolov5:app
