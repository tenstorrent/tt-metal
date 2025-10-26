#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 models.demos.yolov8x.web_demo.server.fast_api_yolov8x:app
