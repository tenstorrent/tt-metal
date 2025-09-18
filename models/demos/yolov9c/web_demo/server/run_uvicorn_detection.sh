#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 models.demos.yolov9c.web_demo.server.fast_api_yolov9c_detection:app
