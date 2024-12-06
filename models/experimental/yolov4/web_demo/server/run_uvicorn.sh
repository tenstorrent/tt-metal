#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 models.experimental.yolov4.web_demo.server.fast_api_yolov4:app
