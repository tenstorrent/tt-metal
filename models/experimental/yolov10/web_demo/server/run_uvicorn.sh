#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 models.experimental.yolov10.web_demo.server.fast_api_yolov10:app
