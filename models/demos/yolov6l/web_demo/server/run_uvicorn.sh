#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 models.demos.yolov6l.web_demo.server.fast_api_yolov6l:app
