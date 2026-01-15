#!/usr/bin/env bash

uv pip install -r tt_metal/python_env/requirements-dev.txt > /tmp/install.log 2>&1
uv pip install ttnn-* >> /tmp/install.log 2>&1

mv /tmp/install.log "$TT_MULTI_USER_GALAXY"

sleep infinity
