#!/usr/bin/env bash

pip install -r tt_metal/python_env/requirements-dev.txt > /tmp/install.log 2>&1
pip install ttnn-* >> /tmp/install.log 2>&1

mv /tmp/install.log "$TT_MULTI_USER_GALAXY"

sleep infinity
