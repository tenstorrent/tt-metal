#!/usr/bin/env bash

pip install -r tt_metal/python_env/requirements.txt
pip install ttnn-*

echo "Creating $TT_MULTI_USER_GALAXY"
touch "$TT_MULTI_USER_GALAXY"

sleep infinity
