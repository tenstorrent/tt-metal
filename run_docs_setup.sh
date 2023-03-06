#!/bin/bash

set -eo pipefail

GIT_DIR=$(pwd)

doxygen
cd docs
python3 -m venv env
source env/bin/activate
pip install -r requirements-docs.txt
deactivate
cd $GIT_DIR
