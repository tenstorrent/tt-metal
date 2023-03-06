#!/bin/bash

set -eo pipefail

cd docs
python3 -m venv env
source env/bin/activate
pip install -r requirements-docs.txt
