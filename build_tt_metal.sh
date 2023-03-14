#!/bin/bash

set -eo pipefail

make clean
make build
source build/python_env/bin/activate
