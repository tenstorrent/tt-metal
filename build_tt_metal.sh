#!/bin/bash

set -eo pipefail

make build
source build/python_env/bin/activate
