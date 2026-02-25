#!/bin/bash
set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Checking docs build..."

cd $TT_METAL_HOME/docs
uv pip install -r requirements-docs.txt
make clean
make html
