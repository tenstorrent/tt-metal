#!/bin/bash
# Recreate the blx01 venv lost with /var/tmp/fasth3/t48 (deleted 2026-10-13 04:34 UTC). Writes only under /var/tmp/fasth3.
set -eo pipefail
F=/var/tmp/fasth3
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache UV_PYTHON_INSTALL_DIR=$F/home/uv-python
export PATH=$F/uvboot/bin:$PATH
cd $F/t422/b
ionice -c3 nice -n 19 ./create_venv.sh --python-version 3.10 --env-dir $F/t460/python_env --force --skip-compat-check || echo "[venv460] create_venv rc=$? (editable install may fail on the broken .git; PYTHONPATH covers it)"
$F/t460/python_env/bin/python -c "import torch, pytest; print('torch', torch.__version__)"
echo VENV460_OK
