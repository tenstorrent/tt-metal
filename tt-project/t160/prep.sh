#!/bin/bash
# prep.sh: light login-node prep (network-bound, no compile): python venv on /data with a /data-resident
# managed python (login-node home is not visible to compute nodes), plus CMake configure so the CPM
# downloads happen here. The compile itself runs inside the batch job on the dit node.
set -eo pipefail
F=/data/smarton/fasth3; W=$F/t48
trap 'echo "PREP_RC=$? $(date -u +%FT%TZ)" >> $F/prep.log' EXIT
grep -q '^STAGE_RC=0' $F/stage_src.log
cd $W; [ "$(git rev-parse HEAD)" = bf7db12a149bc1e8bf0559350ec2c874e86dc431 ]
export UV_PYTHON_INSTALL_DIR=$F/uv-python UV_CACHE_DIR=$F/uv-cache PYTHON_ENV_DIR=$F/python_env
export PATH=$HOME/.local/bin:$PATH
command -v uv || bash scripts/install-uv.sh
uv python install 3.10
[ -x $PYTHON_ENV_DIR/bin/python ] || uv venv --link-mode copy --managed-python --python 3.10 $PYTHON_ENV_DIR
source $PYTHON_ENV_DIR/bin/activate
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match setuptools==80 wheel==0.45.1
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-build-isolation -r tt_metal/python_env/requirements-dev.txt
nice -n 19 bash build_metal.sh --release --configure-only --cpm-source-cache $F/cpm-cache
rm -rf $F/uv-cache
echo PREP_OK
