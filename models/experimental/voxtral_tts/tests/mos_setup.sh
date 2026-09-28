#!/bin/bash
# DistillMOS in an ISOLATED venv, for tests/test_mos.py and tests/mos_score.py. It depends on
# torchaudio 2.11, and installing torchaudio into the main venv breaks transformers, which takes the
# WER scorers down with it (bringup repo BUG-6). A separate venv makes that impossible rather than
# merely unlikely. Run once per box: /tmp does not survive a re-provisioned reservation.
set -e
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mosvenv-uvcache}"
uv venv --python 3.10 /tmp/mosvenv 2>&1 | tail -2
VIRTUAL_ENV=/tmp/mosvenv uv pip install distillmos soundfile numpy 2>&1 | tail -4
/tmp/mosvenv/bin/python -c "import distillmos, torchaudio; print('  distillmos ok, torchaudio', torchaudio.__version__)"
echo MOSSETUPDONE
