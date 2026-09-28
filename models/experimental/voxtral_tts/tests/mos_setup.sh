#!/bin/bash
# DistillMOS in an isolated venv for tests/test_mos.py and tests/mos_score.py: its torchaudio breaks
# transformers in the main venv (VOXTRAL_TTS_BUGS.md BUG-6). Run once per box; /tmp does not survive
# a re-provisioned reservation. See VOXTRAL_TTS_GATES.md [mos-01].
set -e
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mosvenv-uvcache}"
uv venv --python 3.10 /tmp/mosvenv 2>&1 | tail -2
VIRTUAL_ENV=/tmp/mosvenv uv pip install distillmos soundfile numpy 2>&1 | tail -4
/tmp/mosvenv/bin/python -c "import distillmos, torchaudio; print('  distillmos ok, torchaudio', torchaudio.__version__)"
echo MOSSETUPDONE
