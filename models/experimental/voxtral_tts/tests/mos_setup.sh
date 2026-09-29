#!/bin/bash
# DistillMOS in an isolated venv for tests/test_mos.py and tests/mos_score.py: its torchaudio breaks
# transformers in the main venv (VOXTRAL_TTS_BUGS.md BUG-6). Run once per box; /tmp does not survive
# a re-provisioned reservation. See VOXTRAL_TTS_GATES.md [mos-01].
set -eo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mosvenv-uvcache}"
uv venv --python 3.10 /tmp/mosvenv 2>&1 | tail -2
# Pinned: MOS_FLOOR in test_mos.py holds for this stack. see VOXTRAL_TTS_GATES.md [mos-01]
VIRTUAL_ENV=/tmp/mosvenv uv pip install distillmos==0.9.1 xls-r-sqa==0.1.0 transformers==5.17.0 torch==2.14.0 \
    torchaudio==2.11.0 librosa==0.11.0 soundfile==0.14.0 numpy==2.2.6 2>&1 | tail -4
/tmp/mosvenv/bin/python -c "import distillmos, torchaudio; print('  distillmos ok, torchaudio', torchaudio.__version__)"
echo MOSSETUPDONE
