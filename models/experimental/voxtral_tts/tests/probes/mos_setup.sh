#!/bin/bash
# DistillMOS in an ISOLATED venv. It depends on torchaudio 2.11, and STATUS 2 records that
# installing torchaudio into /opt/venv breaks transformers and takes score_quality_set.py with it.
# A separate venv makes that impossible rather than merely unlikely.
set -e
export UV_CACHE_DIR=/tmp/claude-1211416647/-localdev-lserbedzija/5377abb4-5495-4786-b60f-1d18a8772305/uvcache
uv venv --python 3.10 /tmp/mosvenv 2>&1 | tail -2
VIRTUAL_ENV=/tmp/mosvenv uv pip install distillmos soundfile numpy 2>&1 | tail -4
/tmp/mosvenv/bin/python -c "import distillmos, torchaudio; print('  distillmos ok, torchaudio', torchaudio.__version__)"
echo MOSSETUPDONE
