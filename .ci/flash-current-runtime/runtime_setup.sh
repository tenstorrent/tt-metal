set -euo pipefail
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1
SOURCE_ROOT=$1
WORK_ROOT=$2
PACKET_ROOT=$3
WHEEL=$4
test ! -e /dev/tenstorrent
test ! -e "$WORK_ROOT/venv"
cp -a --no-preserve=ownership /opt/venv "$WORK_ROOT/venv"
PYTHON="$WORK_ROOT/venv/bin/python"
"$PYTHON" -c 'import sys,torch,pytest; assert sys.version_info[:3]==(3,12,12); print(sys.version,torch.__version__,pytest.__version__)'
uv pip install --python "$PYTHON" "$WHEEL" transformers==5.16.1
uv pip freeze --python "$PYTHON" > "$WORK_ROOT/runtime-evidence/environment.txt"
"$PYTHON" "$PACKET_ROOT/model_config.py" "$WORK_ROOT/runtime-evidence/model-config"
"$PYTHON" "$PACKET_ROOT/runtime_host.py" "$SOURCE_ROOT" "$WORK_ROOT/runtime-evidence/host"
