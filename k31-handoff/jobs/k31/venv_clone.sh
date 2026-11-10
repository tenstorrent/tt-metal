#!/bin/bash
# venv_clone.sh <src tree> <dst tree>: no PyPI or GitHub from exabox since the 2026-11-08 incident, so create_venv.sh
# cannot run. Copy <src>/python_env into <dst>, repoint every text path from <src> to <dst> (the editable ttnn
# install included), and give <dst> a tree-local interpreter (<dst>/.python) copied from this node's uv store.
set -euo pipefail
S=$1 T=$2
V=$(sed -n 's/^version_info = //p' "$S/python_env/pyvenv.cfg")
PY=$(ls -d $HOME/.local/share/uv/python/cpython-$V-linux-x86_64-gnu 2>/dev/null | head -1)
[ -n "$PY" ] || { echo "no cpython-$V in $(hostname)'s uv store"; exit 2; }
rm -rf "$T/python_env.new" && cp -a "$S/python_env" "$T/python_env.new"
grep -rlI --null "$S" "$T/python_env.new" | xargs -0 -r sed -i -E "s#$S([/\"' :;]|\$)#$T\\1#g"
mkdir -p "$T/.python" && rm -rf "$T/.python/$(basename $PY)" && cp -a "$PY" "$T/.python/"
BIN=$T/.python/$(basename $PY)/bin
ln -sfn "$BIN/python${V%.*}" "$T/python_env.new/bin/python"
sed -i "s|^home = .*|home = $BIN|" "$T/python_env.new/pyvenv.cfg"
rm -rf "$T/python_env" && mv "$T/python_env.new" "$T/python_env"
EXCL=$(git -C "$T" rev-parse --git-path info/exclude); mkdir -p "$(dirname "$EXCL")"
grep -qx '/.python/' "$EXCL" 2>/dev/null || echo '/.python/' >>"$EXCL"
left=$(grep -rlI "$S/" "$T/python_env" | head -3 || true)
echo "venv cloned: python -> $(readlink "$T/python_env/bin/python"); leftover refs to $S: ${left:-none}"
"$T/python_env/bin/python" -c 'import sys, loguru, torch; print(sys.version.split()[0], "loguru+torch ok")'
