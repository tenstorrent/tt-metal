#!/usr/bin/env bash
# Make a python env with transformers==5.16.1 without touching python_env: copy it, then install into the copy.
#   QWEN38_PY   target interpreter (default $TT_METAL_HOME/python_env_q38fn/bin/python); its env dir is the copy target.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd -- "$HERE/../../../../.." && pwd -P)}"
PY="${QWEN38_PY:-$TT_METAL_HOME/python_env_q38fn/bin/python}"
src="$TT_METAL_HOME/python_env"
dest="$(dirname -- "$(dirname -- "$PY")")"
[[ "$dest" != "$src" ]] || { echo "refusing to modify $src; set QWEN38_PY to another env" >&2; exit 2; }
[[ -x "$src/bin/python" ]] || { echo "no $src/bin/python: build first (./build_metal.sh, create_venv.sh)" >&2; exit 2; }

if [[ -x "$dest/bin/python" ]]; then
    echo "$dest exists, only (re)installing transformers"
else
    [[ ! -e "$dest" ]] || { echo "$dest exists but is not an env" >&2; exit 2; }
    echo "copying $src -> $dest"
    cp -a -- "$src" "$dest"
    # venv scripts carry the absolute env path in their shebang / activate files
    grep -rIl --null -F -- "$src" "$dest/bin" 2>/dev/null | xargs -0 -r sed -i "s#$src#$dest#g"
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$dest/bin/python" "transformers==5.16.1" \
        || "$dest/bin/python" -m pip install "transformers==5.16.1"
else
    "$dest/bin/python" -m pip install "transformers==5.16.1"
fi

"$dest/bin/python" - <<'PY'
import importlib
for m in ("transformers", "torch", "tokenizers", "safetensors", "httpx"):
    try:
        print(f"{m} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"{m} MISSING ({e})")
PY
echo "Use:  export QWEN38_PY=$dest/bin/python"
