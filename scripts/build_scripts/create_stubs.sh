#!/bin/bash

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

sanity=0
echo "!! Did you build metal and create python_env? !!"
source $PYTHON_ENV_DIR/bin/activate
python3 -c "import tt_lib" >/dev/null 2>&1; sanity+=$?
python3 -c "import ttnn" >/dev/null 2>&1; sanity+=$?
if [ $sanity -ne 0 ]; then
    echo "!! tt_lib or ttnn not found. Please build+install metal and create python_env first !!"
    exit 1
fi

echo "Generating stubs"
stubgen -p ttnn._ttnn -o ttnn --include-docstrings
sed -i 's/\._C/tt_lib/g' tt_eager/tt_lib/__init__.pyi
