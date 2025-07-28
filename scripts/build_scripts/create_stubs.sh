#!/bin/bash

PURPLE="\e[35m"
GREEN="\e[32m"
YELLOW="\e[33m"
RESET="\e[0m"

if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

sanity=0
echo -e "${YELLOW}!! Did you build metal and create python_env? !!${RESET}"
source $PYTHON_ENV_DIR/bin/activate
python3 -c "import tt_lib" >/dev/null 2>&1; sanity+=$?
python3 -c "import ttnn" >/dev/null 2>&1; sanity+=$?
if [ $sanity -ne 0 ]; then
    echo -e "${YELLOW}!! tt_lib or ttnn not found. Please build+install metal and create python_env first !!${RESET}"
    exit 1
fi

echo -e "${PURPLE}Generating stubs...${RESET}"
stubgen -m tt_lib -m tt_lib.device -m tt_lib.profiler -m tt_lib.tensor \
        -m tt_lib.operations -m tt_lib.operations.primary \
        -m tt_lib.operations.primary.transformers -o tt_eager

echo -e "${GREEN}Processed tt_lib modules. Stubs generated under tt_eager/tt_lib${RESET}"

stubgen -p ttnn._ttnn -o ttnn
echo -e "${GREEN}Processed ttnn modules. Stubs generated under ttnn/ttnn/_ttnn/${RESET}"

sed -i 's/\._C/tt_lib/g' tt_eager/tt_lib/__init__.pyi
echo -e "${GREEN}Stubs updated with tt_lib references${RESET}"
