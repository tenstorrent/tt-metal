#!/bin/bash
set -eo pipefail

# ----------------------------
# Colors and Log Helpers
# ----------------------------
RESET="\e[0m"
PURPLE="\e[35m"
GREEN="\e[32m"
YELLOW="\e[33m"
RED="\e[31m"

log_info()       { echo -e "${PURPLE}[INFO] $*${RESET}"; }
log_success()    { echo -e "${GREEN}[✓] $*${RESET}"; }
log_warn()       { echo -e "${YELLOW}[WARN] $*${RESET}"; }
log_error()      { echo -e "${RED}[ERROR] $*${RESET}"; }

# ----------------------------
# Python Environment
# ----------------------------
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi

# ----------------------------
# Sanity Check
# ----------------------------
log_warn "!! Did you build metal and create python_env? !!"
source "$PYTHON_ENV_DIR/bin/activate"

sanity=0
python3 -c "import tt_lib" >/dev/null 2>&1 || sanity=$((sanity+1))
python3 -c "import ttnn" >/dev/null 2>&1 || sanity=$((sanity+1))

if [ $sanity -ne 0 ]; then
    log_error "tt_lib or ttnn not found. Please build + install metal and create python_env first!"
    exit 1
fi

# ----------------------------
# Generate Stubs
# ----------------------------
log_info "Generating stubs..."
stubgen -m tt_lib -m tt_lib.device -m tt_lib.profiler -m tt_lib.tensor \
        -m tt_lib.operations -m tt_lib.operations.primary \
        -m tt_lib.operations.primary.transformers -o tt_eager
log_success "Processed tt_lib modules. Stubs generated under tt_eager/tt_lib."

stubgen -p ttnn._ttnn -o ttnn
log_success "Processed ttnn modules. Stubs generated under ttnn/ttnn/_ttnn/."

# ----------------------------
# Update Stubs
# ----------------------------
sed -i 's/\._C/tt_lib/g' tt_eager/tt_lib/__init__.pyi
log_success "Stubs updated with tt_lib references."
