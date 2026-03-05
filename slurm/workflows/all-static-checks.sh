#!/usr/bin/env bash
#SBATCH --job-name=all-static-checks
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/all-static-checks.yaml
# Runs all static analysis checks: pre-commit hooks, SPDX license validation,
# kernel count limit, doc spell-check, forbidden imports, symlink validation,
# sweeps workflow verification, and codeowners validation.
# No device access required — runs on build partition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image ci-build

setup_job
trap 'cleanup_job $?' EXIT

log_info "Running all static checks"

# --- Pre-commit hooks ---
log_info "[1/8] Running pre-commit hooks"
docker_run "$DOCKER_IMAGE" "\
    pip install pre-commit && \
    pre-commit run --all-files
"

# --- SPDX license check ---
log_info "[2/8] Checking SPDX licenses"
docker_run "$DOCKER_IMAGE" "\
    python3 -c \"
import yaml, pathlib, re, sys
ignore = yaml.safe_load(pathlib.Path('.github/spdx_ignore.yaml').read_text())
ignored_paths = set(ignore.get('files', []))
errors = []
for f in pathlib.Path('.').rglob('*'):
    if f.is_file() and str(f) not in ignored_paths and f.suffix in {'.py', '.cpp', '.h', '.hpp', '.sh'}:
        text = f.read_text(errors='replace')[:2000]
        if 'SPDX-License-Identifier' not in text:
            errors.append(str(f))
if errors:
    print(f'Missing SPDX headers in {len(errors)} files:')
    for e in errors[:20]: print(f'  {e}')
    sys.exit(1)
print('All files have SPDX headers')
\"
"

# --- Kernel count limit ---
log_info "[3/8] Checking metal kernel count"
docker_run "$DOCKER_IMAGE" "\
    count=\$(find tt_metal/kernels/ -type f | wc -l) && \
    echo \"Kernel count: \$count (max: 8)\" && \
    if (( count > 8 )); then echo 'FAIL: kernel count exceeds limit'; exit 1; fi
"

# --- Doc spell-check ---
log_info "[4/8] Running doc spell-check"
docker_run "$DOCKER_IMAGE" "\
    TT_METAL_HOME=\$(pwd) docs/spellcheck.sh
"

# --- Forbidden imports ---
log_info "[5/8] Checking forbidden imports"
docker_run "$DOCKER_IMAGE" "\
    ttnn_count=\$(grep -Rnw 'tests/tt_metal' -e 'ttnn' | wc -l) && \
    if (( ttnn_count > 11 )); then echo \"FAIL: ttnn references in tt_metal tests: \$ttnn_count\"; exit 1; fi && \
    tt_lib_count=\$(grep -Rnw 'tests/tt_metal' -e 'tt_lib' | wc -l) && \
    if (( tt_lib_count > 0 )); then echo \"FAIL: tt_lib references: \$tt_lib_count\"; exit 1; fi && \
    tt_eager_count=\$(grep -Rnw 'tests/tt_metal' -e 'tt_eager' | wc -l) && \
    if (( tt_eager_count > 10 )); then echo \"FAIL: tt_eager references: \$tt_eager_count\"; exit 1; fi && \
    echo 'Forbidden imports check passed'
"

# --- Sweeps workflow verification ---
log_info "[6/8] Verifying sweeps workflow"
docker_run "$DOCKER_IMAGE" "\
    pip install pyyaml && \
    python tests/sweep_framework/framework/sweeps_workflow_verification.py
"

# --- Broken symlinks ---
log_info "[7/8] Checking for broken symlinks"
docker_run "$DOCKER_IMAGE" "\
    broken=\$(find . -type l ! -exec test -e {} \; -print) && \
    if [ -n \"\$broken\" ]; then echo \"Broken symlinks found:\"; echo \"\$broken\"; exit 1; fi && \
    echo 'All symlinks are valid'
"

# --- Codeowners validation ---
log_info "[8/8] Validating CODEOWNERS"
docker_run "$DOCKER_IMAGE" "\
    if command -v codeowners-validator &>/dev/null; then \
        codeowners-validator --checks=files,duppatterns,syntax --experimental-checks=avoid-shadowing; \
    else \
        echo 'codeowners-validator not available, skipping'; \
    fi
"

log_info "All static checks passed"
