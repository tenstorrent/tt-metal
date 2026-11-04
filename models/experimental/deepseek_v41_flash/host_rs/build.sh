#!/usr/bin/env bash
# Build the dsv41_host Python extension (host-only; does not touch the device) and place it beside this script,
# where models/experimental/deepseek_v41_flash/tt/host_rs.py loads it from.
#
#   models/experimental/deepseek_v41_flash/host_rs/build.sh          # build
#   models/experimental/deepseek_v41_flash/host_rs/build.sh test     # also run the Rust unit tests
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
python="${PYTHON:-$repo/python_env/bin/python}"
export PYO3_PYTHON="$python"

cd "$here"
cargo build --release
cp -f target/release/libdsv41_host.so dsv41_host.abi3.so

if [[ "${1:-}" == "test" ]]; then
    LD_LIBRARY_PATH="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
        cargo test --release --no-default-features
fi
echo "built $here/dsv41_host.abi3.so"
