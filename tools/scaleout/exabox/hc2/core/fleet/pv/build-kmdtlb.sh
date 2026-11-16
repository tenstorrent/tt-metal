#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# build-kmdtlb.sh: rebuild kmdtlb.code from kmdtlb.py (a maintainer step after editing kmdtlb.py; runs never call it).
# run-ttx embeds Python 3.7, so the hook is compiled and marshalled by Python 3.7 (docker python:3.7-slim). The code
# object is a small wrapper that runs kmdtlb.py in its own module (run-ttx's scripts share __main__) and calls main().
# It prints the pv.pin rows to paste (kmdtlb.py, kmdtlb.code and the repacked run-ttx-kmdtlb).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
image=${KMDTLB_PY37_IMAGE:-python:3.7-slim}
docker run --rm --network none -u "$(id -u):$(id -g)" -v "$here:/pv" -w /pv "$image" python3 -I -c '
import marshal, sys
assert sys.version_info[:2] == (3, 7), sys.version
src = open("kmdtlb.py").read()
wrapper = (
    "import sys as _kmdtlb_sys, types as _kmdtlb_types\n"
    "_kmdtlb_m = _kmdtlb_types.ModuleType(\"kmdtlb\")\n"
    "_kmdtlb_m.__file__ = \"kmdtlb.py\"\n"
    "exec(compile(%r, \"kmdtlb.py\", \"exec\"), _kmdtlb_m.__dict__)\n"
    "_kmdtlb_sys.modules[\"kmdtlb\"] = _kmdtlb_m\n"
    "_kmdtlb_m.main()\n"
    "del _kmdtlb_sys, _kmdtlb_types, _kmdtlb_m\n" % src
)
compile(src, "kmdtlb.py", "exec")
open("kmdtlb.code.tmp", "wb").write(marshal.dumps(compile(wrapper, "pyi_rth_kmdtlb.py", "exec")))
'
mv "$here/kmdtlb.code.tmp" "$here/kmdtlb.code"
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT
for f in kmdtlb.py kmdtlb.code; do
  echo "$f sha256=$(sha256sum "$here/$f" | cut -d' ' -f1) bytes=$(stat -c %s "$here/$f")"
done
python3 -I "$here/repack.py" "$here/run-ttx" "$here/kmdtlb.code" "$tmp/run-ttx-kmdtlb"
