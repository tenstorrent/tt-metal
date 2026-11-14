# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run tenstorrent/.github's cachew-checkout migration helper on this repository.

The helper (actions/cachew-checkout/migrate.py) is fetched from the commit
this repository pins in .github/cachew-checkout.json, so it always matches the
action the workflows run. Arguments are passed through:

    python3 .github/scripts/utils/cachew_checkout.py --write
    python3 .github/scripts/utils/cachew_checkout.py --write --action-sha <sha>
    python3 .github/scripts/utils/cachew_checkout.py --check
    python3 .github/scripts/utils/cachew_checkout.py --verify-pins

The tt_metal/tt-llk tree is excluded in the config: its workflows run in
tenstorrent/tt-llk. Set CACHEW_MIGRATE_PY to a local migrate.py to use that
instead of downloading (offline, or to test a helper change).
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / ".github/cachew-checkout.json"
HELPER_URL = "https://raw.githubusercontent.com/tenstorrent/.github/{sha}/actions/cachew-checkout/migrate.py"


def main():
    local = os.environ.get("CACHEW_MIGRATE_PY")
    command = [sys.executable, None, "--root", str(ROOT), *sys.argv[1:]]
    if local:
        command[1] = local
        return subprocess.call(command)
    sha = json.loads(CONFIG.read_text())["action_sha"]
    url = HELPER_URL.format(sha=sha)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            helper = response.read()
    except urllib.error.URLError as error:
        print(f"Could not download the migration helper from {url}: {error}", file=sys.stderr)
        print(
            "Check action_sha in .github/cachew-checkout.json (it must be a pushed commit of tenstorrent/.github"
            " that has actions/cachew-checkout/migrate.py), or set CACHEW_MIGRATE_PY to a local copy.",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "migrate.py"
        path.write_bytes(helper)
        command[1] = str(path)
        return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
