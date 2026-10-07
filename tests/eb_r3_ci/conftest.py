# CI only: the device opens on a CI runner (GITHUB_ACTIONS) or locally under scripts/hwlock.sh.
import os
import sys

if not (os.environ.get("HWLOCK_HELD") or os.environ.get("GITHUB_ACTIONS")):
    sys.exit("not under hwlock")
