#!/bin/bash

set -euo pipefail

# Use main's version for ALL conflicted submodules:
git checkout --ours tt_metal/third_party/

# Add all resolved submodules:
git add tt_metal/third_party/

# Continue the rebase:
git rebase --continue
