#!/usr/bin/env bash
# Verify each leg's PYTHONPATH actually wins before spending device time on it.
# This is the guard for run.sh note (3): the failure mode is not an ImportError, it is
# silently importing the other tree, which reads as a clean result.
cd /home/ttuser/.claude/jobs/05e31507/tmp
VENV=/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv/bin/python
for R in \
  /home/ttuser/dev/muse-glimmer/tt-metal/.claude/worktrees/pristine-ab \
  /home/ttuser/dev/muse-glimmer/tt-metal/.claude/worktrees/dflash-perf
do
  echo "=== requested: $R"
  PYTHONPATH="$R" "$VENV" -c "
import os, models.autoports.meta_models_muse_glimmer_30b as p
print('    resolved:', os.path.realpath(list(p.__path__)[0]))
" 2>&1 | grep "resolved:"
done
