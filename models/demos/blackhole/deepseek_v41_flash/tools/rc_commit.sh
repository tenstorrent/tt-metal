#!/bin/bash
# usage: rc_commit.sh "<message>"  (background commit with retries; hooks are slow over NFS)
cd /mnt/tt-data/ssinghal/wt/pf_reconf
for i in 1 2 3 4 5; do
  git add models/demos/blackhole/deepseek_v41_flash && git commit -q -m "$1

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && { echo committed; git log --oneline -1; exit 0; }
  sleep 20
done
echo FAILED
