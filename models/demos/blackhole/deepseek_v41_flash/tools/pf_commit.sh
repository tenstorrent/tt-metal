#!/bin/bash
# usage: pf_commit.sh "<message>"  (adds tracked changes + new package files, commits with retries)
cd /mnt/tt-data/ssinghal/wt/pf_mhc
git add models/demos/blackhole/deepseek_v41_flash
for i in 1 2 3; do
  timeout 1700 git -c user.name=ssinghal -c user.email=ssinghal@tenstorrent.com commit -q -m "$1

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && break
  git add models/demos/blackhole/deepseek_v41_flash
done
