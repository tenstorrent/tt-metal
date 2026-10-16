#!/bin/bash
# seeded vs unseeded long aime24-p1 generations; NO resets
cd /home/ubuntu/ssinghal/olmo3; source tt-metal/python_env/bin/activate
for sd in 7 123; do
  OLMO_SEED=$sd OLMO_MAX_NEW=6200 ./run_demo.sh seed$sd aime24-p1 2>&1 | tail -1 | cut -c1-120
done
echo seed-runs-done
