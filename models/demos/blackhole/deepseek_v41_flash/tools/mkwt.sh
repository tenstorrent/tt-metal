#!/bin/bash
# usage: mkwt.sh <name>  -> /mnt/tt-data/ssinghal/wt/<name>: overlay of the main repo where only the deepseek_v41_flash package is a private editable copy
set -e
M=/mnt/tt-data/ssinghal/tests/tt-metal; W=/mnt/tt-data/ssinghal/wt/$1; PK=models/demos/blackhole/deepseek_v41_flash
rm -rf "$W"; mkdir -p "$W"
for e in $(ls -A $M | grep -v '^models$'); do ln -s $M/$e $W/$e; done
mkdir -p $W/models/demos/blackhole
for e in $(ls -A $M/models | grep -v '^demos$'); do ln -s $M/models/$e $W/models/$e; done
for e in $(ls -A $M/models/demos | grep -v '^blackhole$'); do ln -s $M/models/demos/$e $W/models/demos/$e; done
for e in $(ls -A $M/models/demos/blackhole | grep -v '^deepseek_v41_flash$'); do ln -s $M/models/demos/blackhole/$e $W/models/demos/blackhole/$e; done
cp -a $M/$PK $W/$PK
cat > $W/run.sh <<R
#!/bin/bash
# usage: $W/run.sh "<cmd>"  -- like /tmp/devrun.sh but imports the package from this private copy ($W/$PK); TT_METAL_HOME stays the main repo
exec flock -w 7200 /tmp/dsv4_dev.lock bash -c "cd $W && source $M/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$M PYTHONPATH=$W:$M MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && \$1"
R
chmod +x $W/run.sh
