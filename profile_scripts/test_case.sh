#!/bin/bash

python3 profile_scripts/analytical_model.py --mode read --pre-issue-overhead 17 --NIU-programming 6 --non-NIU-programming 43 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --NOC-data-width 32 --transfer-size 2048 --buffer-size 65536 --comment 1

python3 profile_scripts/analytical_model.py --mode write --pre-issue-overhead 12 --NIU-programming 6 --non-NIU-programming 37 --round-trip-latency 93 --head-flit-latency 1 --flit-latency 1 --NOC-data-width 32 --transfer-size 2048 --buffer-size 65536 --comment 1

# NOC bound
python3 profile_scripts/analytical_model.py --mode single-core --read-issue-latency 49 --write-issue-latency 43 --compute-latency 400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 2048 --block-num 256  --comment 1

# Compute bound
python3 profile_scripts/analytical_model.py --mode single-core --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 2048 --block-num 256  --comment 1

# MULTI_CORE
python3 profile_scripts/analytical_model.py --mode multi-core --operation matmul --NOC-data-width 32 --DRAM-channels 8 \
    --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --DRAM-round-trip-latency 1600 --Tensix-round-trip-latency 95 \
    --transfer-size 2048 --tile-size 32 --reuse-threshold 16 \
    --batch-size-A 32 --block-num-A 1 --block-height-A 256 --block-width-A 128 \
    --batch-size-B 1 --block-num-B 1 --block-height-B 128 --block-width-B 256

# MULTI_CORE_Y
python3 profile_scripts/analytical_model.py --mode multi-core --operation matmul --NOC-data-width 32 --DRAM-channels 8 \
    --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --DRAM-round-trip-latency 1600 --Tensix-round-trip-latency 95 \
    --transfer-size 2048 --tile-size 32 --reuse-threshold 16 \
    --batch-size-A 32 --block-num-A 1 --block-height-A 128 --block-width-A 4096 \
    --batch-size-B 1 --block-num-B 1 --block-height-B 4096 --block-width-B 4096

# MULTI_CORE_XY
python3 profile_scripts/analytical_model.py --mode multi-core --operation matmul --NOC-data-width 32 --DRAM-channels 8 \
    --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --DRAM-round-trip-latency 1600 --Tensix-round-trip-latency 95 \
    --transfer-size 2048 --tile-size 32 --reuse-threshold 16 \
    --batch-size-A 32 --block-num-A 1 --block-height-A 4096 --block-width-A 4096 \
    --batch-size-B 1 --block-num-B 1 --block-height-B 4096 --block-width-B 4096
