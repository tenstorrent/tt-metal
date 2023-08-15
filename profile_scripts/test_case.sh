#!/bin/bash

python3 profile_scripts/analytical_model.py --mode read --pre-issue-overhead 17 --NIU-programming 6 --non-NIU-programming 43 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --NOC-data-width 32 --transfer-size 2048 --buffer-size 65536 --comment 1

python3 profile_scripts/analytical_model.py --mode write --pre-issue-overhead 12 --NIU-programming 6 --non-NIU-programming 37 --round-trip-latency 93 --head-flit-latency 1 --flit-latency 1 --NOC-data-width 32 --transfer-size 2048 --buffer-size 65536 --comment 1

python3 profile_scripts/analytical_model.py --mode compute --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 16384 --block-num 256  --comment 1

python3 profile_scripts/analytical_model.py --mode compute --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 2048 --block-num 256  --comment 1

python3 profile_scripts/analytical_model.py --mode multi-core --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 2048 --block-num 256  --multi-core-mode DRAM --core-range-x-dim "0-2" --core-range-y-dim "0-2"
