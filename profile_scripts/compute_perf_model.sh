#!/bash/bin

python3 profile_scripts/analytical_model.py --mode compute --read-issue-latency 49 --write-issue-latency 43 --compute-latency 2400 --round-trip-latency 95 --head-flit-latency 1 --flit-latency 1 --CB-producer-consumer-sync-latency 56 --NOC-data-width 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 16384 --block-num 256
