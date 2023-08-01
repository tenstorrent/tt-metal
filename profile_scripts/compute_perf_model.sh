#!/bash/bin

python3 profile_scripts/analytical_model.py --mode compute --read-issue-latency 49 --write-issue-latency 43 --unpack-latency 200 --math-latency 2000 --pack-latency 200 --round-trip-latency 96 --flit-latency 1.01 --transfer-rate 32 --transfer-size-A 256 --transfer-size-B 2048 --transfer-size-write 256 --block-size 16384 --block-num 16
