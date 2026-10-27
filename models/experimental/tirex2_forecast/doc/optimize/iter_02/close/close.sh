set -e
cd /home/andy/bringup/tirex2_base
git checkout -q 7cda792fec3
MODEL_PERF_JSON=/home/andy/bringup/tirex2/models/experimental/tirex2_forecast/doc/optimize/iter_02/close/bench_7cda792.json python -m pytest -q models/experimental/tirex2_forecast/tests/perf/test_forward_perf.py -s > /home/andy/bringup/tirex2/models/experimental/tirex2_forecast/doc/optimize/iter_02/close/bench.log 2>&1 || echo BENCH_FAIL
python -m pytest -q models/experimental/tirex2_forecast/tests/pcc > /home/andy/bringup/tirex2/models/experimental/tirex2_forecast/doc/optimize/iter_02/close/suite.log 2>&1 || echo SUITE_FAIL
tail -1 /home/andy/bringup/tirex2/models/experimental/tirex2_forecast/doc/optimize/iter_02/close/suite.log
