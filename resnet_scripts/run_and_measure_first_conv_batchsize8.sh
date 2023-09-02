source build/python_env/bin/activate
./tt_metal/tools/profiler/profile_this.py -c "pytest tests/python_api_testing/unit_testing/test_resnet50_first_conv.py::test_resnet50_first_conv[25-8-False]"
python tt_metal/tools/profiler/process_ops_logs.py -i tt_metal/tools/profiler/logs/ops
rm -rf  tt_metal/tools/profiler/logs/
cat output/ops/profile_log_ops.csv | awk -F, '{print $13}'
