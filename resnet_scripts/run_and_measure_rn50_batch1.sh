source build/python_env/bin/activate
./tt_metal/tools/profiler/profile_this.py -c "pytest tests/python_api_testing/models/resnet/test_metal_resnet50.py::test_run_resnet50_inference[1]"
python tt_metal/tools/profiler/process_ops_logs.py -i tt_metal/tools/profiler/logs/ops
rm -rf  tt_metal/tools/profiler/logs/
