# tt_metal profiler

## Tests

### Automation tests for the profiler module

If new changes were made to the post processing scripts, please run automated test to make sure previously supported
measurements still have support.

Automated test are run by:

1. Follow the tt-metal dev get_started and general installation guide and make sure `PYTHONPATH`
   and other tt-metal environment variables are set. Activate the python environment as suggested by the guides.
2. For testing the collection side of the profiler module, run the following commands under `TT_METAL_HOME`:

```
    cd $TT_METAL_HOME
    pytest -svvv tests/tt_metal/tools/profiler/test_device_profiler.py
```
