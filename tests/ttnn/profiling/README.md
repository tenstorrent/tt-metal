# Profiling host dispatch time (overhead)

There are multiple scripts used to profile host dispatch time for operations in tt_lib. These scripts should be used with fast dispatch mode. They measure time directly from python, and they also parse `Tracy` output and provide joined results.


## profile_host_overhead.py

`profile_host_overhead.py` script goes through multiple `tt_lib` operations and measures host dispatch time.
It can be called with:

```
pytest tests/ttnn/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile
```

In this case `host_overhead_profile` is the output folder.

You can profile only one op with:

```
pytest tests/ttnn/profiling/profile_host_overhead.py --input-method cli --cli-input host_overhead_profile::tt_lib.tensor.isclose
```

After the script is finished profiling results are in the designated output folder in file `host_overhead_profiler_output.csv`. Content of output csv might look like:

```
op,count,python min dispatch time (ms),python mean dispatch time(ms),python mean dispatch + sync time (ms)
tt_lib.tensor.add,8,0.87,0.89,1.08
tt_lib.tensor.sub,8,0.94,0.94,1.1
tt_lib.tensor.mul,8,0.82,0.85,1.0
tt_lib.tensor.div,8,0.79,0.81,1.1
tt_lib.tensor.hypot,8,3.08,3.22,3.52
tt_lib.tensor.squared_difference,8,0.93,0.97,1.16
tt_lib.tensor.logaddexp,8,1.06,1.34,1.79
tt_lib.tensor.logaddexp2,8,1.06,1.49,1.79
tt_lib.tensor.atan2,8,40.93,43.69,43.35
```

Columns:
* `op`: Op being profiled
* `count`: Number of profile runs.
* `python min dispatch time (ms)`: Minimum measured dispatch time (overhead).
* `python mean dispatch time(ms)`: Mean of measured dispatch times.
* `python mean dispatch + sync time (ms)`: Total time needed to run the op (both dispatch and kernel time). Measured after syncronize.


## profile_host_overhead_with_tracy.py

`profile_host_overhead_with_tracy.py` script profiles host dispatch time both from python and using `Tracy`, and joins results to a single .csv output file.

You can run it with:

```
python tests/ttnn/profiling/profile_host_overhead_with_tracy.py -o host_overhead_profile -c final.csv
```

It profiles all tt_lib ops and saves measurement results to `host_overhead_profile/final.csv`. Output might look like:

```
op,count,python min dispatch time (ms),python mean dispatch time(ms),python mean dispatch + sync time (ms),C++ mean dispatch time (ms)
tt_lib.tensor.add,40,0.85,0.93,1.21,0.96
tt_lib.tensor.atan2,40,37.16,40.51,42.17,40.36
tt_lib.tensor.div,40,0.87,0.92,1.18,0.96
tt_lib.tensor.hypot,40,3.32,4.41,4.79,4.07
tt_lib.tensor.isclose,40,21.13,23.17,23.16,22.3
tt_lib.tensor.logaddexp,40,0.83,0.85,1.19,0.9
tt_lib.tensor.logaddexp2,40,0.85,0.9,1.17,0.95
tt_lib.tensor.mul,40,0.86,0.98,1.18,0.98
tt_lib.tensor.squared_difference,40,0.91,1.1,1.39,1.08
tt_lib.tensor.sub,40,0.88,0.92,1.21,0.98
```

Added column is `C++ mean dispatch time (ms)` which is host time parsed from `Tracy` output and mean taken from those measurements.


## test_host_overhead_ci.py

`test_host_overhead_ci.py` is a unit test which can be run in CI to check if there were no regressions in host dispatch times. It compares measured time with reference times for each op. It can be run with:

```
pytest tests/ttnn/profiling/test_host_overhead_ci.py
```
