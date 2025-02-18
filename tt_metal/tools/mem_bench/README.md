# tt mem_bench

Utility to measure host and device bandwidth on Tenstorrent devices.

## Build

Tools are included in `tt_metal` builds. Using a release build is required for accurate perf measurements.

## Usage

By default, each test is run for 5 iterations and only basic tests are executed. All test patterns can be executed by specifying `--full`. Additional run parameters are listed below.

Tests will report host bandwidth and/or device bandwidth. If device bandwidth is reported, then the average of all cores is reported as well as bandwidth for just a single core.

> [!NOTE]
Reducing the `tt_metal` library log level by exporting `TT_METAL_LOGGER_LEVEL=fatal` will increase the readability of the output.


```
./build/tools/mem_bench --help
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=`<integer>x` OR `<float>s` ]
          [--benchmark_min_warmup_time=<min_warmup_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_dry_run={true|false}]
          [--benchmark_enable_random_interleaving={true|false}]
          [--benchmark_report_aggregates_only={true|false}]
          [--benchmark_display_aggregates_only={true|false}]
          [--benchmark_format=<console|json|csv>]
          [--benchmark_out=<filename>]
          [--benchmark_out_format=<json|console|csv>]
          [--benchmark_color={auto|true|false}]
          [--benchmark_counters_tabular={true|false}]
          [--benchmark_context=<key>=<value>,...]
          [--benchmark_time_unit={ns|us|ms|s}]
          [--v=<verbosity>]
          [--help] Shows this help message
          [--full] Run all tests
```
