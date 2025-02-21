# tt mem_bench

Utility to measure host and device bandwidth on Tenstorrent devices.

## Build

Tools are included in `tt_metal` builds. Using a release build is required for accurate perf measurements.

## Usage

By default, each test is run for 5 iterations and only basic tests are executed. All test patterns can be executed by specifying `--full`. Additional run parameters are listed below.

Tests will report host bandwidth and/or device bandwidth. If device bandwidth is reported, then the average of all cores is reported as well as bandwidth for just a single core.

> [!NOTE]
The `tt_metal` library log level can be adjusted by exporting `TT_METAL_LOGGER_LEVEL=fatal|info|error|debug`.

> [!NOTE]
On NUMA systems, the host page for the device's command queue data is pinned on the memory node closest to where the device is located. If `tt_metal` is run on a different node then bandwidth will degrade because it'll need to cross sockets. Therefore, it's important to run `tt_metal` on the closest node. On Linux, the execution policy can be set using `numactl`. E.g., if the device is located on node 0, then `numactl --cpubind=0 --membind=0 <command>` will allocate resources closer to the device.

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

Counters
          bytes_per_second: Aggregate Host copy to hugepage bandwidth. 0 if not measured.
          dev_bw: Average device core PCIe pull bandwidth. 0 if not measured.
```
