# NOC Estimator

A tool for estimating NOC bandwidth and latency for various data movement patterns on Tenstorrent hardware. The estimator uses empirically measured latency data (collected by the data movement test suite) to predict performance for a given set of NOC parameters via interpolation.

This is useful for:
- Predicting data movement costs when designing kernels
- Comparing unicast vs multicast strategies for a given workload
- Estimating the impact of transaction sizes, barrier frequency, or destination count on NOC throughput
- Evaluating performance differences between architectures (Wormhole B0, Blackhole)

## Usage

```cpp
#include <tt-metalium/experimental/noc_estimator/noc_estimator.hpp>

using namespace tt::tt_metal::experimental::noc_estimator;

NocEstimatorParams params{
    .mechanism = NocMechanism::UNICAST,
    .pattern = NocPattern::ONE_TO_ONE,
    .num_transactions = 64,
    .transaction_size_bytes = 2048
};

// Get estimate (throws std::runtime_error on failure)
NocEstimate result = estimate_noc_performance(params);

double latency = result.latency_cycles;
double bandwidth = result.bandwidth_bytes_per_cycle;
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mechanism | NocMechanism | UNICAST | UNICAST, MULTICAST, or MULTICAST_LINKED |
| pattern | NocPattern | ONE_TO_ONE | Communication pattern (see below) |
| memory | MemoryType | L1 | L1, DRAM_INTERLEAVED, or DRAM_SHARDED |
| arch | Architecture | WORMHOLE_B0 | WORMHOLE_B0 or BLACKHOLE |
| num_transactions | uint32_t | 64 | Total number of transactions |
| num_transactions_per_barrier | uint32_t | 1 | Number of transactions issued between sync barriers |
| transaction_size_bytes | uint32_t | 512 | Size of each transaction in bytes |
| num_subordinates | uint32_t | 1 | Number of destination cores (for *_ALL patterns) |
| same_axis | bool | false | Whether src and dst have one shared axis |
| stateful | bool | false | Whether the transfer uses stateful mode |
| loopback | bool | false | Whether loopback is enabled |
| noc_index | uint32_t | 0 | Which NOC to use (0 or 1) |

### Patterns

| Pattern | Description |
|---------|-------------|
| ONE_FROM_ONE | Read: 1 source -> 1 destination |
| ONE_TO_ONE | Write: 1 source -> 1 destination |
| ONE_FROM_ALL | Read: N sources -> 1 destination |
| ONE_TO_ALL | Write: 1 source -> N destinations |
| ALL_TO_ALL | N sources -> N destinations |
| ALL_FROM_ALL | Read: N sources -> N destinations |
| ONE_TO_ROW | Write: 1 source -> N destinations (row) |
| ROW_TO_ROW | Write: N sources (row) -> N destinations (row) |
| ONE_TO_COLUMN | Write: 1 source -> N destinations (column) |
| COLUMN_TO_COLUMN | Write: N sources (column) -> N destinations (column) |

### Mechanism

- **UNICAST**: Individual transactions to each destination
- **MULTICAST**: Hardware multicast (one transaction to multiple destinations)
- **MULTICAST_LINKED**: Hardware multicast with loopback enabled

## Examples

### Simple unicast write
```cpp
NocEstimatorParams params{
    .pattern = NocPattern::ONE_TO_ONE,
    .num_transactions = 100,
    .transaction_size_bytes = 1024
};
NocEstimate result = estimate_noc_performance(params);
```

### Multicast to 2x2 grid
```cpp
NocEstimatorParams params{
    .mechanism = NocMechanism::MULTICAST,
    .pattern = NocPattern::ONE_TO_ALL,
    .num_transactions = 32,
    .transaction_size_bytes = 4096,
    .num_subordinates = 4  // 2x2 = 4 destinations
};
NocEstimate result = estimate_noc_performance(params);
```

### DRAM interleaved read on Blackhole
```cpp
NocEstimatorParams params{
    .pattern = NocPattern::ONE_FROM_ONE,
    .memory = MemoryType::DRAM_INTERLEAVED,
    .arch = Architecture::BLACKHOLE,
    .num_transactions = 64,
    .transaction_size_bytes = 8192
};
NocEstimate result = estimate_noc_performance(params);
```

### Comparing linked vs unlinked multicast
```cpp
NocEstimatorParams params_unlinked{
    .mechanism = NocMechanism::MULTICAST,
    .pattern = NocPattern::ONE_TO_ALL,
    .num_transactions = 16,
    .transaction_size_bytes = 2048,
    .num_subordinates = 4
};

NocEstimatorParams params_linked = params_unlinked;
params_linked.mechanism = NocMechanism::MULTICAST_LINKED;

NocEstimate unlinked = estimate_noc_performance(params_unlinked);
NocEstimate linked = estimate_noc_performance(params_linked);
// linked.latency_cycles will typically be lower
```

### Estimating impact of barrier frequency
```cpp
// 128 transactions, but only 64 can be in-flight before a barrier
NocEstimatorParams params{
    .pattern = NocPattern::ONE_TO_ONE,
    .arch = Architecture::BLACKHOLE,
    .num_transactions = 128,
    .num_transactions_per_barrier = 64,
    .transaction_size_bytes = 2048
};
NocEstimate result = estimate_noc_performance(params);
```

## Convenience Functions

```cpp
// Get just bandwidth
double bw = estimate_noc_bandwidth(params);

// Get just latency
double lat = estimate_noc_latency(params);
```

## Notes

- Set as many parameters as possible to get the most accurate estimate, some parameters can significantly impact predicted performance
- The estimator auto-initializes from the bundled YAML latency data on first call
- Estimates are based on empirical measurements from the data movement test suite
- `estimate_noc_performance` throws `std::runtime_error` if no match is found for the given parameters
- When an exact parameter match isn't available, the estimator uses N-dimensional interpolation and parameter relaxation to find the closest estimate
