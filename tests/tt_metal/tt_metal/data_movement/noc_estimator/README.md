# NOC Estimator

Estimates NOC bandwidth and latency for various data movement patterns on Tenstorrent hardware.

## Usage

```cpp
#include "noc_estimator.hpp"

using namespace tt::noc_estimator;

// Set up your parameters
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
| mechanism | NocMechanism | UNICAST | UNICAST or MULTICAST |
| pattern | NocPattern | ONE_TO_ONE | Communication pattern (see below) |
| memory | MemoryType | L1 | L1 or DRAM |
| arch | Architecture | WORMHOLE_B0 | WORMHOLE_B0 or BLACKHOLE |
| num_transactions | uint32_t | 64 | Total number of transactions |
| num_transactions_per_barrier | uint32_t | 1 | Number of transactions issued between sync barriers |
| transaction_size_bytes | uint32_t | 512 | Size of each transaction in bytes |
| num_subordinates | uint32_t | 1 | Number of destination cores (for *_ALL patterns) |
| same_axis | bool | false | Whether src and dst have one shared axis |
| linked | bool | false | Use linked multicast (multicast only) |

### Patterns

| Pattern | Description |
|---------|-------------|
| ONE_FROM_ONE | Read: 1 source -> 1 destination |
| ONE_TO_ONE | Write: 1 source -> 1 destination |
| ONE_FROM_ALL | Read: N sources -> 1 destination |
| ONE_TO_ALL | Write: 1 source -> N destinations |
| ALL_TO_ALL | N sources -> N destinations |
| ALL_FROM_ALL | Read: N sources -> N destinations |

### Mechanism

- UNICAST: Individual transactions to each destination
- MULTICAST: Hardware multicast (one transaction to multiple destinations)

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

### DRAM read on Blackhole
```cpp
NocEstimatorParams params{
    .pattern = NocPattern::ONE_FROM_ONE,
    .memory = MemoryType::DRAM,
    .arch = Architecture::BLACKHOLE,
    .num_transactions = 64,
    .transaction_size_bytes = 8192
};
NocEstimate result = estimate_noc_performance(params);
```

### Linked multicast
```cpp
NocEstimatorParams params{
    .mechanism = NocMechanism::MULTICAST,
    .pattern = NocPattern::ONE_TO_ALL,
    .num_transactions = 16,
    .transaction_size_bytes = 2048,
    .num_subordinates = 4,
    .linked = true
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

- All parameters have sensible defaults, set only what you need
- The estimator auto-initializes on first call
- Estimates are based on empirical measurements from the data movement test suite
- estimate_noc_performance throws std::runtime_error if data is missing or no match is found
