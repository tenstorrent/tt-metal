# NOC API Latency Tests

This test suite measures the latency (in cycles) of various NOC API calls using the experimental dataflow 2.0 API.

## Dispatch Mode
These tests use **Fast Dispatch (Mesh Device API)** with `GenericMeshDeviceFixture` for optimal performance measurement.

## Test Description

The tests measure the time taken by NOC API calls themselves, excluding barrier synchronization overhead. Each test:
1. Performs a sweep over number of transactions (1 to 256)
2. Performs a sweep over transaction sizes (configurable)
3. Measures cycles for the API calls only (barriers are outside the measurement window)
4. Uses experimental NOC 2.0 API (`experimental::Noc`, `experimental::UnicastEndpoint`, etc.)

## Test Parameters

- **num_transactions**: Number of NOC transactions to perform (1-256)
- **transaction_size**: Size of each transaction in bytes
- **kernel_type**: Type of NOC operation to measure (compile-time argument)

## Kernel Types

Different kernels test different NOC operations:

1. **Unicast Write**: `noc.async_write()` - measures unicast write API call latency
2. **Unicast Read**: `noc.async_read()` - measures unicast read API call latency
3. **Multicast Write**: `noc.async_write_multicast()` - measures multicast write API call latency
4. **Stateful Write**: `noc.set_async_write_state()` + `noc.async_write_with_state()` - measures stateful write API latency
5. **Stateful Read**: `noc.set_async_read_state()` + `noc.async_read_with_state()` - measures stateful read API latency

## Test Cases

| Test ID | Description |
|---------|-------------|
| 700 | Unicast Write Latency |
| 701 | Unicast Read Latency |
| 702 | Stateful Write Latency |
| 703 | Stateful Read Latency |
| 704 | Multicast Write Latency 2x2 |
| 705 | Multicast Write Latency 5x5 |
| 706 | Multicast Write Latency All Cores |

## Output Metrics

Unlike other data movement tests that measure bandwidth, these tests report:
- **Cycles per transaction**: Total cycles / number of transactions
- **Total cycles**: Total measurement window in cycles

## Implementation Details

- Uses `DeviceZoneScopedN()` profiling markers to measure API call latency
- Barriers are placed outside the profiling zone to exclude synchronization overhead
- All kernels use experimental dataflow 2.0 API
- Compile-time kernel selection based on test parameters
