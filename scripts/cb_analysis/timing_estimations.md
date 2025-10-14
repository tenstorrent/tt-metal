# OP level performance insights - Compute vs DM time

## The idea
Use `cb_wait_front()` and `cb_reserve_back()` wait time from Compute and Data Movement kernel to get some OP level performance realted insights:

1. Determine if OP is DM or Compute bound
2. Measure DM and Compute processing time

### DM vs Compute bound OP detection
In order to determine if OP is DM or Compute bound we should analyze waiting time spent in CB functions for Compute, DM Reader and DM Writer kernel. The decision should be made based on detecting where bottleneck is located e.g. which thread is waiting the most on CB calls:
1. `cb_wait_front()` from Compute kernel(executed on Unpacker thread) - bottlneck is in DM Reader kernel and it is limited by speed of reading from DRAM.
2. `cb_reserve_back()` from Compute kernel(executed on Packer thread) - bottleneck is in DM Writer kernel and it is limited by speed of writing to DRAM.
3. `cb_wait_front()` from DM Writer kernel - bottleneck is in Computer kernel on Packer thread. Determination of Compute kernel bottleneck root cause would require more detailed analysis since in most cases there are 3 threads running and double baffering of inputs/outputs is applied.
4. `cb_reserve_back()` from DM Reader kernel - bottleneck is in Compute kernel on Unpacker thread. Determination of Compute kernel bottleneck root cause would require more detailed analysis since in most cases there are 3 threads running and double baffering of inputs/outputs is applied.

### Measuring DM and Compute kernel processing time

Processing time of DM or Compute kernel in this context represent actual time kernel is doing meaningfull job, not waiting on CB to have new data or free space to store already processed data.

In terms of DM kernel, processing time considers number of cycles spent by NOC to read/write data from/to DRAM as well as cycles required for implementation of appropriate DM scheme(synchronization between cores, multicasting, etc.) depending on the OP.

In terms of Compute kernel, processing time is number of cycles required for Compute kernel(and Tensix engine) to read input data (assuming it is available in core's L1 memory), process it using Tensix engine and store results back in L1.

To estimate processing DM and Compute processing time we can use following simple formulas:
```
DM Reader time ~ DM Reader kernel time - cb_reserve_back() wait time
DM Writer time ~ DM Writer kernel time - cb_wait_front() wait time
Compute time ~ Compute(Unpacker) kernel time - cb_wait_front() wait time
```

Compute time estimation needs to be elaborated in more details since `cb_wait_front()` and `cb_reserve_back()` are executed on different threads(Unpacker and Packer) and they can happen in parallel.

## Testing and validation of methodology

### Test environment

Kernel modification based on eltwise test case, kernel work: cb, compute/noc, wait

Types of test: baseline, zone, counter


### Artifical test scenarios

### Real test scenarios

### Results
