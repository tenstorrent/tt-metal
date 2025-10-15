# OP level performance insights - Disabling DM/Compute

## The idea
The main idea is to disable either Data Movement (DM) or Compute functionality in the kernel, which eliminates the overhead of waiting for data or free space in CBs. By using this approach, Compute or DM kernels will do only primary processing task. Two potential approaches:

1. Kernel Functionality Disabling:
    - Disable DM or Compute functionality
    - Non-disabled kernel perform only core processing
    - Measure processing time using a kernel profiler
2. Synchronization Elimination:
    - Disable synchronization between DM and Compute kernels
    - Each kernel operates independently
    - Measure processing time using a kernel profiler

Any potential approach should be generic and automatic meaning that it can be applied for any available DM or Compute kernel by simple method that doesn’t require any code changes(e.g. commenting out parts of code and similar).

## Tested methods
### Mock kernels for data movement
#### Idea
Mock DM kernel to remove actual NOC transactions and leave only CB related synchronization, in this way Compute kernels can run without DM overhead, similar can be applied vice versa.
#### Problem
Many of the data movement kernels are complicated, and creating clone kernels that mock them would be a big effort, also maintaining mock kernels would be additional overhead for developers

### Disabling NOC operations (Making them NOOP)
#### Idea
Self-explanatory, just remove NOC operations from DM kernels and measure Compute time using profiler.
#### Problem
Doesn't really work because the NOC is used both in the firmware (example for profiler), and making the NOC operations NOOPs would break the runtime (hang/no profiler data)

### Mock CB operations
#### Idea
Make CB operations like data is always available/there is always free place in CB so there is no wait overhead in DM/Compute kernels.
#### Problem
This would maybe work, but would also skew the measurements a lot because CBs aren't only used for Reader -> Compute and Compute -> Writer, but also for Compute -> Compute (Storing intermediate values) and we don't want to eliminate waiting time for Compute -> Compute.

### Disable NOC operations in Kernel using #define
#### Idea
Create debug header that will replace NOC operations with empty function. Include debug header as last include in files where NOC operations should be skipped. Similar can be done for compute in order to measure DM.
#### Problem
This almost works, the header contains #define noc_async_read(...)
This would work almost always because the preprocessor would remove calls to noc function inside of the reader/writer kernel C++ file, but it breaks if we call NOC functions with non-default template params (ex noc_asyc_read<something>()) because the preprocessor cannot remove these calls. This would require manual intervention from the person trying to measure the compute performance

### Disable NOC in Kernel using __attribute__((weak))
#### Idea
Provide the correct implementation marked as a weak symbol so that the firmware can be linked with it, but include overrides for the kernel build that will override the original implementation.
#### Problem
This solution doesn't work because a lot of the NOC functions are marked inline __attribute__((always_inline)), which doesn't work together with __attribute__((weak)) because of linking.
