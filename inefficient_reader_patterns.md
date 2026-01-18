Inefficient writes can take several forms. Typically they are overly heavy barrier selection or they are using barriers earlier than needed.

# Overly heavy barriers
In most cases, `noc_async_write_barrier` is not needed. `noc_async_writes_flushed` can typically be used instead.

## Valid Uses
1)  `noc_async_write_barrier` is typically needed is when a payload is written to one core, and the barrier is used to ensure that the payload landed at the destination before sending a signal (semaphore) to another (3rd) core that may/may not consume the data from the second core. T

2) At the end of a kernel

## Invalid Uses
Every other use case is very likely invalid/inefficient. It is common to call a write flush before popping from a CB.


