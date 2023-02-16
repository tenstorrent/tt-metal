

noc_async_read_barrier
======================

This blocking call waits for all the outstanding enqueued `noc_async_read` calls issued on the current Tensix core to complete.
After returning from this call the noc_async_read queue will be empty for the current Tensix core.

Return value: None
