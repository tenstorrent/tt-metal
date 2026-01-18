Inefficient writes can take several forms. Typically they are overly heavy barrier selection or they are using barriers earlier than needed.

# Overly heavy barriers
In most cases, `noc_async_write_barrier` is not needed. `noc_async_writes_flushed` can typically be used instead.

## Valid Uses
1)  `noc_async_write_barrier` is typically needed is when a payload is written to one core, and the barrier is used to ensure that the payload landed at the destination before sending a signal (semaphore) to another (3rd) core that may/may not consume the data from the second core. T

2) At the end of a kernel

## Invalid Uses
Every other use case is very likely invalid/inefficient. It is common to call a write flush before popping from a CB.


## Inefficient Examples
1) The following writer code is inefficient because we are unnecessary barriering in an inner loop. Since we haven't changed CB state and there are no additional data dependencies we can move the barrier to right before the call to cb_pop_front.

Additionally, the call can be changed to the `flush` variant.
```
                cb_wait_front(cb_output_id, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_output_id);
                for (uint32_t outer_dim_id = 0; outer_dim_id < outer_dims_from_forward; outer_dim_id++) {
                    uint32_t dst_stick_id = (outer_dim_id + outer_dims_to_receive +
                                             (outer_dims_to_keep_end - outer_dims_to_keep_start + 1)) *
                                                num_sticks_per_outer_dim +
                                            stick_start_id;
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
                        noc_async_write(l1_read_addr, dst_noc_addr, stick_size);

                        dst_stick_id++;

                        noc_async_write_barrier();
                    }
                }
                cb_pop_front(cb_output_id, 1);
```

# Unsafe uses
Any time we are performing a write (to noc or fabric) and the write is sourced from a CB, the write MUST have some sort of barrier before a call to cb_pop_front of that CB. At the very minimum, the contents of the write should be flushed from L1 (write flushed call or full barrier) before cb pop front. 



 