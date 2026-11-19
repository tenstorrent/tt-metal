# AUTOTRIAGE

## Diagnosis

Apparent full-context stall is repeated host JIT compilation, not a device hang. The original run completed with LONG_CONTEXT_PASS at 16:20:54 UTC. Do not reduce context or reset the device.

## Triage evidence

`triage/full_long.txt`: no running model ops; dispatch waits for producer commands; Ethernet health passes. `triage/full_long_host_stack.log`: main thread waits `sync_build_steps`; compiler workers wait on changing compiler child PIDs. Direct process inspection found active `kernels/sdpa`, `reader_interleaved`, `writer_interleaved` compilation. Original `full_long.log` records only 320/2110 JIT hits and passing PCC at all required long windows.

## Source evidence

`sdpa_program_factory.cpp` at lines 1224–1236: scalar chunk offset makes compiled Sk=offset+Sq; Skt and k_num_chunks enter reader/compute compile arguments. Device tensor offsets instead use a fixed full-page-table maximum and runtime causal bounds.
`paged_fill_cache_program_factory.cpp` around line 340: page_table_stick_size enters the writer compile arguments. The model passed `page_table[:,position//32:]`, changing table width every chunk even though only n/32 entries are required for that fill.
Independent read-only hypothesis agent `/root/slice_cache_audit` verified these and distinguished ordinary TILE slice program entries (hash includes starts/ends) from kernel binaries (aligned starts are runtime arguments). Do not replace ROW_MAJOR page-table slices with tensor slicing: that implementation reads bound tensors to host.

## Proposed fix

Model-local exact-width chunk page-table slice, and tensor `chunk_start_idx_tensor` for SDPA. Keep full input, causal history, BF16 policy, target context and unaligned continuation behavior unchanged. No shared runtime changes.

## Uncertainty

Dynamic-offset path must pass the focused and advertised-context device gates. No device correctness/performance claim follows from source inspection alone.
