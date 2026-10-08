# r03-b02-a02: land the all-gather stats scratch in L1 (mesh-coherent L1-interleaved buffer) instead of DRAM

## Motivation
After this lineage's combine fixes (r03-b0x-a01), the post-AG critical path is: forwarder F_FABRIC
(fabric multicast of the packed stick page + wait for the peers' fused atomic incs + local write barrier)
-> go -> each worker reads its 4 x 128 B gathered sticks (8 x 64 B reads) -> combine (0.38 µs) -> POST/drain.
r02-b02-a04 measured go -> sticks-in-L1 at ~0.67 µs (8 x 64 B DRAM round trip) and showed address pre-staging
does not help: it is the DRAM read latency itself. Its reflection (#3) names the only fix: make the gathered
data land in L1. The landing is also on the AG path itself: the forwarder's own local packet write is a DRAM
write it barriers on before releasing go, and every peer's fused write+atomic is a DRAM write followed by the
inc (flush=true), so the remote EDM's DRAM write latency sits inside every chip's F_FABRIC wait.

## Mechanism
`make_stats_tensor_spec` (the single source of truth for the persistent gathered-stats scratch, used by the
test's `dit_fused_distributed_rmsnorm_create_stats_buffer`, `compute_output_specs` and validate) allocates the
scratch as `INTERLEAVED, BufferType::L1` instead of DRAM. It is still a mesh-coherent MeshBuffer (same
address on every chip, same bank->core mapping), so the fabric multicast addresses are unchanged in kind.
Kernels need no change: the forwarder and worker writer already address it through `TensorAccessor`
(`TensorAccessorArgs(stats_dram_buffer)`), which resolves L1 banks the same way it resolves DRAM banks.
Files: `device/dit_fused_distributed_rmsnorm_program_factory.cpp` (spec + comments; validate message).
Host-side change -> rebuild.

## Why this is not a repeat
No earlier node touched where the AG lands. r02-b02-a04 (multicast go + pre-staged stick read addresses)
attacked the release fan-out and the address math, both measured as non-costs; it explicitly identified the
DRAM round trip as the cost and L1 landing as the fix. The r03 siblings all work on the combine chain in
compute; this is on the data-movement side of the same post-AG critical path, and it stacks with them.

## Expected effect and risk
Expected: go -> sticks ready shrinks from ~0.67 µs toward an L1-L1 round trip (~0.3 µs), and F_FABRIC may
shrink by the DRAM-vs-L1 write-ack difference on the local and remote landing writes. A fixed -0.3..-0.6 µs
on every shape (+2..4%). Risks: (1) L1 alloc: 2 buffers x 4352 B per bank at the top of L1, far below any
clash with this op's CBs; (2) trace/ping-pong safety is unchanged (same persistent-buffer protocol, only
the memory changes); (3) if remote chips' L1 bank->NoC mapping differed the sticks would land on the wrong
core and accuracy would collapse (PCC fail) — mesh devices use a uniform virtual grid, so not expected.
Read it with the r02-b02-a04 `go.py`-style W_AGWAIT end -> W_DRAIN start gap and F_FABRIC duration.
