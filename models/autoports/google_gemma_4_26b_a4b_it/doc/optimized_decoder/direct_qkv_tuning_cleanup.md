# Direct QKV subblocks and optional setup cleanup

Applied atomically after the parent reported hardware idle. Runtime SHA256:
`4a9050497d5d4734eb40715d8942adfc33413b958358a209dbbd31c9132a146e`.
`direct_qkv_tuning_cleanup.patch` records the applied diff. Compilation and
`git apply --check` passed before application. Both `DirectQKV.__call__` and
`LanePartitionQKV.__call__` retain exactly the prior AST. No device commands.

## Per-weight subblocks

With an explicit grid, `qkv_subblock_w=0` now chooses the largest divisor of each
projection's `per_core_N` from4,3,2,1. The lane setup validates each weight's
config independently; direct decode inherits it with `per_core_M=1`. Existing
nonzero choices and all defaults are unchanged. Factory `qkv_grid="auto"` keeps
its existing fixed subblock choice, so comparisons should pass an explicit grid.

| Separate topology | Grid | Per-core N tiles | Selected subblock widths |
| --- | --- | --- | --- |
| Sliding Q/K/V widths4096/2048/2048 | 8x8 | 2/1/1 | 2/1/1 |
| Sliding Q/K/V | 8x4 | 4/2/2 | 4/2/2 |
| Full Q/K widths8192/1024, tied V | 8x8 | 4/1 | 4/1 |
| Full Q/K, tied V | 8x4 | 8/1 | 4/1 |

These are source/arithmetic predictions, not new device timing claims.

## Optional setup cleanup

`qkv_direct_cleanup=False` is the unchanged default. Enabling it requires a
direct backend and an exact `BroadcastQKV` or `TiedQKV` source type. Unknown
source implementations are rejected before their references are changed.

For packed interleaved weights only, the factory's two phase copies come from
the same source quantization. The cleanup shares the prefill weight when dtype,
logical shape, layout and memory config match. Separate and DRAM-sharded decode
weights retain their distinct storage. Cleanup does not change numerical policy.

The source proof for releasing obsolete references is:

1. `DirectQKV` handles every logical M=1 call itself; it does not call the lane or
   broadcast decode implementations.
2. The delegated lane M>1 branch returns before accessing `lane_mask` and calls
   its source with the same M>1 tensor.
3. `BroadcastQKV.__call__` M>1 uses `.weight` directly; `.rows` is read only in
   M=1. `TiedQKV` adds output duplication after the same branch.

Consequently cleanup assigns the broadcast source's `.rows` and lane wrapper's
`.lane_mask` to `None`. It **does not call `ttnn.deallocate`**; any other tensor
alias retains its own storage reference. Policy field
`qkv_decode.setup_cleanup` records sharing and reference release. These fields
do not claim measured allocator savings.

This version releases persistent duplicates after construction; it does not
claim to eliminate the earlier setup-time quantization or peak allocation.
Maximum-context and public-contract tests remain required before enabling it
in a final default. The parent already planned those tests.
