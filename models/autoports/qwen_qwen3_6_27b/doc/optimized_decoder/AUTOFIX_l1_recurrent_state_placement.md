# Persistent L1 recurrent state: placement and capacity audit

Source-only investigation against frozen `candidates/delivery118.py`; no device
execution and no implementation changes. The proposal is **blocked with the
current B32 compute placement**. This does not reject a coordinated, separately
tuned decode/prefill partition or smaller batches. A whole-decoder timing win
would still be required. Removing state DRAM traffic changes the roofline byte
denominator; it does not by itself satisfy the 70% requirement.

## Matmul owners are a rectangle, not just the active workers

`ttnn/cpp/ttnn/operations/matmul/device/factory/`
`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1371-1396`
collects bank-adjacent workers, then takes their **bounding rectangle**. Its X,
W0/W1 and OUT DataflowBufferSpec objects (around 1456-1510) have no narrower
target-node override. The `bank_cores_rect` WorkUnitSpec at 1704-1708 assigns all
three kernels and their buffers to that rectangle, including idle role-zero
cores. Counting only 16 or 32 active workers understates the conflict.

The measured `geometry106.log:19` allocation failure identifies MLP down and
rectangle `[0-0 - 6-9]`; its static region ends at byte 1,312,448. This was the
capture-lifetime failure repaired in geometry V2, not an L1-state experiment.
It supplies a real allocation extent for the unchanged default R2 matmul.
`.../matmul/device/utilities/matmul_utilities.cpp:494-539` builds worker lists
deterministically, with reader index outermost. With the same device/NoC/bank
count, R4 contains the R2 worker prefix, so its rectangle cannot create more
disjoint owners. On this 11x10 grid **at most 40 cores** are outside every
default matmul rectangle. The exact R4 rectangle could reduce that number;
the impossibility below already holds at the optimistic 40.

Blackhole physical L1 is 1,536 KiB/core (`tt_metal/hw/inc/internal/tt-1xx/`
`blackhole/dev_mem_map.h:33`). Firmware and the default 69 KiB kernel-config
reservation further reduce allocatable memory (`tt_metal/llrt/hal/tt-1xx/`
`blackhole/bh_hal_tensix.cpp:44,69-70`).

| B32 per-core quantity | Bytes | KiB |
| --- | ---: | ---: |
| BF16 state: ceil(1536 heads / 40 owners), 32 KiB/head | 1,277,952 | 1,248 |
| Current recurrence CBs, BF16 state variant | 360,448 | 352 |
| Sum before firmware, allocator, other tensors | 1,638,400 | 1,600 |
| Physical L1 | 1,572,864 | 1,536 |
| Physical overrun | 65,536 | 64 |

The state is 50,331,648 bytes (48 MiB) total. CB sizes come directly from
`candidates/gdn_sfpu_decay_predict.py:660`: input/state/output rings remain
allocated on **every recurrence compute core**, including proposed state
owners. Even zero system reservation requires at least 42 owners. The CPU
check below extracts that CB dictionary and checks these lower bounds.

The table assumes whole heads per owner. Allowing a height shard to split a
head at a tile-row boundary improves the bound to 154 tile rows * 4 tiles/row *
2,048 bytes = 1,261,568 bytes/owner. With current recurrence CBs this still
exceeds physical L1 by **49,152 bytes (48 KiB)** before all reservations. Thus
whole-head alignment is not necessary for the rejection. The absolute
zero-padding average state size also exceeds the remaining L1 capacity.

## Accessor legality is not the blocker

The generic reader/writer build `TensorAccessorArgs(state)` and address state
by logical tile page `head*16+i`; they do not encode DRAM bank arithmetic.
`tt_metal/hw/inc/api/tensor/tensor_accessor.h` supports sharded distributions
and both L1/DRAM through DSpec. The current Python helper explicitly asserts
DRAM for all inputs (line 629); that check would need to allow HEIGHT_SHARDED
L1 for state only. Rebuilding accessor compile arguments from the actual state
and keeping it in generic_op's registered tensor list preserves the existing
runtime-address tracking. Actual L1-sharded read/write, reshape aliasing,
in-place publication and trace replay remain **unverified**.

One legal-looking decode-only topology is explicit recurrence compute on
`x=0..6,y=0..9`, with state owners outside the union of all matmul rectangles.
At the optimistic 40 owners, this removes recurrence CBs from owners and leaves
288 KiB/owner before reserved memory. The current `cores=70` option does NOT
construct that rectangle: it takes 70 row-major cores in an 11-column grid.
An explicit coordinate-list change is needed. Reducing compute from 110 to 70
cores also increases heads/core; no latency benefit is assumed. An actual R4
rectangle must be recovered before assigning owner coordinates.

## Prefill must coexist with the same live allocation

`delivery118.py:1027-1061` keeps the published state object stable, casts the
BF16 state to FP32 for native prefill, then publishes BF16 back into the caller's
buffer. With L1 input, the initial implicit-memory-config typecast would itself
be unsuitable. A device-only boundary can instead be expressed as:

```python
dram_bf16 = ttnn.to_memory_config(published.recurrent, ttnn.DRAM_MEMORY_CONFIG)
work_fp32 = ttnn.typecast(dram_bf16, ttnn.float32,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Existing native prefill and continuation, operating on work_fp32.
rounded = ttnn.typecast(work_fp32, ttnn.bfloat16,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
ttnn.interleaved_to_sharded(
    rounded, published.recurrent.memory_config(),
    preallocated_output=published.recurrent,
)
```

The last keyword is supported by the MemoryConfig overload in
`ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/`
`interleaved_to_sharded_nanobind.cpp:77-93`. This is an API/source sketch, not a
tested publication implementation. Conv carry, arbitrary logical sequence
lengths and chunk ordering must remain unchanged.

**The public L1 state still occupies owner memory throughout prefill.** A
temporary DRAM copy does not solve coexistence. Freeing/reallocating the public
state would invalidate the stable allocation and captured decode addresses.
Current prefill programs use the 11x10 grid. The 2D projection path in
`delivery118.py:464-490` uses K-block 8; for the packed 544-tile width this gives
50 N tiles/core. Its weight CB alone is 8*50*1088*2 = 870,400 bytes when K has
multiple blocks (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:418-425`),
well beyond the optimistic owner margin. Flat correction still calls this
`_linear_raw` path (`delivery118.py:2139-2146`).

Native GDN uses explicitly placed producer/receiver sets, but its config has
no arbitrary core-range parameter. The placement must be checked/retuned too.
For Ct=1,Kt=Vt=4,NV=1, receiver-only CBs are 81 FP32 tiles = 331,776 bytes,
already more than 288 KiB; handoff and system bytes are additional.
See `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/`
`chunk_gdn_fused_program_factory.cpp:159-250` and
`chunk_gated_delta_rule_config.hpp:38` for geometry controls. This receiver
bound applies wherever an NV1 receiver overlaps an owner; it is not a claim
that every calibrated placement necessarily overlaps every proposed owner.

## Decision and next experiment prerequisites

No drop-in B32 L1-state probe was produced: it would knowingly fail the physical
bound above. This conclusion combines existing allocation evidence with source
capacity arithmetic, rather than treating a first API error as rejection.

A future stage-local experiment can remain legal if it first defines a joint
decode/prefill partition: recover exact R4/MLP rectangle union; exclude owners
from recurrence compute; retune all large prefill matmuls and native GDN placement
to coexist; implement the stable preallocated publication boundary; then probe
actual L1 allocation, exact own-input recurrence, changed-input trace replay,
prefill/continuation and whole-layer latency. Do not reduce advertised context
capacity for this optional optimization. Smaller batches may have enough room
under the existing placement, but require allocator validation; the CPU batch
table is only a lower bound. A correct DRAM-state path may remain the fallback
for batches without a verified profitable L1 placement.

Reproduce the source-only check:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_l1_state_budget_cpu.py
```

Artifact: `candidates/l1_state_budget_cpu.json`. No PCC, device latency or
watcher result is claimed for this unimplemented topology.
