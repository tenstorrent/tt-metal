# PLE resident-prefill capacity: scope of the finite boundary argument

This is a source review, not a device test or stage verdict. The reviewed path is
batch 1, the complete resident packed PLE table, the recorded expert datatype,
all 32 physical prefill signatures plus decode 1 prepared before large I/O, and
the current public prefill/decode/repeated-prefill lifecycle. It is not a claim
about arbitrary allocator history, batch-capacity products, or an optimal implementation.

The closed `ple_capacity_private_public_4768_v1.json` records two complete public
passes of 149 chunks each, separated by traced decode. The fresh 4769 record fails
on its first actual chunk: a 20,971,520-byte allocation needs 2,621,440 bytes per
bank; 4,446,016 bytes per bank are free but the largest block is 2,485,376 bytes.
Thus 4768 is an observed current-profile boundary; this 4769 failure is allocator
fragmentation, not exhaustion of aggregate free DRAM. Neither record alone is an
HF correctness result.

The separate physical reason some reduction is unavoidable is stronger and
independent of that boundary: advertised 262144-token hidden input alone occupies
5,368,709,120 bytes, while the recorded loaded model/state leaves 514,902,528 bytes
before all-tail program preparation. This does not establish 4768 as an optimal
or universal maximum, and no such claim is needed.

## Why finite tail coverage is meaningful

`tt/functional_decoder.py:145` divides every positive logical length into
32-row chunks and at most one tail. For lengths above 32 it allocates one complete
output and releases each temporary chunk immediately after its queued writer.
It neither retains previous chunk outputs nor builds a length-sized list of
intermediates. Lengths 1..32 use the already-prepared direct chunk path.

`tt/chunk_input.py` and `tt/chunk_output.py` use fixed generic-op descriptors;
source/destination addresses, offset and count are runtime arguments. Their
standard BF16 tile page size and UINT32 token-word page size do not depend on
the complete prompt length. Therefore more offsets do not introduce more slice
programs or persistent program resources. PLE/GDN state sizes are fixed; each
decoder math pass has one of the prepared 1..32 physical row counts.

The length-dependent resident I/O is two BF16 tiled buffers of
`ceil(N/32)*32*10240*2` bytes each, plus `N*16*4` token-word bytes. Other prepared
resources and the stable decode inputs are fixed by this profile. Terminal
lengths 4737..4768 share the same 97,648,640-byte hidden/output allocation size
and exercise every tail residue; token storage ranges 303,168..305,152 bytes.
Smaller tiled-length buckets reduce both large I/O buffers, and execute fewer
copies of the same chunk program. This is a bounded resource/shape argument,
not an assumption that binary-search observations prove monotonicity.

## Supported statement after the pending checks

Fresh boundary success/failure, a passing terminal 32 lifecycle sweep, complete
literal-HF public checks at 4768 and 4767, and the existing short/boundary tests can
support a documented `1..4768` range for this prepared current implementation.
There is no concrete source hazard here requiring exhaustive execution of every
integer length. The argument is an engineering inference from finite coverage
and bounded allocations, not a theorem that every allocator history is monotone.
If terminal/HF checks fail, that failure must be resolved rather than hidden by
this argument. A stronger all-count workspace sweep is corroboration with a
distinct allocation ordering, not a replacement public contract.

Report 4768 as the largest observed passing current-profile boundary with a fresh
adjacent failure. Preserve higher/lower historical profile observations, state
that advertised single-call input is physically impossible with resident
weights, and leave room for a different implementation to support more.
No additional hardware requirement beyond the pending terminal/HF checks is
introduced by this review.
