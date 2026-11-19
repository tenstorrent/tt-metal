# Focused PLE resident-prefill capacity assessment

This is CPU/source analysis for independent review, not stage review or acceptance.
No device work, runtime change, accepted-capability field, or numerical threshold is
changed by this report.

**Conclusion:** the available finite tests and bounded-allocation source argument are
adequate to propose a supported **1..4768 logical-token range for the recorded batch-1,
all32-prepared resident-prefill profile**. No additional capacity experiment is presently
necessary before stage review. The hard physical reason some advertised-prefill reduction
is required and the evidence for the current implementation's tested boundary are
separate claims. Neither establishes a universal optimal implementation or allocator
history.

## Physical obstruction versus measured boundary

The HF configuration still advertises262144. One BF16 input of shape
`[1,1,262144,10240]` alone occupies **5,368,709,120B**. The recorded resident PLE
model/state profile has **514,902,528B** free before all-tail program preparation;
input alone exceeds that by4,853,806,592B, before its full output or decoder workspaces.
This is a byte-level physical obstruction to the advertised *single resident public
call* with the current fully resident packed table/BFP8-expert representation. It is
not a runtime-duration or convenience reduction. It does not limit streaming state
context, which is a separate advertised-context test.

The fresh public4768 lifecycle passes. Fresh4769 fails in the **first actual32-row
chunk** while the MLP hyper-connection projection requests20,971,520B: each of8banks
needs2,621,440B, has4,446,016B total free, but its largest free block is2,485,376B.
That boundary failure is **contiguous-allocation fragmentation**, not aggregate DRAM
exhaustion.4768 is the largest observed allocatable boundary of this fixed prepared
profile, not a claim that every alternative tensor lifetime, allocator, representation
or implementation must fail above4768. The task does not require an optimization-stage
search over all such alternatives.

## Why arbitrary logical lengths have finite support evidence

`tt/functional_decoder.py:144` accepts positive logical lengths without alignment
restrictions. Above32, it allocates the full output once and traverses every offset
in chunks of at most32; `del chunk` follows its last-consumer write. At most one
chunk output is retained. `tt/chunk_input.py` and `tt/chunk_output.py` pass offset,
valid count and addresses as runtime arguments to fixed descriptor programs. The
prior length/offset-dependent native-slice resource growth was repaired; retained
`chunk_input_final_v5.json` passes exact reads/writes, all counts, first unseen trace,
and program/allocation stability. GDN/PLE histories have fixed allocation sizes.
SparseMoE intermediates use fixed expert and physical-tile geometry; token choices
affect values/sparsity, not the full resident input/output sizes.

For the long public branch, resident prompt I/O is two BF16 buffers of
`ceil(N/32)*32*10240*2` bytes each plus `N*16*4` token-word bytes. N4737..4768 share
97,648,640B per BF16 buffer and cover all32 terminal residues; smaller tiled buckets
reduce these large allocations and execute fewer copies of the same prepared chunks.
This is a resource-envelope inference, not a claim that arbitrary allocator histories
are mathematically monotone.

The actual evidence is stronger than a binary search alone:

- `ple_capacity_residues_private_buffers_v1.json`:320 observed probes; for each
  residue r1..32,4736+r passes and4768+r fails. All post-probe allocator baselines are
  identical (341,782,016B free,41,217,152B largest block per bank).
- `ple_capacity_private_terminal_4768_v1.json`:all32 lengths4737..4768 pass the
  full-resident-I/O chunk/decode/repeat lifecycle. These are **proxies**, using two
  offset-zero full chunks plus the actual tail; they are not32 complete prefills.
- `ple_capacity_private_public_4768_v1.json`:two actual full public traversals of
  all149 offsets, with decode capture/replay between them; no output retained by
  the observation hook. `ple_capacity_private_public_4769_v1.json` is the fresh
  adjacent full-public failure, not an inferred search failure.
- `ple_public_hf_4768_private_v1.json` and `ple_public_hf_4767_private_v1.json`:
  complete real-weight literal-HF outputs, every whole/contribution row>=.995,
  three traced decodes, exact repeated output/state and clean runtime audits.
  Minimum prefill contribution PCC is.9951774922/.9956877869 respectively.
- `final_capacity_qsa_closure_v3.json`:all six records reproduce their declared
  outcome/source/log checks. Its exact unreachable QSA-only source exception does
  not excuse changed PLE runtime, failed metrics or physical claims.

## Exact proposed final contract wording

> PLE full resident public prefill supports logical lengths1..4768 in the validated
> batch-1 profile: fully resident group10 packed table, BFP8 experts, BF16 activations,
> all32 prefill/hash signatures and decode1 prepared before large prompt allocation,
> and the recorded state/trace lifecycle.4768 is the largest tested passing boundary;
> a fresh4769 call fails a contiguous DRAM allocation. The advertised262144-token
> single resident call is physically impossible in this profile because its input
> alone requires5.37GB versus0.515GB free. This is not an architectural maximum or a
> claim about an optimized alternative implementation.

The4768 lifecycle **does not retain a decode trace during the repeated maximum-size
prefill**: it releases the trace/output first (`probe_decoder_capacity.py:596` onward;
`probe_ple_decoder.py:417` onward). Independently tested live-trace interleaving uses
shorter requests. Do not claim the Cartesian product of maximum prefill, live trace,
and normal batch5. Hash/preparation subsets also have a different supported profile.

## Remaining work is not an additional capacity sweep

Independent review must accept or reject the stated finite range/resource inference
and physical-reduction scope. The full advertised PLE state-context run remains a
separate completion requirement. The previously retained3707 fresh contribution
outlier requires its existing model-specific numerical review; it must not be hidden
by either boundary accuracy or this capacity argument. If that numerical control is
rejected, correctness remains work, but another allocator sweep would not resolve it.
No unmeasured universal monotonicity, maximum trace-resident prefill, or optimal-memory
claim is proposed. Existing accepted-context fields should remain unchanged until the
independent review decides these explicitly scoped claims.
