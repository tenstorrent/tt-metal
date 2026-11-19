# Overlap decode-convolution reader, compute and writer with two CB slots

The parent reports approximately 87 us for the bit-exact custom convolution
in profile80. The current one-entry input CBs remain occupied until compute
receives the writer's completion token, so the reader cannot prepare the next
task while the current task computes and writes carry. This candidate changes
only scheduling/storage, preserving every convolution and SiLU instruction.
No device speedup or correctness result is claimed here.

## Immutable artifacts

- `candidates/conv_decode_sfpu_pipeline2.py`: standalone helper, same public
  `conv_decode_sfpu(token, history, taps, output=None, cores=None)` signature.
- `candidates/make_conv_pipeline2_candidate.py`: freezes an integrated candidate
  from an existing decoder already using the custom convolution.
- `candidates/conv_pipeline_on_exact82.py`: generated from exact82; replaces its
  one custom-convolution call with `conv_decode_sfpu_pipeline2`.
- `candidates/check_conv_pipeline2_cpu.py` and `conv_pipeline2_cpu.json`: source
  identity, task ownership, and exhaustive finite-state scheduling proof.

The helper's source-string globals use `_CONV_PIPE2_*`; the generator renames
its Python function before appending. Existing convolution, norm and GDN globals
are not rebound. The generator refuses overwrites and checks the expected call
appears exactly once. Main/shared source and all previous candidates are intact.

## Scheduling repair

CB0..7, CB9 (output), and CB10 (writer done tokens) each grow from one page to
two. Reader-private CB8 scratch remains two pages. Total CB allocation increases
from 24 KiB to 44 KiB/core. The reader's executable source is unchanged: reserve
one entry in every input CB, gather/read all operands, barrier, publish entries.
It can now prepare the second task while compute or writer uses the first.

Compute processes groups of two tasks, with a final singleton if necessary.
For group slot0 it waits for one input entry; for slot1 it cumulatively waits
for two. It selects `copy_tile(...,slot,...)`, runs exactly the old four-product
left-associated FP32 sum, BF16 RNE, native BF16 SiLU, and output pack, then
publishes each output immediately. Thus the writer can drain output0 while
compute works on output1. Only after both outputs are published does compute
wait for two writer completion tokens, pop two entries from every input CB,
and consume the tokens. The final singleton uses counts of one.

Grouping is essential: each complete input group starts at the ring base, so
slot1 never indexes beyond a wrapped CB end. Popping two entries wraps exactly;
the only pop-one occurs at program end. Compute remains the sole input-CB
consumer/acknowledgment owner, and each done token follows a complete writer
NoC barrier. Input storage is therefore retained until both compute and writer
have finished reading it.

## Why a depth-only edit would be wrong

`CircularBuffer::get_read_ptr()` returns the calling processor's local
`fifo_rd_ptr` (`api/dataflow/circular_buffer.h`). `cb_pop_front` advances that
local pointer and the shared acknowledgment counter; it does not advance a
second processor's local pointer. The old writer never pops input CBs, so its
carry read pointer remains at the ring base. This was harmless with one slot.
With two slots, using the unmodified pointer would repeatedly write slot0 carry.

The new writer explicitly uses offset `((task-start)%2)*2048` from its unchanged
carry-CB base. It does not pop or wait on those input CBs. Output publication is
the proof that compute has already observed the relevant input entries. The
writer still waits/pops output normally, performs every output/history write,
waits for the NoC barrier, then publishes one done token. Only compute consumes
the input entries, preventing conflicting acknowledgment ownership.

Each `(user,chunk)` task reads and updates a distinct 1024-channel history
slice. Other tasks may concurrently read/update other chunks or users without
intersection. Within one task all three old history slices are read before any
carry write, because output publication follows complete input availability.
The raw current token and taps are read-only. This allows reader lookahead even
though the history tensor itself is updated in place.

## Host validation completed

Python syntax compilation passes for helper, generator and generated decoder.
The source checker confirms the reader is identical after removing comments,
and the complete arithmetic/pack span is identical after accounting for the
input tile index change from zero to slot. The NoC calls retain all five
mandatory arguments.

The finite-state model enumerates every reachable interleaving of input
publication, output computation/publication, writer NoC start, writer barrier
completion, and compute group acknowledgment. Per-core task counts
1,2,3,4,5,8,10,20,40,80,160,320 cover singleton tails and repeated wraps. Across
4577 states it finds no deadlocks, live-slot overwrites, wrong carry slot,
output overwrite, or acknowledgment before writer completion. It separately
validates unique task/history ownership for B1/B3/B32 and eight requested core
counts, including partitions whose global start is odd.

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_conv_pipeline2_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_pipeline2_cpu.json
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_conv_pipeline2_candidate.py --base exact82.py --output conv_pipeline_on_exact82.py
```

These are abstract/source checks, not an emulation of TRISC timing or device
compilation. The existing CPU convolution mapping proof also applies because
reader addresses and arithmetic are unchanged.

## Required parent-owned device validation

The existing real-tap/captured-input comparator accepts the new helper:

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_conv_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_decode_sfpu_pipeline2.py --require-baseline-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_pipeline2_probe.json
```

Also run a focused `--cores 1 --batch 3 --steps 3` probe to exercise repeated
ring wraps and a core partition with many tasks; default hardware may assign
only two or three tasks/core. Required evidence remains device compilation,
zero output differences, exact in-place carry, changed-input/output reuse,
watcher, trace behavior, unchanged 512-step gate, and measured latency.

Pairs have a synchronization gap between groups, and B1 cores with only one
task gain no overlap. The candidate does not remove the 64 small token reads
per task. Benefit depends on how much of the measured convolution latency is
reader work versus compute/NoC writing; it must be measured. More aggressive
ring protocols are deferred until this minimal repair is evaluated.
