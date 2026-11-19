# Prepared Q/K expansion hypothesis

The accepted `integrated134.py` selects `gdn_sfpu_update_readout.py`. Its
recurrence repeatedly broadcasts the same four prepared Q and K column tiles
while traversing each 16-tile state matrix. The measured predecessor's
recurrence cost was 532 us; updated decoder timings remain around 1.82–1.84 ms.
This candidate targets redundant SFPU traversal, without changing state traffic
or partitioning. No hardware result is claimed here.

## Source diagnosis and change

Each value head performs 16 K broadcasts for decay/prediction, 16 K broadcasts
for updates and 16 Q broadcasts for readout. Grouped Q/K is cached across three
heads, giving 144 full-tile broadcasts per group. Every full-tile broadcast
contains eight row bands. Native column broadcast performs an eleven-instruction
replay and a ten-instruction rotate/add tail per band; four tail slots also
carry useful loads or stores. Those dependency chains consume SFPU issue slots
inside the recurrent state traversal.

`gdn_sfpu_expanded_qk.py` expands each prepared Q/K column tile once, immediately
after its existing transpose. It uses the same native mask/rotate/add broadcast
and writes the result into all four eight-column slots in each band. The eight
existing FP32 CB13/14 tiles now retain fully replicated rows. Their allocation,
formats, producer/consumer counts, caching and full-FP32 unpack remain unchanged.
The two active fused primitives read the prepared row directly into LREG_BCAST.
Their original matrix arithmetic, stores, accumulation order and reduction tree
are unchanged.

This exchanges 144 hot-loop broadcasts for eight preparation broadcasts per
three-head group. It also adds four stores per preparation band. Accounting for
four useful instructions formerly hidden in each broadcast tail, the static
estimate is 17 removed SFPU instructions per hot-loop band, less 26 instructions
per new preparation band: approximately 17,920 SFPU instructions per group.
This is an issue-work estimate, not a latency estimate: unpack, pack, NoC,
instruction scheduling and preparation overhead can limit or erase the gain.
For ungrouped B1 there are 48 removed broadcasts versus eight preparation
broadcasts per head. No extra DRAM or L1 allocation is required.

The native layout contract is in
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h`:
each load/store slot spans four rows by eight columns, raw offsets are 0/2/16/18
within a paired-face band, bands start every four raw addresses, and the bottom
tile half starts at 32. Expansion writes disjoint bands in place. Its full FP32
pack/unpack stores the already-broadcast values, preserving the incumbent's
broadcast rounding and signed-zero behavior rather than assuming a raw scalar
copy is numerically interchangeable.

## Frozen artifacts and source checks

- Helper: `candidates/gdn_sfpu_expanded_qk.py`.
- Whole candidate: `candidates/integrated_expanded_qk134.py`.
- Generator: `candidates/make_gdn_expanded_qk_candidate.py`; only the selected
  recurrence dispatch changes, with a namespaced helper appended.
- Source/CPU checker: `candidates/check_gdn_expanded_qk_cpu.py`.
- Result: `candidates/gdn_expanded_qk_cpu.json`.

The checker passed. It establishes identical reader/writer strings, Python host
functions and policies, identical actual matrix arithmetic/store instruction
order, and an identical per-head loop including reductions, BF16 rounding and
CB lifetime. Six finite scales including signed zero and subnormals exercise the
native broadcast placement model. Python compilation passed. These checks do
not establish device instruction timing, compiler behavior or end-to-end accuracy.

## Parent-serialized validation

Run the existing immutable exact comparator against the accepted helper:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_recurrence_exact_comparison \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_expanded_qk.py \
  --reference models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_update_readout.py \
  --state-dtypes bf16 fp32 --output /tmp/expanded_qk_exact.json
```

Require zero output/state bit mismatches across all 54 cases. If that passes,
measure whole-decoder B32 short/512 and B1 short using
`integrated_expanded_qk134.py`, including recurrence device latency. Retain only
if the measured gain is consistent and the whole-decoder accuracy gate passes.
No device was opened and no main/shared source was edited by this investigation.
