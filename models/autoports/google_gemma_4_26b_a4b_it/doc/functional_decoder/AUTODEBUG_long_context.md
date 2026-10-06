# Maximum full-context decode investigation

The integrated layer 5 run at length 262144 passes sampled prefill PCC
0.9962776617 but fails traced decode at positions 262143/262142/262143 with
PCC 0.1016041513/0.1148314142/0.1016041513. The repeated position is stable.
Evidence: `long_full_262144_final.json` and its adjacent log. Sliding attention
at the same logical length passes. The unchanged threshold is 0.995.

The local decode attention gathers every full-attention cache page. At this
length the caller owns 8192 pages, each `[2,32,512]` BF16. Gathered K/V have
shape `[1,2,262144,512]`; each of two QK operations produces
`[1,1,8,262144]` FP32 scores. The following softmax reduction and probability
times V multiplication span 262144 tokens. Sliding attention gathers at most
33 pages/1056 tokens, so its successful maximum-length result does not validate
these wide operations.

## Initial source checks

- Cache row addresses now use UInt32 shift/add. Root's `cache_indices.json`
  proves exact row addresses through physical page 262144, but does not test
  the complete embedding/permute/gather chain at 8192 selected pages.
- Float32 minimum and comparisons select the SFPU in
  `binary_ng_device_operation.cpp`; page-offset truncation is not supported
  by that source finding.
- The tiled single-core gather reader has a UInt16 local offset, but its
  selector switches to the multi-core path above 60 tiles. The production
  page table and index are row-major and use the separate RM implementation.
  The tiled UInt16 observation therefore does not establish this bug.
- Inspected generic reduction wrapper and embedding token reader use UInt32
  bounds/indices for this input. No proven width limit has been identified.

## Focused experiment

`tests/probe_long_attention.py` calls the production attention operation using
the model-derived head geometry, BF16 pages, a random physical page table,
and the exact maximum sequence extent. Instrumentation compares logical page
IDs, physical row indices, embedding contents, permuted K/V, QK matmul,
masking, max, exp, sum, division, and probability times V against CPU
operations on the identical accelerator inputs. A separate complete CPU
attention comparison checks the whole chain. This is diagnostic host
instrumentation, not a proposed runtime path.

## Verified result and retained repair

The first wrong operation is row-major `ttnn.gather` on the page table.
Logical page IDs are exactly `[0,...,8191]`; the gathered physical IDs are
corrupted (baseline PCC -0.00106912, maximum integer error approximately
1.15e9). This occurs before cache reads or attention arithmetic. Evidence:
`long_attention_probe.json` / `.log`.

Replacing only this identity gather with its input table makes all 524288
cache row indices, both cache embeddings, and both permuted K/V tensors
bitwise correct. The complete 262144-token attention result then has PCC
0.99999976096 against CPU. The same-input QK, softmax components, and weighted
V multiplication are accurate. Evidence: `long_attention_identity_gather.json`
/ `.log`. Long attention arithmetic and cache embedding are refuted as the
cause of this catastrophic failure.

The bounded no-cache probe `tests/probe_page_gather.py` confirms the selector
boundary: widths 128 and 1920 are exact; 1921, 2048, and 8192 are corrupt.
At width 8192, 6139 IDs differ. Direct table use is exact at every width.
This matches `gather_device_operation.cpp` selecting the row-major multi-core
factory only above 1920 indices. This identifies the native operation and
shape-dependent path; its internal C++ defect was not repaired in this task.
Evidence: `page_gather_boundary.json` / `.log`.

The retained model change in `tt/precise_attention.py` uses the supplied
physical page table directly for full attention, which selects every logical
page in table order. Sliding selection is unchanged. Real full-layer
4096-token/128-step decode passes at minimum PCC 0.9997738820, and real batch32
decode passes every slot at minimum 0.9998285179. Both report clean runtime
audits and equal repeated traces. The parent owns full-layer maximum and
near-maximum reruns after device handoff; those remain pending here.

The attention diagnostic now accepts `--native-full-gather` to reintroduce
the rejected operation on the integrated implementation, optionally paired
with `--identity-full-gather` for the same A/B override. These switches are
diagnostic only and do not affect normal decoder execution.
