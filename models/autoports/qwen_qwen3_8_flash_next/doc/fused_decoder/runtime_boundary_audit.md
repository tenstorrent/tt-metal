# Measured runtime boundary

The selected final default is bound by `wrap_final_plan.json` and its recorded
runtime source identity. It enables the exact residual writer and compact
physical-count-2 expert routing; `final_v5_source.json` is historical. The public decoder API,
request state and paged-cache lifecycle are inherited. The selected components
are asserted by `test_fused_decoder.py` and recorded by each `run_fused_probe`
selection sidecar; final acceptance must use completed final-source runs.

All `torch` imports and `from_torch` calls in `tt/fused_decoder.py` are inside
constructors or preparation methods: checkpoint regrouping/permutation, expert
packing and immutable reduction/route tables. There is no `to_torch` call in
that file. The inherited full-forward runtime audit forbids host conversions
and Torch tensor work during measured model calls. Host reference checks,
request uploads and state snapshots are outside the measured interval.

| Remaining movement | Contract / selected path |
|---|---|
| MoE prefill activity mask to row-major | Native sparse matmul consumes row-major per-tile activity. Compact physical counts 1/2 use prepared indexed-mode sparsity; the index list, not mask pruning, controls the dispatched experts. |
| MoE prefill score tensor to row-major | Dedicated weighted reduction consumes per-token row-major scores. The selected cleanup replaces the preceding TILE permutation with a reshape of the row-major result. |
| QSA position words to TILE arithmetic and row-major integer indices | Request position/cache-update interfaces own integer row-major state; RoPE arithmetic consumes tiled selected tables. These conversions change representation for their consumers. |
| QSA embedding output to TILE | Native table lookup produces selected row-major cos/sin rows; rotary kernels require tiled operands. |
| Precise projection reduction reshapes | Reduction outputs carry one value per logical row/output column in a different tile geometry. These are not identity layout round trips. The QKV kernel writes the final tile geometry directly; other projection kernels retain their necessary output conversion. |
| Residual-stream read grouping | The four-stream and hidden-width axes exchange order before normalization. This read-side permutation is not a metadata-only identity. The selected write kernel directly emits the final four-stream geometry and removes the former write-side permutation. |

The selected compact GDN recurrence reader/writer folds head repetition, scalar
expansion and padded-token layouts. The previous metadata-only row-major round
trips are absent from that path. The rejected native convolution and decode-head
adapters retain experimental conversion code behind disabled options; they are
not selected by the final default. In particular the decode QKV-head/concat
adapters' L1-to-DRAM reshards are disabled. No selected runtime reshard or host
fallback is introduced by this stage.

Official per-op tables list actual executed layout operations. Their presence
is not itself a redundant-conversion finding: the representation-changing
boundaries above must be distinguished from the removed identity conversions.
The selected residual writer removes its write-side layout intermediates while
retaining both BF16 rounding boundaries. Any further reader/layout fusion is
hand-off work under the owner wall cap, not a claim that a remaining conversion
is an identity operation.
