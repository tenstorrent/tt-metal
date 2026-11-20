# Numerics

The numerics classes the model code and the tests use, and the relation each served path has to the others.

## Classes

- **Bitwise**: identical bytes, tensors or token streams. The same request on the same build repeats bitwise, and a
  request without sampling fields gets the greedy stream the acceptance replay measures, on a sampling server too.
- **ULP**: the same math re-associated or fused (an eltwise chain in one pass, a reduction in another order).
- **Component**: the chain's precision points move (fp32 state kept, a rounding removed); judged against the `tt/`
  torch reference at the component, not bitwise.
- **Tolerance class**: two paths whose greedy streams agree up to a near-tie and can differ after it. The device is
  in this class against the CPU reference.

`ttnn/fused/registry.py` records each fused kernel's class. A kernel serves by default only when it is bitwise
against the op chain it replaces, or component class with a recorded component proof; `QWEN38_FUSED_OFF=<names>`
runs the op chains instead.

## The acceptance gate

`--acceptance` replays the CPU greedy records in `tools/acceptance/greedy-prompts.jsonl` before the server listens:
twelve records, plus the slab record `document` on a server with a prefill slab (a server without one skips it by
name). `--require-json-96` refuses to serve unless the `json` record matches the CPU for all 96 tokens. The other
records leave the CPU stream after a near-tie; each record's first divergence index is written to `acceptance.json`
in the run directory. The records carry their own system turn in their prompt ids.

## Relations between paths

- **GDN chunk recurrence**: main's `ttnn.transformer.chunk_gated_delta_rule` in every chunked form, the slab, the
  verify-rows wrap and the lanes with the fold off. Its arithmetic differs from the source's in the last bits, so
  greedy streams are not the source's word for word.
- **Prefill forms**: the slab body is tolerance class against the chunk bodies. A prompt shorter than 128 rows runs
  32-row chunks in every chunked form, so the `json` record is the same in all of them ([PREFILL.md](PREFILL.md)).
- **MTP**: the verify rows compute the target in a multi-row shape, so the committed stream can part from plain
  decode at a near-tie. Sampled requests draft by exact speculative sampling: each emitted token follows the
  distribution plain sampling draws from that row.
- **Lanes**: a lane's stream equals the single stream's; the `--lanes` start-up gate replays the acceptance records
  through the lanes and requires it (`acceptance-lanes.json`). A forced `</think>` from the thinking budget is a
  tolerance event: the lane feeds it through the verify body, the single stream through the 1-row step.
- **Prompt-end snapshot**: the restore copies the snapshot back bitwise (checked at every start). It reproduces the
  state of the request that captured it: `qwen38.snapshot_schedule` is `chunked` when that state is a fresh
  prefill's, and `forced-tail` when parts of it came from 1-row steps, which agree with a fresh prefill to rounding.
- **Two DRAM readers per bank** on dies harvested in different columns: every device uses the reference device's
  bank-to-worker placement (`decode_dram_workers_placement`, named in the READY record), so some dies read a bank
  from a further worker. Placement does not change the arithmetic.
- **`gr_recip_last`** (component class): `bf16(x * gamma/4)` goes into the matmul without the chain's spill, the
  rsqrt scales the fp32 partial, and the gate operand has one bf16 rounding where `gr_read`'s chain has two.
- **Device sampler**: draws from the same law as the host sampler over the same candidate row.
- **Images**: tolerance class against the CPU reference; text requests are unchanged with the tower resident.
- **BF4 experts**: the cache holds the host packer's bytes; every start re-packs one expert and compares it bitwise.

## Buffer lifetimes

`ttnn.reshape` of a ROW_MAJOR tensor (or of a TILE tensor that keeps its last dimension) is a view over the same
buffer under a new tensor id. Freeing the source frees rows the view still reads, so `ttnn/contracts.py`'s
`same_buffer` decides whether a reshape's source may be freed; tensor ids must not decide it.

## Reference columns

`tools/reference/` freezes the teacher-forced corpus `Q38-REF-v1` with its sha256s.
`tools/qwen38_reference_corpus.py` produces a column per source (`hf`: Transformers `qwen4_exp` on the CPU, bf16
weights and fp32 LM head; `oracle`: `tt/` with bf16 or BF4-emulated experts; `device`: a served chain's agreement
records) and `score` compares two columns (top-1 / top-5 agreement, clear-margin top-1, truncated KL over the shared
top-32 support, first divergence). The HF column is the reference.
