# Text generator trace I/O

The text generator prepares persistent inputs and outputs before its first
capture. Warmup and capture both copy model results into the same explicit
output buffers. Capture-local activations are released; they are not returned
to consumers or exempted from allocation tracking.

Prefill variants share an input or output only when their model lane, semantic
role, tensor spec and mesh topology agree. Decode outputs use a separate group.
Decode tokens and positions retain separate storage per mode/bucket because
device sampling updates them between steps. Sharing these stateful inputs would
discard a continuing request's device-produced token or position.

Device results are **borrowed**. Their values remain valid until another writer
to the same compatible output group runs. Keeping a Python tensor reference
does not keep an earlier value. To retain a result:

- Enqueue its host read on command queue zero before the next writer. The
  generator's prefill readback and `read_decode_output(async_read=True)` do this;
  wait for the returned event before consuming an asynchronous decode read.
- Or copy it into separate device storage allocated before the first capture,
  warming that explicit-destination copy before capture too. Enqueue the copy
  before the next writer. Each simultaneously retained value needs its own
  destination.

The generator serializes execution on command queue zero. Concurrent host
submissions or cross-queue consumers need external ordering and are not supported
by this buffer-sharing contract.

Prepare the complete configured set before capture. Eager decode warmup stages
all requested decode variants; traced prefill warmup stages its configured
lengths and batch sizes. The first prefill call also stages both available
decode modes. A prepared batched prefill or alternate decode variant may be
captured lazily because its persistent storage already exists. An unprepared
variant is rejected before model preparation allocates device memory, including
when `skip_precompile=True`. That flag does not make late allocation safe.
Expanding the set requires releasing live traces and rebuilding the model and
generator with the complete configuration.

`test_prepared_trace_io.py` checks ownership and tracker enforcement with small
device operations. `test_prepared_trace_io_model.py` runs all 64 Qwen3-32B layers
on T3K, compares traced results against eager results across prefill/decode
alternation, and checks queued reads, retained copies and stable program caches.
Run both with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.

This applies to the base text generator's prepared paths. Models that replace
capture or warmup must prepare their own full set before delegating to these
paths. The separate Galaxy generator and its GCB/prefetcher lifecycle require
their own integration.
