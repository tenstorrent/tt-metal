# AutoFix Report

## Starting Evidence

- Source report: `AUTOTRIAGE.md`, generated from the exact TP4 topology-probe
  hang on 2026-07-18.
- Original failing command:

  ```bash
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  RUN_MULTICHIP_DECODER_TOPOLOGY_PROBE=1 timeout 600 \
  pytest -q -s --tb=short \
    models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
    -k fractured_residual_topology_probe
  ```

- The separate reduce-scatter, local add, distributed RMSNorm, all-gather, and
  BFP4 QKV matmul path completed and matched the replicated boundary. The fused
  `all_gather_matmul_async` call did not complete. GDB placed the host in
  `FDMeshCommandQueue::finish_nolock`; all four ARC heartbeats remained live.

## Hypothesis Experiments

- Hypothesis: the fused operation uses a TP8-only transfer ledger.
  Experiment: reproduce the exact hang; compare the passing standalone
  all-gather path; build the producer/consumer ledger from
  `all_gather_matmul_async_program_factory.cpp` and `MatmulOpReceiver`.
  Result: the factory hardcodes four transfers and the receiver applies two
  directions. For TP4 QKV, eight advertised slices times four K-blocks/slice
  cannot equal the actual sixteen K-blocks. Existing fused tests use TP8, where
  that same hardcoded count is correct.
  Verdict: verified.
  Evidence artifact: `AUTOTRIAGE.md`.
  Fix: the TTNN core operation must derive transfers as `ring_size / 2`, reject
  odd rings under the current equal-two-direction protocol, and validate the
  gathered-K block ledger before launch. That core edit is outside this goal's
  explicit autoport-only write scope and was not made.
  Verification: the standalone TP4 fractured boundary remains the safe
  model-local experiment. It passed with PCC above 0.99997 on all ranks and
  measured 0.085782 ms versus 0.084890 ms for the selected replicated boundary.

- Hypothesis: links, worker count, chunking, BFP4, or the matmul core geometry
  caused the hang.
  Experiment: evaluate each parameter against the receiver equality.
  Result: none changes the false slice count; varying them would only launch the
  same invalid ledger.
  Verdict: refuted.
  Fix: none.

## Final Status

- Status: platform limitation isolated; no failing fused operation remains in
  the model-stage test path.
- The decoder keeps replicated residuals with asynchronous TP4 all-reduces.
  This is both the fastest runnable measured boundary and the only trace-safe
  model-local option under the current TTNN core contract.
- Every terminated reproduction was followed by `tt-smi -r`, a four-device
  inventory check, and a successful `MeshShape(1, 4)` open/close smoke.
- Remaining upstream follow-up: make generic fused AGMM ring-size-aware and add
  TP4 regression coverage before reconsidering a fractured stacked contract.
