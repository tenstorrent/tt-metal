# V6 host-only source and public-contract audit

`tests/audit_fused_snapshot_v6.py` is a new companion audit; the original checker and all PCC/performance/device gates are unchanged. It does not import Torch, TTNN, the decoder, or its test helpers, and performs no hardware work. It parses source, then executes only the hash-reviewed host planner and prefill dispatcher with plain Python stand-ins.

Commands (from repository root):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_snapshot_v6 --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_snapshot_static_audit_8cda_v2.json
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_snapshot_v6 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_snapshot_static_audit_production_v5.json
```

Both pass. Output paths deliberately reject overwrite. The earlier `v6_snapshot_static_audit_8cda.json` is retained as the initial, less comprehensive audit; `_v2.json` is the complete evidence.

## Coverage

- All 34 non-setup functions/methods in the 8cda candidate are inspected, including `__call__`, helper constructors, pure shape/scheduling functions, nested functions and the inherited `Chunk64StateGraph.__call__` alias. This is a superset of runtime-reachable Python helpers. There is no class-wide setup exemption.
- Ten individually identified setup functions/methods may pack/upload host tensors, but only when their full AST matches the explicitly pinned V5 or 8cda reference. Runtime calls into those constructors/functions or setup methods are rejected. Any subsequent setup change, including a compact-LUT setup repair, requires explicit review and a new pinned allowance.
- Runtime Torch/NumPy references, host transfer/readback methods, runtime imports, dynamic execution, setup calls, and obvious host I/O are rejected. Top-level imports, initialization expressions, class decorators/bases, literal constants and the reviewed state-method alias are constrained. Functional decoder imports/references and all test imports are rejected globally.
- Public argument signatures match functional, with exactly the already reviewed V5 `prepare_prefill.chunk_size` hint-default change from 256 to 4096. No other signature delta is accepted. KV allocation/page-table, position preparation, batch methods and decode entrypoint bodies match functional exactly. Seven full-attention guards must contain the complete original V5 method body and return before linear repairs; full-attention implementation and remaining shared V5 methods remain exact.
- Each layer kind passes 40,392 positive host-plan cases, spanning starts/residues 0–65, aligned and non-aligned lengths, 256/4096 boundaries, maximum public context 262144, positive physical hints and invalid-bound rejection. Logical coverage is contiguous and exact; absolute positions never shift. Full-attention plans equal V5; linear plans use chunks64 relative to each public call, including internal final singletons.
- 168 mocked public-prefill executions check input slices/output lengths, absolute positions, unchanged public singleton recurrence, internal singleton chunk behavior, initial recurrent cache rounding, cache-boundary flags, projection K, invalid input-plan lengths and allocated full-attention context bounds. The 8cda cases include 21 internal linear singleton dispatches and 7 public linear singleton dispatches.
- Eleven mutation controls are rejected: helper host readback; Torch alias; runtime setup constructor; dynamic execution; test import; modified setup body; module-level host call; changed full-kind guard; changed functional KV method; changed public signature; added public sequence-alignment restriction.

## Limits and source identity

This establishes Python source restrictions and host control-flow contracts. It does not prove tensor numerical behavior, native TTNN implementation behavior, absence of arbitrary external-library side effects, C++ inline-kernel correctness, physical device capacity, numerical determinism, watcher cleanliness or latency. Optional diagnostic observer callbacks are not invoked by this audit. Hardware acceptance still requires the original device gates. The result explicitly sets `acceptance_eligible=false`.

The context JSON still advertises/supports 262144 without reduction. Its existing fused hardware-validation hash matches production V5 and **does not match 8cda**; the audit reports that mismatch explicitly instead of transferring physical-capacity evidence between source versions.

Pinned source hashes and per-method AST hashes appear in each JSON. Final audit source SHA256: `2de52ebca258d1780691d1b1accf22db5354ad36aefe77d6bcc6f869cca9c18a`.

Candidate SHA256: `8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa`.
Production/reference V5 SHA256: `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`.

No production, frozen candidate, shared/native source or original checker was edited.
