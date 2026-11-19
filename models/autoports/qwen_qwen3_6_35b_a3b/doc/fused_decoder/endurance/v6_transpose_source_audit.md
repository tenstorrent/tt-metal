# Supplemental transpose-reader source audit

Candidate `85adacf5df348f815e684a6ad1b1a36ecf12e862b72ef156681bfabb86dccd92` is assessed independently of selected production44f87. This host-only audit does not select the candidate, run hardware, or alter the production entrypoint, frozen audit chain, compatibility manifest or capacity harness.

`tests/audit_fused_transpose_v6.py` pins the new helper AST and reader string, then reverses only these changes in the candidate snapshot:

1. `Chunk64StateGraph.__call__`, line2073: `self.transpose_mm(kb,k,transpose_b=True)` restores `self.mm(kb,t.transpose(k,-1,-2))`.
2. The same method, line2094: `self.transpose_mm(weighted_k,v_new,transpose_a=True)` restores `self.mm(t.transpose(weighted_k,-1,-2),v_new)`.
3. `Chunk64OutputGraph.__call__`, line2351: `self.transpose_mm(q,k,transpose_b=True)` restores `self.mm(q,t.transpose(k,-1,-2))`.
4. Remove exactly one trailing constructor statement from `Chunk64StateGraph.__init__`, and exactly the two appended reader/helper definitions.

Each call is matched exactly once inside its specified class/method. Whole-module AST equality to the frozen44f87 reference is required after these narrow inverses. The complete `FusedDecoder` AST is independently compared unchanged; this equality is a proof, not an exemption from runtime scanning. The existing corrected matmul class and compute/writer source strings remain identical.

The helper's constructor assigns references, the existing compute string and `last_plan=None`, and queries mesh grid metadata. It allocates no tensors. Its constructor and `__call__` stay inside the original runtime syntax audit; **no new setup exemption is introduced**. Only the already-reviewed `Chunk64StateGraph.__init__` setup hash changes, after proof that its sole addition is this nonallocating constructor call.

The reader keeps increasing-K iteration, circular-buffer schedule, FP32 payload transport and the corrected-FMA compute unchanged. Its new booleans select physical tile/face addresses for transposed logical operands. The Python adapter adjusts only logical matrix dimensions and compile-time flags before calling the original shape planner/scheduler. The original broadcast leading-axis stride semantics remain. All three selected call sites use tiled FP32 operands. Arbitrary positive public sequence tails continue through the original chunk padding/trim and planner; none is newly constrained by matrix tile geometry.

Exact pins:

- Helper AST: `ad92b3e06964b793f0daa48c6259c0f465315b75c2d5cc88cf63a417ef923b01`.
- Constructor AST: `b7a0368ac991fbcbbe48b9083a3275247e6f7f824abd90b0be82377c3bcc3412`.
- New reader string: `06c4ade37b4af14aa1acac5f48fdf284ce0318a1f581084b102981f5898593ec`.
- Frozen upstream decay auditor: `11c88ceb2dd3ea2855e0e6ce5c913e1f4ec875c12159e25309b3b737b606466b`.

The audit executes the existing44f87 source-audit chain, then scans actual85ad runtime functions and public contracts. Mock planner/dispatch evidence is explicitly transferred through identical method ASTs; no new device or tensor execution is claimed. Original21, transport9, convolution/norm14, decay8 controls are retained, with11 additional transpose controls: runtime readback, constructor upload/allocation, extra graph setup call, unrelated decoder edit, each of the three transpose flags, reader and compute mutations, and an extra call at the wrong site. Host transfer controls reject independently through the runtime syntax checker, not only fixed hashes.

Run with a fresh output path:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_transpose_v6 --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_transpose_integration.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transpose_85ad_source_audit.json
```

Evidence: [machine audit](v6_transpose_85ad_source_audit.json), [integration inverse proof](v6_transport_conv_norm_decay_transpose_integration_build.json), and [component/full-chunk assessment](v6_matmul_transpose_integration_assessment.md). The assessment preserves both the early slow attention samples and the later matched component timings; this source audit does not dismiss the former or infer whole-decoder speed from the latter.

Limits remain unchanged: no Python source audit proves native-library behavior, hardware arithmetic, allocator feasibility, PCC, determinism, Watcher cleanliness, or whole-decoder latency. No persistent allocation is added; the helper retains40960 bytes/core of temporary CBs. Public context262144 and cache dtypes/layouts remain unchanged, but source analysis is not a substitute for final capacity evidence. Whole-decoder testing and parent selection remain separate.
