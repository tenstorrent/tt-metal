# Functional decoder work log

## 2026-07-15 — IR classification and emit

- Confirmed a clean stage baseline at commit `66533e5bc32` on branch `mvasiljevic/model/openai-gpt-oss-20b`.
- Ran `classify_graphs.sh` over the supplied IR directory. Compiler graphs `g0`/`g2` classify as prefill (`fill_cache=48`), while `g1`/`g3` classify as decode (`paged_update_cache=48`, decode SDPA=24). The matching runtime wrappers report the same roles.
- Selected compiler graphs `g0` and `g1`; runtime wrappers were not used as translation inputs.
- The converter initially rejected signed `si32` `topk`/`scatter` dimension attributes. Normalized only those attribute spellings to `i32` in temporary copies and ran `ir_to_emit.sh`. Original MLIR was not edited.
- Read the raw MLIR alongside the emit. Confirmed batch 1, prefill sequence 17, decode sequence 1, cache length 128, head dimension 64, and a `1x4` TP mesh.
- Segmented representative layer 12 using the repeating `rms_norm_24`/`rms_norm_25` pair. The flat graph has 49 RMSNorm calls, consistent with 24 decoder layers plus one final norm.

## 2026-07-15 — TP collapse and implementation

- Collapsed 4-way head partitions and all-reduces to full dense Q/K/V/O math on canonical Hugging Face weights: 64 Q heads, 8 KV heads, hidden size 2880.
- Collapsed eight local experts per rank to all 32 dense experts and removed runtime collectives. Preserved top-4 routing, interleaved gate/up slicing, clamp limits, BF16-rounded sigmoid coefficient `1.703125`, and expert reduction.
- Added `FunctionalDecoder(LightweightModule)` with host-only const-eval in `from_state_dict`, BF16/TILE/DRAM runtime defaults, linear cache construction, prefill and decode methods, and no runtime host fallback.
- Preserved the emitted batch. Both paths validate public runtime shapes before dispatch.

## 2026-07-15 — Device gate and correctness iterations

- `tt-smi -ls --local` found four Blackhole devices. The first P300 pair repeatedly failed mesh open with stale sysmem (`expected NOC 0x1000000000000000`, observed address offset by `0x40000000`) even after one bounded `tt-smi -r`; all UMD lock files reported free and there were no device-owning processes.
- `TT_VISIBLE_DEVICES=2,3` opened and closed a `1x1` mesh successfully on the second complete P300 pair. All subsequent hardware commands were serialized on this pair.
- First emitted-shape prefill reached SDPA but exposed K sequence 32 versus V sequence 17. Restored the emitted post-RoPE logical slices; synthetic sequence-17 PCC then passed at `0.9999856`.
- With the final high-fidelity compute boundary, synthetic prefill passed at `0.9999946` for sequence 17 and `0.9999975` for sequence 128.
- Initial real prefill PCC was `0.9799969`. Isolation showed the dense MoE reached `0.9994637` when fed the exact Hugging Face attention residual; a `0.9993340` attention residual was enough to change routed experts.
- Applied the emitted high-fidelity norm/matmul/attention boundaries with FP32 destination accumulation. Final official real-weight PCC is `0.9997057` for prefill and `0.9996046` for one decode step at cache position 17.

## Gate commands

```bash
python -m py_compile \
  models/autoports/openai_gpt_oss_20b/tt/functional_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py

TT_VISIBLE_DEVICES=2,3 python \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py 21248

TT_VISIBLE_DEVICES=2,3 pytest -q --capture=tee-sys -o junit_logging=all \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/functional_decoder/test_results.xml \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py

python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/openai_gpt_oss_20b \
  --hf-model models/demos/gpt_oss/configs/gpt-oss-20b \
  --stage functional_decoder --require-contract --strict-caps
```

Stage-review verdict and local commit SHA are recorded after the final gates.

## 2026-07-15 — Exact device recovery record

The device skill's recovery ladder was followed before any model test. Exact commands and outcomes:

| Step | Command | Exit | Result |
| ---: | --- | ---: | --- |
| 1 | `timeout 60 tt-smi -ls --local` | 0 | Found four Blackhole p300c devices, UMD IDs 0–3 at PCI BDFs `01:00.0` through `04:00.0`, arranged as two board pairs. |
| 2 | `timeout 60 python - <<'PY'` with `ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)` | 1 | All-device selection failed the sysmem check: expected `0x1000000000000000`, observed `0x1000000040000000`. |
| 3 | `timeout 180 tt-smi -r` in a PTY | 0 | Bounded host-wide reset completed in 19.4 seconds; only low-power warnings were emitted. |
| 4 | `timeout 60 tt-smi -ls --local` | 0 | The same four boards enumerated after reset. |
| 5 | repeat the all-device `open_mesh_device` smoke | 1 | The same sysmem mismatch persisted, so reset did not recover the first pair. |
| 6 | `/proc/*/maps`, file-descriptor, and hugepage ownership scans; `timeout 60 build/tools/umd/lock_virus` | 0 | No device-owning process was found; all 21 UMD locks were `FREE`; devices 0–3 were detected. |
| 7a | `timeout 60 env TT_VISIBLE_DEVICES=0 python - <<'PY'` with the mesh smoke | 1 | A one-chip subset of a P300 reported a CUSTOM cluster without a mesh descriptor; this was not treated as sysmem recovery evidence. |
| 7b | `timeout 60 env TT_VISIBLE_DEVICES=0,1 python - <<'PY'` with the mesh smoke | 1 | The first complete pair reproduced the sysmem mismatch. |
| 7c | `timeout 60 env TT_VISIBLE_DEVICES=2,3 python - <<'PY'` with the mesh smoke | 0 | Opened the complete second pair, printed `MESH_SMOKE_OK`, and closed normally. |

One host-wide reset was enough to show that reset did not alter the first-pair failure. A second reset was considered but not run because it would be another host-wide mutation with no new isolation value, while the second complete pair passed. All model tests and capacity probes were subsequently serialized on `TT_VISIBLE_DEVICES=2,3`; device close remained normal after successful runs and expected capacity failures.

## 2026-07-15 — Independent-review remediation and capacity boundary

- The initial independent stage review returned `more-work-needed`. Its principal finding was that 128 was merely the emitted cache size, not the largest feasible context required by the stage contract. It also requested a fresh final suite, durable exact PCC output, an explicit dual-position decode invariant, and the detailed recovery evidence above.
- Added `tests/functional_decoder_capacity_probe.py`. Each fresh process constructs the complete dense decoder, allocates caches, executes the full prefill including all 32 experts, transfers the output to the host to synchronize, checks its shape, and closes the mesh.
- A bounded search passed at sequence lengths 512, 1024, 2048, 4096, 8192, 16,384, 20,480, 20,992, and 21,248. It failed at 32,768, 24,576, 22,528, 21,504, 21,376, 21,312, 21,280, and exactly 21,249.
- Thus 21,248 is the largest complete prefill that fits and 21,249 is the first logical length that does not. The latter pads to a physical TILE length of 21,280 and fails at the MoE up-value `ttnn.slice`: total request 3,922,329,600 bytes, 490,291,200 bytes per bank, 566,160,768 bytes free per bank, but only 457,193,856 bytes in the largest contiguous free block. The allocator reported a 4,272,341,376-byte bank size.
- Set `SUPPORTED_CONTEXT = 21_248`, reject larger caches before weight loading, and added the first-above-bound constructor test. The default remains the emitted cache length 128.
- Added a synthetic sequence-256 PCC case to cross the layer-12 sliding-window boundary. Documented that host `cache_position` and device `cache_position_tensor` must encode the same position.
- Updated the context contract with the exact boundary, the descriptor and allocator capacities, and the full-context 45-GiB gate/up lower bound. The 21,249 failure was collected immediately before adding only the protective constructor guard; the executed runtime was otherwise unchanged.

## 2026-07-15 — Final pre-review gates

- Re-ran the complete capacity probe at the enforced maximum after adding the guard: `CAPACITY_PROBE_PASS seq_len=21248 output_shape=(1, 21248, 2880)`, exit 0, followed by a normal device close.
- Final pytest command passed 6/6 tests in 7.71 seconds with zero failures, errors, or skips. `doc/functional_decoder/test_results.xml` contains the exact captured PCC output:
  - synthetic prefill 17: `0.9999945678956855`;
  - synthetic prefill 128: `0.9999974798163773`;
  - synthetic prefill 256: `0.9999979815485808`;
  - official real-weight prefill 17: `0.9997057201235178`;
  - official real-weight decode at position 17: `0.9996046254703057`.
- `py_compile` passed for the implementation, PCC tests, and capacity probe.
- The strict context gate passed: `target=131072, supported=21248 (DRAM-limited)`.

## 2026-07-15 — Independent stage rereview

- `$stage-review` inspected the stable staged artifact, both forge skill contracts, device evidence, selected raw MLIR and emits, classifier results and hashes, source/AST runtime audits, the JUnit result, capacity arithmetic, and the strict context checker.
- Final verdict: `clean-pass`.
- Required work: none. Other concerns: none. Material hard-check gaps: none.
- Residual risks are limited to the documented single-device dense-MoE capacity boundary, representative layer-12 validation, and caller enforcement of equal host/device decode positions.

## 2026-07-15 — Commit-hook compliance

- The first commit attempt created no commit: repository hooks applied `isort` and an XML final newline, then required the repository `expect_error` fixture instead of `pytest.raises`.
- Replaced the one bound-check context manager with `expect_error`; runtime implementation behavior was unchanged. Re-ran `py_compile`, the strict context checker, and the complete hardware suite after that test-source change. All six cases passed again with the same exact PCC values and zero skips.
- Re-ran the independent stage review on the hook-clean snapshot before committing.

## Local commits

- Reviewed functional-decoder artifact: `88aa9f0d75fecc65be3b034d1f9ea3fa9348df26` (`Add IR-derived GPT-OSS functional decoder`).
- This work-log-only SHA record is committed separately so the reviewed artifact commit can be named exactly.

## 2026-07-16 — Multichip provenance

- Retro-generated [`multichip_provenance.json`](multichip_provenance.json) from the selected compiler `g0` prefill and `g1` decode IRs for representative layer 12.
- The capture uses a `1x4` mesh with TP degree 4: 64/8 query/KV heads become 16/2 per rank and 32 experts become 8 per rank.
- The layer collective set is `ttnn.mesh_partition`, ring-sum `ttnn.all_reduce`, and a shared const-eval `ttnn.all_gather` for the MoE router-scatter base, all on cluster axis 1. This was static IR analysis only; no TT device was opened.
