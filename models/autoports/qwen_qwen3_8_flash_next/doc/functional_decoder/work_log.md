# Functional decoder work log

Stage 1 only, Qwen/Qwen3.8-Flash-Next. Worktree started clean at
`d7e788cd88199501a8546ddb7bc381a601af47a9`. No push is authorized.
Status: in progress; initial real-weight layer-0 and component gates pass, but stage acceptance is incomplete.

## Startup, 2026-11-12

- Enabled skills: tt-model-bringup 0.1.23 and tt-autodebug 0.1.8, confirmed
  with `codex plugin list`. Read installed startup and selected skills.
- Active Python: checkout `python_env/bin/python`. Bootstrapped pip using
  `python -m ensurepip --upgrade`. Transformers 5.12.1 lacked `qwen4_exp`.
  Installed `transformers==5.22.2` and `tt-perf-report==1.5.0`, then pinned
  `huggingface-hub>=1.31,<2` to retain diffusers compatibility. `python -m pip
  check` reports no broken requirements. See `logs/dependency_install.log`,
  `logs/hub_compat.log`, `logs/pip_check.log`.
- Official sources checked: HF target API, HF Transformers qwen4_exp source,
  and QwenLM/Qwen3.8-Flash-Next README. Target revision is
  `812d96e10a0dd82b69d749e68693698cf6c36daa`. Cached snapshot has no `main`
  reference, so loading uses its exact path. Downloaded only missing layer-3
  shards 80, 81, 82 with `hf_hub_download` at that revision.
- `fuser /dev/tenstorrent/*` showed no owner. `timeout 60 tt-smi -ls --local`
  listed four Blackhole p300c chips. All stage device commands select chip 0
  and run serially under timeout; no resets or other users' process kills.
- Default open with `TT_VISIBLE_DEVICES=0` failed before dispatch because
  a single visible P300 chip needs a custom mesh descriptor. Setting
  `TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`
  fixed the open/ones/add/synchronize/close smoke. The first attempted env
  spelling `TT_METAL_MESH_GRAPH_DESC_PATH` was ineffective. Exact outcomes:
  `logs/mesh_smoke.log`, `logs/mesh_smoke_single.log`,
  `logs/mesh_smoke_descriptor.log`. Grid: 11×10. The motherboard fallback
  warning uses bus ID as tray ID; the 1×1 descriptor smoke succeeds.
- Allocator reports 8 banks × 4,272,341,376 free bytes = 34,178,731,008 bytes.
  See `logs/dram_capacity.log`. This is measured empty-device allocatable
  capacity, not a failed model allocation or supported-context proof.
- Hook command before the smoke: `pre-commit run --files
  models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/logs/dependency_install.log`.
  Before the storage probe: same command selecting `tests/probe_ple_storage.py`.
  Hook logs: `logs/startup_hooks.log`, `logs/probe_hooks.log`.

## Architecture and capacity investigation

HF decoder residual width is 10,240 (four streams of 2,560). Each block uses
gated residual mixing before attention and MoE. Layer 0 is Gated DeltaNet;
layer 1 adds PLE; layer 3 uses QSA with compressed-block selection, not ordinary
dense attention. There are 512 routed experts, top 10, plus a gated shared
expert. The advertised context is 262,144.

The PLE n-gram table is 128 checkpoint shards × 2,500,012 rows × 160 BF16
elements = 102,400,491,520 bytes. This table alone exceeds chip capacity.
The complete BF16 layer-1 weights total 107,654,129,368 bytes. This is a
storage investigation; no context reduction or omitted layer is accepted.

Reproducible metadata command (substitute the absolute snapshot path):

```bash
python models/autoports/qwen_qwen3_8_flash_next/tests/inspect_checkpoint.py \
  --snapshot "$SNAPSHOT" \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/checkpoint_inventory.json
```

`tests/probe_ple_storage.py` uses the first 4,096 real rows of PLE shard 0 and
32 selected indices. It tests BF16/BFP8/BFP4 storage and embedding/gather.
Runtime command:

```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
timeout 180 python models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_storage.py \
  --snapshot "$SNAPSHOT" \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_storage_probe.json
```

Results and failures: `ple_storage_probe.json`, `logs/ple_storage_probe.log`.
BF16 embedding and gather PCC 1.0. BFP8 embedding PCC 0.9999709, but its
wrapper expands the whole table to BF16; capacity remains unresolved.
BFP4 embedding fails layout validation. Both compressed gather results have
non-finite PCC. BFP4 storage roundtrip PCC is 0.9933436. These are component
diagnostics, not decoder acceptance scores or proof quantization cannot pass.

AutoDebug launch required environmental repairs: the default child sandbox
cannot create user namespaces; retried with explicit sandbox skip under the
current unrestricted execution authorization. The user-local Codex lacks its
code-mode host; disabling the host failed closed, and a newer host paired with
the old CLI had incompatible IPC. Removed that temporary host symlink. The
final retry puts the installed matching npm CLI/host directory first on PATH.
Only agent-owned, inspection-only failed child processes were terminated;
no device work was interrupted. Logs: `logs/autodebug_*.log`.

## Required evidence remaining

Implementation, real-weight full-decoder PCC, all sequence/batch/continuation
and page tests, traced repeated replay, allocation tracking, deterministic
replay, fallback audit, warmed performance, watcher, and stage review remain
unmet. No stage pass or stage completion commit is claimed.

## AutoFix implementation and initial validation

All device commands below used chip 0 and the p150 mesh descriptor from startup,
with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
for traced correctness probes. `SNAPSHOT` denotes the pinned absolute snapshot
under `/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/`.
Each changed Python file was checked by `pre-commit run --files ...` before its
hardware run. Hardware commands remain serialized; CPU quantization overlaps
only host work.

- `timeout 1200 python tests/probe_decoder.py --snapshot "$SNAPSHOT"
  --output doc/functional_decoder/linear_decoder_probe.json` (paths relative
  to the autoport directory). Real layer 0, S=33, prefill PCC 0.9999147654;
  three changed-input traced decode steps 0.9999481440, 0.9999409914,
  0.9999201894. Log: `logs/linear_decoder_probe_serializable.log`.
  Earlier failures were Python Shape slicing and MeshTraceId JSON conversion;
  their distinct logs remain. They did not establish numerical failures.
- `timeout 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_ngram_hash
  --snapshot "$SNAPSHOT" --device --batch 2 --max-sequence 257
  --output .../ngram_hash_device.json`: exact IDs/history pass over seven
  scenarios and three reused physical-bucket traces. CPU arithmetic exhaustively
  checks 744960 products and compares 65600 HF IDs. Log:
  `logs/ngram_hash_device.log`.
- `timeout 600 python tests/probe_packed_embedding.py --output
  doc/functional_decoder/packed_embedding_probe.json`: exact NF4 reconstruction
  at S=1,7,31,32, changed inputs and repeated replay. Full-domain address checks
  pass, including addresses where naive UINT32 row*20 would overflow. Initial
  logical reshape incorrectly used padded volume; fixed with logical shape
  product. Logs: `logs/packed_embedding_probe.log` (failed),
  `logs/packed_embedding_probe_reshape.log` (passed). The analogous run with
  `--snapshot "$SNAPSHOT"` passes real shard-0 rows: `packed_embedding_real_small.json`.
- `timeout 14400 python tests/probe_packed_embedding.py --prepare-only
  --snapshot "$SNAPSHOT" --cache-dir /home/ttuser/qwen38-flash/packed_ple_nf4_absmax
  --output doc/functional_decoder/packed_ple_cache.json`: all 128 shards
  streamed successfully to a persistent external cache. Total: 25600122880
  packed bytes + 640004096 scale bytes. No full FLOAT32 table was constructed.
  Log: `logs/packed_ple_cache.log`. This establishes setup completion only.
- `timeout 600 python tests/probe_ple_device.py --snapshot "$SNAPSHOT"
  --output doc/functional_decoder/ple_device_boundaries.json`: real-weight PLE
  arithmetic with setup-prepared embeddings passes lengths 1,7,31,32,33,65,
  sequential traced decode, independent reset replay and fresh prefill equality.
  This explicitly excludes hash/table lookup and is not full decoder evidence.
- `timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa
  --snapshot "$SNAPSHOT" --device --prefix 0 --steps 9
  --output .../qsa_device_short.json`: paged QSA component passes PCC
  0.9999837–0.9999964 with exact masks and permuted pages. Initial -inf masks
  produced zero SDPA output (non-finite PCC), while all Q/K/V inputs were finite.
  Replacing the sentinel with -1e9 restores the masked online-softmax reduction.
  Logs: `logs/qsa_device_short.log`, `logs/qsa_diagnostic.log`,
  `logs/qsa_device_short_finite_mask.log`.
- QSA prefix2048 pruning probe initially hits an explicit scatter guard:
  UINT32 TILE scatter-index rows above256 are unsupported. The repair splits
  the exact512 selected indices into two256 scatters; validation is pending.
  Original log: `logs/qsa_device_pruning.log`.

`tt/functional_decoder.py` now composes PLE/device hashing/packed lookup and
QSA as well as GDN. Tuple inputs/states expose independent request batching.
These integrations require further hardware validation before any stage pass.
Full context, batch32, contribution-sensitive continuation tests, runtime audit,
watcher, profiling, independent review and local stage commit remain required.

## Full PLE and direct lookup repair

`timeout 1800 python tests/probe_ple_decoder.py --snapshot "$SNAPSHOT"
--output doc/functional_decoder/ple_decoder_probe.json` completed normally at
2026-11-12 17:02:40 UTC, with exit1 for numerical acceptance. All real layer1
weights and the entire packed embedding table fit simultaneously. Prefill PCC
0.9960616827 (contribution0.9957634211); decode PCCs 0.9947828054,
0.9965517521,0.9955986738. First decode fails the unchanged0.995 gate.
Six replay outputs, all four reset state buffers, and repeated prefill agree
bit-for-bit. The approximately1791-second command duration includes setup,
compilation, CPU reference and repeated passes; it is not reported as kernel
performance. No timeout, reset or hardware recovery occurred.

Live source/tt-triage investigation shows native gather scans all6250030 source
tiles per output tile. The second capture proves progress into traced decode.
`AUTOTRIAGE.md` records this finite-work diagnosis and exact captures. AutoFix
is testing a generic TTNN direct indexed NoC reader, preserving the full table
domain. Separately, the failing decode triggers complete-layer CPU comparison
of absmax and least-squares NF4 scales; the numerical gate is not relaxed.

Full-context CPU QSA fixture preparation is running with all262144 token
outputs. It has a source/config/snapshot-keyed persistent cache so subsequent
device repair does not require regenerating the reference. `weight_stats.json`
now records all209 used tensors (including128 embedding shards), actual shapes,
dtypes, means and population standard deviations; the saved target config is
unchanged. Normal synthetic pytest paths use these statistics without download.

## Direct device kernels and QSA integration

The following commands use the same serialized chip0 environment and trace
allocation tracking. Paths below are relative to the autoport directory.

- `timeout 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_striped_gather --output
  doc/functional_decoder/striped_gather_device.json`: all15 combinations of
  UINT32/BF16/FLOAT32 and five logical shapes pass exact eager, rebound-buffer,
  trace, changed-index and repeat checks. Log: `logs/striped_gather_device.log`.
- `timeout 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_chunk_output --output
  doc/functional_decoder/chunk_output_device.json`: partial-tile output writes
  and untouched-row canaries pass exactly for two changed-value replays.
  Integrated into full decoder prefill to avoid retaining all chunk outputs
  plus a separate concatenation allocation.
- `TT_METAL_WATCHER=10 timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_packed_embedding
  --full --snapshot "$SNAPSHOT" --cache-dir /home/ttuser/qwen38-flash/packed_ple_nf4_absmax
  --output doc/functional_decoder/packed_embedding_full_direct.json`: complete
  resident table, all128 shards, S1/7/31/32, exact reconstruction and replay.
  Watcher log under `watcher/packed_full/generated/watcher/` has no fatal,
  invalid-address, overflow or sanitize messages. Log:
  `logs/packed_embedding_full_direct.log`.
- `TT_METAL_WATCHER=10 timeout 600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_selected_cache
  --page-sizes 32,64,256 --queries 1,3,32 --pages 4 --output
  doc/functional_decoder/qsa_selected_cache_device.json`: all9 cases pass exact
  K/V, sentinel rows, permuted page ownership, refreshed indices/page tables,
  and repeated replay. Watcher directory `watcher/selected_kv/` is clean.
- `timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill --snapshot "$SNAPSHOT"
  --seq-len 33 --continuation-cut 17 --output
  doc/functional_decoder/qsa_chunked_device_33.json`: minimum attention-output
  row PCC0.99998075, three decode replays above0.999984. This run's randomly
  generated four-page table happened to be identity; subsequent probes force
  a nonzero first page and do not claim this run alone proves permutations.
- `timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder --snapshot "$SNAPSHOT"
  --seq-lens 1 31 32 33 --output doc/functional_decoder/qsa_decoder_device.json`:
  full real layer3 passes all boundary lengths, contribution gates and traced
  decode. Full prefill PCC at31/32/33 is0.99993032/0.99993056/0.99984664.
- `timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill --snapshot "$SNAPSHOT"
  --seq-len 2053 --continuation-cut 2049 --output
  doc/functional_decoder/qsa_chunked_device_2053.json`: minimum prefill row
  PCC0.99994290 across2053 rows; resumed rows minimum0.99995345. Deliberately
  dropped history fails the same gate at0.91823208. Repeated decode is exact,
  and the same live decode trace remains correct after interleaved prefill.
- `timeout 180 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_trace_tracker --output
  doc/functional_decoder/trace_alloc_tracking/positive_control.json`: deliberately
  live post-capture allocation is rejected with the exact unsafe-buffer message;
  deallocation permits clean replay. Log is beside the JSON.

## Remaining precision failures

`gdn_device_1025.json` fails long GDN attention despite deterministic replay:
minimum row PCC0.94654554, final decode0.97519588. The unchanged HF reference
uses1024-token chunks; full advertised-context CPU fixture preparation completed
and is cached externally. `gdn_cpu_numerics_1025.json` rules out HF chunk-vs-
recurrent disagreement (minimum0.99999583) and ordinary BF16 convolution
rounding (minimum0.99994707). Native recurrence core isolation is the next
discriminating device test. No failed gate is accepted or threshold relaxed.

PLE CPU diagnostics reproduce the native decoder's first failing decode.
Least-squares NF4 scales alone fail. Grouped independently trained codebooks
improve one seed but group40 fails separate continuation holdouts. Group20
passes those holdouts with exact BF16 MoE; fitting its metadata requires a
separate expert-storage candidate, which still needs device accuracy evidence.
All failed and successful diagnostics remain explicit, separate artifacts.

## Recurrence isolation, grouped reconstruction, and batch 32

Commands below run from the checkout with `python_env/bin/python`,
`TT_VISIBLE_DEVICES=0`,
`TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`,
`TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`,
and `PYTHONPATH=.`. `$SNAPSHOT` is the pinned snapshot listed above. Each log
is `logs/<output-stem>.log` unless stated otherwise.

- `timeout 900 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_grouped_embedding --snapshot "$SNAPSHOT" --codebook-report models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_decoder_smaller_group_continuation.json --groups 20 --lengths 1,7,31,32 --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/grouped_embedding_device.json`
  passes exact CPU reconstruction, eager execution, changed-input trace replay
  and repeated-input determinism at all four lengths. It validates group-20
  addressing and reconstruction, not full decoder numerical acceptance.
- `timeout 600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_gdn_core --fixture /home/ttuser/qwen38-flash/gdn_core_1025.pt --chunks 1025,32 --backend model --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/gdn_core_model_1025.json`
  compiled and closed cleanly but failed numerically. PCC aggregates were NaN;
  a subsequent diagnostic revision distinguishes nonfinite values from zero
  variance. The first candidate is not integrated into the decoder. Source
  inspection identified missing unpack-format reconfiguration after selecting
  exact FP32 unpack; the revised candidate is being checked separately in
  `gdn_core_model_reconfig_1025.json`.
- `timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch --snapshot "$SNAPSHOT" --layer 3 --batch 32 --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/qsa_batch32.json`
  passes all 32 real-weight prefill and traced decode cases, including whole
  output and contribution gates, independent lengths 1/2/3, page mappings,
  current positions, runtime audit and exact repeated replay.

`linear_contract.json` remains a failed gate: continuation windows and their
negative controls pass, but extra per-row prefill contribution checks expose
router-sensitive rows. `router_rounding_probe.json` shows BF16 router-logit
rounding alone does not fix those rows; equal boundary logits also have
different HF/TT top-k tie ordering. No router rounding change was retained.

## FP32 recurrence repair and advertised GDN context

The final alias candidate keeps FPU source CBs in Default mode and uses
shared-allocation precise-copy aliases for carried FP32 state. Independent
read-only source review checked alias capacity, FIFO ownership and transitions.
The old native recurrence remains available only through the diagnostic runner;
`GatedDeltaNet` now calls the model-local `fp32_recurrent_gated_delta_rule`.

Core isolation passes at one and 32 tokens. At 1025 tokens the state minimum
row PCC is 0.99998879. A tiny-magnitude head has an ill-conditioned PCC that
is retained in the diagnostic JSON; `gdn_core_cpu_transient_tf32.json` labels
its separate cancellation simulation as hypothetical. Stage acceptance uses
the actual attention outputs, not a claim that every head-level metric passes.

With the standard device/tracking environment above:

```bash
timeout 900 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_gdn_context --snapshot "$SNAPSHOT" --length 1025 --fixture-cache /home/ttuser/qwen38-flash/gdn_cpu_prefill_1025.pt --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/gdn_model_device_1025.json
```

All 1025 attention rows pass (minimum 0.99992985), final traced decode passes
at 0.99997807, and repeated output is exact. The old native implementation's
minimum was 0.94654554 on the same saved HF fixture.

The advertised-context run additionally used `TT_METAL_WATCHER=10` and
`TT_METAL_LOGS_PATH=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/watcher/gdn_full_context`:

```bash
timeout 3600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_gdn_context --snapshot "$SNAPSHOT" --length 262144 --fixture-cache /home/ttuser/qwen38-flash/gdn_cpu_prefill_262144.pt --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/gdn_model_device_262144.json
```

All 262144 attention rows pass, minimum PCC 0.9998751879; the nonaligned
262143-token prefix has the same minimum. Traced final decode at absolute
position 262143 passes at 0.9999824762, repeated output is exact, and runtime
audit passes. Watcher has 3789 lines and no fatal, invalid, overflow, sanitize
or assertion entries. This is the explicitly reduced real-weight attention
harness, excluding token-local hyper-connections and MoE; it is not a full
decoder 262144-token performance claim. Logs use matching stems under `logs/`.

## Complete layer-0 rounding regression passes

The native HF matched control (`probe_hf_chunk_router.py`, `hf_chunk_router.json`)
refutes chunking as the cause of the failed row: full 65 vs chunk 32 has minimum
contribution PCC 0.99999887, token-at-a-time 0.99998981, and expert sets never
change. The reference remains exact BF16 checkpoint weights.

`probe_decoder_intermediates.py` records exact-HF component substitutions in
`decoder_intermediates_norm.json`, `decoder_intermediates_conv.json` and
`decoder_intermediates_residual.json`. GDN now matches HF's separate BF16 norm
rounding and FP32 gate, and accumulates the depthwise convolution before BF16
rounding. Residual writes preserve the separate BF16 multiply/add boundaries
with explicit FP32 computation. In the exact-input control this changes element
equality from about 89% to 100%, with zero maximum error. No router policy or
threshold changes were made.

Final complete layer-0 command, with the standard chip/tracking environment,
`TT_METAL_WATCHER=10` and
`TT_METAL_LOGS_PATH=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/watcher/linear_contract_rounding`:

```bash
timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract --snapshot "$SNAPSHOT" --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/linear_contract_rounding.json
```

Passes 1/31/32/33/65 then short-reset sweep: minimum whole-row PCC 0.99961138,
minimum contribution 0.99958813. Continuations at 31/32/33 pass above 0.99995
and dropped-history controls fail the same gate. Three changed-input traced
decode contributions are 0.99999046 / 0.99998128 / 0.99988490. Independent
requests, repeated reset, the same live decode trace after interleaved prefill,
and runtime audit pass. Watcher has 1147 lines and no suspicious entries.
The log is `logs/linear_contract_rounding.log`.

The shared residual-write correction invalidates the earlier QSA full-decoder
measurement as a final result; it is retained as a labeled baseline. Attention-
only QSA evidence is unaffected. Full GDN context must also be refreshed after
its norm/convolution change. Both follow-ups remain explicit work.

### Evidence packaging (2026-11-12)

The two detailed intermediate diagnostics exceed the repository 500 KB file gate. Their original JSON files remain at their documented local paths; deterministic gzip copies and `compressed_evidence.json` preserve their exact bytes and SHA256 for the local commit. Raw Tracy capture directories remain local and are excluded by the model-local `.gitignore`; filtered performance tables, compact provenance and compressed CSV artifacts are retained for review. No evidence was deleted.

### Full group20 table and independent PLE context reference

Full group20 cache preparation completed: 320001536 rows, words 25600122880 bytes, BF16 scales 5120024576 bytes, total 30720147456 bytes. `packed_embedding_group20_cache_verification.json` verifies 1152 real boundary rows across all128 shards against independently packed checkpoint values. This establishes packing parity, not full decoder numerical acceptance.

The exact-BF16 layer1 reduced PLE/GDN CPU fixture computes all262144 rows from empty state, including a literal last-token decode; `ple_cpu_prefill_262144.json` records source/config hashes and the 202.85-second reference run. The 25.55GB fixture is external at `/home/ttuser/qwen38-flash/ple_cpu_prefill_262144.pt`. The reduced device harness preserves state across explicit outer upload/check boundaries, audits every device-only pass, and separately gates PLE and attention contributions. It does not establish a resident single-call full-decoder prefill limit; `probe_decoder_capacity.py` and a complete decoder followup remain required.

### Advertised QSA prefill failure

`probe_qsa_prefill --seq-len262144 --decode-steps0 --fixture-cache /home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt` completed all262144 device rows from empty state in1291.6985 host seconds. `qsa_device_262144.json` reports position262144 and finite output, but minimum per-row PCC0.183295384 atrow222275, so it FAILED. AggregatePCC1.0 is not accepted as evidence of correctness; the per-row gate caught the issue. The HF row has ordinary magnitude(std0.02824,maxabs0.12598), ruling out a near-zero reference row. A targeted prefix-state diagnostic is being prepared. No context reduction or changed threshold is accepted.

### Final GDN context regression passed

With the final norm/convolution rounding fixes, `probe_gdn_context --length262144 --fixture-cache /home/ttuser/qwen38-flash/gdn_cpu_prefill_262144.pt --output .../gdn_final_device_262144.json` passed all262144 outputs and the nonaligned262143prefix at minimum rowPCC0.9999221563; final valid traced decodePCC0.9999903440, repeatedoutput exact, runtimeaudit clean. Command used the standardchip0 environment, timeout3600, TT_METAL_WATCHER=10 and unique `watcher/gdn_final_context` logpath. Watcher log has no error/invalid/overflow/assert/sanitize/corruption matches. This reduced state-path harness excludes token-local MoE, which is covered by `linear_contract_rounding.json`.

### Group10 packed scales and batch regressions

`probe_grouped_embedding --spec doc/functional_decoder/ple_group10_scale8_wide_hypothesis.json --small-cache-dir /home/ttuser/qwen38-flash/ple_group10_small_device_fixture --lengths1,7,31,32` passes exact eager, traced, changed-ID and repeated lookup, including maximum global scale-word address. `group10_scale8_device.json` and watcher/group10_scale8_small record the result; watcher is clean. This uses a small real-row table and does not claim full-table residency. The complete new cache is being prepared separately; group20 remains preserved.

`probe_batch --layer0 --batch32` fails request3 decode(contribution0.98349082) and request23 prefill thirdrow(contribution0.99241823); deterministic replay and runtime audit pass. `linear_final_batch32.json` is a failed diagnostic, not acceptance. Exact-request intermediate and router-rounding controls are being prepared.

`pytest -q .../test_synthetic_decoder.py -k test_ple_arithmetic_only` passed both actual-shape synthetic tests (1 and33tokens) in4.71s; `synthetic_ple_arithmetic.xml` and logs/synthetic_ple_arithmetic.log retain evidence. This does not replace the full resident-table real PLE decoder gate.

`probe_qsa_output_assembly --fixture /home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt` exactly copied all262144 rows through the actual slice/paged_fill_cache/reshape assembly sequence, zero mismatched rows (`qsa_output_assembly_262144.json`). Exact-HF-prefix injection at222272 makes the32-rowQSA window pass(minPCC0.99964678), while the CPU hypothetical one-block selection swap reproduces row222275PCC0.18362785. Indexer prefix precision is the active hypothesis, not a proved runtime fix.

### Router rounding and QSA indexer causal isolation

The full TT-reconstructed compressed-indexer prefix at222272 reproduces the bad row222275 (PCC0.18313965) while retaining exact HF K/V prefix; all other31 rows pass. `qsa_context_tt_index_prefix.json` and its external tensor capture isolate the failure to indexer selection. Command: `probe_qsa_context_window --fixture /home/ttuser/qwen38-flash/qsa_context_window_222272.pt --device --tt-prefix-indexer --capture /home/ttuser/qwen38-flash/qsa_context_tt_index_prefix_222272.pt`, standard chip0 environment, timeout1800.

`probe_qsa_projection_precision` compared native BF16, FP32 output, FP32 operands, and splitK512/256 on242 original32row chunks. All retain roughly85.72% exact raw-key BF16 equality; none fixes the underlying projection discrepancy. Report `qsa_projection_precision.json`, log `logs/qsa_projection_precision.log`, external capture `/home/ttuser/qwen38-flash/qsa_projection_precision.pt`. These remain diagnostic candidates, not accepted runtime fixes.

`batch32_intermediates_device.json` proves both batch failures persist with exact HF MLP input, but rounding FP32 router logits through BF16 restores the HF-selected expert sets. Applied only that dtype boundary before top-k/softmax; GR-read perturbations are not justified. `router_cpu_ties.json` independently characterizes the installed HF top-k ordering on95 real and8193 synthetic tied rows. Native TT top-k already matches the two failing selections after BF16 rounding; no custom tie kernel was added. The full32-request regression is running as `linear_router_batch32.json` with watcher and live-trace allocation checks.

### Exact-input GR arithmetic controls

`probe_gr_arithmetic --snapshot SNAPSHOT --output .../gr_arithmetic_exact_inputs.json` isolated each operation with exact HF inputs for both GR modules, prefill/decode, real requests3/16/23. Native BF16 SiLU, sigmoid and four-stream mean differ from HF (minimum exact fractions0.8656,0.625,0.9234); computing these operations in FP32 and casting at each HF BF16 boundary makes every tested value exactly equal. BF16 product and power-of-two scaling already match100%. Applied only FP32 nonlinear/mean intermediates to GR.read; product/scaling unchanged. This is independent source-semantic evidence, not a choice based on aggregate decoder PCC. Full regressions remain required.

Router BF16 rounding alone fixes original requests3/23 but fails request16(prefill contribution0.99059349); runtimeaudit/determinism pass. `router_request16.json` proves native TTtopk on exactHF logits selects a different tied expert. Synthetic layer0S1 decode1 similarly failsPCC0.99465424 and exact-logit topk picks198 instead ofHF53; three other full-decoder synthetic cases pass. `synthetic_intermediates_device.json` preserves causal controls. A device-only HF tie-policy repair is being implemented.

`qsa_projection_sfpu.json` isolates the accurate alternative: SFPU FP32 products + balanced addition tree have FP32rms1.08e-8 versus CPUFP64 and99.956% BF16 equality toHF, compared to native matrix projection85.46%. This is a one-chunk diagnostic; complete indexer and262144 gates remain open.

### Complete revised PLE cache

The group10/scale-byte full cache finished at `/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide`. `packed_embedding_group10_scale8_cache_verification.json` independently verifies1152 boundary rows across all128 shards and payload checksums. Word/scalestream total30720147456bytes; manifestSHA256642f0e2382b182aba6d1eee24fc2e1ab1ef53d84d30b7284a246cbb7d1680792. All1440 below-positive-scale groups are exactly90 checkpoint padding rows beyond the valid hash domain320001446; no valid-domain group is outside the LUT range. Device residency, full decoder/BFP8 experts and largest feasible prefill remain pending.

Final regression launches will use `tests/run_recorded.py`, a host-only subprocess wrapper retaining exact argv, selected TT environment, UTC timestamps, exitcode, logfile hash and all model-local runtime/test source hashes. It does not alter the requested command or environment. Original diagnostic evidence remains separately labeled by its runtime generation.

### Precise indexer window and remaining decoder regressions

The full TT indexer prefix rebuild to222272 with `PreciseIndexerProjection` passed all32 continuation rows: minimumPCC0.9996484259 (`qsa_precise_window.json`, external capture `/home/ttuser/qwen38-flash/qsa_precise_window_222272.pt`). Prefix rebuild took701.60s with watcher enabled; watcher has no error/assert/invalid/overflow/sanitize/corruption matches. This window still used native indexerRoPE; a subsequent separately proved precise128-wideRoPE correction requires final-combination regression. Main256-wideattentionRoPE remains unchanged.

`qsa_score_rope_precision.json` isolates native matrix-dot error from the four-head reduction: native reduction agrees with CPU on identical TT dots to about1e-6, while matrix-dot maximumerror is0.0276 on exact operands. No head-reduction change is justified. Exact-input precise indexerRoPE matches HF bitwise. These component controls do not replace full-context acceptance.

`linear_semantic_batch32.json` still fails request16prefill0(contribution0.99018145),23prefill2(0.99239552),29prefill2(0.99172610), after source-faithful GR arithmetic and device ATen tie selection. Runtimeaudit and repeatedoutputs pass. Accurate router-projection and upstream-input controls remain work. No failed row is excluded.

### Full group10 PLE contract and remaining precision work

`ple_group10_contract.json` passes with the full 30,720,147,456-byte packed table resident and BFP8 experts. Lengths 1/31/32/33/65 have minimum per-row contribution PCC 0.9963022470; continuation starts 1/31/32/33 pass, including negative controls. Three changed traced decode inputs have minimum contribution PCC 0.9975129366. Interleaved prefills, repeated outputs/state, 39 audited passes and trace-allocation tracking pass. Watcher is clean. Exact command and source hashes are in `provenance/ple_group10_contract.json`. Capacity, batch32 and long-context gates remain open.

`router_projection_intermediates.json` separates two causes: request23 with exact HF MLP input still fails with native matrix projection (0.992423058 contribution PCC) and passes with SFPU projection (0.999960244). Accurate projection alone leaves all three batch failures unchanged; literal CPU routing on actual TT mixed inputs also selects the changed experts. Thus upstream activation drift is independently relevant. A controlled boundary bisection with precise projection is next; failed rows remain in the acceptance suite.

`qsa_sfpu_scores.json` compares native and SFPU dots on exact HF operands for the critical 32-row window. Native scores retain the critical self block at rank511, though other top512 boundaries differ. This is a component diagnostic, not advertised-context acceptance; the final precise-projection/precise-RoPE combination remains to be tested over all rows.

`ple_capacity_group10.json` measures empty DRAM34178731008bytes and resident model free541166080bytes. The full262144 input alone fails its5368709120-byte allocation. Binary search with resident input/output/token words and two real chunk outputs finds6496 rows passing,6497 failing an83886080-byte chunk allocation. This optimistic candidate excludes trace reservation and is not an accepted reduction; complete public prefill/HF and neighboring nonaligned validation remain required. Exact command/source manifest: `provenance/ple_capacity_group10.json`.

`qsa_continuation_boundaries_precise.json` passes all cuts3/4/5/31/32/33 using the final precise indexer projection/RoPE combination, with minimum resumed-rowPCC0.9999787211. Each cut includes negative dropped-history controls, changed traced decode, deterministic replay and live prefill interleaving. Watcher/qsa_continuation_precise has570lines and no error/assert/invalid/overflow/sanitize/corruption matches. Exact command/source hashes: `provenance/qsa_continuation_boundaries_precise.json`.

### Precise GR/router integration

`gr_projection_intermediates.json` holds precise routing fixed. With exact HF post-attention input, precise MLP down/up projections restore mixed-input BF16 equality to100% for requests16/23 and99.987% for29; minimum contribution PCC becomes0.99998605/0.99998587/0.99998987. Applied the same bounded FP32 SFPU projection arithmetic in `tt/precise_residual_projection.py` and wired GR down/up plus `PreciseRouterProjection` into `decoder_components.py`. Original native matrices are not retained, avoiding duplicate device storage. Accurate injection projection changes none of the tested full-path results, so native injection remains.

Actual-path accurate attention+MLP GR still fails requests16/23, while29 passes. Request23 attention mixed input is now bit-exact, localizing its remaining divergence inside GDN. Test-local BF16 convolution SiLU and beta sigmoid controls are prepared; GDN runtime is unchanged until their device evidence. Existing full-decoder contracts and PLE capacity must be refreshed after this shared change. The running QSA262144 test is attention-only and unaffected by GR/router changes.

### Full-decoder final-context bridges prepared

`tests/probe_ple_full_context.py` keeps the full PLE decoder/table resident while advancing its actual PLE/hash/GDN prefix from raw inputs, then traces public `decode_forward` at262143. Every prefix contribution/hash is checked against the exact CPU fixture; the full final output/contribution, all four state buffers and repeats are gated. `ple_full_context_hf_bridge.json` proves the independent full HF final-token state path matches the existing fixture bit-for-bit. This harness does not claim full-layer prefix output or single-call full-length prefill.

`tests/probe_qsa_full_context.py` builds an independent HF prefix from real projections and executes two complete HF final forwards at262143. `qsa_full_decoder_final_cpu.json` records successful reference preparation. Device validation will consume only a provenance-checked export of TT-produced prefix state, never HF cache contents, and run two changed full decoder inputs through one trace with reset/repeat gates. `probe_qsa_prefill --prefix-state-cache` can export its checked262143 prefix before the final decode; every state tensor, page permutation, source key and runtime source hash are retained. This bridges reduced attention context evidence to complete public decode, not to full-decoder long-prefill.

### Runtime source audit supplement

Reviewed model-local torch/from_torch/to_torch call sites: they belong to checkpoint packing, weight loading, RoPE/state construction, and explicit test boundaries. The measured runtime remains covered by `runtime_audit` (TorchFunctionMode, forbidden TTNN conversions, throw_exception_on_fallback) plus traced execution. The router reads its C++ source once at module import; selection itself uses device NoC/dataflow kernels. Source inspection of `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp` confirms `nnz=None` selects `get_batch_from_reader`, with sparsity consumed by device readers; no host tensor readback is present in that factory/device-op path. Shape/address queries are host metadata, not model-value fallback.

`qsa_precise_device_262144.json` FAILED after completing all262144 rows in1376.4427 host seconds. FP64 globalPCC0.9999557527 does not override the row gate:78 rows below0.995, minimum0.956661582 at205531. Full actual output is preserved at `/home/ttuser/qwen38-flash/qsa_precise_device_262144.pt`; exact command/source hashes in `provenance/qsa_precise_device_262144.json`. Position262144, allfinite. Precise projection/RoPE removes the earlier catastrophic floor but is insufficient. Native score precision is being tested on the worst windows before another full run. No threshold/context reduction accepted.

### GDN nonlinear proof and PLE bridge harness fix

`gdn_nonlinear_intermediates.json` proves FP32-evaluated BF16 SiLU and beta sigmoid match literal HF on identical operands in every tested case (100% equality, maxerror0); native SiLU equality is about88.3% and sigmoid87.5–91.0%. Integrated only those HF rounding boundaries in `GatedDeltaNet`; precommit passes. Combined controls fix requests16/23 but move request29 across a router boundary. `gdn_projection_intermediates.json` substitutes literal HF CPU projections on actual device inputs solely as a diagnostic; even all five substitutions leave request29 failing. Recurrence/norm same-input controls are the next causal step. No host projection substitution exists in runtime.

`ple_full_context_smoke.json` aborts at a host check after device prefix computation: Torch cannot promote UINT32 hash IDs against INT64 expected IDs. Fixed the shared test comparison to cast actual IDs to INT64; IDs are nonnegative below2**32, so this preserves exactness. The original failed log/provenance are preserved. No decoder source or CPU fixture identity changes for this fix; rerun is required.

### Score arithmetic causality and current PLE bridge smoke

`qsa_score_windows_causal_summary.json` connects exact-input native score changes to the full TT failures at205531/165867/215431: CPU HF attention driven by native TT scores yields0.9566743/0.9697891/0.9744914, matching the full-device failures. SFPU scores restore all128 window rows' exact HF selected sets; the three worst rows exceed0.9999999. `qsa_sfpu_scores_block65536.json` confirms the larger1GiB product scratch fits but gives no meaningful latency improvement (0.5266s versus approximately0.530s for8192 keys). Retain bounded8192-key scratch for runtime integration. Full-context acceptance remains open.

`ple_full_context_smoke_v2.json` passes the65-token bridge with the complete current decoder and packed table resident: minimum prefix PLE/GDN/contribution PCC0.997939/0.997222/0.997206; final public traced decode contribution0.9978442893. Exact hashes, repeated output and all four repeated state buffers pass, five runtime-audited passes, watcher enabled. This short smoke does not replace the advertised-context run or largest public prefill. Exact command/source identities are in `provenance/ple_full_context_smoke_v2.json`.

`gdn_deep_intermediates_v2.json` completes the repaired causal harness; the first run's Torch reshape type error is preserved separately. Same-input HF gated norm makes no difference; replacing recurrence on actual operands fixes request29. Request16 then exposes a separate upstream effect, so no broad runtime change is accepted without further controls.

`qsa_score_runtime.json` passes actual runtime scoring at65536 compressed keys for1/3/7/31/32 query rows, including changed input and repeated traces. Maximum score error1.91e-6; top512 sets all exact; fallback audit clean. Watcher timings are not substituted for final profiler tables. `qsa_continuation_precise_scores.json` passes cuts3/4/5/31/32/33 with minimum resumed-rowPCC0.9999787211 and all dropped-history negative controls detected. `watcher_current_precision_summary.json` records clean watcher scans for score runtime, continuation, and PLE bridge smoke.

`ple_capacity_precise_gr.json` gives current resident free514951680bytes and optimistic candidate5792;5793 fails83886080-byte workspace allocation. Fresh-process upper-bound confirmation and complete public prefill are required; no reduction accepted. The capacity harness now supports explicit single-length probes, decode-input reservation, and nonaligned tails.

`gdn_sfpu_request29_first.json` fails JIT compilation before the candidate kernel executes: three low-level SFPU functions were incorrectly qualified with ckernel::. Corrected against current Blackhole LLK declarations in the global namespace; failed command/log retained. Existing runtime default remains the previously tested kernel until candidate validation.

### Exact GDN recurrence candidate passes causal cases

`gdn_sfpu_request29_first_v2.json` compiles and passes after the LLK namespace correction: first-token state is bit-exact and output maximumerror5.82e-11. `gdn_sfpu_request29_three.json` matches HF with maximum outputerror1.164e-10 and stateerror4.657e-10 for both3-token and1-token chunks. `gdn_sfpu_decoder_focused.json` then runs the actual full decoder with native GDN projections and the SFPU recurrence: requests16/23/29 minimum contributionPCC0.9999635816/0.9999588728/0.9999684691, matching the earlier CPU-recurrence causal controls. The model-local candidate uses exact FP32 loads, products and reductions for state reads and rank-one updates. No norm or bulk-projection substitution was necessary. Default integration is justified; full regressions, long contexts and perf remain open.

`qsa_precise_scores_batch32.json` passes complete real-weight layer3 prefill and traced decode for32 independent requests with lengths1/2/3 and permuted page tables. Position checks, deterministic replay and fallback audit pass. Source/command provenance is retained.

`linear_sfpu_batch32.json` runs all32 requests with the new recurrence default. All prefill rows pass (minimum contribution0.9961789846), but request3 decode fails: wholePCC0.9928228855, contribution0.9835732579. Deterministic replay and fallback audit pass. This is an unaccepted numerical regression; exact-prefix and activation-boundary controls are next. The earlier focused3-request pass did not cover this decode case.

`qsa_full_boundary_smoke.json` passes the new64-token full-prefill branch plus63-token prefix and final traced decode at full physical capacity. A device-only checkpoint of five mutable buffers before the final32 rows lets the long harness independently test aligned full prefill and the31-row tail from the same actual TT prefix. No HF state is injected.

`synthetic_sfpu_decoders.xml` passes all6 real-shape offline tests. Layer0 whole-prefill minimumPCC0.9995278120 and three decode outputs per length pass; layer3 and PLE arithmetic smokes pass. This synthetic PLE scope explicitly excludes full embedding/hash composition, which is validated separately with real weights.

`router_request3_decode_sfpu.json` isolates the remaining real-weight decode mismatch: replacing only the HF prefix conv/recurrent state restores contributionPCC0.9999722838; replacing current attention mixed input alone leaves0.9835728407. Exact post-attention and exact MLP boundaries pass. Prefix-state source accuracy is the next repair target, not current-token GR arithmetic or nondeterministic trace replay.

`request3_prefix_projection.json` rejects output-pack-only repair: FP32 matmul output followed by BF16 cast leaves QKV equality85.4102% prefill/85.2832% decode unchanged. Neither prefix conv nor recurrent state alone restores decode; both together do. `request3_projection_combinations.json` tests phase-specific and combined real HF projection controls. QKV+output passes both phases; QKV+Z+output still fails while all projections pass. Candidate selection therefore requires same-input projection arithmetic evidence, not only expert-route outcome. Device K-block/F32-sum and boundedSFPU controls are prepared.

`ple_sfpu_contract.json` passes all short full-table boundary/continuation/traced-decode gates with the exact recurrence default. `qsa_final_profile.json` executed finite outputs but Tracy rejected incomplete data because10000-op profiler storage filled across warmups and measured passes. Original raw/log artifacts are preserved; no timing is accepted from that run. Added profiler drains after warmups and outside each signposted window in `profile_decoder.py`; the v2 rerun has no dropped-marker warning and is processing its report.

`qsa_upstream_windows.json` reconstructs actual TT prefix state once through222304 tokens and runs four full runtime attention windows. All128 output rows pass (including the prior catastrophic self-key case); no HF state/operands are injected. Batched prefix-cache writes are a clearly labeled reduced diagnostic, not final full-context acceptance.

QSA profile v2 exposes a separate host instrumentation limit: Tracy reports "Too many source locations" (32K) and saves before PERF_DECODE markers. Device markers are complete, but the initial report generator silently accepted missing signposts and counted unrelated rows. Marked `tracy/qsa/final/{provenance,latency_summary}.json` rejected; no decode timing accepted. `profile_decoder.py` now requires separate prefill/decode runs; `report_performance.py` requires the matching mode and exactly one pair of signposts before rendering. This keeps each capture below the source-location limit. Original raw evidence preserved.

Source inspection identifies a supported profiler switch: `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1` disables device-zone duplication into the Tracy GUI while preserving device CSV collection (`rtoptions.cpp` and `DeviceProfiler::pushTracyDeviceResults`). Subsequent separated-mode captures will set it to avoid32K dynamic GUI source locations. `run_recorded.py` now retains this setting in provenance. Profiler buffers still drain outside measured windows.

### Trace packaging AutoDebug and corrected QSA profile

Fresh inspection-only AutoDebug report: trace_capture_debug/AUTODEBUG.md. Verified source stores assembled trace bytes in uint32_t while uploading the full size_t vector; the batch32 precise-GDN failure is consistent with exceeding4GiB, but no actual failing byte count was recorded. A model-local hybrid projection candidate retains256-output BF16 parent blocks, using256 for single-token and32 device slices for prefill, preserving the K addition tree and persistent payload. Candidate batch32 test is running; no shared tt-metal source change.

Correction: linear_precise_gdn_batch32.json does NOT pass all prefill rows. Request16 contribution aggregate0.9941862822 and token0 0.9906083941 fail0.995, found by AutoDebug; prefill completion is not acceptance. Separate causal numerical investigation remains open.

qsa_decode_profile completed with device-zone Tracy duplication disabled and separate decode capture. Strict signpost validation finds exactly8096 operations. tracy/qsa/decode/latency_summary.json records kernel192.929773ms + gaps5.797338ms =198.727111ms from Device Time/Op-to-Op Gap microseconds. Human table and filteredCSV retained; previous combined profile remains rejected.

### Precise GDN projection trace workaround

linear_gdn_hybrid_batch32.json completes capture and two repeated complete batch32 decode replays with runtime audit and watcher enabled. Every decode row passes and outputs are bit-exact between restored-state repeats. Prefill request16 retains exactly the earlier failing PCC; other rows pass. Thus the hybrid block policy fixes capture packaging without concealing the numerical gate. Integrated the tested256-output BF16 parent/256 single-token/32 multi-token policy in tt/precise_gdn_projection.py. Same weight payload and K reduction tree; no shared trace-code edit. This supports a graph-size-sensitive workaround, not a measured proof of the old4GiB byte count. request16_precise_gdn_bisection probes the remaining accuracy failure.

`request16_precise_gdn_bisection.json`: baseline contribution minimum0.9906088114; exact HF attention mixed input alone0.9806512594; CPU a/b unchanged, z/all/full-GDN controls0.990599–0.990604. Only exact full-GDN arithmetic with exact attention mixed input restores0.9999860525 and exact GDN output. These controls identify combined upstream GR and GDN arithmetic drift; no host control is accepted as runtime.

`qsa_prefill_profile` completes separate-mode capture with all markers. Strict report:8926ops,916.626334ms kernels +5.622145ms gaps =922.248479ms. Together with decode, final QSA warmed performance artifacts are under tracy/qsa/{prefill,decode}; no GUI-device-zone duplication. QSA pruning-boundary continuation2053 at cuts2047/2048/2049/2051/2052 is running.

### Matched original-HF rounding control

`gr_same_operand_rounding_cpu.json` identifies attention GR down[0,0,34] at request16: exact FP64 dot2.1640585417408147 is below BF16 midpoint2.1640625; TT FP32 tree and Torch FP32/FP64 round to2.15625, literal oneDNN AVX512 BF16 GEMM rounds to2.171875. `gr_backend_sensitivity_cpu.json` changes only this projection accumulation in an otherwise unchanged HF decoder. FP32,FP64,and balanced-tree controls reproduce contribution minimum0.99061042 versus TT0.99060839, while whole-output remains>=0.995 and the same expert365 replaces405. Requests3/23 remain exact and29 remains>0.999989; literal reruns are exact. No runtime injection or rounding bias added.

`probe_batch --gr-backend-control` explicitly classifies only fresh-prefill contribution anomalies when literal whole aggregate/every-row>=0.995, an independent unchanged-weights/cache HF-FP64 control reproduces the anomaly at each affected granularity, and TT passes all whole/contribution>=0.995 comparisons against that control. Decode and continuation contribution gates stay literal/strict. Default tests remain strict. Source-only independent AutoFix check by gdn_precision_review found no additional semantic/gate defect; this is not stage review or a strict literal-contribution pass. The earlier CPU report hash was stale after final formatting; regenerated `linear_batch32_gr_control_cpu_final.json` with full provenance. Final device control gate remains pending.

`qsa_pruning_continuation.json` passes all2053-token tests at cuts2047/2048/2049/2051/2052 with current precise scoring. Minimum resumed-rowPCC0.9999546409; history-drop negative controls fail as required. Runtime audit and final-position checks retained. Exact command/source provenance in provenance/qsa_pruning_continuation.json; elapsed708.88s.

`linear_final_batch32.json` passes the explicitly selected matched-control policy on final integrated hybrid projections: every literal whole-output prefill row>=0.9951083660, every literal decode whole>=0.9999881387 and contribution>=0.9999741912. Two restored-state complete batch32 trace replays are bit-exact; audit enabled and watcher run retained. Literal contribution strict status remains false for request16 fresh-prefill aggregate/row0 only; `gr_backend_carried` and independent control metrics retain this fact. No decode or continuation exception. Command/provenance records `--gr-backend-control`; elapsed121.46s.

Final integrated-source short gates: `synthetic_final_decoders.xml` passes all6 real-config synthetic tests in113.14s; `linear_final_contract.json` passes real-weight lengths1/31/32/33/65 plus continuation1/31/32/33, negative controls, repeated/changed/interleaved traced inputs and allocation/runtime audit, with watcher enabled (114.15s). Source/argv/log hashes are in matching provenance JSON.

`ple_final_contract.json` passes the final integrated real-weight full-resident table contract with BFP8 experts: lengths1/31/32/33/65, continuation and negative controls, changed traced inputs, exact restored-state repeats, runtime/allocation audit and watcher. Elapsed136.16s; full provenance retained. Largest public prefill and full advertised-context bridge remain pending.

Final long-harness smokes pass: gdn_final_bridge_smoke.json (65 tokens, full model resident, changed complete final-token decodes,24.34s) and ple_final_bridge_smoke.json (65 tokens, full table/model resident, all raw hash/PLE/GDN prefix checks and public final decode,33.83s). Both include watcher and runtime/allocation audits. These are harness smokes, not advertised-context acceptance.

`ple_final_capacity_search.json` reports3520/3521 with final precise projections and reserved decode inputs. This is NOT accepted as the physical bound: residentfree514,951,680B falls across failed search probes (485,293,056B before3521;480,694,272B afterward), so accumulated program/failed-allocation resources bias the search. Fresh-process checks are required. Failure is an83,886,080B workspace allocation; exact allocator per-bank bounds retained.

### PLE prepared-lifecycle capacity investigation

Fresh-process allocation probes before the MoE lifetime repair passed5664 and failed5665, but the complete public5664 run failed its first prefill. Released three dead private MoE references (`gate`, `up`, `activated`) before subsequent large allocations; arithmetic, layout and precision are unchanged. `ple_public_5664_moe_lifetimes.json` now passes every5664 prefill row (whole minimum0.9954739551, contribution minimum0.9952924836), all three traced decodes, and six restored-state replays. Repeated prefill still fails allocating83,886,080B: per-bank free22,237,888B but largest contiguous block9,618,560B versus required10,485,760B. This is not accepted as a capability bound.

`ple_capacity_lifecycle_5664.json` passes its chunk-capacity/decode-capture/replay/repeated-chunk sequence after preparing the small prefill and decode paths before allocating large input/output buffers. Trace release restores its allocation; persistent decode setup changes free memory from506,485,248B to501,575,168B. This distinguishes allocation order from a trace leak. The capacity harness is explicitly a proxy, not full-prefill accuracy evidence. Added optional `probe_ple_decoder --prepare-programs` to test the same preparation order in the complete real-weight public pass; existing cold-allocation failures remain preserved.

`ple_public_5664_prepared.json` passes the complete public5664 prefill, all three traced decodes, bit-exact replay/state repeats, and bit-exact repeated prefill after trace release. All runtime audits pass;200.44s. Setup warms only actual prefill32/decode1 shapes before full input allocation and restores all four request states exactly. This confirms allocation order as the remaining failure mechanism. The largest prepared-lifecycle boundary and near-limit nonaligned correctness still require evidence. `ple_batch32_moe_lifetimes` fails prefill allocating83,886,080B with per-bank free17,638,720B/largest10,354,688B; smaller traced batches are under test.

`ple_batch3_moe_lifetimes` completes traced replay deterministically but fails fresh-prefill request2,row0 contribution0.9949769974 (literal whole0.9971301556). Batch4 also encounters a trace-address collision:69,803,008B overlap, not yet an unavoidable physical bound. Source confirms binary multiply permits aliasing its nonbroadcast left operand. Replaced the private80MiB weighted-down copy with `output_tensor=down` and released down after reduction. `ple_batch4_inplace_moe` now captures/replays deterministically; all prefill metrics are identical to the previous run and all decode contributions exceed0.99753445. The isolated fresh contribution failure remains open.

`public_continuation_layer3` initially failed allocation tracking because the test first compiled device-to-device input refresh after capture. Added exact BF16/UINT32 refresh-signature warmup before capture, then restored original inputs; no program-cache exemption. `public_continuation_layer3_v2` passes strict full-decoder continuation (minimum resumed contribution0.9999672174), strong dropped-history negatives, live independent prefill, two changed inputs per request, repeats/state isolation, and page-table/cache remapping through the same trace with exact logical-state equality. Watcher and runtime/allocation audits enabled. `public_continuation_layer0_v2` also passes; layer1 validation pending.

Independent unchanged-HF/actual-packed-table control (`ple_batch4_quantization_cpu.json`) reproduces request2,row0 contribution0.9949601293 with exact BF16 experts; whole0.9971231222. Quantization changes router expert434 to20. Optional `--ple-quantization-control` preserves default strict failure, literal whole>=0.995, literal decode/resume contribution>=0.995, independent literal rerun equality and exact matched-failure granularity. All TT-versus-packed-control comparisons must independently pass>=0.995. This is explicitly carried fresh-prefill contribution evidence requiring stage review, not a literal contribution pass or runtime reference injection.

Current in-place MoE capacity probes pass aligned7648/fail7680 and7649, but7647 with a31-token tail preparation fails. Diagnostic `ple_capacity_tail7647_diagnostic.json` proves the failure occurs in the first32-row chunk, in GR-down balanced reduction, because the40MiB original product coexists with two20MiB slice copies and its next20MiB result. Matched the existing precise-GDN lifetime pattern: release consumed products after slicing and before addition. No arithmetic change. `ple_capacity_gr_lifetime_7647.json` moves the limit to the actual80MiB MoE down output (free19,921,088B/bank, largest7,864,320B). Thus aligned-only capacity is not accepted as the arbitrary-length range. Final boundary, complete public validation, and refreshed trace checks remain work.

### 2026-11-12 21:38 UTC — continuous logical-range capacity policy

Candidate-specific preparation exposed a distinction between aligned and non-aligned capacity: aligned 7648 passed, but 7647 failed after its extra tail signature consumed persistent resources. Those observations are profile-specific diagnostics, not a continuous supported range. The final capacity policy prepares all physical prefill lengths 1–32 (32 first), plus decode1, before large input/output allocation. The capacity probe then exercises all 32 physical lengths with the full candidate I/O resident, captures/replays decode, releases the trace, and repeats all lengths. Fresh boundary probes and complete real-weight public prefill/HF comparison remain required. `--prepare-all-tails` implements this same preparation in the public correctness harness. Runtime arithmetic is unchanged.

The batch5 allocator snapshots identify a 358,875,136-byte trace reservation. Source inspection additionally proves original and clone 80 MiB MoE reduction buffers coexist for logical one-row decode. Together with the resident model these exceed device DRAM; detailed diagnosis is recorded separately. Batch4 on the final residual-projection lifetime source still needs its refreshed controlled correctness run.

### 2026-11-12 21:47 UTC — all-tail capacity bracket

All32 preparation retains 340,782,592 bytes free before large I/O, versus 514,951,680 after model/state construction. Candidate6144 fails a40MiB allocation;4096 fails80MiB;3584 and3328 fail later physical tail paths despite earlier full chunks fitting. Fresh-process3200 fails count25; fresh3136 and3168 pass every count1–32, decode capture/replay, and the repeated all-tail plan. Final3169/3167 fresh probes are running. These are capacity candidates, not yet a public accuracy/capability acceptance. All exact arguments, source hashes and failures are in provenance/ple_capacity_alltails*.json and matching logs.

### Chunk input preparation investigation (2026-11-12 22:26 UTC)

The terminal-residue capacity sweep exposed retained programs for new logical input shapes/offsets. No context limit is accepted from those proxy results. `chunk_input_native_growth.json` isolates native slices using only setup-time `from_torch` inputs: all324 exact-copy checks completed,338 retained programs and5,537,792 persistent DRAM bytes after deallocating input/output tensors. This is diagnostic evidence, not a functional pass. Exact command/source/log hashes are in `provenance/chunk_input_native_growth.json`.

Fresh AutoDebug launch diagnostics are retained: original sandbox preflight failed because user namespaces are disabled; explicit skill-authorized unsandboxed child then failed because `/home/ttuser/.local/bin/codex`0.147 lacked its code-mode helper. Per-child `features.code_mode_host=false` also failed routing tools. Only that owned, nonhardware Codex child PID1780068 was terminated with SIGTERM; no device process/reset was involved. Retry `chunk_input_autodebug_host_fixed` prepends `/home/ttuser/qwen38-flash/codex-venv/lib/python3.12/site-packages/codex_cli_bin/bin` to PATH, selecting the installed matching Codex0.163 and helper. No shared configuration/install files were edited. The fresh report remains required before runtime repair.

### Streaming profiler feasibility (2026-11-12 22:31 UTC)

`provenance/ple_streaming_prefill_smoke_before_chunk.json` completed exit0 in106.43s on the pre-chunk-repair runtime. This is **not accepted performance evidence**: boot explicitly reports streaming capture unavailable. The direct read-only health/triage snapshot reports Blackhole firmware bundle19.8.1.0; `tt_metal/llrt/firmware_capability.cpp` requires19.12.0.0 to avoid DRAM-core firmware collision. No force override, reset, or firmware change was attempted. The test completed normally; no hang is established. Triage's configuration/stack modules lacked the default inspector path (actual files are under generated/inspector), so the partial snapshot establishes firmware/health only. The profile's28,073 prefill dispatcher IDs and diagnostic host time are not device-duration evidence. Streaming final jobs must be replaced by a supported measured path; investigating whole-trace profiling with small classic buffers.

### First bounded-I/O device results (2026-11-12 22:38 UTC)

`chunk_output_final_v4.json` passes exact canaries after two changed-input trace replays. `public_continuation_layer0_v4.json` passes full real-HF gates including unseen65/97/129-token requests under one live decode trace (106.59s). Runtime candidate identities are preserved in `chunk_input_debug/runtime_after_sources.json`.

`chunk_input_final_v4.json` is retained as a failed harness run: it incorrectly read freshly allocated outputs immediately after record-only trace capture, before any execution. All three actual trace replays matched exact BF16/UINT32 values and zero padding; all four intervening eager plans and writer canaries were exact, with zero program-cache/retained-DRAM growth from the three fixed I/O programs. The first-capture test will explicitly execute its initial replay before inspection, then preserve the live eager/replay challenge and exhaustive sweep. No runtime change or threshold relaxation follows from this test-order correction.

### Bounded reader validation and preserved PLE failures (2026-11-12 22:49 UTC)

`chunk_input_final_v5.json` passes the corrected record-only capture test: its first explicit replay executes the unseen N97/start64/count7 plan after only fixed-count32 preparation. All 644 exhaustive cases, all32 writer counts and38 negative host contracts pass. The three prepared I/O programs retain49,152 bytes before capture and after trace release, with no persistent growth across the changed eager plans or exhaustive sweep. Four total replay calls include the initial execution and three subsequent replays; changed contents, stable captured addresses, zero padding, repeated inputs and writer canaries are exact. The reader's watcher contains8,906 lines with zero configured scan hits. This is a component pass, not stage acceptance. Exact command, source hashes and closed exit0 log are in `provenance/chunk_input_final_v5.json`.

Both `public_continuation_layer1_v4.json` and `public_continuation_layer1_v5_diagnostic.json` remain **strict failed runs** (exit1). The diagnostic rerun preserves the65-token fresh row2 contribution PCC0.9949381948 and129-token fresh row1 PCC0.9947543740. Whole-output rows remain above0.995; resumed/decode/state checks are separately recorded and do not turn these fresh-contribution failures into passes. The second run exports actual device outputs for matched diagnosis. `ple_public_quantization_cpu.json` and `ple_public_backend_cpu.json` are independent CPU controls, not accepted device evidence or threshold changes; quantization alone does not explain the65-token failure. Further matched backend investigation remains work. The two device watcher logs contain4,083 and4,109 lines respectively, with zero configured scan hits.

`evidence_archives.json` now binds13 additional lossless deterministic XZ archives (3,427,605 original bytes to130,432 archive bytes), including the two failed public reports and both CPU reports. Every archive round-trips to its recorded SHA256; raw originals remain local. Three exact-path ignores cover the reader's raw log, large JSON and copied watcher; their compressed evidence remains tracked. Active `ple_whole_trace_prefill_v1` outputs were not touched. No runtime, harness, acceptance manifest, README or context contract changed in this packaging step.

### Matched PLE backend control (2026-11-12 22:52 UTC)

`ple_public_backend_matched.json` completes one fixed control across lengths 65, 97 and 129: actual packed embedding rows plus FP64 accumulation at the same six GR down/up and GDN QKV/output projection boundaries implemented by runtime FP32 trees. This is a backend sensitivity control, not a bitwise emulation of those trees. Both saved TT repetitions meet whole-output and contribution PCC >= 0.995 against this control (minimum per-row whole 0.9981886148; contribution 0.9978749156). The control reproduces both failing literal fresh rows; the original strict literal failures remain failures. The record preserves input/device-output hashes, all policies and unchanged downstream HF operations. No runtime or threshold changed. Its result, closed exit0 log and provenance are archived losslessly (1,459,110 original bytes to 72,848 bytes); only the large raw JSON receives an exact-path ignore. The unresolved whole-trace profiler files remain untouched.

### 2026-11-12 bounded-I/O capacity environment correction

`ple_capacity_bounded_io_search_v1` was launched with Watcher10. Its all32 preparation is much slower than earlier non-watcher capacity probes and debug-instrumented program storage can change the apparent resident bound. Preserve this diagnostic, but do not infer the final normal-runtime capability from it. Re-run the bounded-I/O search and exact public boundary in the normal unprofiled environment; representative separate watcher-clean public/reader tests already exist and remain required. No target was terminated or device reset.

### Partial capacity deadline snapshot (2026-11-12 23:20 UTC)

`capacity_debug_deadline_snapshot/manifest.json` preserves nine read-only copies taken non-atomically at23:20:12 UTC while the Watcher-instrumented search was progressing. Captured provenance still says `running`; the partial JSON contains16 completed candidate probes, including allocation failures and successful proxy lengths2079,2335,2367. These are diagnostic observations, not an exact maximum, continuous supported range or normal-runtime capacity acceptance. The snapshot does not establish a hang or final timeout outcome. The parent planned to let the existing bounded timeout end naturally at23:23:26 UTC and perform a trivial mesh smoke before another device job; this packaging step did not signal, reset or access hardware.

All nine copies and their manifest are archived deterministically with SHA256 verification:27,304,450 original bytes to1,058,248 compressed bytes. Snapshot watcher scan:50,260 lines, zero configured hits, explicitly partial rather than a completed watcher-clean run. Three exact-path ignores cover only large raw snapshot copies; all originals and active source files remain untouched. The preceding Watcher-versus-normal-runtime capacity correction remains in force.

### 2026-11-12 Watcher capacity diagnostic timeout

`ple_capacity_bounded_io_search_v1` ended by its original timeout at1800.19s (exit124), while the final2368-token candidate was in progress. Completed proxy lifecycles passed2079,2335,2367;2369 failed allocation. These are debug-instrumented diagnostic results only, not accepted normal-runtime or complete-public capacity. No hang was established: candidate progress continued until the deadline. The pre-timeout non-atomic logs/inspector snapshot is retained and hash-verified; no manual signal or reset was issued. `/dev/tenstorrent/0` had no remaining owner afterward. A bounded trivial addition/open/close health check was launched before any further device job; its result is recorded separately.

The post-timeout trivial-op mesh smoke passed in2.03s with exact addition and clean close (`post_capacity_debug_mesh_smoke.json`); no reset was required. The isolated device-only trace-profiler canary then passed in4.34s (`trace_profiler_canary_v1.json`): numerical output exact, two native replay ordinals, all eight FW/kernel endpoints match independently read known-core L1 timestamps exactly. This proves the basic producer/drain/export path works in this environment; it does not yet explain the rejected full-layer capture. The uninstrumented bounded-I/O capacity search v2 is now running; final acceptance remains open.

Closed timeout/health/canary packaging adds11 hash-verified archives:5,053,114 original bytes to165,148 XZ bytes. The final closed Watcher log contains56,619 lines and zero configured scan hits; that observation does not convert the exit124 partial proxy into an accepted capacity run. Final2368 remains unverified. The native canary CSV is retained separately from rejected full-layer timing; all eight known-core endpoints are diagnostic evidence only. Originals are preserved, and only the large copied Watcher log receives an exact-path ignore. Active normal-capacity v2 outputs and profiler helper sources were not touched.

### Normal-environment strong capacity proxy (2026-11-12 23:28 UTC)

`ple_capacity_bounded_io_search_v2` completed exit0 in222.13s with all32 physical signatures prepared, reserved decode inputs and the decode/repeated-sweep lifecycle. Its strongest resident-I/O all-count proxy passed2528 and failed2529; this search repeats offset0 chunks and is not complete public execution or HF acceptance. The candidate boundary is conservative diagnostic evidence only: no public capacity, continuous supported range or advertised-capability reduction is accepted. Fresh exact public boundary, non-aligned lengths and real HF comparison remain required. Result/log/provenance are losslessly archived with hashes; the active public-proxy v3 files were left untouched.

### Public-tail proxy and native full-layer prefill timing (2026-11-12)

`ple_capacity_public_proxy_search_v3` completed exit0 in61.63s. With all32 signatures prepared, its actual-tail allocation proxy passes3712 and fails3713. This narrower proxy differs from v2's stronger all-count resident-I/O sweep; neither establishes complete public execution, a continuous accepted range or HF accuracy. Final public capacity remains pending.

`layer1_prefill_device_trace_v2` and its report both completed exit0. The device-only capture validates all110 participating cores across two replay ordinals, exact eager/replay output and all four request states. The human-readable `tracy/layer1_prefill_device_trace_v2/report/prefill_perf_report.txt` reports459,938 microseconds for one warmed whole32-row prefill trace (kernel-envelope elapsed459.9376904ms; firmware envelope459.9388384ms). Internal gaps are included; per-operation durations and a kernel-sum breakdown are unavailable. The independent native CSV, loaded-source/protocol metadata and report provenance are retained. This successful full-layer device-only path does not resolve the earlier host-Tracy source-location overflow and empty-device-CSV cause, and establishes no blanket firmware diagnosis. Fifteen closed evidence archives total1,709,109 original bytes and68,704 XZ bytes. Active decode capture files remain untouched.

### Terminal tail holes and closed decode profiling (2026-11-12)

`ple_capacity_terminal_public_v4` completed exit0 in120.27s, testing every terminal residue3681..3712 under all32 preparation. Its proxy passes29 lengths but fails3707/3709/3710 at physical tail27/29/30. All failures allocate the80MiB sparse down-projection output (`decoder_components.py:168`): each bank requires10,485,760 bytes contiguous;20,606,080 bytes are free per bank, but the largest blocks are8,779,392/8,428,672/8,361,088 bytes. Identical post-candidate free/largest-block snapshots refute cumulative retained growth in this sweep. These are current allocation-history fragmentation failures, not proof of total DRAM exhaustion. Aligned3712 success therefore cannot establish a continuous range through3712; no limit is accepted. Fresh complete-public3707 validation remains separate work. Source supports a caller-supplied sparse-matmul output as a possible bounded-workspace diagnostic, but no such repair or capacity gain is established here.

`layer1_decode_device_trace_v2` and report completed exit0. The native whole-trace decode report measures48.5546889ms (table48,555 microseconds), including internal gaps, with no per-op timing claim. Fifteen new immutable archives cover the terminal result/log/provenance and decode capture/report/native CSV:2,823,098 original bytes to77,444 XZ bytes. The large raw decode log has an exact-path ignore; originals and the live full-public3707 run remain untouched.

### Fresh complete-public tail failure (2026-11-12)

`ple_capacity_public_3707_v5` completed its wrapper with exit0 in79.70s, but its single capacity probe is explicitly **failed**. The real public loop reached tail index115/start3680/count27 before the same80MiB down-output contiguous allocation failed. This reproduces the terminal hole in a fresh process with actual prefix traversal, rather than only the three-chunk proxy. It remains fragmentation evidence for the current implementation, not an accepted capability reduction. Result/log/provenance are archived with exact hashes; the active real-HF3712 run is untouched. A separate test-only reserved-output experiment is being prepared; no runtime change or success is claimed here.

### Real-HF aligned3712 public validation (2026-11-12)

`ple_public_hf_3712_v5` passed in159.74s with all32 physical signatures prepared, resident packed lookup and BFP8 experts. All3712 prefill rows meet strict literal HF gates: minimum whole PCC0.9961261411 and contribution PCC0.9959619986. Three decode contributions have minima0.9975516685/0.9981430849/0.9977936547. One capture and six replays preserve exact repeated outputs and all four states; repeated full prefill and runtime fallback audits pass. Result/log/provenance are hash-verified and archived. This validates that aligned length and its continuation; the reproduced shorter tail holes still prevent accepting a continuous prefill range or declaring final capacity. Active layer0 profiling remains untouched.

### Six native whole-trace profile modes recorded (2026-11-12)

All six device-only capture/report pairs now close exit0 on the recorded bounded-I/O runtime. Each validates native endpoint coverage across110 participating cores and two replay ordinals, exact eager/replay output and request state, and the runtime audit. One warmed whole-trace sample per mode measures: layer0 prefill32 424.197505ms / decode1 47.196239ms; layer1 PLE prefill32 459.937690ms / decode1 48.554689ms; layer3 QSA prefill32 919.896021ms / decode1 194.279724ms. Layers0/3 use BF16 experts; layer1 uses BFP8 experts. Internal gaps are included; these are neither per-op kernel sums nor statistical latency distributions.

Exact artifacts are `tracy/layer{0,1,3}_{prefill,decode}_device_trace_v2/profile.json`, native `raw/.logs/profile_log_device.csv`, and `report/` human-readable tt-perf-report tables, CSVs, latency summaries and provenance. Capture/report commands and source hashes are in matching `provenance/*_device_trace_v2{,_report}.json`. The last four modes add48 verified archives (6,822,107 original bytes to237,244 XZ bytes); the sole new raw-file ignore is the large layer3-prefill log. These measurements do not close final source identity if a later capacity repair changes runtime. The original host-Tracy overflow/empty-CSV capture remains rejected and unresolved. README was not modified in this packaging step; capacity and final review remain open.

### Closed layer0 public continuation v6 (2026-11-12)

`public_continuation_layer0_v6` passed strict literal-HF gates in106.35s: one capture/twelve replays, clean runtime fallback audit, exact repeated outputs and request states, and isolation while independent prefills interleave with the live trace. Its watcher has3,574 lines and zero standard or broad fatal/assert/invalid/overflow/out-of-bounds/corrupt/sanitize scan hits. Result/log/provenance, copied watcher and original kernel/inspector maps are archived in11 entries (4,448,673 original bytes to201,352 XZ bytes). This refresh does not establish long-context or final stage acceptance. Runtime and the active sparse-output workspace control were untouched.

### Closed QSA public refresh and workspace controls (2026-11-12)

`public_continuation_layer3_v6` passed strict literal gates in91.16s with one capture/thirteen replays, clean runtime audit/allocation tracking and the live isolation/repeat/page-remapping checks. Its watcher contains2,579 lines with zero standard or broad scan hits. `sparse_down_workspace_control_v1` passed in30.40s:96 rank4 eager comparisons and15 trace replays, with856 watcher lines/zero hits. This isolated control does not establish integrated decoder capacity.

`ple_capacity_reserved_public_3707_v6` and `ple_capacity_reserved_shape_v7` failed test guards before capacity execution (19.87s/19.98s), exposing the actual rank6 activation shape. `sparse_down_workspace_control_v2` then failed in26.86s after96 rank4 eager comparisons: with both sparse-input flags true, the native down output remains rank4 despite rank6 input. The control's output-shape assumption was wrong; none of these failures proves a physical capacity bound or validates a workspace repair. Its watcher also has856 lines/zero scan hits. All failures remain preserved while the test-only v3 correction is prepared; runtime is unchanged. Eighteen archives total675,032 original bytes to64,628 XZ bytes, with no additional raw ignores.

### Persistent sparse-output reservation refuted (2026-11-13)

`sparse_down_workspace_control_v3` passed in55.86s:192 eager A/B/A comparisons and30 traced executions cover rank4/rank6 inputs while preserving the native rank4 output. Reserved addresses, exact values and skipped-expert zeroing pass; watcher1,711 lines has zero standard/broad hits. This proves isolated arithmetic/alias correctness, not memory benefit.

`ple_capacity_reserved_public_3707_v9` wrapper exited0 in35.17s, but its capacity probe failed on the **first32-row public chunk**. Holding the extra80MiB output pool caused an earlier20MiB balanced-sum slice in `precise_residual_projection.py:57` to fail: free2,074,752 bytes per bank is below required2,621,440 (total free16,598,016 versus20,971,520 requested). Thus persistent reservation is rejected as this memory fix; it moves failure earlier and is not integrated into runtime. This does not establish the ordinary implementation's maximum continuous range. Aligned3712 remains validated, tail holes remain relevant, and per-residue/public boundary work continues. Seven immutable archives total417,150 original bytes to28,860 XZ bytes. The active ordinary terminal v10 run is untouched.

### Earlier fixed-tail capacity hole reproduced (2026-11-13)

`ple_capacity_terminal_2560_v10` closed exit0 in122.88s;31 of32 terminal proxy lengths pass, while2555 fails count27. `ple_capacity_residue_search_v1` closed exit0 in100.98s with candidate pass/next-failure pairs27:2523/2555,29:3293/3325,30:3550/3582. Fixed-residue binary search assumes monotonicity and remains candidate-only.

Fresh `ple_capacity_public_2555_v11` wrapper closed exit0 in65.74s but its probe **failed** after actual public traversal at tail79/start2528/count27. The80MiB down buffer requires10,485,760 contiguous bytes per bank;18,518,144 bytes are free but the largest block is7,864,320. This confirms an earlier fragmentation hole than the aligned3712 success. No accepted continuous bound or capability reduction follows. README/context now distinguish tested aligned3712+3 decode from accepted range, which remains unset. Nine archives total1,342,847 original bytes to43,144 XZ bytes; the active query-row control is untouched.

### Exact GR query-row split control (2026-11-13)

`gr_query_rows_control_v1` passed in 143.89s: 384 exact eager A/B/A comparisons across all four real GR matrices and Q1..32, plus 120 trace executions over 20 plans with changed inputs and intervening eager calls. Runtime audit and allocation tracking pass. Observed padded product geometry for down projections is 40MiB at Q32 versus 20MiB at Q16. First K-half slice sizes of 20MiB versus 10MiB are source-derived, not measured allocation peaks. This proves the isolated schedule preserves tested arithmetic and reduces product geometry; it does not yet prove integrated capacity benefit or a runtime repair. The watcher has 5,093 lines with zero standard/broad scan hits. Four verified archives total 840,661 original bytes to 33,292 XZ bytes. The active split-only full-public3707 experiment remains untouched.

### Split-only full-public3707 lifecycle rescue (2026-11-13)

`ple_capacity_split_public_3707_v1` completed exit0 in126.81s and its capacity probe passed. Both complete public prefill passes visit116 chunks/all3707 logical rows with finite output, including the previously failing27-row tail; warmed/captured/replayed decode and the repeated-prefill lifecycle complete. This uses the test-only GR down query-row split, without the rejected persistent80MiB reservation. It demonstrates a capacity rescue for that constant-input lifecycle, not real-HF accuracy, a final continuous range or an integrated runtime repair. The ordinary failed3707 evidence remains unchanged. Result/log/provenance are archived with hashes; active `ple_capacity_residues_split16_v1` is untouched.

### Offline six-profile closure (2026-11-13)

`performance_evidence_offline_v1` closed exit0 in6.42s and passes all six current `device_trace_v2` modes. The CPU-only verifier independently reparses warm/measured native endpoints, regenerates exact tt-perf-report tables/CSVs and checks the complete runtime inventory plus selected harness dependencies against recorded capture/report invocations. Result/log/provenance are archived. This verifies current-source evidence, not stage acceptance; a later PLE memory repair would invalidate source closure until refreshed. Active residue-search files were untouched.

### Split-only all-residue candidate search (2026-11-13)

`ple_capacity_residues_split16_v1` closed exit0 in656.08s with320 probes across32 residues. Except residue26, each search reports largest passing candidate3680+r and next observed failure3712+r. Residue26 instead passes3450 and fails3482 at tail26: the80MiB down allocation needs10,485,760 contiguous bytes per bank, while22,173,568 are free but the largest block is10,482,176 (3,584 bytes short). All post-probe baselines agree at344,100,352 total free bytes and40,763,904 largest-block bytes per bank, so this run shows no cumulative retained growth.

The test-only schedule resolves the prior tested27/29/30-tail holes but does not establish a continuous3712 range. Binary-search monotonicity, full-public boundary and HF accuracy remain separate requirements; no bound or runtime repair is accepted. Result/log/provenance are archived and the two large raw files receive exact-path ignores. Active combined split-plus-reservation3707 files remain untouched.

### Combined persistent reservation rejected (2026-11-13)

`ple_capacity_split_reserved_public_3707_v1` wrapper closed exit0 in35.56s, but the probe failed on its first Q32 chunk before MLP. Combining the query split with a persistent80MiB down pool overlaps GDN's beta reshape temporary: the6MiB allocation needs786,432 bytes per bank, while1,315,968 are free but the largest block is565,248. This is fragmentation, not total-free exhaustion (10,527,744 total free versus6,291,456 requested). The combined persistent policy is rejected; no runtime integration or accepted range follows. Result/log/provenance are archived. A separate test-only per-call reservation at SparseMoe entry is being prepared to avoid overlap with GDN/GR; no result or memory benefit is claimed for it yet.

### Historical PLE public continuation v6 control evidence (2026-11-13)

`public_continuation_layer1_v6` closed exit0 in 134.87s with its explicit PLE control enabled. The report records `passed=true`, `literal_hf_strict_passed=false`, and `fresh_backend_all_matched_passed=true`; classification remains subject to stage review. Fresh 65-row row2 and 129-row row1 literal contribution PCC failures remain 0.9949381948 and 0.9947543740 in both repetitions. Against the fixed actual-packed-lookup plus six-projection FP64 CPU control, all fresh 65/97/129 whole/contribution rows pass; minima are 0.9981886148/0.9978749156. The control preserves its packed manifest and probe/helper/HF identities, nine actual-cache parity checks, 4,656 selected rows, and an exact independent literal rerun. This is not a strict literal pass.

The historical run records one capture, 12 replays, allocation tracking and a clean fallback audit. Its 4,135 watcher lines have zero standard or broad scan hits. Eleven result/log/provenance/watcher/map sources (6,068,077 bytes) are preserved as deterministic XZ (282,828 bytes). The recorded pre-replay discarded-read field is preserved without reinterpretation; the harness cleanup removes those reads without changing numerical gates, and pending v7 evidence will replace v6 after runtime freeze. No final acceptance is inferred.

### Test-only split plus per-call reservation at3707 (2026-11-13)

`ple_capacity_split_percall_public_3707_v1` closed exit0 in 125.83s with probe PASS. With query-row splitting at16 and down-output reservation scoped to each SparseMoe call, both complete public constant-input passes execute all 116 chunks/3,707 rows, including tail27; decode lifecycle completes. This is test-only capacity/lifecycle evidence, not HF accuracy, current-runtime integration, a continuous supported range, or final acceptance. Its result/log/provenance (1,150,406 bytes) are archived to18,260 bytes, preserving originals; the large raw result receives an exact-path ignore. The active per-call-alone experiment remains untouched.

### Per-call reservation alone rejected at3707 (2026-11-13)

`ple_capacity_percall_public_3707_v1` wrapper closed exit0 in79.68s but the probe fails at actual public tail27, index115/start3680. Without the query-row split, the per-call down-output reservation leaves the activation `multiply(silu(gate), up)` unable to allocate20MiB: each bank needs2,621,440 bytes, has4,877,440 free, but its largest block is2,119,168. This is fragmentation, not total-free exhaustion. The smaller policy is unsuccessful; among the per-call policies only split16 plus per-call reservation has a complete3707 capacity/lifecycle pass, and neither has accepted HF/continuous-range evidence or runtime integration. Result/log/provenance are archived with exact source hashes and originals preserved. The active combined-policy all-residue search remains untouched.

### Combined per-call search does not improve the candidate range (2026-11-13)

`ple_capacity_residues_split_percall_v1` closed exit0 in 683.51s with 320 probes over all32 residues. Its candidate boundaries exactly match split-only: residue26 passes3450/fails3482; every other residue passes3680+r/fails3712+r. At3482 the failure moves from the80MiB down output to the20MiB activation multiply: per bank6,444,928 bytes are free but the largest block2,617,856 is3,584 below the2,621,440-byte request. Reservation relocates the allocation failure without extending these observed boundaries. This remains a proxy under an unproved binary-search monotonicity assumption; no uniform3712, accepted bound, HF capacity or runtime integration follows.

The three closed sources (12,493,582 bytes) are archived to63,428 bytes; raw originals remain and the large result/log have exact-path ignores. README and context status now distinguish the ordinary2555 failure, aligned3712 HF pass, test-only residue26 hole and rejected persistent GDN overlap. The active in-place activation control is pending and untouched.

### Exact real SparseMoe activation reuse control (2026-11-13)

`expert_activation_control_v1` closed exit0/PASS in178.97s. The test-only policy reuses the private native sparse gate output for SiLU and its product with up; shared-expert computation is unchanged. All96 ordinary-versus-reuse A/B/A comparisons across Q1..32 are exact and finite, and36 trace executions over six plans preserve changed/repeated inputs and independent eager interleaving. All240 observed SiLU/multiply dispatches return the exact input address and report zero allocated-byte delta. Runtime audit and trace allocation tracking pass. These are complete real SparseMoe scheduling-equivalence checks, not HF full-decoder/capacity acceptance or runtime integration.

The5,209 watcher lines have zero standard/broad scan hits. Four result/log/provenance/watcher sources (709,630 bytes) are archived to18,644 bytes with originals preserved. The active activation-only complete-public3707 experiment is separate and untouched.

### Activation reuse alone reaches a later reduction allocation failure (2026-11-13)

`ple_capacity_activation_public_3707_v1` wrapper closed exit0 in79.63s, but the probe fails at actual public tail27/index115/start3680. Activation and sparse-down allocation pass; native `sum(down)` then requests an80MiB clone. Each bank has12,610,688 bytes free but only6,157,952 contiguous versus10,485,760 required: fragmentation, not total-free exhaustion. The isolated exact activation control remains valid, but this smaller policy alone does not rescue the full3707 lifecycle. Result/log/provenance are archived with originals preserved; no runtime integration, HF/capacity acceptance or bound follows. The active activation-plus-per-call experiment and proposed private-padding/aligned-view reduction control remain separate and unverified here.

### Activation plus per-call reservation still fails reduction cloning (2026-11-13)

`ple_capacity_activation_percall_public_3707_v1` wrapper closed exit0 in79.95s but the probe fails at the same actual tail27/index115/start3680 native `sum(down)`80MiB clone. Per bank12,610,688 bytes are free, while the largest block8,779,392 remains below the10,485,760-byte request. Reservation changes the fragmentation geometry but does not remedy this clone allocation. Result/log/provenance are archived with exact identities and originals preserved; the large raw result receives a scoped ignore. No runtime change, accepted bound or HF capacity result follows. The proposed private-padding/aligned-view reduction control remains pending.

### Exact private reduction and measured clone removal (2026-11-13)

`private_reduction_control_v1` closed exit0/PASS in35.50s. This weight-free control uses the real512-expert/2560-wide BF16 down geometry:96 eager comparisons across Q1..32 and36 trace executions preserve valid rows bit-exactly while clearing NaN/Inf/finite poisoned padding. Runtime fallback audit and trace allocation tracking pass. All31 nonaligned counts reduce observed additional peak live allocation from84,213,760 to163,840 bytes in actual NORMAL-dispatch allocation graphs, not a source-only estimate. The aligned32 path remains native.

Source inspection confirms the test helper clears padding only on the private down buffer, aliases its physical Q32 storage, calls unchanged public `sum(dim=1, keepdim=True)` without compute-configuration overrides, then restores logicalQ by an alias view. Native generic reduction otherwise clones a nonaligned caller tensor before in-place padding fill (`generic_reductions.cpp:640`). The helper performs no Torch conversion or host numerical fallback inside the pass; allocator metadata observations are test instrumentation. This establishes the isolated shape/precision/lifetime control, not full decoder HF accuracy or accepted capacity, and no runtime edit follows yet.

All64 ordinary/private raw allocation graphs plus result/log/provenance/watcher are archived individually (68 sources,1,863,638 bytes→139,456 bytes); originals remain byte-exact. Watcher1,137 lines has zero standard/broad scan hits. The active activation-plus-private-sum complete-public3707 run is separate and untouched.

### Private activation/reduction lifecycle passes; PLE integration awaits refreshed gates (2026-11-13)

`ple_capacity_activation_private_sum_public_3707_v1` closed exit0/PASS in124.57s. Without GR splitting or output reservation, the test-only combined policy completes both116-chunk/3,707-row constant-input public prefills and decode lifecycle. Three sources (735,614 bytes) are archived to16,340 bytes with originals preserved; this is capacity/lifecycle evidence, not HF acceptance.

The corresponding minimal repair is now integrated in `SparseMoe`, defaulting `reuse_private_buffers=False` and enabled by the public decoder only when `has_ple`. Private gate SiLU/multiply preserve their BF16 boundaries; private weighted-down padding is cleared before explicit alias views around the same native sum. The isolated controls support exactness, while integrated real-HF3707 and repaired-runtime all-tail/batch capacity remain pending. No accepted context bound follows. README/context now mark the six device_trace_v2 timings and offline source closure historical; the old batch5 physical argument relied on the removed clone and no longer bounds this allocation path. Refreshed performance and batch evidence are required.

### Integrated real-HF3707 literal contribution failure retained (2026-11-13)

`ple_public_hf_3707_v6` closed exit1 in161.19s with report PASS=false. All3,707 whole-output rows pass: global PCC0.9980743815, minimum row55 PCC0.9950039280. Contribution global PCC is0.9980376773, but row55 is the sole failing contribution row at0.9947104757. All three decode contribution PCCs exceed0.997059999; six replays/one capture, exact repeated outputs/states and repeated prefill, and every runtime audit pass. Passing lifecycle and whole-output metrics do not erase the literal contribution failure, and no accuracy/capacity acceptance is claimed.

Result/log/provenance are archived with failure status and originals intact. An independent actual-packed CPU control is being investigated; no explanation or carried classification is asserted yet. The active default-runtime all32-residue search is separate and untouched. Runtime remains frozen.

### Integrated default-runtime residue candidates (2026-11-13)

`ple_capacity_residues_private_buffers_v1` closed exit0 in580.67s with320 probes and no experimental flags. Every residue r1..32 reports last passing candidate4736+r and next observed failure4768+r; no holes appear among these observed candidate boundaries. After each probe the allocator returns to341,782,016 total free bytes and41,217,152 largest contiguous bytes per bank. The first candidate failure4769 occurs in the MLP GR projection's first20MiB slice: per bank4,446,016 bytes are free, but the largest block2,485,376 is below2,621,440 required. This is fragmentation, not a demonstrated aggregate physical limit.

Result/log/provenance are archived; originals and exact hashes are preserved. Binary-search monotonicity remains an assumption. Fresh complete public4768/4769, terminal-tail and HF boundary checks remain pending; no accepted continuous bound or capability reduction follows. The pending CPU capacity-closure helper likewise does not establish a hard physical limit or accept a reduction. Active boundary runs are untouched.

### Fresh public4768 passes; next4769 reproduces fragmentation (2026-11-13)

Default integrated `ple_capacity_private_public_4768_v1` closes exit0/PASS in150.70s with298 checks across two complete149-chunk prefills and traced decode lifecycle. Fresh `ple_capacity_private_public_4769_v1` closes its wrapper exit0 in62.57s but probe FAIL occurs on actual public firstchunk/index0/count32 at the MLP GR20MiB slice, reproducing the observed contiguous-allocation shortfall. Thus4768 is a completed constant-input public lifecycle candidate, while4769 is an observed implementation-specific allocation failure, not a demonstrated aggregate physical limit. Real-HF boundary/neighbor and terminal checks remain required; no accepted cap or accuracy follows.

Also archived the closed CPU-only `ple_prefill_backend_3707_v1`,4768_v1 and4767_v1 results/logs/provenance plus derived intermediate/worst-row diagnostics. The3707 packed control reproduces a row55 contribution discrepancy and also has a separate row33 whole/contribution failure; it does not by itself establish TT/control acceptance or excuse the device failure. The4768/4767 CPU controls pass their all-row metrics but are not device results; BFP8 expert arithmetic is not modeled. External tensor identities remain recorded, with the3707 tensor hash independently verified during packaging. Eighteen sources total393,716 bytes archive to60,288 bytes. The active real-HF4768 run is untouched.

### Integrated4768 strict real-HF pass (2026-11-13)

`ple_public_hf_4768_private_v1` closes exit0/PASS in182.49s. Every literal prefill row passes: minimum whole PCC0.9954640871 and contribution PCC0.9951774922. Three traced decode contribution PCCs are0.99843637/0.99815441/0.99803068, reaching logical context4771. Runtime audits and exact repeated output/state/prefill checks pass. Result/log/provenance are archived; the exported `/home/ttuser/qwen38-flash/ple_device_4768_private.pt` remains external with its SHA256/size recorded in the archive index.

This establishes the tested full public HF context, not yet an accepted continuous bound. HF4767 and terminal32 checks remain pending. `tests/capacity_contract_source_review.md` explains a conditional prepared-profile1..4768 engineering argument from fixed chunk resources and finite tails, while preserving4768 as the largest observed boundary, not an optimal/universal maximum. The separate advertised single-call hidden input requires5,368,709,120 bytes versus514,902,528 free after the recorded loaded model/state; this establishes an advertised resident-input physical obstruction, not that4769 fragmentation is aggregate exhaustion. Capability reduction remains unaccepted pending the required checks and review.

### Integrated nonaligned4767 strict real-HF pass (2026-11-13)

`ple_public_hf_4767_private_v1` closed exit0/PASS in199.45s. All literal prefill rows pass, minimum whole PCC0.9958494904/contribution PCC0.9956877869; three traced decodes pass with minimum contribution PCC0.9978838141 through context4770. Audits and exact repeated outputs/states/prefill pass. Result/log/provenance are archived; external `/home/ttuser/qwen38-flash/ple_device_4767_private.pt` SHA256/size are recorded in the archive index. This complements the aligned4768 literal pass. The active current-runtime terminal32 sweep and offline matched comparison remain separate; supported range/capability reduction is not accepted yet.

Closed CPU `ple_prefill_backend_4768_matched_v1` result/log/provenance are also archived. Its saved TT outputs pass all4768 literal and packed/six-projection control rows; this is additional matched evidence, not a changed gate or classification. The4767 matched comparison remains active and untouched.

The4767 matched CPU comparison subsequently closed exit0 in17.49s (`ple_prefill_backend_4767_matched_v1`) and is archived with result/log/provenance. Prompt/IDs/literal reference identities match; all TT-versus-literal/packed/six-projection whole and contribution rows pass. Minimum TT/packed contribution is0.9974862511. This supplements the strict device pass without rewriting acceptance; the terminal sweep remains untouched.

### Current terminal32 public-chunk lifecycle coverage closes (2026-11-13)

`ple_capacity_private_terminal_4768_v1` closed exit0/PASS in130.56s: all32 lengths4737..4768 pass their full-resident-I/O public-chunk, traced-decode and repeated-chunk lifecycle. Scope is explicitly `chunk_proxy`, not32 complete public prefills: repeated offset0 chunks plus the actual terminal offset/count are exercised. This complements the complete two-pass public4768 lifecycle, adjacent4769 failure, and strict real-HF4767/4768 passes.

Together with fixed-resource/source reasoning in `tests/capacity_contract_source_review.md`, these records support an engineering range claim for1..4768 under the specified batch1/all32-prepared allocation profile; they do not establish a universal allocator-history theorem or optimal maximum. Formal helper closure is still pending source/dependency checks, and capability reduction/final stage acceptance is not promoted here. Result/log/provenance are archived; the earlier3707 literal failure is preserved and its active export rerun remains untouched.

### 3707 export rerun preserves the literal failure (2026-11-13)

`ple_public_hf_3707_private_v1` closed exit1 in155.13s and exactly repeats contribution row55 PCC0.9947104757003457 as the sole failure. Whole-output rows, three decodes, repeated outputs/states/prefill and runtime audits pass. This does not alter the failed literal accuracy classification. Result/log/provenance are archived, and external `/home/ttuser/qwen38-flash/ple_device_3707_private.pt` SHA256/size are recorded for the independent matched CPU comparison. The earlier failure remains intact; active matched comparison and Watcher batch5 files are untouched.

### Current batch5 normal pass and batch6 trace failure (2026-11-13)

`ple_private_batch5_controlled_v1` closes exit1 in41.96s under Watcher with trace high-water overlap1,111,040 bytes total; this is not an aggregate-impossibility proof. Its1,136 watcher lines have zero standard/broad scan hits. `ple_private_batch5_normal_v1` closes exit0/PASS in23.44s with deterministic output and runtime audit; `literal_hf_strict_passed=false` and explicit matched-control finding/classification-review requirements remain preserved. Normal batch6 (`ple_private_batch6_normal_v1`) then closes exit1 in23.69s with end-trace overlap44,092,416 bytes total. Result/log/provenance are archived, alongside the batch5 watcher. Current batch6 aggregate-resource analysis remains pending; no harder bound is inferred from overlap alone.

### Capacity source/provenance closure and3707 matched attribution (2026-11-13)

Closed CPU `capacity_search_import_closure_v1` and `final_capacity_summary` records plus the capacity input manifest are archived. The helper validates six current-runtime capacity/HF records and explicitly proves the search's sole dependency mismatch consists of unused CLI-export/import additions, with unchanged imported definitions. It reports `evidence_closed=true` while `continuous_range_proven=false`, `hard_physical_limit_proven=false`, `capability_reduction_accepted=false` and stage acceptance not evaluated. The separately reviewed prepared-profile source/resource argument remains necessary; the helper does not supply a stage verdict.

Closed CPU `ple_prefill_backend_3707_matched_v1` and selected-row diagnostic are archived. Exact saved TT-versus-packed outputs pass all3,707 whole/contribution rows (minima0.99812449/0.99795272); row55 contribution is0.99991131. The literal row55 failure remains0.9947104757, and the CPU control's separate row33 literal failure remains visible. This is attribution evidence, not a strict literal pass or automatic carried classification. All active full-context QSA files remain untouched.

### Current batch6 total-DRAM obstruction and normalbatch5 scope (2026-11-13)

`ple_private_batch_capacity_bound.json` and its readable report are archived. Recorded resident model33,659,240,448B + actual six-request trace399,966,208B + mandatory six state payloads26,739,072B + simultaneously live expert activation/down100MiB gives34,190,803,328B, exceeding device DRAM34,178,731,008B by12,072,320B. The proof excludes the removed80MiB sum clone and conservatively omits inputs/outputs/routing/alignment, so it is distinct from the44,092,416B native trace-address overlap.

Normalbatch5 is the largest tested passing tuple profile, with explicit packed-control pass and literal strict flag false retained. Watcherbatch5 is instrumentation-limited and does not reinstate the superseded four-request maximum. README/context now state this current-policy physical bound and distinguish it from maximum-prompt capacity or hypothetical alternate representations/schedules. Final controlled-finding/stage review remains required. Active QSA evidence is untouched.

### Full QSA all-row gate fails before boundary continuation (2026-11-13)

`qsa_final_device_full_boundary` closed exit1 after5,713.40s. All262,144 attention prefill outputs are finite; global PCC from FP64 streaming sums is0.9999747948. Exactly one row fails the0.995 gate: row105584 PCC0.9944236875; the next worst row58531 is0.9958395362. The harness asserts this all-row gate before restoring/running the nonaligned boundary branch, tail or final decode. Their requested fields are not execution evidence; no prefix state export was produced. This is a failed attention-layer result, not accepted full-context or full-decoder evidence.

Failing result/log/provenance are archived. Actual output `/home/ttuser/qwen38-flash/qsa_final_device_262144.pt` remains external with byte size and independently computed SHA256 in the archive index. AutoFix CPU/AutoDebug investigation remains active; no cause, threshold change or reduction is inferred. Active GDN full-context files and hardware are untouched.

### QSA cutoff diagnosis and launcher recovery (2026-11-13)

The fresh source-only AutoDebug launcher first failed its sandbox preflight because this host disallows unprivileged namespaces (`qsa_full_context_autodebug_v1`). The explicitly authorized retry (`v2`) selected the old CLI, whose code-mode helper was absent. After verifying that PID 2603810 was this run's CPU-only Codex child, it was terminated with SIGTERM; no device process was terminated or reset. Retry `v3` prepends the installed matching CLI/helper directory `/home/ttuser/qwen38-flash/codex-venv/lib/python3.12/site-packages/codex_cli_bin/bin` and runs with `AUTODEBUG_SKIP_CHILD_SANDBOX=1`. All three logs and provenance records are retained. The fresh report remains pending.

CPU-only controls (`qsa_failed_row_selection_cpu.json`, `qsa_two_row_precision_cpu.json`, `qsa_two_row_midpoint_cpu.json`) localize a plausible cause without changing runtime or acceptance. On row 105584, a single near-cutoff block replacement raises saved-TT agreement from 0.994424 to 0.999973. Uniform FP64-rounded and balanced-FP32 indexer projections both predict that replacement using otherwise unchanged HF query normalization, RoPE and keys. The critical exact projection is -0.1333008091605734, just below the BF16 midpoint -0.13330078125; literal CPU BF16 GEMM rounds to -0.1328125, while the more exact controls round to -0.1337890625. Row 58531 provides an independent analogous control. Score precision and query-normalization reduction controls on identical operands retain the original selected sets. These are source-bound CPU observations, not proof of actual device-selected IDs or a passing full-context gate.

A uniform all-row FP64 indexer reference control is running separately. It must verify the original full selection hash before reusing any unchanged attention output. A test-only exact-runtime prefix-state recovery harness is being prepared to exercise the boundary checks omitted by the failed run. It must preserve the failed all-output result and prove state/output equivalence; it cannot relabel that run as passed. Chip 0 remains exclusively owned by the GDN full-context test during this CPU work.

### Uniform FP64 indexer control does not close the QSA gate

`qsa_fp64_indexer_control_full_v2` completed its CPU-only run in 313.45 seconds. The original full selected-ID SHA matches exactly, and all 2,092 changed-selection rows reproduce literal attention bit-for-bit before comparison. The uniform FP64 indexer policy changes 40,465 projection elements across 37,552 rows. It resolves row 105584 but exposes row 16748: TT-versus-control PCC 0.99354633. Literal HF versus the FP64 policy itself falls below 0.995 at both rows (0.99358871 and 0.99446824). Thus changing reference policy is not a passing all-row repair. The original strict failure and the new control failure remain explicit. Device projection/selection discrimination and the row-16748 historical-key rounding investigation remain required; no threshold or runtime change has been made.


### Current-source GDN advertised-context bridge passes (2026-11-13)

`gdn_final_full_context` closed exit 0 after 2286.24 seconds. The actual attention module advances fresh TT state through 262143 real, 2560-wide prefix rows with the complete layer0 weights resident. Every prefix attention output passes: minimum PCC 0.99997919798. The original final attention row at position 262143 also passes, PCC 0.99999868870. This checks all 262144 attention rows and the nonaligned prefix; it omits full decoder/MoE computation for prefix rows and makes no full-decoder long-prefill claim.

Two independent 10240-wide final inputs pass through public `FunctionalDecoder.decode_forward` in one capture and four A/B/A/B replays, restoring the actual TT prefix state each time. Minimum whole/contribution PCC is 0.999996995696/0.999996959935. Repeated outputs and both conv/recurrent states are bit-exact; changed inputs produce changed outputs. All 264 runtime-audited passes complete with trace allocation tracking enabled. Watcher was disabled for this normal-mode run; no new Watcher-clean claim is inferred.

Result, exact log and recorded provenance are preserved and compressed deterministically: three new archive entries, 96357 original bytes to 14020 XZ bytes. The multipart archive index now contains 494 entries. The GDN long gate validates 44 explicit checks and 27 current source identities; its status is `declared_checks_satisfied_review_required`, not stage acceptance. README/context identify the completed reduced-prefix/full-final-decode scope. Other pending gates, active QSA work, runtime and hardware remain untouched.


### QSA actual-device cause and boundary recovery (2026-11-13)

The fresh AutoDebug v3 report completed successfully. Direct captures now close the causal gap: `qsa_failed_projection_device_v1` shows the production balanced-FP32 projection agrees with its CPU arithmetic model at all four named coordinates, while literal oneDNN BF16 projection differs at sensitive midpoints. `qsa_window_105568_current_v1` confirms the device selects block25012 instead of23259 at row105584; top-k agrees with CPU on identical scores, all2049 IDs are valid/unique/causal, and cache gathering is exact. Independent HF attention using those actual IDs matches saved full-run TT output at PCC0.9999731261 (`qsa_actual_capture_diagnosis_v1`). The original all-row PCC failure remains open.

Uniform FP64 projection is rejected as a repair: `qsa_fp64_indexer_control_full_v2` fixes105584 but creates row16748 PCC0.993546327 against TT. Literal HF itself differs from that uniformly higher-precision reference below0.995 on those rows. Projection order therefore matters; higher precision alone is insufficient. Source-derived fixed K1024 blocks with high-then-low pair accumulation reproduce literal BF16 on307200 tested projections (`qsa_projection_orders_cpu_v3`); a full-history CPU check and standalone device candidate are pending. No per-row correction, reference rewrite or threshold relaxation has been applied.

`qsa_boundary_recovery_v1` passes an explicitly reduced state-reconstruction path: short controls prove all five mutable states bit-exact to ordinary forward, final32 outputs are bit-exact to the saved full TT run, the31-row tail has minimum HF PCC0.9999137521, and one traced attention capture/four A/B/A/B replays pass with exact repeated states. The exported recovery state has its own version2 provenance and retains `strict_full_prefill_passed:false`. `qsa_full_decoder_recovery_v1` then passes full public layer3 decode at position262143 on two changed inputs, one capture/four replays; minimum whole/contribution PCC0.9999422431/0.9999366403, clean audit and allocation tracking. No HF state is injected. These boundary results do not clear the failed full-prefill accuracy gate.


### Strict layer 0 public continuation v7 closes current-source checks (2026-11-13)

`public_continuation_layer0_v7` closed exit 0 in 107.59 seconds, with `literal_hf_strict_passed=true` and no backend-control exception. Independent 31/33-row prefixes continue for 8/7 rows; all 15 resumed rows meet literal whole/contribution PCC, with contribution minimum 0.9999600053. Both dropped-history controls fail as required. All 24 request/decode comparisons meet the literal gates, minimum contribution PCC 0.9999452233.

Independent fresh prefills of lengths 65/97/129 first execute while the same tuple decode trace remains live. Across both repetitions, all 582 rows pass literal whole/contribution PCC (minima 0.9985389113/0.9980036616). Repeated outputs/states and request isolation are exact. One capture and 12 replays acknowledge two outputs and perform zero output readbacks before the first replay; runtime fallback and allocation tracking checks pass. This is the documented short live-trace resource profile, not maximum-context prefill or maximum-batch evidence.

The closed Watcher log contains 3700 lines with zero standard or broad fatal/assert/invalid/overflow/out-of-bounds/corrupt/sanitize scan hits. Result/log/provenance, watcher log and seven kernel/inspector maps are archived: 11 new entries, 4467164 source bytes to 204840 XZ bytes. Both layer 0 continuation/unseen-length bindings satisfy their 18/64 declared checks and current source identities; stage review remains required. Historical archives remain intact, and running layer 1 v7/device work was not touched.


### Layer 1 public continuation v7 retains explicit controlled findings (2026-11-13)

`public_continuation_layer1_v7` closed exit 0 in 148.49 seconds with explicit `--ple-backend-control`, `literal_hf_strict_passed=false` and `fresh_backend_all_matched_passed=true`. Fresh 31/33 prefixes, 15 resumed rows and 24 decode comparisons pass strict literal whole/contribution gates. Resumed/decode contribution minima are 0.9967340231/0.9969496727; both dropped-history controls detect lost state.

All 582 independent live-prefill rows across lengths 65/97/129 and two repetitions pass literal whole output (minimum 0.9960471988). Literal contribution still fails length65 row2 at 0.9949381948 and length129 row1 at 0.9947543740, identically on repetition. The fixed independent actual-packed-lookup/six-FP64-projection control reproduces both failures; all TT/control whole/contribution rows pass (minima 0.9981886148/0.9978749156). Nine actual-cache parity checks and bit-exact independent literal reruns are recorded. This classification is subject to stage review, not a blanket strict pass or reduced threshold.

One capture/12 replays acknowledge two outputs, with zero capture-output readbacks before first replay. Repeated outputs/states and inter-request isolation are exact; runtime fallback/allocation audits pass. Watcher contains 4380 lines and zero standard/broad scan hits. Result/log/provenance, watcher and seven maps add 11 archives (6060985 source bytes to 286584 XZ bytes); historical records remain intact. Layer1 continuation/unseen bindings satisfy 65/1863 declared checks with current source identities while explicitly preserving literal strict false. Active linear-contract/device work is untouched.


### Current GDN short contract and additional PLE batch 4 Watcher closure (2026-11-13)

`linear_final_contract_v4` closed exit 0 in 143.39 seconds. Fresh lengths 1/31/32/33/65 have minimum literal whole/contribution PCC 0.9996449947/0.9996082187. Resumed windows after positions 1/31/32/33 and changed decode inputs pass, with contribution minima 0.9999601245/0.9999952912. All four dropped-history negative controls, exact reset/replays, interleaved prefill and independent-request isolation pass. One capture/12 replays and 39 runtime-audited calls are recorded. Watcher: 3249 lines, zero standard or broad scan hits.

`ple_final_batch4_controlled_v4` closed exit 0 in 52.12 seconds, batch 4 with lengths 1/2/3/1. Literal whole-output passes all rows; decode contribution minimum is 0.9975344539. Fresh request 2 row 0 contribution is 0.9949769974, retained under the explicit actual-packed-lookup control; `literal_hf_strict_passed=false`. All matched whole/contribution rows pass (fresh contribution minimum 0.9992804527, decode minimum 0.9999366403), with exact repeats and runtime audit. Watcher: 1400 lines, zero standard or broad scan hits. This instrumented short profile does not change the separate normal batch 5/physical batch 6 bound.

Both result/log/provenance and Watcher/kernel maps are deterministically archived: 22 new entries, 7712023 original bytes to 375472 XZ bytes. Original and historical evidence are preserved. Current source-bound gates satisfy their declared checks, with the PLE controlled finding explicit and stage review still required. No runtime/test edits or hardware were performed during this closure.


### Current GDN batch 32 controlled evidence (2026-11-13)

`linear_final_batch32_v4` closed exit 0 in 131.11 seconds under Watcher 10, with 32 heterogeneous requests (63 prefill rows and 32 decode rows), BF16 experts and explicit `--gr-backend-control`. Literal whole-output minimum is 0.9951083660; decode contribution minimum is 0.9999741912. Fresh request 16 row 0 contribution remains 0.9906083941; the independent attention-GR-down-only FP64 control reproduces 0.9906105995 versus literal HF and matches TT at contribution PCC 0.9999851584 for that row. All matched rows pass; literal strict remains false and the finding requires stage review. Independent literal rerun, deterministic outputs and runtime audit pass.

Watcher contains 4002 lines with zero standard/broad scan hits. Eleven archived result/log/provenance/Watcher/map files preserve 2985567 original bytes in 147968 XZ bytes. The current source-bound batch gate satisfies 390 declared checks; no stage acceptance is inferred. Runtime and test sources were not changed.


### PLE short contract current-source closure (2026-11-13)

`ple_final_contract_v4` closed exit 0 in 188.10 seconds under Watcher. All literal whole/contribution gates pass for fresh lengths 1/31/32/33/65, resumed windows after positions 1/31/32/33 and changed decode inputs. Fresh whole/contribution minima are 0.9965662956/0.9962700009; resumed/decode contribution minima are 0.9965371490/0.9977497458. Six live-trace interleaved prefills preserve both states, with exact repeated outputs and conv/recurrent/PLE-conv/token-history buffers. One capture/7 replays, allocation tracking and 39 audited calls pass. This strict input set does not reclassify separate public/batch controlled fresh findings.

Watcher has 4415 lines and zero standard/broad scan hits. Eleven result/log/provenance/Watcher/map archives preserve 4700964 bytes in 223244 XZ bytes. The short report has 89 declared checks. Its original QSA full-file hash remains stale after the separately validated two-line constructor change; `linear_qsa_dependency_closure_v1.json` proves that change is unreachable for the recorded linear layer while every shared runtime hash remains identical. Raw gate evaluation still fails closed until this narrow closure is integrated; stage acceptance remains pending.


### Explicit linear-layer source closure after QSA-only integration (2026-11-13)

`evaluate_gate` now supports only an explicit `linear_qsa_constructor_v1` opt-in for the preserved QSA full-file mismatch. It checks exact historical/current QSA and frozen 22-file inventory bindings, permits only the two local constructor replacements, and calls the live record/branch verifier. All 21 other runtime files must remain byte-identical. Results preserve original hashes, current hashes, exact replacement proof and controlled numerical findings. There is no blanket runtime-hash waiver.

Eight CPU tests cover successful closure and rejected wrong layer, extra QSA/import-time edit, shared router edit, stale binding, unknown closure kind, missing source and failed numerical gate. The CPU-recorded `linear_qsa_gate_evaluations_v1` closed exit 0: all 12 selected gates satisfy declared checks, including the six short/public/batch reports, their two unseen-length gates, GDN reduced-prefix/full-public-final-decode bridge, normal PLE batch 5, and full-public PLE 4768/4767 accuracy. Long prefill releases decode trace/output before its repeated maximum prefill; short live-trace coverage remains a separate profile. No existing literal fresh-contribution failure is reclassified.

The raw PLE contract QSA mismatch noted above is now resolved by this explicit source proof, not by replacing historical provenance. Compound capacity reports without layer fields and deliberate failed batch captures require the separately owned harness/outcome-specific source closure; success-only proof does not apply to them. Physical capacity, capability reduction and whole-stage acceptance remain review decisions.


The final live source-closure inventory is `linear_qsa_gate_evaluations_v2.json` (CPU recorded exit 0), superseding v1 after tightening the source helper. It additionally requires the current runtime inventory to be exactly the frozen 22 files plus four pinned ordered-projection files, rejecting any extra module/import-time initializer; it verifies the pinned target config selects linear attention for the recorded layer. Historical standalone candidate files, if recorded by unrelated short runs, are retained as explicitly unreachable historical identities. All 12 selected gates still satisfy declared checks; eight CPU tests include the extra-module rejection. The v1 report remains historical. No runtime/device rerun or acceptance waiver occurred.

### Ordered QSA projection repair and full-context rerun (2026-11-13)

The source-derived indexer projection reproduces the pinned CPU backend's independent FP32 accumulation blocks and BF16 rounding boundary. Q=1 uses five K=512 blocks; Q=2..32 uses K=1024/1024/512, with high-then-low BF16 pair accumulation and ordered partial additions. The full CPU experiment compares all 167,772,160 BF16 indexer outputs over 262144 rows bit-for-bit against the literal reference (`qsa_blocked_projection_full_cpu_v1`). This is arithmetic identity evidence, not device attention acceptance.

The first device candidate exposed two BF16 midpoint differences despite bit-exact FP32 accumulators. Direct packing disagreed with literal nearest-even rounding; device FP32 output followed by explicit `ttnn.typecast` resolves both. The production operator therefore preserves that explicit cast. V1/V2 failures and source snapshots remain intact (`ordered_projection_pack_autofix.md`, `ordered_projection_pack_review.md`). V3 passes all Q=1..32 at ten independent starts, all 352 changed/repeated trace replays, stable program count 33, preallocated output identity, runtime audit and a clean 860-line Watcher log. `ordered_projection_operator_closure_v1` verifies the source, log, result and historical-source bindings; the subsequent module/class documentation-only edit preserves the executable AST.

Production integration changes only the QSA constructor's imported class and its construction; shared runtime files remain unchanged. The independent 32-row diagnostic window now passes row105584 at PCC 0.9999760135 and all rows at minimum 0.9999231424 (`qsa_window_105568_ordered_v1`). This window injects an HF prefix for diagnosis and does not establish end-to-end context accuracy.

The normal full-context rerun `qsa_ordered_device_full_boundary_v1` began at 04:29:28 UTC. It builds TT state from empty, keeps the original PCC>=0.995 per-row gate, and will test all262144 attention rows before boundary tails, traced final attention decode and standard prefix export. At 04:51 UTC it remains running; dispatch progress is not an accuracy result. Its original failed predecessor is preserved. Exact recorded commands, including the separate source-bound CPU preflight and complete public final-decoder bridge, are in `plans/qsa_ordered_repair_commands.json`. The bridge accepts only a closed passing producer and its normal version1 prefix; recovery-version2 state cannot discharge this gate. PLE advertised-context, remaining QSA short/synthetic checks, current performance, independent stage review and commits remain pending.

CPU evidence-helper regression: `python_env/bin/python -m pytest --noconftest -o addopts='' --import-mode=importlib --tb=short -q models/autoports/qwen_qwen3_8_flash_next/tests/{test_archive_manifest_cpu,test_qsa_bridge_preflight_cpu,test_trace_profile_cpu,test_streaming_profile_merge_cpu,test_private_reduction_cpu,test_capacity_residue_search_cpu}.py` passes 29 tests and 27 subtests (3.46 seconds). The first ordinary pytest invocation loaded the repository conftest's TTNN import and caused nine no-TTNN-import assertions; removing default addopts alone additionally lost namespace-package imports. The explicit CPU invocation above disables conftest and retains importlib mode. No helper source, production runtime or device run changed to obtain the pass.

### Final evidence aggregation preparation (2026-11-13)

`final_cpu_helpers_v1` records the combined CPU-only suite with `--noconftest --import-mode=importlib`: 84 tests and 60 subtests pass in 3.84 seconds (recorded wrapper4.82seconds). Its command, all source identities and closed log are retained. `semantic_gate_inventory_v1` applies14 explicit all-of dependency mappings without replacing30 concrete gates;50total gates currently include13 satisfying declared checks,31unmet and6pending inspection/review/commit entries. Missing, cyclic, duplicated, empty and failing dependencies cannot produce a passing rollup. This inventory is not a stage-review verdict.

`close_watcher_evidence.py` and `final_watcher_command_plan.json` are ready for the final CPU closure. They require all three actual closed successful layer runs, enabled Watcher/absent profiler, source and command-log identity, nonempty complete worker dumps, attach/detach coverage, dump completion counts and failure-signature scans. Current layer0/1 logs validate; pending layer3 prevents a summary. No final Watcher-clean aggregate is fabricated. Five additional immutable CPU inventory/log/provenance archives bring the current manifest to678 entries; original historical evidence remains intact.

### Repaired QSA advertised-context and normal public bridge closure (2026-11-13)

`qsa_ordered_device_full_boundary_v1` completed exit 0 in 6135.046658 seconds. From initially empty actual TT state, all 262144 literal-reference attention rows pass PCC >= 0.995: minimum 0.9982089996 at row 222943, global FP64 PCC 0.9999755086, zero failed/nonfinite rows. The actual device checkpoint at position 262112 supports the full 32-row ending, then a restored 31-row ending to prefix 262143 (minimum 0.9999139905), and traced final attention decode (0.9999723434). Full-prefill, checkpoint/tail and trace fallback audits pass. Recorded elapsed/host-wall values are diagnostic durations, not final warmed device performance measurements.

`qsa_ordered_repaired_row_cpu_v1` separately reads the saved full-run output and original HF fixture with CPU mmap: historical failing row 105584 is now 0.9999759609057268 (FP64 PCC), max absolute error 0.0009765625. Its source, command, producer result/log/provenance and external files are hash-bound: actual output `qsa_ordered_device_262144_v1.pt` SHA256 `047c388ff15222e06b30f2674e538a5953c01bf53bc855c2bec7590f7b20a01b`; normal TT prefix `qsa_ordered_tt_prefix_262143_v1.pt` SHA256 `4b4f2e63626e084ea4018e4dcc900130f32197e64bdb764019feed54084882e3`. The original full-run failure and rejected alternate FP64 reference are unchanged; this is a runtime repair with the literal oracle and threshold retained.

`qsa_ordered_full_decoder_v1` completed exit 0 in 11.90369 seconds. It reuses that source-verified normal version 1 TT prefix and runs complete public layer 3 decode at position 262143 for two changed inputs in A/B/A/B order. Minimum literal whole/contribution PCC is 0.9999422431/0.9999366403; one capture/four replays, repeated outputs and all five states exact, fallback audit clean and allocation tracking enabled. The CPU preflight v1 metadata `KeyError: snapshot` occurred before hardware opened and remains failed evidence; repaired preflight v2 passes, with six CPU guard tests. Historical recovery-version2 diagnostics are not used for this normal-prefix result. The prefix remains attention-only, not full long-prefix decoder/MoE execution.

Current QSA public continuation v7, short contract v4 and batch 32 v4 also satisfy their literal gates. Public capture acknowledges two outputs with zero pre-first-replay readbacks. Batch minimum prefill/decode contribution PCC is 0.9986094236/0.9995502234. Their Watcher scans contain respectively 2010/1465/2435 lines and zero broad failure-signature hits. The closed `final_watcher_summary` verifies all three representative layers, completed dumps, current source closure and profiler absence; no normal long-run Watcher claim is inferred.

Deterministic XZ packaging adds 60 entries (standard results/logs/provenance, failed/repaired CPU preflights, compact closures and Watcher kernel/inspector maps). All 738 manifest entries verify roundtrip and available original identity; manifest SHA256 `c4cf9ff5c617814995841171d7a4ca20eaebf353ee23d963e4a2ae96e87f52bb`. Originals remain present and scoped large-raw ignores preserve the repository size policy. Both advertised-QSA gates evaluate as `declared_checks_satisfied_review_required`, with no schema gap; artifact digests are recorded at closure, not falsely attributed to original producer provenance. PLE advertised state-path/final decode, synthetic refresh, six final performance reports, capacity/control assessment, independent stage review and local commits remain outstanding. No whole-stage acceptance is inferred.


## Closed synthetic v4 evidence

Synthetic pytest closed exit 0 in 109.9891897 seconds. The separate CPU summary verifies exactly six expected cases, no failures/errors/skips, all 26 current runtime sources plus the test/generator, original log identity, XML command/timestamps and explicit XML SHA256 `eb65826fe0931fc47c9dae5d1636fadc2ec44e8f4a44829344ec4dffbde56bdc`. Scope: four complete synthetic GDN(layer 0)/QSA(layer 3) decoder tests at lengths 1/33 with traced decode, plus two PLE arithmetic-only tests. No full resident synthetic PLE decoder claim. Minimum recorded PCC is 0.9996175169944763. The final_synthetic_ci leaf satisfies its declared checks, with independent review still required.

Exact recorded commands:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_recorded --record models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/provenance/synthetic_final_decoders_v4.json --log models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/logs/synthetic_final_decoders_v4.log -- timeout 900 python_env/bin/python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_synthetic_decoder.py --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/synthetic_final_decoders_v4.xml
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_recorded --record models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/provenance/synthetic_final_decoders_v4_summary.json --log models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/logs/synthetic_final_decoders_v4_summary.log -- timeout 120 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.summarize_synthetic_evidence --xml models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/synthetic_final_decoders_v4.xml --pytest-provenance models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/provenance/synthetic_final_decoders_v4.json --xml-sha256 eb65826fe0931fc47c9dae5d1636fadc2ec44e8f4a44829344ec4dffbde56bdc --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/synthetic_final_decoders_v4.json
```

Artifacts: `synthetic_final_decoders_v4.{xml,json}`, `synthetic_final_decoders_v4_closure.json`, original/summary `provenance/` and `logs/` records. The closed Watcher log has 3143 lines and zero matches for `fatal|assert|invalid|overflow|out[ _-]*of[ _-]*bounds|corrupt|sanitize`; kernel-name/ELF and inspector maps are preserved. Seven evidence records plus 8 Watcher/map files were deterministically XZ archived with SHA/byte checks and exact roundtrip; originals remain intact. Archive count is 753. This closes synthetic coverage only; stage acceptance and other pending runs remain separate.


## Final PLE advertised state context and exact formatting transition (2026-11-13)

`ple_final_full_context` closed normally at `2026-11-13T07:03:43.590959+00:00`, exit 0,
2596.324586790055 seconds. Exact command and pre-format source identities:
`provenance/ple_final_full_context.json`; output `ple_final_full_context.json`;
log `logs/ple_final_full_context.log`. Every one of 262143 reduced prefix rows was
compared from initially empty device state with the full decoder/table resident.
PLE/GDN/attention-contribution minima are respectively 0.9976900815963745,
0.9957841634750366 and 0.9957735538482666, with zero failed/nonfinite rows and exact
hash IDs. The complete public final decode at position262143 (context262144) passes
whole/contribution PCC 0.997859124756483/0.9978260896469693. One capture/two identical-input
replays restore the actual TT prefix before each replay; output and all four state buffers
are exact. Final hash/history matches HF and all1028 runtime audit passes close.
This is262143 reduced prefix rows plus a complete final public token, not262144 separately
read-back reduced component rows, nor full MoE on every prefix row. Changed-input evidence
is provided by the short public continuation tests. This normal run has no Watcher claim.

After the process closed, the four exact canonical clang-format19.1.4 outputs from
`plans/format19_transition_v1.json` were applied to GDN compute/SFPU, ordered projection
reader and router header. Arithmetic, statement order and predicates are unchanged;
three existing GDN `if(emit_state)` bodies and ten existing router single-statement bodies
receive braces. Preserved before/after bytes and exact formatter/config identities support
an explicit source-semantic closure only, not identical compiled binary/timing claims.
Historical process hashes remain immutable. New final performance measurements use the
actual formatted runtime. The new format verifier rejects any unlisted edit or partial
transition; the QSA prefix loader remains strictly source-bound for future invocations.

New archive restoration source/test files were added and formatted. Explicit pinned
Black23.10.1/isort5.13.2 checks cover all149 Python files, clang-format19.1.4 dry-run/Werror
covers all8 native files; all pass. Full scoped repository precommit passes without
modifications. `final_cpu_helpers_postformat_v2` closes exit0 in5.359210240188986 seconds:
105 tests and84 subtests pass (pytest4.32 seconds). Exact CPU command/log/source provenance
is `provenance/final_cpu_helpers_postformat_v2.json`; source remains frozen for final
performance. This verifies helper behavior and does not replace independent stage review.


## Final formatted-runtime performance and source closure

All six `layer{0,1,3}_{prefill,decode}_device_trace_v3` captures and their `_report`
commands close exit0. Batch1, prefill32, decode1 after a32-token prefix; full QSA262144
capacity, BF16 activations, BF16 experts for0/3 and BFP8/full resident packed PLE for1.
Actual native whole-trace milliseconds (internal gaps included):

| Layer | Prefill | Traced decode |
|---|---:|---:|
|0|424.2814822222222|47.216009629629625|
|1|455.6046666666667|48.11629037037037|
|3|969.7737237037038|249.90783925925925|

Each `tracy/layer{layer}_{mode}_device_trace_v3/report/` contains native raw CSV,
whole-trace ops CSV, filtered tt-perf-report CSV, human-readable table, latency and
conversion provenance; compressed artifacts preserve exact bytes. Process records
`provenance/layer{layer}_{mode}_device_trace_v3{,_report}.json` bind executed argv,
environment, source and log hashes. `plans/qwen_perf_v3_commands.json` preserves the
planned sequence. `performance_evidence_offline_v3` closes exit0 in6.1703658127225935s;
all six gates pass native endpoint reparse, exact regenerated report/latency comparison
and source/log closure. This is one warmed measured replay per mode, not a latency
statistic or per-op breakdown. Watcher is disabled during all profiling.

The prepared CPU refresh sequence in `plans/postformat_cpu_refresh_commands.json`
closes all five commands exit0: capacity and batch closures v4, Watcher
`final_watcher_summary_format_v2`, synthetic `synthetic_final_decoders_v4_format_summary_v2`,
and `linear_qsa_gate_evaluations_v3`. The latter has12/12 declared source/numerical gates
satisfied. Raw synthetic source identity remains false after formatting; exact source
format-equivalence is explicitly true. Earlier hashes/results remain immutable.
`postformat_gate_binding_update_v1.json` records the guarded binding transition.

A live scan then found two older component records still bound to the pre-private-buffer
`decoder_components.py` identity. The exact-format proof correctly refused to waive that
computational edit. The owner chose fresh input/output-copy reruns with all26 current
runtime sources rather than broadening the source exception. Commands are in
`plans/component_current_source_refresh_v1.json`; v6/v5 results will be bound only after
successful closure. No runtime or test source changed for these reruns.

Archive batches preserve785 then899 total entries, all verified. See
`initial_postformat_packaging_v1.json`, `second_postformat_packaging_v1.json`, and their
archive verification records. Original bytes, failed records and secondary source
archives are preserved. Final commit staging must include the exact manifest payloads
hidden by generated/raw ignore rules; force-add only those hash-bound paths, not raw trees.


## Current-source component reruns closed

`chunk_input_final_v6` closes exit0 in307.9321167138405s; exact copies/physical padding,
all644 cases,38 invalid host contracts, all32 writer counts, unchanged3program/49152B
persistent baseline and zero sweep growth pass. First unseen-plan capture and changed/
repeated fixed-plan replay pass (one capture/four replays). `chunk_output_final_v5` closes
exit0 in3.924505041446537s; both changed-input replay outputs preserve all canaries exactly.
Executed commands/logs/current source identities are in their matching provenance
records and `plans/component_current_source_refresh_v1.json`. Both use Watcher10 and
allocation tracking with no program-cache exclusion. The two final gates now bind all26
actual current runtime hashes plus their harness/audit sources, with no formatting
exception. The earlier computationally stale records remain historical evidence.

The owner inspected the public source/config/state/page/position contract and adopted
the prepared PLE1..4768 resident-call inference, separately from the262144 state context.
Actual preliminary inspection and scoped capacity rationale are retained in
`owner_source_capacity_inspection.md`, `owner_preliminary_inspection_snapshot.json` and
`owner_context_interpretation.md`. These are owner judgments for independent review,
not an independent verdict or a numerical waiver. The final context fields now distinguish
reduced-prefix rows (262144/262143/262144), complete public tested prefill extents
(129/4768/129), and complete final public decode context262144 for all three kinds.


## Pre-review gate inventory and immutable-text packaging

The first `pre_review_gate_inventory_v1` CPU wrapper exited1 because it iterated the
DAG evaluator's result mapping as a list (`AttributeError: str has no attribute items`).
No hardware or validator change occurred. The wrapper was corrected to consume
`evaluate_gates(...).values()`; v2 closes exit0 in6.988985943607986s with48 declared
checks satisfied and only `stage_review`/`local_commit_shas` pending. Failed wrapper
log/provenance is retained; it is not a model failure or a suppressed gate.

The installed commit hook formats text. Read-only preflight identified26 immutable
XML/official table/triage/diff originals with meaningful historical whitespace or EOF
bytes. Packaging preserves them exactly as hash-bound archives and scopes their raw
originals out of Git, instead of rewriting provenance inputs. Six final performance
Markdown views expose the actual official tables with only trailing display whitespace
removed; original TXT bytes and conversion hashes remain unchanged. This is a storage/
readability change, not new measurement evidence. Final archive and view identities are
recorded by the packaging report. The owner inspection is refreshed after these final
link/packaging updates; independent stage review remains pending.


## Final staged checks before independent review

The full staged commit hooks initially reformatted a historical diagnostic snapshot,
`triage/ple_full/qsa_diagnostic_source.py`. Its exact staged historical bytes were
restored, losslessly archived and excluded as a raw original from the index. The
failed hook output is preserved. Eight secondary Watcher archives were verified and
staged; three oversized assembled CSV archives remain ignored local duplicates of
byte-identical, tracked multipart streams. No runtime or test source changed. See
`staged_hook_packaging_remediation_v1.json`. All958 archive entries verify and all988
manifest storage paths are staged. The complete staged commit checks then passed
exit0; exact output is `logs/pre_review_commit_hooks_v2.log`
(SHA256 `df217d17613a3159372ec3e68bc0b6fd77ae79c27fa50aca266e1d21a4c9d4bf`).

The final owner inspection and pre-review inventory are refreshed as
`final_semantic_inspection_v2` and `pre_review_gate_inventory_v3`, each with matching
recorded command/provenance/log paths. Independent review remains a separate gate;
these records cannot grant stage acceptance.


## Independent round1 finding and clean-timing remediation

`stage_review_round1.md` returns **more-work-needed** with one P2 finding: the tracing
skill requires separate performance runs with allocation tracking and diagnostics off.
The reviewer accepted the scoped numerical controls, reduced-prefix context composition,
physical capacity profiles and other evidence; it requested no decoder repair or
unrelated hardware rerun. Historical v3 timings remain valid instrumented records.

Only `tests/profile_trace_decoder.py`, `report_trace_performance.py` and
`test_trace_profile_cpu.py` changed. Their prior bytes are preserved under
`tests/diagnostic_sources/clean_timing_v1/`. Explicit `--clean-timing` rejects enabled
tracking/diagnostics, records effective flags, and uses no runtime-audit wrappers.
Exact eager/replay output and all-state comparisons remain outside timing. The offline
reporter validates the explicit clean regime and rejects diagnostics, bad state
comparisons and unknown regimes; historical tracked records retain their original mode.
Final performance gates require clean mode and false tracking/diagnostics/audit flags.
No runtime file changed. Scoped precommit passes; `clean_timing_cpu_v1` closes exit0
with10 CPU tests in2.008s, including negative setting and replay-state checks.

All12 serial chip0 commands in `plans/qwen_perf_v4_commands.json` close exit0, recorded
in `plans/qwen_perf_v4_execution.json` and individual run provenance/logs. Every capture
is a fresh process with tracking/tracebacks explicitly0, Watcher absent and
`--clean-timing`; native trace profiler instrumentation remains enabled for measurement.
No reset or termination occurred. Both warm and measured replay output/state equal
eager exactly for all six modes. Final device milliseconds are:

|Layer|Prefill32|Traced decode1 after32|
|---|---:|---:|
|0|424.1376088888889|47.199168888888885|
|1|455.555837037037|48.11890740740741|
|3|969.7029740740741|249.93035703703706|

`performance_evidence_offline_v4` closes exit0 in5.854029951617122s: all six native
endpoint/worker, regenerated CSV/table/latency and source/log checks pass. Historical
v3 bindings are retained in `plans/performance_v3_historical_bindings.json`. New exact
TXT/CSV/provenance and readable table views are under each v4 report root. These are
single warmed whole-trace intervals including device gaps, not per-op timings or a
latency distribution. A fresh diff-only round2 review will assess the fix and evidence.

Clean timing packaging is recorded in `clean_timing_v4_packaging_v1.json` and
`clean_timing_v4_archive_verification_v1.json`: all1053 archive entries verify, all1083
manifest storage paths are staged, and six readable v4 table views retain exact raw/
archive/view identities. No original capture/report bytes changed. The follow-up owner
inspection/inventory are `final_semantic_inspection_v3` and
`pre_review_gate_inventory_v4`, with separate process provenance and logs.


## Independent round2 acceptance

Fresh diff-only reviewer `/root/functional_stage_review_round2` returns **clean-pass**
in `stage_review_round2.md`, with no Required Work. It independently checks the changed
harness/reporter, all six clean timing regimes and real argv,12 closed serial processes,
exact output/state comparisons,110-core endpoint inventories, native interval arithmetic,
report/source/log identities and unchanged26-file runtime. The three original numerical,
state-context and physical-profile limitations remain accepted only in their stated
scopes. This stage makes no full-model, multichip or serving claim.

`stage_review_acceptance_v1.json` and its successful recorded CPU command bind the actual
review reports and current runtime hashes. Final gate bindings now point to this actual
independent verdict. The complete staged checks passed before review; local stage-owned
checkpoint commits follow, with no push.


## Local checkpoint and final disposition

Repository `/home/ttuser/qwen38-flash/tt-metal`, branch `main`:
`747f83791c95c757bf4f3d439dfc3e768d4f5c9b` —
`Add Qwen3.8-Flash-Next functional TTNN decoder`. Command:
`git commit -m 'Add Qwen3.8-Flash-Next functional TTNN decoder'`. It closes exit0 with
all installed commit hooks passing. All2231 changed paths are under the model autoport.
`local_commits.json` and its recorded CPU verifier prove the actual Git object/parent,
model-only scope, all26 committed runtime hashes and committed independent review.
No push was performed.

The final bookkeeping is committed separately as
`Record Qwen3.8 functional-decoder acceptance and checkpoint`. A commit cannot store its
own SHA in its tree; the implementation SHA above is logged here and that follow-up
metadata SHA is reported in the final handoff. This preserves a clean worktree without
a self-referential SHA claim.

All50 declared gates are evaluated in `final_gate_inventory_v1.json`, with matching
command/log/provenance. The independent verdict is **clean-pass, round2**; the original
reviewed limitations remain explicit. Functional-decoder stage only is complete. No
optimized-decoder, multichip, full-model or vLLM work was started.
