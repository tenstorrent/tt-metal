# Exact weighted-tail reader integration candidates

Two standalone candidates are frozen for parent-owned whole-decoder qualification. Neither candidate is selected by this preparation; production was not edited.

| Candidate | Base | SHA256 |
|---|---|---|
| `v6_transport_conv_norm_decay_interleaved_tail_integration.py.snapshot` |44f87 independent|`50b86061c351cf8e1d7628f9e55fbc00d1711eabd2b5e6779c74db88d35c3d58`|
| `v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot` |85ad composed|`d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`|

Builder `tests/v6_build_weighted_norm_interleaved_snapshot.py`, SHA256 `0d6d4f820fb862d37fd462bd782db2f9212bbd230c90b74d0629b383f381025e`. Each adjacent `_build.json` pins component source/gate, base/output, exact transferred helper/kernel ASTs and source hashes. Adjacent `.builder.py.snapshot`, `.reader.cpp.snapshot`, and `.compute.cpp.snapshot` preserve exact bytes. Pinned autoflake2.3.1/isort5.13.2/Black23.10.1 were applied before freezing; both candidates pass exact formatter AST comparison and all pinned checks.

## Component transfer and exact scope

Parent device result `v6_weighted_norm_interleaved_device.json/.log` passed all16 complete-adapter/debug fixtures plus2 L1/DRAM signed-zero/subnormal/midpoint transport controls and closed cleanly. All FP32 tail debug/weighted outputs and complete BF16 output bits match;100 checked traced replays per adapter remained exact.

The integration transfers the measured `DirectWeightedNormTail.__call__` into the existing `WeightedNormTail`, inlining only its baseline-object aliases and renaming the static reader constant. Reversing those aliases exactly restores the measured method AST. Existing setup constructor is untouched: gamma/config/core ranges remain identical; no wrapper object, extra setup allocation, table duplication, or runtime diagnostic dependency is introduced.

The compute source changes only the two input self-publication lines. Raw reader publication replaces them; arithmetic/rsqrt/unpack/pack/BF16 rounding is unchanged. It reads4core+1sum physicalFP32 tiles directly, after unchanged native square/sum producers. Gamma/output/debug buffer layouts remain unchanged. `LocalSiluMultiply` is AST-identical, preserving compact table/lookup/rounding/WIDTH output. `FusedDecoder` is AST-identical, preserving all decode/public-singleton/prefill/full-kind branches. A precise inverse restoration of the method, compute constant, and appended reader reproduces the **entire** selected base AST exactly, including85ad transpose fusion in the composed candidate.

## Measured component timing and limits

Completeadapter includes core BF16 rounding/reshape/FP32 widening, native square/sum, weighted tail, compactSiLU, every runtime placement and finalWIDTH output. Inputs are actual L1 core/Z; setup/readback/output projection excluded equally. Candidate measured first.

- Traced median: baseline0.0655262964ms → candidate0.0638425583ms (about2.57% improvement). Five100-total-replay batches passed exact output checks. Candidate sample range0.063836109–0.063849101ms; baseline0.065460638–0.065654051ms.
- Eager median: baseline0.207236037ms → candidate0.190384220ms (about8.13% improvement). **Warmup trend remains:** candidate samples0.249738,0.203369,0.190384,0.186728,0.180966ms decrease across all five samples; baseline0.261149,0.221835,0.207236,0.199030,0.200443ms also trends downward. Three warmups did not establish an eager plateau. These are observed medians, not a definitive steady-state eager gain. Whole-decoder repeated timing must decide selection.

Parent separately qualified85ad transpose source with full layer34 cached8192+33 correctness and S25610prefill/60decode: PF478.206970263ms versus44f87484.135794919ms (1.22% improvement); decode1.449556323 versus1.450283686ms is noise-compatible with its unchanged decode graph. These results concern85ad only, not either new weighted-tail candidate. Original transpose attention7.247840ms anomaly remains preserved and cause-unproven.

## Footprint and source-audit contract

Old transient input-shard storage:4×4096 +1×4096 =20,480bytes on core0. New local input CBs: exactly4/1FP32 pages =20,480bytes on core0. Gamma, weighted output, debug output shapes and compact table29,952bytes/core are unchanged. Persistent addedL1/DRAM bytes are zero; state/KV allocation and public context contract are unchanged. A reader kernel and descriptor/runtime metadata are new; no claim of byte-identical total program-metadata footprint is made. Parent capacity qualification remains the final physical-capability gate.

New helper AST: `ea3a9bd0d5f1c673822f6454a0535f025af17659d1ac64871a6c9a4ac893d2f4`; new call AST `0bb1cae71b4aed64e5d8311ce6b080070c0e29e2bca3c3e5faddb875ed03bfdf`. Existing constructor AST `582b69dfbed36c930b713be87cf143df1f0f4f57bf2c0fe50cbb127278ea6045` is unchanged and needs no new setup exemption. Source audit should pin `WEIGHTED_NORM_INTERLEAVED_READER` plus modified `WEIGHTED_NORM_COMPUTE_SOURCE` and audit the full weighted runtime. No blanket allowlist or existing checker changes are part of this task.

Rebuild fresh outputs with `--compose` for85ad or omit for44f87:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_build_weighted_norm_interleaved_snapshot.py \
  --compose --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_tail_composed_fresh.py.snapshot \
  --report models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_tail_composed_fresh_build.json
```

Whole-decoder exact/PCC/semantic checks, representative-kind coverage, warmed/traced performance and capacity are pending parent qualification. Component exactness and fewer dispatches alone do not select this candidate.
