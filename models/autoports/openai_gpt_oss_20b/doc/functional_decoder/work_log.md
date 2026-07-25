# Functional decoder work log

Date: 2026-07-25 UTC

## Translation source

The pre-generated EmitPy packages were read directly:

| File | SHA-256 |
| --- | --- |
| `/home/mvasiljevic/emit-gptoss/g0_prefill/main.py` | `7849914c47e92f71576a3e2536e103c47cb8335cf8b4f676cfaf7b91d8656df3` |
| `/home/mvasiljevic/emit-gptoss/g0_prefill/consteval.py` | `e6208274566107b69b8fa1d70ccd4f05f310fe691382ba0e01779d26ba934166` |
| `/home/mvasiljevic/emit-gptoss/g1_decode/main.py` | `eb2e65bc45b0b60f9963ca3fefa977de5116c9190d0537a0f29e8557afbae691` |
| `/home/mvasiljevic/emit-gptoss/g1_decode/consteval.py` | `5024dcaa993f1bd26397ba7b7945ce9083673147a111859f7ac871e0f40cd952` |

No `ir_to_emit.sh`, MLIR conversion, emit regeneration, or other graph
generation command was run.

## Segmentation and semantic checks

- Classified `g0_prefill` from `fill_cache`: batch 1, sequence 17, cache
  length 128.
- Classified `g1_decode` from `paged_update_cache` plus decode SDPA: batch 1,
  sequence 1, cache length 128.
- Counted 49 RMSNorm sites in each flat full-model graph.
- Selected middle layer 12 of 24, bounded by the two layer RMSNorm sites and
  `model.model.layers.12.*` keys.
- Segmented prefill lines 3879-4089 and decode lines 3318-3488.
- Cross-checked epsilon `1e-5`, scale `0.125`, Q-K-V fusion order, RoPE, sink
  placement, cache mutation, sliding-window behavior, residual placement,
  exact biased SwiGLU order, FP32 router, and top-4 routing against the
  Hugging Face decoder.
- Collapsed the 1x4 TP graph to full dense Hugging Face weights on a 1x1 mesh.
  All source collective placement and emitted dtypes are retained only in
  `multichip_provenance.json`.
- Reconciled every TP-local representative-layer transient in both source
  ranges. The provenance now records 19 sharded transient tensor groups in
  addition to the parameter, persistent/auxiliary, boundary, and collective
  inventories.
- Preserved HiFi4/FP32 accumulation only on the two emitted RMSNorm sites.
  Every emitted-default projection, attention, router, and expert operation
  uses the framework default compute-kernel configuration.

## Static and device commands

Static compilation and runtime audit:

```text
python -m py_compile \
  models/autoports/openai_gpt_oss_20b/tt/functional_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py

pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_runtime_contract_and_no_host_fallback \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py::test_supported_context_bound_is_enforced_before_weight_loading
```

Result: PASS, 2 tests.

Device health and bounded open/close:

```text
timeout 60 tt-smi -ls --local
TT_VISIBLE_DEVICES=2,3 timeout 60 python <1x1 open/close smoke>
```

Result: PASS. Four local Blackhole P300c chips were enumerated; the selected
endpoint opened and closed a 1x1 mesh. No reset was required.

Complete functional suite:

```text
TT_VISIBLE_DEVICES=2,3 \
HF_HOME=/home/mvasiljevic/hf-cache \
timeout 1800 pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/functional_decoder/test_results.xml
```

Result: PASS, 6 tests in 7.66 seconds.

| Test | Output PCC |
| --- | ---: |
| synthetic prefill S=17 | 0.9999918033830795 |
| synthetic prefill S=128 | 0.9999962546971618 |
| synthetic prefill S=256 | 0.9999969718379738 |
| official real prefill S=17 | 0.9933185739429331 |
| official real decode at position 17 | 0.9993172395569213 |
| official real prefill S=256 sliding primer | 0.9913088292368427 |
| official real prefill S=256 key cache | 0.9999469748427011 |
| official real prefill S=256 value cache | 0.9999538928229443 |
| official real decode position-256 key update | 0.9999342857373951 |
| official real decode position-256 value update | 0.999948804078287 |
| official real decode position-256 attention residual | 0.9994680591941338 |
| official real decode at position 256 | 0.9994801582161599 |

Capacity boundary:

```text
TT_VISIBLE_DEVICES=2,3 \
HF_HOME=/home/mvasiljevic/hf-cache \
timeout 600 python \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py 21248
```

Result:
`CAPACITY_PROBE_PASS seq_len=21248 output_shape=(1, 21248, 2880)`.

```text
TT_VISIBLE_DEVICES=2,3 \
HF_HOME=/home/mvasiljevic/hf-cache \
timeout 600 python \
  models/autoports/openai_gpt_oss_20b/tests/functional_decoder_capacity_probe.py \
  21249 --probe-above-supported-expecting-dram-oom
```

Result: `CAPACITY_PROBE_EXPECTED_DRAM_OOM seq_len=21249`. The failing
3,922,329,600-byte DRAM allocation required 490,291,200 bytes in each of eight
banks; the largest free block per bank was 457,193,856 bytes. The mesh closed
normally, and a post-probe `tt-smi` health check passed.

## AutoDebug and sliding-boundary repair

The first independent review requested a decode control beyond layer 12's
128-token sliding window. The new control prefills positions 0-255 and decodes
position 256, for which only keys 129-256 are eligible.

The initial causal/sliding call failed at the attention residual with PCC
`0.397`. Correcting the Hugging Face sliding-cache reference raised the
isolated result to `0.583`, but it still failed. Two further isolated controls
refuted cache capacity and attention-sink hypotheses:

- the prefill K/V slices for positions 129-255 matched Hugging Face above
  `0.99994`;
- an emitted-form explicit additive mask produced the same failure;
- disabling the sink in both references produced PCC `0.580`.

The `$autofix` investigation invoked fresh `$autodebug` analysis. It found
that `RotaryEmbeddingDeviceOperation::compute_program_hash` omits
`token_idx`, while the program factory bakes that value into runtime offsets
and declares no dynamic override. A decode at position 256 therefore reused
the program compiled by the earlier position-17 decode.

The retained repair selects the cosine/sine row on device and calls rotary
embedding with constant index zero. This is runtime TTNN computation only:
there is no host transfer, layout conversion, fallback, or collective.
Afterward, the newly written position-256 K/V cache rows reached
`0.9999343`/`0.9999488` PCC, attention reached `0.9994681`, and the complete
decoder reached `0.9994802`. The source diagnosis is in `AUTODEBUG.md`.

## Stage gates

- Context-contract checker: PASS for both the exact requested command and the
  HF-aware `--require-contract --strict-caps` command; target 131,072,
  supported 21,248, DRAM-limited.
- Independent stage review round 1: MORE WORK NEEDED. All three required
  findings were addressed: compute policy, beyond-window decode evidence, and
  exhaustive sharded-transient provenance.
- Independent stage review round 2: MORE WORK NEEDED on one documentation-only
  contradiction. The README now identifies Q/K/V as column-parallel, O as
  row-parallel, and all gate/up/down expert tensors as expert-parallel over
  the 32-to-8 expert axis, matching the emit and structured provenance.
- Independent stage review round 3: CLEAN PASS with no required work. The
  reviewer revalidated implementation, tests, JUnit, context checks, source
  hashes, all 16 collectives, and the parameter/transient provenance.
- Local checkpoint SHA: pending.
