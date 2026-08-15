# AutoFix proposal: precision isolation and decode-QKV DRAM reader

Date: 2026-07-28 UTC

Scope: `AUTODEBUG.md` findings 4 and 5 only. This is a source-only proposal;
no TT hardware command was run and no implementation or test file was edited.

## Starting evidence

- Checkout `b9e6c242a34011e3daeebab9207fbb5b79750f39` is the recorded stage
  base and current `HEAD`.
- `git merge-base --is-ancestor
  7aa26e4b1f274867bcea5ff6ea99295f961d89b1 HEAD` exits zero.
- Commit `7aa26e4b1f2` is the kernel-side Blackhole BF16
  DRAM-interleaved fix for `nlp_create_qkv_heads_decode`. The current program
  factory still selects its aligned scratch path when
  `input_is_dram && 16 * element_size < dram_alignment`.
- The current reader accepts arbitrary tile-aligned head widths. Gemma4's
  head dimensions 256 and 512 therefore exercise 8 and 16 head tiles,
  respectively; they are not excluded by the API.
- The decoder's one `correctness_compute_config` is HiFi4, exact math, FP32
  destination, and is passed to norms, SDPA, output/dense/router projections,
  router softmax, and expert matmuls. The recorded motivating contrast is only
  sliding layer 0 decode PCC `0.993209` at defaults versus `0.995073` with
  the blanket policy.

## Hypothesis 4: only a subset of operations needs elevated precision

### Test-only policy injection

Do not add production environment branches. In the isolation test, use
`pytest.MonkeyPatch.context()` to wrap these Python-visible TTNN operations:

| Policy group | Wrapped operation(s) |
| --- | --- |
| `norm` | `ttnn.rms_norm` |
| `sdpa` | prefill SDPA and paged decode SDPA |
| `dense_o` | `ttnn.linear` only when the weight object is `o_proj`, `mlp_gate`, `mlp_up`, or `mlp_down` |
| `router` | `ttnn.linear` with `router_proj` and `ttnn.softmax` |
| `experts` | `ttnn.matmul` |

Each wrapper must copy `kwargs`, then either replace
`compute_kernel_config` with the group's config or remove the keyword when
the policy value is `None`. Object identity (`weight is candidate`) avoids
tensor equality and distinguishes QKV, which must remain at framework
defaults. Assert that every enabled group was hit. This overrides the
blanket production keyword without changing the decoder.

For diagnosis only, wrap `_attention_decode`, `_dense_mlp`,
`_router_weights`, and `_moe_decode` and copy their returned tensors to host.
Record pairwise component PCCs between policies. These host reads are
diagnostic and must not be included in fallback-audit or latency passes.

### Adaptive matrix

Use one frozen real-weight layer-0 input, real paged prefix/cache, page table,
RoPE, and current position for every run. Reinitialize cache state identically
between policies.

| Run | norm | SDPA | dense/o-proj | router | experts |
| --- | --- | --- | --- | --- | --- |
| `D` | default | default | default | default | default |
| `H` | high | high | high | high | high |
| `A` | high | high | default | default | default |
| `B` | default | default | high | high | high |

Here `default` means the compute-config keyword is absent. `high` is the
current HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`,
`packer_l1_acc=False` policy.

Recursively split every half whose PCC improvement over `D` is at least
`0.0002`: `norm` versus `sdpa`; `dense_o` versus `router+experts`; then
`router` versus `experts`. Keep measured PCC values even when all runs pass.

For each contributing leaf, run this factorial while holding exact math and
packer settings constant:

| Fidelity | FP32 destination |
| --- | --- |
| LoFi | off |
| HiFi4 | off |
| LoFi | on |
| HiFi4 | on |

Construct each explicit policy with
`ttnn.init_device_compute_kernel_config(device.arch(),
math_fidelity=..., math_approx_mode=False, fp32_dest_acc_en=...,
packer_l1_acc=False)`. Then A/B `math_approx_mode=True/False` only at the
winning fidelity/FP32 setting, because the original blanket change also
changed approximation mode.

### Verdict and retained policy

- A narrowed policy is proven only if three identical runs are deterministic,
  final HF PCC is at least `0.995` in every run, and its minimum PCC is no
  more than `0.0002` below blanket `H`.
- A group with improvement below `0.0002` and no boundary-crossing effect is
  refuted and returns to framework defaults.
- If no narrower subset passes reproducibly, retain the blanket exception and
  record that result rather than inferring which knob matters.

### Minimum serialized hardware sequence

1. Run the exact layer-0 sliding batch-1 reproducer containing `D/H/A/B` and
   adaptive leaves. This is the only broad isolation run.
2. Run the selected leaf's fidelity/FP32 factorial, followed by exact-versus-
   approximate math.
3. Freeze the minimum policy and rerun real-weight layer 0 and layer 5 paged
   prefill/decode PCC.
4. Rerun traced decode for both layer kinds at batch 1 and batch 32. Do not
   combine this with the QKV DRAM/L1 experiment.

Suggested command selector:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q -rA \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
-k precision_policy_isolation
```

Write `precision_policy_isolation.json` with source/test SHA or dirty diff
hash, seeds, layer, batch, cache/page-table/current-position contract, full
policy ledger, final PCC, component pairwise PCCs, repeated-run values, hit
counts, and the selected minimum. The normal PCC and trace artifacts then
prove the retained policy across layer kinds and serving batches.

## Hypothesis 5: the whole-QKV L1 promotion is obsolete

### Exact-shape op test

Add a model-local test for BF16 DRAM-interleaved inputs at batch 32:

- sliding: Q heads 16, KV heads 8, head dimension 256;
- full: Q heads 16, KV heads 2, head dimension 512.

Run each three times to cover program-cache reuse. Compare every Q/K/V output
to the Torch split with PCC at least `0.9999`. Also run the same host input
from L1 as a control and require DRAM-versus-L1 PCC at least `0.9999`.

### Decoder A/B without a production branch

For the existing real-weight batch-32 traced test, monkeypatch
`ttnn.to_memory_config` only during capture. Bypass the call only when all of
these are true:

- requested config is interleaved L1;
- input config is interleaved DRAM;
- shape is `[1, 1, 32, layer_kind.qkv_width]`.

Return the input unchanged for that exact call and delegate every other call
to the original operation. Assert exactly one bypass per decoder invocation.
This produces the DRAM-reader graph without editing runtime source. Baseline A
uses the unpatched decoder and therefore promotes to L1.

For both layer kinds require:

- HF PCC at least `0.995` for A and B;
- A-versus-B PCC at least `0.9999`;
- deterministic repeated replay;
- no watcher error in the later final watcher gate.

Measure at least 20 warmed traced replays and record median and p95. Latency is
supporting evidence, not a correctness gate; investigate a B regression above
5% before removal. The removed whole-tensor L1 allocations are 524,288 bytes
for sliding (`32 x 8192 x BF16`) and 655,360 bytes for full
(`32 x 10240 x BF16`), before allocator overhead.

Suggested serialized selectors:

```bash
pytest -q -rA \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
-k qkv_head_split_dram_exact_shapes

GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q -rA \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
-k qkv_dram_l1_decoder_ab
```

Write `qkv_head_split_dram_sliding.json`,
`qkv_head_split_dram_full.json`, and
`qkv_dram_l1_decoder_ab_{sliding,full}.json`. Include commit-fix ancestry,
input/output shapes, dtype and memory configs, per-output PCC, cross-policy
PCC, repeatability, bypass hit count, warm latency statistics, and calculated
L1 bytes removed.

### Source cleanup after both A/Bs pass

Apply only this runtime deletion; the output head sharding and later
RoPE/SDPA-required L1 layouts are unrelated and remain:

```diff
         xqkv = ttnn.linear(x, self.weights.qkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
-        # Blackhole's interleaved DRAM reader for nlp_create_qkv_heads_decode
-        # corrupts later batch rows (tt-metal #16667). The L1 reader path is
-        # required for correct batch-32 decode and remains untuned/interleaved.
-        xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
         qkv_head_mem_config = _make_decode_height_sharded_memory_config(
```

Then rerun the unpatched real-weight batch-32 traced cases, the final watcher
gate, and the normal traced latency evidence. If either exact-shape DRAM op
case fails, retain the promotion and record the kernel gap instead of applying
the cleanup.

## Current status

Both hypotheses remain unverified until the serialized hardware A/Bs run.
The ancestry and source/API evidence make the QKV cleanup likely, while the
minimum precision policy cannot be predicted responsibly from the one existing
whole-layer contrast.
