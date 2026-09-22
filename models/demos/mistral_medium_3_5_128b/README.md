# Mistral-Medium-3.5-128B — sequence-parallel chunked prefill on Blackhole Galaxy

Prefill bring-up of the text stack of `Mistral-Medium-3.5-128B` (`Mistral3ForConditionalGeneration`,
text config `ministral3`) on a 32-chip Blackhole Galaxy, following
`method/MODEL_BRINGUP_RECIPE.md` through P2. The graded output is per-layer K/V PCC against a CPU
golden trace, one-shot and multi-chunk.

Scope is prefill only. Decode, serving (adapter, manifest, the two-process runner), KV migration
and perf tuning are out of scope and absent, not stubbed — see [Known gaps](#known-gaps).

## Model

Everything here is read from the checkpoint's `config.json` (vendored at `reference/config.json`)
and asserted at build time, not configured; the prefill spec carries only the target hardware,
parallelism, chunking, dataformats and PCC thresholds.

| | |
|---|---|
| Layers | 88, all dense — no MoE, no routed or shared experts |
| Hidden | 12288 |
| Attention | GQA, 96 Q heads / 8 KV heads (group 12), `head_dim` 128 |
| MLP | SwiGLU, intermediate 28672, no biases |
| Norm | plain Mistral RMSNorm, eps 1e-5, pre-attention and pre-MLP |
| Vocab | 131072, untied `lm_head` |
| RoPE | YaRN, theta 1e6, factor 64, original context 4096, `beta_fast` 4 / `beta_slow` 1 |
| Context | 262144 max position embeddings |
| Checkpoint | fp8 e4m3 with a **per-tensor scalar** `weight_scale_inv`, on 7 projection suffixes |

Not present, and so not implemented: QK-norm, attention biases, sliding-window attention, MoE.
The checkpoint also carries a 48-layer Pixtral `model.vision_tower`; the loader's
`model.language_model.` prefix excludes it, and `tests/unit/test_checkpoint_loader.py` asserts both
that the tower is there and that no name the loader builds can reach it.

`config.json`'s `rope_parameters.llama_4_scaling_beta` is 0, which makes the Llama-4 Q-scale
factor identically 1.0 for this checkpoint — the reason the D1 row for it is skipped rather than
implemented.

## Topology and parallelism

| | |
|---|---|
| Mesh | 8x4 `bh_galaxy`, opened once per test session, `l1_small_size=1152` |
| SP | 8, on the mesh **rows** (axis 0) — the sequence dimension |
| TP | 4, on the mesh **cols** (axis 1) — heads, MLP intermediate, vocab |
| Fabric | `FABRIC_1D`, 1d/linear CCL topology, `num_links=2` |

Activations are `[1, 1, tokens/8, hidden_size]`: the sequence is SP-sharded across the rows and
`hidden_size` is **replicated** across the TP cols. Weights are replicated across the SP rows and
sharded only over TP=4, which is why the per-chip weight footprint is the model total / 4 and not
/ 32 — the constraint that sizes this bring-up (see [Memory](#memory)).

Dataformats, from the spec:

| Tensor | Format |
|---|---|
| Activations | `bfloat16` |
| Weights (attention, MLP gate/up/down) | `bfloat8_b` |
| KV cache | `bfloat8_b` |
| Embedding table | `bfloat16` — a deviation, see [Known gaps](#known-gaps) |

The KV cache is block-cyclic over the SP rows with a period equal to the chunk size;
`tt/attention/kv_cache.py` is the single description of that layout and
`cache_row_index` is its inverse permutation, used by every read-back in the package. Capacity is
rounded **up** to a whole number of chunk periods: the spec's 262144 is not a multiple of 5120
(`262144 % 5120 == 1024`), so the allocation is 266240 tokens (52 chunks).

## Architecture of the package

```
reference/     torch ground truth — modeling.py (the CPU model), checkpoint.py (fp8 loader +
               the lazy CheckpointStateDict), golden.py (the prepared CPU trace), model_config.py
               (config.json + the binding prefill spec), regenerate.py
tt/            the device implementation — model.py, layer.py, mlp.py, rms_norm.py, embedding.py,
               lm_head.py, rope.py, ccl.py, config.py, runtime.py (the P2 chunk runtime),
               attention/{prefill,dense_sp,operations,kv_cache,weights,config}.py
tests/         conftest.py (session mesh), device_utils.py (sharding + read-back conventions),
               unit/ (D/M/P block tests), galaxy_prefill_kv_pcc.py (the P1/P2 row),
               test_prefill_acceptance.py (the pipeline acceptance contract)
utils/         substate.py (the state-dict sub-tree hook), fabric_env.py, general_utils.py
bringup_log.jsonl   the recipe's append-only process log
```

Two seams are worth knowing before reading any PCC number in this file.

**HF half-split vs Meta interleaved.** `tt/attention/weights.py` permutes the `q_proj` / `k_proj`
rows at load time so RoPE can be applied with the interleaved (Meta) convention the ttnn op
implements. Device K is therefore Meta-interleaved while the reference and the golden trace store
it HF half-split. `GoldenTrace.layer_kv_meta()` is the one place that permutation is undone for a
comparison; comparing raw correlates at ~1/head_dim (measured 0.0148). V is unrotated and needs no
permutation.

**Causal-prefix equivalence.** A query at position `p < n` attends exactly keys `0..p` whether the
run is `n` tokens or 10240, so the first `n` positions of a 10240-token golden trace are the
correct answer for an `n`-token run. Every reduced-length diagnostic in this package relies on
that; acceptance does not, because it runs the whole trace.

## Numerics

`pcc_target` 0.99 is what the bring-up aims at; `pcc_lower_bound` 0.85 is what every test asserts.
Values are Pearson correlation in fp64 against the torch reference (D/M rows) or the CPU golden
trace (P rows).

### Decoder and whole model, random weights (D1-M3)

| Row | Measured PCC | Before the fidelity fix |
|---|---|---|
| `rms_norm` | 0.99999 | 0.99999 |
| SwiGLU | 1.00000 | 1.00000 |
| dense MLP | **0.99992** | 0.99950 |
| attention, SP | **0.99992** | 0.99663 |
| attention, replicated | **0.99994** | — |
| decoder layer | **0.99878** | 0.99606 |
| decoder layer, chunked | **0.99874 / 0.99862** | 0.99623 / 0.99568 |
| embedding, 1D | bit-exact vs `torch.gather` | bit-exact |
| embedding, 2D | **wrong** — see [Known gaps](#known-gaps) | "bit-exact" (untested length) |
| `lm_head` | 0.99982 | 0.99982 |
| whole model hidden state | 0.99076 |
| whole model KV, layer 0 | k 0.99978 / v 0.99979 |
| whole model KV, layer 1 | k 0.99584 / v 0.99586 |
| chunked vs one-shot | 0.99529 |
| logits | 0.99054 |

The whole-model rows run full width (12288 / 96-8 / 28672 / 131072 on 8x4) at a reduced **depth**
of 2 layers: a full-depth random-weight torch reference is ~250 GB of host bf16 and hours of CPU
forward. Full depth is measured against the golden trace instead, which is what P1/P2 are.

**Read the two columns together — that is the point of keeping the old one.** Every "before" value
above passes a 0.99 bar, and the model built from them measured `v 0.552` at 88 layers. Two defects
hid behind those numbers, and neither was visible in any per-block row:

1. Every projection matmul omitted `compute_kernel_config`, so it ran at ttnn's default `LoFi` with
   `math_approx_mode=True` — the *lowest* fidelity the hardware has, not a neutral setting. One
   matmul at LoFi still correlates 0.999; 88 layers of it do not. `tt/compute.py` is now the single
   default and the "after" column is the same tests re-run.
2. The 2D embedding corrupted 0.7% of token rows (below), which no block row indexes deeply enough
   to see.

The general lesson, recorded in `tt/compute.py` and `bringup_log.jsonl`: **a per-block PCC cannot
see a fixed per-layer bias, and depth is the only test that can.** A 0.1%/layer error is invisible
at one block and fatal at 88.

### Real weights against the CPU golden trace (P1-P2)

Trace: `/mnt/models/mistralai/Mistral-Medium-3.5-128B/golden/synthetic_10240`, 10240 tokens, all
88 layers, bf16.

Full depth and full width: 88 layers, hidden 12288, 96/8 heads, intermediate 28672, 10240 tokens,
SP=8 x TP=4 on the 32-chip `bh_galaxy`, `FABRIC_1D` with `Topology.Linear`. Real weights streamed
from the fp8 checkpoint and cast to the spec's `bfloat8_b`. P2 is the same trace in 2 chunks of
5120, reading its own KV cache back for the second chunk.

| layer | P1 one-shot k / v | P2 chunked k / v |
|---|---|---|
| 0 | 0.999971 / 0.999930 | 0.999971 / 0.999930 |
| 8 | 0.999913 / 0.999486 | 0.999907 / 0.999362 |
| 16 | 0.999915 / 0.999247 | 0.999907 / 0.999091 |
| 24 | 0.999805 / 0.998669 | 0.999776 / 0.998411 |
| 32 | 0.998263 / 0.993338 | 0.998009 / 0.992356 |
| 40 | 0.988678 / 0.960640 | 0.987257 / 0.955580 |
| 48 | 0.989139 / 0.945539 | 0.987908 / 0.938795 |
| 56 | 0.988466 / 0.931132 | 0.987205 / 0.923591 |
| 64 | 0.985579 / 0.937365 | 0.984056 / 0.930688 |
| 71 | 0.989094 / **0.916214** | 0.987916 / **0.907555** |
| 80 | 0.987221 / 0.946105 | 0.985857 / 0.940845 |
| 85 | **0.982830** / 0.946320 | **0.981013** / 0.941701 |
| 87 | 0.996390 / 0.987711 | 0.996055 / 0.986654 |
| **worst of 88** | **k 0.982830 (L85) / v 0.916214 (L71)** | **k 0.981013 (L85) / v 0.907555 (L71)** |

Both **pass** the spec's `pcc_lower_bound` 0.85 with margin. Both are **below `pcc_target` 0.99**
in the second half — 33/88 layers for k and 54/88 for v one-shot, 42 and 55 chunked. Why:

* **Layer 0 is at the precision floor and has nothing left to give.** Recomputing layer 0 on CPU
  with the spec's `bfloat8_b` weights and bf16 activations measures k 0.9999929 / v 0.9999479
  against the same trace. The device measures 0.999971 / 0.999930. There is no device-side error
  at the entry to the stack; what the table shows is 88 layers of that floor compounding.
* **The bf16 CPU reference does the same thing**, just less: run against the same trace at full
  depth it reaches worst k 0.99494 / v 0.98676 — it too decays with depth and it too recovers at
  layer 87. The device curve has the same shape, scaled by the extra `bfloat8_b` weight step the
  spec mandates and the reference does not take.
* **V is consistently the weaker readout.** V is the un-rotated residual projection, so an error in
  the residual stream reaches it directly; K is rotated by RoPE, which mixes channel pairs and
  partially averages the error out. Every row above shows v below k, including the ones at the
  floor.
* **Chunked is uniformly ~0.001 (k) / ~0.009 (v) below one-shot**, never above, with a maximum
  divergence of k 0.001817 / v 0.008659 across all 88 layers. That is the cost of chunk 0's KV
  going through a `bfloat8_b` cache round-trip before chunk 1 attends to it — one extra
  quantization on the path, exactly where the design puts it. The two modes agreeing to 1e-3 on k
  is the evidence that chunking is correct, not merely that it runs.

Run metadata (**not** a perf claim — nothing here was tuned and perf is out of scope): weights on
mesh in 826-827 s, compile 1.8 s one-shot / 4.3 s chunked, prefill 1.3 s / 1.4 s.

Reports: `PREFILL_ACCEPTANCE_OUT` writes the full 88-layer table as JSON for both modes.

## Memory

The binding constraint, measured rather than assumed, because two `MeshDevice` introspection APIs
(`num_dram_channels()` / `dram_size_per_channel()`, and `allocator_statistics()`) do not exist in
this build; the number below comes from allocating 1 GiB DRAM tensors until failure.

| | |
|---|---|
| Usable DRAM per chip | **31.83 GiB** (8 banks x 4272341376 B) |
| Per-chip weights, one layer, `bfloat8_b` | ~350.6 MiB |
| Per-chip weights, 88 layers | **~30.14 GiB** |
| Embedding table, 1D-sharded (hidden on TP only) | ~768 MiB/chip |
| KV cache, 88 slots x 10240 tokens | ~61 MB/chip |
| Total | **~30.97 GiB of 31.83 GiB** |

Under 0.9 GiB of headroom forces three choices in the full-depth runs, all of them recorded in
`bringup_log.jsonl`:

* `with_lm_head=False` — the `[12288, 131072]` head is 1.6 G parameters and produces nothing a K/V
  comparison reads. The head is measured separately at M2 (0.99982).
* the **1D** hidden-only embedding (768 MiB/chip) rather than the 2D vocab-sharded layout
  (96 MiB/chip). This spends 672 MiB/chip of a tight budget to buy correctness: the 2D path is
  numerically wrong on this mesh (see [Known gaps](#known-gaps)). It fits, and the P1/P2 runs
  above are the measurement that says so.
* **no tilized-weight cache.** At `bfloat8_b` the 88 layers are ~130 GB on disk per layout, so
  every run streams the fp8 checkpoint instead.

Host memory is not a constraint but was: a plain `state_dict` of the dequantized checkpoint is
~250 GB. `utils/substate.py` defers to a `state.substate()` method when the state object defines
one, and `reference/checkpoint.py`'s `CheckpointStateDict` implements it by dequantizing **one
layer at a time** (~2.8 GB), which falls out of scope as soon as the layer is on the mesh.

## Reproducing

Every runtime command needs the prepared environment. Export it in each shell:

```bash
export WS=<workspace>/checkout-tt-metal
export TT_METAL_HOME=$WS TT_METAL_RUNTIME_ROOT=$WS LD_LIBRARY_PATH=$WS/build/lib
cd $WS
```

Paths to the real weights and the golden trace come from the environment; both have a package
default and both accept the acceptance interface's names:

| Variable | Meaning |
|---|---|
| `PREFILL_SPEC` | the snapshotted binding spec JSON |
| `PREFILL_WEIGHTS` / `PREFILL_HF_MODEL` / `HF_MODEL` | the real checkpoint directory |
| `PREFILL_GOLDEN_TRACE` / `PREFILL_TRACE_DIR` | the CPU golden trace directory |
| `PREFILL_CHUNKED` | `0` one-shot, `1` multi-chunk |
| `PREFILL_ACCEPTANCE_OUT` | where the acceptance test writes its JSON report |

**Everything except the full-depth rows** (the D/M ladders, ~61 device tests plus the host suite):

```bash
scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests -q -s
```

**Host-only** (no hardware, no `/mnt/models` needed — the checkpoint rows skip):

```bash
PYTHONPATH=$WS $WS/python_env/bin/python -m pytest \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_reference_config.py \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_reference_model.py \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_reference_modeling.py \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_golden_cache.py \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_runtime_contract.py \
    models/demos/mistral_medium_3_5_128b/tests/unit/test_checkpoint_loader.py -q
```

**P1 (one-shot) and P2 (multi-chunk)** — full depth, full width, real weights:

```bash
PREFILL_CHUNKED=0 scripts/run_safe_pytest.sh \
    models/demos/mistral_medium_3_5_128b/tests/galaxy_prefill_kv_pcc.py -q -s
PREFILL_CHUNKED=1 scripts/run_safe_pytest.sh \
    models/demos/mistral_medium_3_5_128b/tests/galaxy_prefill_kv_pcc.py -q -s
```

**Acceptance**, both modes, same trace:

```bash
for m in 0 1; do
  PREFILL_CHUNKED=$m PREFILL_ACCEPTANCE_OUT=/tmp/acceptance_$m.json \
    scripts/run_safe_pytest.sh \
    models/demos/mistral_medium_3_5_128b/tests/test_prefill_acceptance.py -q -s
done
```

The full-depth rows carry `@pytest.mark.timeout(7200)`: the repo's `pytest.ini` sets
`timeout = 300` for every test, and streaming 88 layers onto 32 chips exceeds that before the first
token is embedded. The mark is a hang guard, not a perf budget.

Two diagnostic knobs, read **only** by `galaxy_prefill_kv_pcc.py` and never by the acceptance test:
`MISTRAL_PREFILL_LAYERS` and `MISTRAL_PREFILL_SEQ` reduce depth and length so a wiring change can
be smoke-tested in a couple of minutes. A run with either set logs itself as a diagnostic and is
not a P1/P2 result.

## Known gaps

* **Full-depth PCC clears `pcc_lower_bound` 0.85 but not `pcc_target` 0.99.** Worst v is 0.916
  one-shot / 0.908 chunked at layer 71. [Numerics](#real-weights-against-the-cpu-golden-trace-p1-p2)
  argues this is accumulated precision under the spec's `bfloat8_b` weights rather than a
  localized error — layer 0 sits at the CPU-measured floor, and the bf16 CPU reference decays with
  the same curve shape. That argument is not the same thing as a demonstration: nobody has shown
  this model reaching 0.99 at depth 88 on this hardware at the spec's dataformats, and this
  package does not. The experiment that would settle it is a `bfloat16`-weight full-depth run,
  which does not fit in 31.83 GiB/chip.
* **The 2D vocab-sharded embedding is numerically wrong on this mesh, and the package no longer
  uses it.** Its closing SP reduce-scatter corrupts local sequence indices **1088..1099 of every
  one of the 8 shards** — 76 of 10240 rows, ~0.7%. The damage is positional: it is independent of
  the token ids and of the table contents, and the bad rows come back either exactly zero or
  holding an unrelated table row. That is what rules out the sentinel shift/clamp arithmetic in
  `_lookup_2d`, which would mis-map as a deterministic function of the id.

  It is a `ttnn` collective-level defect, below this package, so it is documented rather than
  fixed. `DEFAULT_SHARD_VOCAB_ON_SP = False` selects the 1D layout, which measures exact;
  `MISTRAL_EMBED_SHARD_VOCAB=1` still selects 2D so the defect stays reproducible, and
  `tests/diag_embed_2d.py` is the reproduction. Cost of the workaround: 672 MiB/chip
  (see [Memory](#memory)).

  **Why it survived to the full-depth run.** `tests/unit/test_embedding_vs_ref.py` asserted both
  layouts bit-exact and passed, because it runs `SEQ = 2048` — 256-row shards, in which local index
  1088 does not exist — with ids from `torch.randint(0, 131072)`, uniform where a real prompt is
  concentrated in the low vocab. A uniform random id is the *weaker* input for a sharded lookup.
  `test_embed_real_prompt_ids` now runs the acceptance length on the trace's own ids and
  strict-xfails 2D. Impact while it was live: layer 0 measured k 0.9960 / v 0.9961 against a
  `bfloat8_b` weight floor of 0.99999, and since a wrong embedding row is wrong in the *residual
  stream*, every later layer inherited it.
* **The embedding table is `bfloat16`, not the spec's `bfloat8_b`.** `ttnn.embedding` is a gather,
  not a matmul; a block-float table would have to be dequantized to be indexed. Every other tensor
  follows the spec exactly.
* **`with_lm_head=False` in the full-depth runs.** See [Memory](#memory). The head's correctness is
  an M2 measurement (0.99982), not a P1/P2 one, and the acceptance contract grades K/V.
* **No tilized-weight cache**, so each full-depth run pays the streaming load. This is a perf
  decision and perf is out of scope; a serving deployment would want the cache and ~130 GB of disk
  per layout.
* **Whole-model random-weight rows are 2 layers deep.** Full depth on random weights has no
  tractable CPU reference; the golden trace covers it.
* **`lm_head` top-1 agreement with the reference is 0.887, not ~1.0.** Top-1 and top-2 of 131072
  near-Gaussian logits differ by ~0.2 sigma and `bfloat8_b` perturbs by ~0.028 sigma, which flips
  ~11% of near-ties. `tests/unit/test_lm_head_vs_ref.py` asserts that the disagreements *are* near
  ties rather than asserting an argmax rate.
* **Only the three runtime methods P2 asks for exist** (`compile`, `make_chunk_input`,
  `prefill_chunk`). The prefill engine's contract also has `capture_trace` (perf),
  `build_kv_chunk_table` / `kv_migration_*` (KV migration) and `set_layer_completion_sink` (the
  pipelined runner) — all out of scope, and no adapter is registered in
  `models/demos/common/prefill/adapter.py`.
* **`fp32_dest_acc_en=True` is a correctness requirement, not a tuning knob**, for `ttnn.rms_norm`
  at hidden 12288: bf16 accumulation of 12288 squares lands ~23% low, scaling the output by
  1.1378x. PCC is scale-invariant, so this is invisible to a PCC threshold — `tests/unit/
  test_norm_vs_ref.py` asserts the absolute output RMS as well. Turning it off drops the decoder
  layer from 0.99878 to 0.9034.
* **Compute-kernel fidelity is explicit everywhere, because ttnn's default is not neutral.**
  Omitting `compute_kernel_config` selects `MathFidelity.LoFi` with `math_approx_mode=True`, the
  lowest setting the hardware has. `tt/compute.py` is the package's single default (HiFi4,
  `fp32_dest_acc_en`, `packer_l1_acc`) and every projection, MLP and `lm_head` matmul passes it.
  The one documented exception is `ring_joint_scaled_dot_product_attention`, which does not
  support `fp32_dest_acc_en`; that is pinned off in `ProgramConfig.get_ring_compute_kernel_config`
  alone, rather than globally as it was before.
* **`bringup_digest.py --lint` could not be run.** The recipe's §7.6 tool
  (`models/demos/common/prefill/tools/bringup_digest.py`) does not exist in this checkout — only
  `adapter.py`, `docs/`, `runners/` and `tests/`. An equivalent inline validator was run instead;
  it reports 6 schema deviations in `bringup_log.jsonl` from earlier sessions (4 `skip` records key
  the row as `what` instead of `row`, one `verify` has `result: "partial"` outside the `pass|fail`
  vocabulary, one `source` record lacks `part`/`chosen`/`envelope`). They are **left unedited**: the
  recipe specifies the log is append-only and that no line is edited or deleted after it is written.
