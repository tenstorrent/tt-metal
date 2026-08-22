# Mistral-Small-4-119B prefill: test suite wiring — final report

Branch `ssalice/mistral4-119b-prefill`, cherry-picked onto `kmabee/prefill-shared-fixes` as
`ssalice/mistral4-tests`. Hardware: 32-chip Blackhole galaxy, 8x4 mesh, SP=8 (axis 0) / TP=4 (axis 1).
Link health at sweep start: 297 UP / 87 DOWN.

## 1. Headline

The wiring is done and mostly green, but **the most valuable output is a failure**: the model's MoE
router is softmax and this stack has no softmax router, and the resulting error accumulates at a
steady ~0.0355 PCC per layer. On a 36-layer model that is not a tolerance question. Details in §5.

## 2. Scope: what was triaged

`pytest --collect-only` over `models/demos/deepseek_v3_d_p/tests/` finds **192 test functions /
97,109 parametrizations / 71 files**. Classification (full detail in `TRIAGE.md`):

| bucket | count | action |
|---|---|---|
| name-hardcoded (model in the function name) | 58 | needs a new function |
| variant-parametrized (model only in IDs) | 36 | one param entry |
| model-free (no model token anywhere) | 98 | nothing to do |

Not applicable **by architecture** — the analogue of "the DSA/HCA ones for DeepSeek":
`sparse_mla/` (5 files, 137 params — mistral4 has no indexer; `resolve_has_indexer()` is
**False**, verified), `dflash_prefill/` (DeepSeek-V4-flash), `torch/test_kimi_k3_mla_reference.py`,
`didt/test_deepseek_v3_128k_matmul.py` and `pcc/test_deepseek_v3_matmul_pcc.py` (hardcoded
DeepSeek 7168-dim matmul shapes).

## 3. Tests added

### New test functions (4)
| test | status |
|---|---|
| `test_prefill_block.py::test_mistral4_prefill_block` | **PASS** (dense + moe) |
| `test_kv_cache_table.py::test_mistral4_kv_cache_table` | **PASS** |
| `pcc/test_ttnn_moe.py::test_mistral4_moe` | **PASS** (by +0.0015) |
| `test_prefill_transformer.py::test_mistral4_prefill_transformer` | smoke **PASS** / pcc **FAIL** |

### Variant added to existing axes (7 files)
`test_mla.py::test_mla_chunked_prefill` (**PASS**, +1404 params) · `cache/test_mla_cache.py` ·
`pcc/test_parallel_embedding.py` · `op_unit_tests/test_prefill_dispatch.py` (**PASS**) ·
`op_unit_tests/test_prefill_combine.py` · `op_unit_tests/test_ttnn_dispatch_combine.py` (**PASS**) ·
`op_unit_tests/test_reduce.py`

### New torch/CPU reference package
`models/demos/deepseek_v3_d_p/reference/mistral_small_4/` — the GLM-5.1 composed-reference pattern,
because all three `reference_*_cls` adapter hooks are genuinely unwirable for this model (signature
mismatches, documented in the adapter). Its router calls HF's own
`Mistral4MoE.route_tokens_to_experts` rather than re-implementing it, and was validated on CPU
against HF `Mistral4MoE` at **PCC 1.0000 (fp32) / 0.9999953 (bf16)**. Also provides
`mistral4_torch_config` (namespace → real `Mistral4Config`), `unpack_stacked_expert_weights` for the
packed `[128, 4096, 4096]` `gate_up_proj`, and `llama4_attn_scale` behind a default-OFF flag.

### Deliberately NOT wired
`pcc/test_moe_gate_prefill2d.py`. An entry there would validate the device gate against a
sigmoid + `noaux_tc` reference the model never uses — a green PCC on the wrong routing rule. Both the
host reference (`tt/moe/validation_helpers.py:27`) and the device op
(`moe_grouped_topk.cpp:23`) hard-reject anything but `sigmoid`/`sqrtsoftplus`, so a correct entry
needs a new bias-free softmax branch, i.e. new mechanism rather than test wiring.

## 4. Results table

| # | test | selector | result | key numbers |
|---|---|---|---|---|
| 1 | `test_mistral4_mla` | `-k 8x4` | **4 PASS** | out PCC 0.9981–0.9993; KVPE 0.9999 |
| 2 | `test_mistral4_kv_cache_table` | — | **1 PASS** | kvpe latent 320 wide, chunk 10880 B |
| 3 | `test_mla_chunked_prefill` | `mistral4 … cpu` (3 sub-8192) | **3 PASS** | out PCC 0.9991–0.9996 |
| 4 | `test_mistral4_prefill_block` | dense | **PASS** | 0.999878 |
| 5 | `test_mistral4_prefill_block` | moe | **PASS** | 0.995178 (thr 0.98) |
| 6 | `test_mistral4_moe` | — | **PASS** | reference_output 0.972469 (thr 0.971) — **+0.0015** |
| 7 | `test_mistral4_prefill_transformer` | smoke, 2L + 5L | **PASS** | forward completes |
| 8 | `test_mistral4_prefill_transformer` | pcc, 2 layers | **FAIL** | lm_head 0.928376 (thr 0.99) |
| 9 | `test_mistral4_prefill_transformer` | pcc, 5 layers | **FAIL** | lm_head 0.769880 |
| 10 | `test_ttnn_dispatch_combine` | `mistral4-640-avg` | **8 PASS** | — |
| 11 | `test_prefill_dispatch` | `mistral4-perf_no_pcc and mesh-8x4` | **8 PASS** | — |
| 12 | `test_prefill_dispatch` | `mistral4-pcc and mesh-8x4` | 12 FAIL | **pre-existing**, see §6 |
| 13 | `test_prefill_combine` | `mistral4 and mesh-8x4` | 32 SKIP | **pre-existing**, see §6 |
| 14 | `test_reduce` / `test_mla_cache` / `test_parallel_embedding` | `-k mistral4` | 8 SKIP | mesh unreachable, see §6 |

**Genuine mistral4 passes: 25 cases. Genuine mistral4 failures: 3 (rows 8–9), all the same root cause.**

## 5. The failure that matters

`test_mistral4_prefill_transformer` per-stage PCC, 5 layers, seq 5120, threshold 0.99:

| stage | PCC | Δ per layer |
|---|---|---|
| `embed` | **1.000000** | — |
| `layer_0` | 0.975813 | −0.0242 |
| `layer_1` | 0.942922 | −0.0329 |
| `layer_2` | 0.906779 | −0.0361 |
| `layer_3` | 0.870295 | −0.0365 |
| `layer_4` | 0.834688 | −0.0356 |
| `norm` | 0.834414 | −0.0003 |
| `lm_head` | 0.769880 | −0.0645 |

Three controls make this a diagnosis rather than a suspicion:
- **`embed` is exactly 1.000000** → embedding, SP/TP sharding, snapshot labelling and the comparator
  are all exact. Every bit of error originates inside the decoder layers.
- **`layer_0` / `layer_1` are bit-identical between the 2-layer and 5-layer runs** → deterministic,
  not run-to-run noise.
- **The marginal loss converges to ≈ −0.0355/layer** after a two-layer transient.

**Cause.** The real model routes with `softmax` (`transformers/models/mistral4/modeling_mistral4.py:226`).
The device gate is `sigmoid`-only — `moe_grouped_topk.cpp:23` `TT_THROW`s on anything else, and
`Mistral4Small119BConfig` declares no `SCORE_FUNC` so it silently takes the `sigmoid` default
(`tt_moe_gate_prefill.py:118`). With the correction bias zeroed, top-4 *selection* still matches
(both scoring functions are monotone in the logit); the top-4 *weights* never do, because after
`norm_topk_prob`, `sigmoid(l_i)/Σsigmoid ≠ exp(l_i)/Σexp`. Context-length-independent.

**Why the one-layer tests didn't catch it.** A single layer costs only ~0.005 PCC
(block test: 0.995178), which passes every threshold in the suite. `test_mistral4_moe`'s
softmax-reference check passes by **+0.0015** — noise. So a green one-layer result is not evidence
the router is correct; the effect only becomes visible over depth.

**Fix (not attempted, deliberately).** A `softmax` `score_func` in `moe_grouped_topk.cpp`, the
matching host branch in `validation_helpers.py`, and `SCORE_FUNC = "softmax"` on
`Mistral4Small119BConfig`. That is C++ kernel work on an op shared by DeepSeek / Kimi / GLM, so it is
a major change, not a test fix. No threshold was lowered, nothing was xfailed, and the reference was
not switched to sigmoid to manufacture green.

## 6. Failures that are NOT mistral4's

Each verified against the **dsv3 baseline** rather than assumed:

- **`test_prefill_dispatch` `-pcc` params → `ZeroDivisionError`.** The file states its own assumption
  (`:440-444`): *"this op test runs on at most 8 chips."* `-pcc` scales experts by `//16`, giving
  mistral4 8 experts over 32 chips = **0 experts/chip** at `tt/moe/init_helpers.py:245`.
  `dsv3-pcc` on `mesh-8x4` fails **identically** (256//16 = 16, also 0). The `-perf_no_pcc` params
  (`//4` → 1 expert/chip) run and pass. Log: `diag_dsv3_dispatch_8x4.log`.
- **`test_prefill_combine` on 8x4 → all skipped.** Its 8x4 entry requires
  `FABRIC_2D_TORUS_XY`; this galaxy has no wrap on the 4-wide TP axis. `dsv3` skips 32/32 with the
  identical message. Log: `diag_dsv3_combine_8x4.log`.
- **`test_reduce` / `test_mla_cache` / `test_parallel_embedding` → skipped.** Their mesh axes offer
  only 4- and 8-chip shapes; on Blackhole those report *"Blackhole only supports 32-device mesh
  configs"*. The mistral4 entries are correct and collect — they are simply unreachable on a 32-chip
  galaxy and would exercise on a Wormhole T3K or a smaller carve.

## 7. Could not complete

| item | why | what unblocks it |
|---|---|---|
| `test_prefill_block_chunked.py` | needs a **golden prefill trace** (a recorded 56320-token reference with 7 named tensor keys). `prefill_trace_default = None`; nothing exists. Also needs a populated weight cache. Verified it **skips cleanly** rather than erroring. | record a golden trace, set `test_prefill_trace_default` |
| `test_prefill_transformer_chunked.py` | same golden trace, plus a 36-layer `num_layers` axis. **This is also the only place ttnn op-trace (`traced`/`notrace`) lives** — needs `trace_region_size = 256 MB`, and is dense-MLA-only, which mistral4 satisfies. | same, then the trace axis is reachable |
| pretrained **MoE** anywhere | `packed_expert_checkpoint = True`: routed experts are one stacked `[128, 4096, 4096]` fp8 tensor, so the pretrained fixture loads **attention only** and `routed_expert_weights` is `None` | split the stacked tensor; `unpack_stacked_expert_weights` in the new reference package already implements the layout |
| `pcc/test_moe_gate_prefill2d.py` | softmax router unsupported on both host and device (§3) | the same `score_func` work as §5 |
| strict MLA PCC vs real `Mistral4Attention` | its `forward` needs a precomputed `position_embeddings` and names its cache `past_key_values`; `run_reference_mla` calls neither way | an adapter shim, or align `run_reference_mla` |
| perf entries in `perf/test_mla_perf.py`, `perf/test_moe_perf.py`, `perf/test_prefill_block_perf.py` | not attempted this pass — these assert against tuned per-model targets, and mistral4 has no established baseline yet. Tracy numbers in §8 are the raw material for setting one. | pick targets from §8 |

**Chunked attention itself is NOT blocked** — that was the open question and it is now answered
empirically. `ring_mla` sees the absorbed widths (`DH = 256+64 = 320`, `VDH = kv_lora_rank = 256`),
not the 128-wide per-head dims, so all five latent-V asserts in
`ring_joint_sdpa_device_operation.cpp:573-595` hold, and `test_mla_chunked_prefill` passes 3/3 on
device. Chunked mode is a plain `is_chunked` constructor kwarg (`tt/mla/mla.py:608-614`); mistral4
was single-shot only because every call site used the default.

## 8. Perf (Tracy)

No rebuild needed — `build_Release` already has `ENABLE_TRACY=ON`
(`cmake/project_options.cmake:7` defaults it ON; `build_metal.sh` only offers `--disable-profiler`,
so you opt *out*). One MLA layer, seq 5120, 8x4, signposted `MLA_START`→`MLA_END`, devices merged:

| share | op | device time | FLOP util |
|---|---|---|---|
| **60.15%** | `RingJointSDPADeviceOperation` | 1005.65 µs | — |
| 13.70% | `MatmulDeviceOperation` (6 ops) | 229.07 µs | **15.55%** weighted (5.5–27.2) |
| 11.05% | `ReduceScatterMinimalAsyncDeviceOperation` | 184.72 µs | — |
| 7.08% | `HighBwAllGatherDeviceOperation` | 118.40 µs | — |
| 6.02% | everything else (norm, concat, rope, slice, heads, KV update) | ~100 µs | — |

**Overall DRAM roofline: 2.9% (15 GB/s).** Slowest-device kernel sum 1.984 ms vs 1.063 ms fastest —
a 1.87x spread.

Reading: one ring-SDPA call is **60%** of the layer; **collectives are 18.1%**, more than matmul;
matmul is 13.7% at ~15.6% FLOP utilisation. At 2.9% of roofline this layer is bound by the SDPA
kernel and the fabric, not by bandwidth or FLOPs.

Two gotchas worth recording: a quoted `-k "a and b"` does **not** survive Tracy's re-exec (pass a
full node ID), and `tt-perf-report` silently produces an **empty** report unless you pass
`--start-signpost` *and* `--end-signpost` explicitly.

## 9. CI

Seven entries in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`, all with timeouts derived
from measured local runs: `mistral4_mla`, `mistral4_kv_cache_table`, `mistral4_mla_chunked`,
`mistral4_moe_ops`, `mistral4_prefill_block`, `mistral4_moe`, `mistral4_prefill_transformer`.
Tokens are auto-derived from the YAML, and all seven validate against
`validate_test_type_selection.py` (bogus tokens correctly rejected). Workflow help text updated.

The known-failing transformer PCC command is left **commented** with its measured numbers and the
condition for enabling it — a permanently-red nightly entry only teaches people to ignore the
pipeline.

**Two things CI wiring cannot make true yet:**
1. **Weights are not staged.** Nothing for this model exists under `/mnt/models` (verified). Both
   `MISTRAL4_HF_MODEL` and a **writable** `TT_MISTRAL4_PREFILL_TTNN_CACHE` must be provisioned; the
   adapter's `ttnn_cache_default` is `""`, so without the latter every run re-converts weights from
   fp8. Locally the checkpoint dir *and* its `tensor_cache_bh_32dev` are both read-only and empty.
2. **`owner_id` is the models-team default** on all seven entries — reassign before enabling.

## 10. Honest caveats

- Every PCC number here is from **random weights**, except `test_mistral4_mla`, which also passes
  pretrained. Pretrained MoE is blocked (§7).
- `test_mistral4_moe`'s comparison is *pessimistic*: `create_gate_weights` emits an
  `e_score_correction_bias` (σ = 0.01) the device consumes but Mistral's router lacks, and that bias
  costs more (Σ(Δw)² = 0.046) than the softmax/sigmoid difference (0.023), changing the 4th-ranked
  expert on ~40% of tokens. `test_mistral4_prefill_block` zeroes it, which is why its number is
  cleaner.
- All reference comparisons are held at **seq ≤ 8192**, where the missing `get_llama_4_attn_scale`
  is exactly 1.0. Above that, ttMLA and the reference agree with each other and not with the real
  model (7–14% on the query) — the second known gap, see `FINDINGS.md` F1.
- No `>8192` accuracy claim is made anywhere in this work.

## 11. The cherry-pick onto `kmabee/prefill-shared-fixes`

Pushed as **`ssalice/mistral4-tests`**. Six commits (two pre-existing MLA bring-up commits + the
three from this work + one merge-adaptation commit), all with original authors and byte-identical
messages.

`prefill-shared-fixes` is 324 commits and 2494 files ahead of where these were authored, so the pick
was not clean. Four adaptations were required — worth reading, because two of them would have
produced **silently failing CI jobs**:

1. **`fabric_profiles` restructuring.** Upstream replaced inline `device_params` dicts with profile
   helpers (`torus_xy_device_params`, `per_axis_topology`). `test_prefill_block.py` and
   `test_prefill_transformer.py` auto-merged with **no textual conflict** yet both raised `NameError`
   at import on the now-undefined `create_fabric_router_config`. Ported to the profile helpers;
   mesh ids `mesh-8x4` → `torus-xy-8x4`.
   **Behaviour difference worth flagging:** the profile also sets
   `reliability_mode = RELAXED_INIT`, which the inline dict did not. That is correct for an 8x4 case
   and matches every sibling test, but it is *not* the configuration the local numbers in §4 were
   measured under.
2. **Two CI `-k` selectors that matched zero nodes.** `mistral4_mla_chunked` still said
   `and fabric2d` after upstream fused two axes into one, and `mistral4_moe_ops` still said
   `mesh-8x4` where the id is now `fabric2d-torus-xy-8x4-2link`. Zero-match `-k` is pytest exit code
   5 → job failure. Both corrected and re-verified against real collected node IDs (3 and 16 nodes).
3. **`test_ttnn_moe.py`** — upstream added its own `test_kimi_k3_moe` at the same spot, plus 7 new
   `run_model` kwargs. Both test functions now coexist; `reference_fn=None` appended last so the
   dsv3/kimi paths are untouched.
4. **`test_mla.py`** — upstream had *deleted* the kimi_k3 trace skip and rewritten that docstring
   (K3 now does run `trace`). Upstream's version kept; only the mistral4 skip added, restated
   against upstream's new `variant.mla_trace_defaults` mechanism since the `discover_traces`
   filtering it originally cited no longer exists.

### Why the results in §4 do not automatically carry over

Two independent reasons a re-run on `ssalice/mistral4-tests` is required rather than optional:

- **`RELAXED_INIT`** now applies to the block and transformer tests (adaptation 1).
- **The branch needs a newer `_ttnn.so`.** Upstream's
  `tt/moe/tt_routed_expert.py:33` references `ttnn.RoutedExpertActivation.SituGlu`, which the
  build used for §4 does not expose (it has only `Silu` and `SwiGluOai`). The branch therefore cannot
  even be collected against the old binary without a shim.

### Re-run: done, and the results hold

The branch was rebuilt from scratch (`build_metal.sh --clean` then `--enable-ccache`; ~4 min compile,
0 errors, `ENABLE_TRACY=ON` preserved) and the sweep re-run against that build.

**19 mistral4 cases passed, 1 failed — the identical pass/fail pattern.** MLA output PCCs are
bit-identical; everything else reproduces to 4+ decimals, so `RELAXED_INIT` and the 324 upstream
commits are numerically neutral for this model. The one exception is `lm_head`, which *improved* by
0.0144 (0.928376 → 0.942780) from an upstream change to the lm-head path; it still fails its 0.99
gate and the per-layer accumulation is unchanged. Full side-by-side in
`test_logs/RESULTS.md`; logs in `test_logs/on_mistral4_tests/`.

So §4 is now **verified** on the deliverable branch, not inherited.

**One trap worth carrying forward.** With a symlinked `python_env`, `PYTHONPATH=$PWD` alone silently
loads the *main repo's* older `_ttnn.so`: the importable package is `<root>/ttnn/ttnn`, so `<root>`
yields only a namespace candidate and `PathFinder` continues on to the shared env's
`ttnn-custom.pth`. You get the wrong binary while believing you tested the new one. The three-entry
form `$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools` is required, and `run_wt.sh` asserts
`'sf-trial' in ttnn._ttnn.__file__` before running anything.

Also worth knowing: on this branch `pytest --collect-only` is **not** device-free — it opens and
starts all 32 chips for ~18 s, via `ttnn.get_num_devices()` and `is_blackhole()` at
decorator/import time.

## 12. Update: the softmax router fix already exists, and halves the error

`GateComputeMode.GPT_DEVICE` is already implemented and **is** Mistral's routing rule —
`_device_gpt_gate` (`tt/moe/tt_moe_gate_prefill.py:864-880`) does `topk(logits + bias)` then
`softmax` over the selected top-k. A k-wide softmax suffices because softmax is monotone
(`topk(softmax(l)) == topk(l)`) and the 128-wide normaliser cancels in `norm_topk_prob`.
CPU-proven max weight error **1.19e-07**, vs the current sigmoid path's **5.34e-01**.

Measured on device, seq 5120, 8x4:

| | `DEVICE_FP32` (sigmoid) | `GPT_DEVICE` (softmax) | `GPT_HOST` |
|---|---|---|---|
| single block, moe | 0.995192 | **0.998185** | — |
| 2-layer `layer_0` | 0.976044 | **0.987098** | 0.987135 |
| 2-layer `layer_1` | 0.943361 | **0.968872** | 0.968953 |
| 2-layer `lm_head` | 0.942780 | **0.968680** | 0.968950 |

**Correcting §5:** I had implied the router was *the* cause. It is a large contributor, not the
whole. The correct rule cuts per-layer error ~62% on one block and ~46% over two layers, but
~0.018 PCC/layer remains and the PCC case still fails at 2 layers.

`GPT_HOST` matching `GPT_DEVICE` to four decimals proves the residual is **not** the gate's
top-k/softmax arithmetic. It sits upstream — most likely the bf16 gate matmul changing expert
selection (CPU-measured: fp32 vs bf16 logits agree on only **98.02%** of top-4 sets).

Both gate modes are now wired as a test axis, so the wrong rule stays visible as a regression
witness rather than being silently replaced.

**Next step, and it looks small:** there is no `GPT_DEVICE` + fp32-logits mode. `DEVICE_FP32` is
fp32 logits with the wrong rule; `GPT_DEVICE` is the right rule with bf16 logits. A
`GPT_DEVICE_FP32` that typecasts before `ttnn.topk` — mirroring what `DEVICE_FP32` already does —
would test the bf16-selection hypothesis directly. The production fix remains a `score_func =
"softmax"` (implemented as `exp`) in `moe_grouped_topk`, ~15 additive lines, which also keeps
padding-aware dispatch and `route_scale` that `GPT_DEVICE` silently drops.

## 13. CI readiness — what actually has to happen before merge

A separate audit found the entries would **not** have gone green. Fixed in this branch:

| # | problem | fix |
|---|---|---|
| 1 | Both checkpoint-dependent entries would attempt a **gated 113 GB HF download** — weights resolve env → `default_local_path` → `shared_path` → `snapshot_download`, and mistral4 sets neither middle option. The fixture raises; it does not skip. | `mistral4_mla` narrowed to `-k "8x4 and random"`; `MISTRAL4_HF_MODEL` **deleted** from `mistral4_mla_chunked` — merely exporting it flipped that entry to pretrained |
| 2 | `test_mistral4_mla` **hard-asserts** on CI without a host-reference cache, and every selected case is `max_sl` + `check_pcc` | added `MISTRAL4_MLA_REF_CACHE`, mirroring the existing `Blaze - MLA` entry |
| 3 | Four entries exported vars they never reach | deleted; five of seven need **nothing** staged |
| 4 | Timeouts were warm-local guesses | tightened to measured values, 137 → 76 min |
| 5 | Header overstated the ttnn weight cache | corrected — none of the seven ever writes it |

### The one blocker I did not fix, because it isn't mine

**`.github/time_budget.yaml` `models.demo.bh_sc1` must be raised.** The gate runs in
`load-test-matrix`, which `multihost-tests` depends on, so an over-budget matrix **blocks every leg
in the pipeline** — including all pre-existing DeepSeek/Kimi/GLM/MiniMax entries, on every nightly.

| branch | requested before | after tightening | budget | required |
|---|---|---|---|---|
| `ssalice/mistral4-119b-prefill` | 572 | **511** | 437 | **511** |
| `ssalice/mistral4-tests` | 616 | **555** | 488 | **555** |

The budget was already at 435/437 before any of this, so no trimming on my side closes it. Last set
by Lukasz Galas (#50977) on the older base and Janko Mitrovic (#52008) on the newer one — their call.

### Still needs a human with `/mnt/models` write access

Nothing in the repo provisions `/mnt/models`; the pipeline only asserts it is readable
(`.github/actions/setup-multihost-job/action.yml:163-168`). Existing caches were placed by hand by
individuals. Two files need staging for the MLA entry: `random_seq5120.pt` and `random_seq25600.pt`
(45 MB + 226 MB), currently at `/tmp/mistral_small_4_119b_mla_ref_cache/`. **Question to ask the
runner owners:** is `/mnt/models` writable from a blaze worker pod, under what uid, and what is the
naming convention for a new model plus its caches?

### And fix `owner_id` before merge
All seven entries carry `U08RL15T4N8` as a placeholder. It only mis-routes Slack on failure (and is
gated to `refs/heads/main`), so it will not fail a job — but it is wrong.

### Closed: `transformers` was not the risk I thought
`tt_metal/python_env/requirements-dev.txt:33` already pins `transformers == 5.12.1` **on main**, and
the CI dev image builds from it, so the `Mistral4*` classes the new reference package imports are
present. No entry is affected.
