# Gemma 4 31B full-model revalidation on 2x Blackhole p300 (4 chips)

Date: 2026-08-14 UTC
Scope: host move only. This records that the Stage 06 full-model surface runs
correctly on different hardware and establishes the perf numbers that later
stages on this host must compare against. It is not a new stage verdict and it
does not change any stage's completion status.

## Why this exists

Every recorded perf and capacity number for Stages 03-11 was measured on four
Blackhole **P150b** boards with a 13x10 worker grid. This host is different
hardware, so the recorded numbers are not valid comparands for work done here.

| | Recorded evidence host | This host |
| --- | --- | --- |
| Boards | 4x Blackhole P150b | 2x Blackhole p300 (`p300c`), 4 chips |
| Mesh | `MeshShape(1,4)`, TP4, `FABRIC_1D` | unchanged |
| Worker grid | 13x10 | **11x10** |
| DRAM views/chip | 8 | 8 |
| DRAM bytes/chip | 34,225,520,640 usable | unchanged |
| Host | reservation container `spawner-exp-d-gemma31` | `qb2-120-p02t03`, 249 GB RAM, 16 physical cores |

Per-chip DRAM geometry is identical (8 views, `dram_grid_size` 8x1), so the
Stage 06 capacity accounting in `../../context_contract.json` carries over
unchanged. The worker grid is **two columns narrower**, which is the only
material hardware difference for program configs and shard geometry.

## Core-grid audit

No code change was required. Every core grid on the full-model path is either
derived at runtime from `mesh_device.compute_with_storage_grid_size()` or is a
fixed grid that fits inside 11x10:

| Site | Grid | Fits 11x10 |
| --- | --- | --- |
| `tt/decode_head_grid.py` | derived; factors the active batch within the worker grid, falls back to `ttnn.num_cores_to_corerangeset(..., row_wise=True)` | yes |
| `tt/model.py` LM head / DRAM cores | derived from `dram_grid_size()` and `compute_with_storage_grid_size()` | yes |
| `tt/multichip_decoder.py` attention/MLP/QKV grids | derived from `grid.x`/`grid.y` | yes |
| `tt/multichip_decoder.py:443` | fixed `(8, 3)` | yes |
| `tt/multichip_decoder.py:988`, `tt/fused_decoder.py`, `tt/functional_decoder.py`, `tt/optimized_decoder.py` | fixed `CoreCoord(8, 4)` | yes |
| `tt/fused_decoder.py:53` | fixed `CoreCoord(11, 10)` | exactly fits |

Batch 32 decode heads factor to 8x4 on both an 11-wide and a 13-wide grid, so
the dynamic decode-head repair from Stage 11 is unaffected by the move.

`tt/fused_decoder.py:53` is the one site with no headroom: it hardcodes
`CoreCoord(11, 10)`, which exactly equals this host's worker grid. It is a
latent failure on any Blackhole part with a narrower or shorter grid. Left
as-is because it is correct here and belongs to the Stage 02 surface.

## Checkpoint provenance

The local weights are a different HF revision from the one in the recorded
evidence, but the weights themselves are identical:

- recorded revision: `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`
- local revision: `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`
- the HF tree API for both revisions lists identical LFS oids for
  `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`,
  `model.safetensors.index.json`, `config.json`, `tokenizer.json`, and
  `tokenizer_config.json`. Only `README.md` differs.
- the staged local copy was checksummed against those oids and matches:
  `186fa361...1637aac`, `b78ae829...946e6da0`, `12bac982...95fd11e6`.

`../readiness_aime24_plain.refpt` therefore remains a valid reference on this
host, and `GemmaTokenizer.chat_template` is still `None`, preserving the
plain-completion prompt contract.

Staging on this host, so `tt/model.py::_resolve_checkpoint()` works unmodified:

```text
/mnt/models/blaze/google/gemma-4-31B                # NFS source, 62,546,177,752 bytes
/home/mvasiljevic/models/google/gemma-4-31B         # verified local copy
~/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89
    -> /home/mvasiljevic/models/google/gemma-4-31B  # symlink
```

A persistent TTNN tensor cache is warmed at
`/home/mvasiljevic/models/tt_cache/gemma4_31b_full` (30 GB). Pass it as
`--tensor-cache` to skip weight reconversion; without it a full 60-layer
construction re-reads and reconverts all 62 GB.

## Two precision regimes

The harnesses do not resolve precision the same way, so the results below span
two configurations. This is existing behavior, not a change made here:

- `run_prefill_check` and `run_teacher_forcing` call `build_generator` without a
  `model_config`, which consults only `GEMMA4_31B_PRECISION_CONFIG`. That
  variable was unset, so they ran the **BF16 LM-head default**.
- `tests/run_full_model_qualitative.py --benchmark-only` falls back to
  `doc/datatype_sweep/selected_precision_config.json` when the variable is
  unset, so token-out ran the **Stage 08 selected `lm_head_bfp8_hifi2`** policy
  (confirmed by `runtime_precision.config_id` in `token_out_no_readback.json`).

Each result is therefore compared against the recorded number for its own
regime.

## Correctness

| Check | This host | Recorded comparand | Verdict |
| --- | --- | --- | --- |
| `tests/test_full_model_contract.py` | 25/25 pass | 23/23 at remediation | pass, superset |
| Reduced two-kind probe (`test_reduced_full_model_prefill_split_greedy_and_trace`) | pass, 134 s | pass | pass |
| Prefill top1/top5/top100, BF16 default | **0.910 / 1.000 / 1.000** | 0.91 / 1.00 / 1.00 (Stage 06) | matches exactly |
| Teacher-forcing top1/top5/top100, BF16 default | **0.920 / 1.000 / 1.000** | 0.91 / 1.00 / 1.00 (Stage 06) | +1 token of 100 |

The single-token teacher-forcing difference is consistent with a different
matmul core split changing BF16 accumulation order on a narrower grid. Top-5
and top-100 are saturated in both cases, so it is not a correctness regression.

## Performance baseline for later stages on this host

Token-out, full 60 layers, batch 1, 149-token prompt, 100 generated tokens,
selected `lm_head_bfp8_hifi2` policy, one discarded warmup then five recorded
repeats. Artifact: `token_out_no_readback.json`.

| Metric | This host (median) | min - max | Recorded Stage 08 post-selection | Delta |
| --- | ---: | --- | ---: | ---: |
| TTFT | **183.596 ms** | 179.804 - 188.403 | 479.707 ms | -61.7% |
| Decode, overall | **29.2096 t/s/u** | 29.1648 - 29.2320 | 24.787 t/s/u | +17.8% |
| Decode, steady | **33.9225 t/s/u** | 33.9211 - 33.9237 | 34.256 t/s/u | **-0.97%** |

**Steady decode is the honest hardware-to-hardware comparison: -0.97%.** Two
narrower columns cost essentially nothing on this path.

The TTFT and overall-decode gains are **not** hardware. Commit `18a8b3fd656`
replaced the prefill-to-decode host token readback with a device token handoff,
which both removes work from the timed window and changes what TTFT measures.
The trace counters prove it: this host records `sampled_token_readbacks: 0` and
`token_device_refreshes: 1`, where the recorded Stage 06 artifact has
`sampled_token_readbacks: 1` and no device-refresh counter. Do not quote the
TTFT or overall-decode deltas as a hardware result.

Steady-decode sample spread is 0.008%, so this baseline is tight enough to
resolve the 10-15% gap trigger that Stage 07 uses.

Teacher forcing, warmed, BF16 LM-head default. This harness reads one prediction
and writes one ground-truth token per step, so it is a separate measurement from
token-out and is not comparable to it. Artifact: `run_teacher_forcing.log`.

| Metric | This host, warmed | Recorded Stage 07 | Delta |
| --- | ---: | ---: | ---: |
| TTFT | **318.01 ms** | 841.55 ms | -62.2% |
| Traced decode | **27.89 t/s/u** | 23.15 t/s/u | +20.5% |
| End-to-end | **25.85 t/s/u** | 19.54 t/s/u | +32.3% |

The Stage 07 comparands were themselves measured on P150b, and part of the
decode gain belongs to commit `18a8b3fd656` rather than to this hardware, so
treat these as this host's baseline rather than as a hardware speedup claim.

Trace counters on every recorded sample, unchanged from the recorded contract:

```text
model_trace_replays      99
token_host_refreshes      0
token_device_refreshes    1
position_host_refreshes   2
rope_host_refreshes       2
page_table_refreshes      0
sampled_token_readbacks   0
full_logits_readbacks     0
```

## Commands

```bash
unset LD_LIBRARY_PATH TT_METAL_RUNTIME_ROOT   # /etc/profile.d/ttop.sh points these at another checkout
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib

pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py

GEMMA4_31B_FULL_MODEL_RUN_REDUCED=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_prefill_split_greedy_and_trace

python -u -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -u -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -u models/autoports/google_gemma_4_31b/tests/run_full_model_qualitative.py \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/mvasiljevic/models/google/gemma-4-31B \
  --prompt-source models/autoports/google_gemma_4_31b/doc/full_model/qualitative/prompts.txt \
  --output-dir <scratch> --benchmark-only \
  --benchmark-reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --benchmark-tokens 100 --benchmark-warmups 1 --benchmark-repeats 5 \
  --benchmark-output token_out_no_readback.json \
  --tensor-cache /home/mvasiljevic/models/tt_cache/gemma4_31b_full
```

`--mesh-device P150_X4` is only a label for `MeshShape(1,4)` in
`models/common/readiness_check/mesh_device.py`; it does not assert board type.
`models/common/modules/tt_ccl.py` independently maps a 4-chip Blackhole system
with an 8-wide DRAM grid to its `P150x4` topology entry, which is why the
multichip CCL path needs no change on p300.

## Artifacts

- `token_out_no_readback.json`: five-sample token-out baseline, per-sample
  values, resolved runtime precision, and trace counters.
- `run_prefill_check.log`: full-stack prefill accuracy.
- `run_teacher_forcing.log`: teacher-forcing accuracy and perf, warmed.
- `run_teacher_forcing_cold.log`: first run on this host, retained only to
  document the cold-JIT measurement trap described below.

## Measurement trap worth knowing

The first teacher-forcing run on a fresh checkout reported **9796.54 ms** TTFT,
**18.07** t/s/u decode, and **6.55** t/s/u end-to-end. That run compiled 115
kernels inside the timed region (`JIT cache stats: 333/448 hits`) while also
streaming 62 GB of weights over NFS.

The identical command after warming reported **318.01 ms** TTFT, **27.89** t/s/u
decode, and **25.85** t/s/u end-to-end at `JIT cache stats: 448/448 hits`: a
30.8x TTFT difference from JIT state alone. Accuracy was **bit-identical across
both runs** (0.920 / 1.000 / 1.000), which is the tell that the gap was
compilation and not computation.

Any perf number collected before the JIT cache and weight staging are warm is
not comparable to the recorded evidence, and no such number should be quoted as
a hardware result. Both logs are retained here so the trap is reproducible.

## Deliberately not done here

- No fresh autoregressive or qualitative HF/TT generation. Stage 07 owns
  refreshing AIME24 prefill, teacher-forcing, and autoregressive evidence, and
  the existing non-degenerate `autoregressive_meta.json` artifacts already
  satisfy the runner-side degeneracy gate. Regenerating them here would be
  duplicated work.
- No change to `../../context_contract.json`. Per-chip DRAM geometry is
  identical, so the Stage 06 capacity plan still holds. The known accounting
  defect in `optimized_full_model_plan` and `datatype_sweep_plan`, which record
  the 2,789,212,160 BFP8 KV **value** count as bytes instead of applying the
  17/16 physical ratio, is left for Stages 07 and 08 to fix as their own
  reviews already flagged.
- No re-run of Stages 03-05. Their per-layer latencies were measured on the
  13x10 grid and remain stale on this host. Stage 07's decoder-stack lower
  bound should be recomputed from measurements taken here rather than quoted
  from `optimized_multichip_decoder`.

## Serving stack is not covered by this baseline

Everything above exercises the model and generator path only. The serving path
is unproven on this host and is the real hardware-adaptation gap for Stages
09-11:

- `vllm` is not installed in `python_env`; `tests/test_vllm_adapter_contract.py`
  fails at import with `ModuleNotFoundError: No module named 'vllm'`. The
  `tenstorrent/vllm` `dev` checkout is staged at `/home/mvasiljevic/vllm`
  (`bf98d55`) but not installed.
- `tt/generator_vllm.py` has never run on this host. It calls
  `prepare_token_out_decode(first_input_tokens=..., ...)` by keyword, so it is
  source-compatible with the signature change in `18a8b3fd656`, but that is a
  source check and not an execution result.
- The 113,280-token serving ceiling in `../../context_contract.json` came from
  per-bank DRAM **allocator probes** on P150b, not from arithmetic. Per-chip
  DRAM geometry is identical here, so it is expected to carry, but it must be
  re-probed before being restated on this host.
- No `tt-inference-server` checkout exists here, and the nine local TTI harness
  fixes recorded in `../../tti_release/RUN_NOTES.md` were never pushed. They do
  not exist upstream: `b803374e04c2460ea3bfabec4bfed832f2af532a` returns
  `422 No commit found` against `tenstorrent/tt-inference-server`, whose
  `v0.18.0` tag resolves normally to `d5913e816ac5`. They must be rebuilt from
  the `../../tti_release/autofix/*/FIX_RESULT.md` diagnoses.

The Stage 11 mandatory Meta accuracy gate is unchanged by any of this. It needs
a canonical base-checkpoint reference that requires H200-class hardware; this
host has 16 physical cores and no CUDA or ROCm device.

## Harness friction worth fixing

`run_prefill_check` and `run_teacher_forcing` accepted no `--tensor-cache`, so
each invocation reconverted all 62 GB of weights before producing a number. Only
`tests/run_full_model_qualitative.py` took `--tensor-cache`.

Fixed in commit `840b8301c40`: `add_build_generator_args()` /
`build_generator_kwargs()` in `models/common/readiness_check/mesh_device.py` add
a shared `--tensor-cache` to `run_prefill_check`, `run_teacher_forcing`, and
`run_autoregressive`, forwarded as `tensor_cache_path`. It is strictly opt-in —
when the flag is absent the kwarg is omitted rather than passed as `None` — so
autoports whose `build_generator` does not accept it are unaffected.
`qwen_qwen3_4b` would reject an unknown key, so unconditional injection was not
safe.

Measured on this host with teacher forcing:

| Run | Wall time | Result |
| --- | ---: | --- |
| without `--tensor-cache` (warm JIT) | 932 s | TTFT 318.01 ms, decode 27.89 t/s/u |
| with `--tensor-cache` | 680 s | TTFT 312.71 ms, decode 27.90 t/s/u |

252 s saved, 27% faster, accuracy bit-identical at 0.920 / 1.000 / 1.000. The
win is bounded because `from_pretrained` always calls `_load_checkpoint_state`,
so the 62 GB safetensors read happens either way; only the torch-to-TTNN
conversion and host tiling are skipped. Eliminating the read would mean skipping
`_load_checkpoint_state` when every cache entry is present, which carries
cache-staleness risk and is model-specific; not attempted.

Full environment and troubleshooting notes for this host, including several
failure modes that only surface after a complete weight load, are in
`../../host_runbook_qb2_p300x2.md`.
