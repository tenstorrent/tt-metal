# Qwen3.5-9B Prefill Optimization — Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Qwen3.5-9B prefill TTFT on Blackhole P150 from 3.4 s to ~2.85 s by moving MLP, attention QKV, and GDN conv-state tensors from DRAM-interleaved to L1-sharded / DRAM-sharded memory configs. Every change must be sized for `chunk_size=2048` (DeltaNet) and `attn_chunk_size=4096` (full-attn) so it scales unchanged to 128 k-token contexts.

**Architecture:** Feature-flagged opt-in to L1-sharded prefill paths, gated on the chunked-prefill chunk size (not total seq length). Existing matmul/op kernels are configured with sharded memory configs and program configs; consumers in the same chunk inherit the layout to avoid implicit `to_memory_config` round-trips. PCC is verified per-sub-item against a pinned prompt set; perf is verified by re-profiling the same `traced_4k` test that produced the baseline.

**Tech Stack:** ttnn (Python wrappers + program configs), pytest, Tracy profiler, `models/demos/blackhole/qwen3_5_9b/tt/*.py`.

**Spec reference:** `docs/superpowers/specs/2026-04-29-qwen35-9b-prefill-optimization-design.md`

**Phases 2, 3, 4 deferred:** This plan covers Phase 0 (pre-flight) and Phase 1 (tactical config wins) only. Each phase is independently shippable per the spec. Phase 2 / 3 / 4 plans will be authored after Phase 1 lands and re-profiling confirms the predicted savings — the actual savings shape later-phase priorities (e.g., if Phase 1 over-delivers, Phase 3.4's high-variance GDN call-fusion investigation may be deprioritized).

---

## File Structure

**Files created:**

- `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py` — pinned-prompt PCC smoke test, runs after every sub-item
- `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_phase1_perf.py` — phase-1 perf regression test (re-runs profiler, asserts matmul-share threshold)
- `docs/superpowers/plans/phase1-baseline.json` — captured baseline metrics for diffing against post-phase numbers

**Files modified:**

- `models/demos/blackhole/qwen3_5_9b/tt/model_config.py` — add chunk-size constants (`DELTANET_CHUNK_SIZE`, `ATTN_CHUNK_SIZE`) and Phase 1 feature flags (`mlp_l1_shard`, `convstate_l1`, `attn_qkv_l1_shard`)
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py` — Phase 1.1: MLP L1-sharded path
- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py` — Phase 1.2 (DeltaNet `mega_fused_weight`) and Phase 1.3 (conv-state pad/concat)

**Files NOT modified in Phase 1** (deferred to follow-ups):

- `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py` — full-attention QKV sharding goes through `models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention.gated_attention_forward_ttnn`, which has its own `memory_config` plumbing. Deferred to its own follow-up plan (Phase 1.2b) since the change crosses module boundaries.
- GDN kernel internals (`gdn_kernel/`) — Phase 3.

---

## Task 0.1: Re-capture the baseline profile and lock the metrics

**Files:**
- Create: `docs/superpowers/plans/phase1-baseline.json`

- [ ] **Step 1: Run the profiler against `traced_4k`.**

```bash
cd /local/ttuser/atupe/tt-metal && \
TT_METAL_DEVICE_PROFILER=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_4k --capture=no
```

Expected: prefill TTFT logged at ≈ 3.4 s ± 5 %; profiler CSV written under `generated/profiler/reports/<timestamp>/`.

- [ ] **Step 2: Run the analysis script on the new profile.**

```bash
LATEST=$(ls -t generated/profiler/reports | head -1) && \
python3 .claude/skills/profiler-report-analysis/scripts/process_profile.py \
  --csv generated/profiler/reports/$LATEST/ops_perf_results_$LATEST.csv \
  --output-dir generated/profiler/reports/$LATEST/analysis/ \
  --sdpas-per-iteration 1
```

Expected: prefill kernel total ≈ 422 ms (per 4-layer block; full-pass ≈ 3.4 s extrapolated). MatMul `pct_of_kernel_time` ≈ 18 %, FPU% on matmul rows ≈ 11–15 %.

- [ ] **Step 3: Capture key metrics into `phase1-baseline.json`.**

Write this exact file:

```json
{
  "test": "traced_4k",
  "ttft_ms_target": 3400,
  "ttft_ms_tolerance_pct": 5,
  "prefill_kernel_total_ms_per_block": 422.21,
  "n_layers_profiled_per_block": 4,
  "total_layers": 32,
  "extrapolation_factor": 8,
  "key_op_pct_of_kernel": {
    "GenericOpDeviceOperation": 60.67,
    "MatmulDeviceOperation": 18.19,
    "ReshapeViewDeviceOperation": 8.16,
    "BinaryNgDeviceOperation": 3.66
  },
  "matmul_avg_fpu_pct": 13.0,
  "phase1_exit_criteria": {
    "matmul_pct_of_kernel_max": 10.0,
    "matmul_avg_fpu_pct_min": 35.0,
    "ttft_ms_max": 2950
  }
}
```

- [ ] **Step 4: Commit.**

```bash
git add docs/superpowers/plans/phase1-baseline.json
git commit -m "phase 1: pin baseline metrics from traced_4k profile"
```

---

## Task 0.2: Confirm long-context tests run before any optimization lands

**Files:**
- Read-only check on existing parameterised tests in `text_demo.py`.

- [ ] **Step 1: Run `traced_32k` end-to-end (PCC + perf, should already work).**

```bash
cd /local/ttuser/atupe/tt-metal && \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_32k --capture=no -x
```

Expected: PASSes; logs TTFT and decode tok/s. If this fails on `main` for reasons unrelated to this plan, STOP and triage before proceeding — long-context regression detection requires this test to be green first.

- [ ] **Step 2: Run `traced_128k` end-to-end.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_128k --capture=no -x
```

Expected: PASSes. If `traced_128k` is too slow or hits OOM on the workstation, fall back to running `traced_64k` and document the substitution by editing the test parameterization in `text_demo.py:171-198` (do not commit the substitution — record it in `phase1-baseline.json` under a new `long_context_test` field instead).

- [ ] **Step 3: Capture both TTFTs.**

Edit `phase1-baseline.json` and add a `long_context_baseline` block:

```json
{
  "long_context_baseline": {
    "traced_32k_ttft_s": <captured_value>,
    "traced_128k_ttft_s": <captured_value_or_null_if_substituted>,
    "long_context_test_used": "traced_128k"
  }
}
```

- [ ] **Step 4: Commit.**

```bash
git add docs/superpowers/plans/phase1-baseline.json
git commit -m "phase 1: capture long-context baseline TTFTs"
```

---

## Task 0.3: Add a pinned-prompt PCC smoke test

This is the test we run after **every** sub-item to gate merging.

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py`

- [ ] **Step 1: Write the test.**

Create `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py`:

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pinned-prompt PCC smoke test for Qwen3.5-9B prefill optimisation.

Used as a per-sub-item gate during the Phase 1 plan. Runs prefill on a fixed
prompt and asserts the top-1 logit and top-5 token IDs match a captured
golden snapshot. Cheap (single 2048-token prefill, no decode).

Regenerate the golden snapshot only when the model itself changes (not for
optimisation work). Use:
    pytest .../test_prefill_pcc_smoke.py::test_capture_golden --capture=no
"""
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model
from models.utility_functions import run_for_blackhole

CHECKPOINT_DIR = "/local/ttuser/atupe/Qwen9b"
GOLDEN_PATH = Path(__file__).parent / "prefill_pcc_smoke_golden.json"
PROMPT_TOKEN_IDS = list(range(100, 100 + 2048))  # deterministic, no tokenizer needed


def _run_prefill(device):
    device.enable_program_cache()
    model = Qwen35Model.from_pretrained(
        device,
        CHECKPOINT_DIR,
        max_batch_size=1,
        max_seq_len=4096,
        n_layers=4,
    )
    token_ids = torch.tensor(PROMPT_TOKEN_IDS, dtype=torch.int32).unsqueeze(0)
    logits = model.prefill(token_ids)
    last_logits = logits[0, -1, :].float().cpu()
    top1 = int(last_logits.argmax().item())
    top5 = last_logits.topk(5).indices.tolist()
    top1_logit = float(last_logits[top1].item())
    return {"top1": top1, "top5": top5, "top1_logit": top1_logit}


@run_for_blackhole()
def test_prefill_pcc_smoke(device):
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden file missing at {GOLDEN_PATH}; run test_capture_golden first")
    golden = json.loads(GOLDEN_PATH.read_text())
    actual = _run_prefill(device)
    assert actual["top1"] == golden["top1"], f"top1 changed: {actual['top1']} vs {golden['top1']}"
    assert actual["top5"] == golden["top5"], f"top5 changed: {actual['top5']} vs {golden['top5']}"
    # Logit value: allow 2% relative drift (BF8/BF4 quant noise per matmul re-tune)
    rel_err = abs(actual["top1_logit"] - golden["top1_logit"]) / max(abs(golden["top1_logit"]), 1e-3)
    assert rel_err < 0.02, f"top1_logit drift: {actual['top1_logit']} vs {golden['top1_logit']} ({rel_err*100:.2f}%)"


@run_for_blackhole()
def test_capture_golden(device):
    """Regenerate the golden file. Only run intentionally."""
    if os.environ.get("ALLOW_GOLDEN_OVERWRITE") != "1":
        pytest.skip("Set ALLOW_GOLDEN_OVERWRITE=1 to regenerate the golden snapshot")
    actual = _run_prefill(device)
    GOLDEN_PATH.write_text(json.dumps(actual, indent=2))
    print(f"Wrote golden snapshot to {GOLDEN_PATH}")
```

- [ ] **Step 2: Capture the golden snapshot (one time).**

```bash
cd /local/ttuser/atupe/tt-metal && \
ALLOW_GOLDEN_OVERWRITE=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py::test_capture_golden --capture=no
```

Expected: writes `models/demos/blackhole/qwen3_5_9b/tests/prefill_pcc_smoke_golden.json` containing `{"top1": ..., "top5": [...], "top1_logit": ...}`.

- [ ] **Step 3: Run the smoke test against the golden.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py::test_prefill_pcc_smoke --capture=no
```

Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py \
        models/demos/blackhole/qwen3_5_9b/tests/prefill_pcc_smoke_golden.json
git commit -m "phase 1: pin PCC smoke test for prefill optimisation work"
```

---

## Task 0.4: Add chunk-size constants and Phase 1 feature flags to `model_config.py`

This single change unlocks every Phase 1 sub-item to be feature-flagged for clean revert.

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py`

- [ ] **Step 1: Add the constants and flags.**

Edit `model_config.py`. After line 78 (the `weight_dtype` / `act_dtype` block), add:

```python
        # --- Phase 1 prefill optimisation: chunk-size constants and feature flags ---
        # These are sized for the existing chunked-prefill chunk sizes used by
        # qwen35_model.prefill_layer_chunked. Sharding decisions key off the
        # chunk size, NOT total sequence length, so 128 k contexts use the
        # same per-chunk L1 budget as 4 k contexts.
        self.deltanet_chunk_size = 2048   # qwen35_model.py: chunk_size=2048
        self.attn_chunk_size = 4096       # qwen35_model.py:194 max(chunk_size, 4096)

        # Feature flags (default OFF — flip to True per sub-item after PCC + perf validation)
        self.mlp_l1_shard = os.environ.get("QWEN35_MLP_L1_SHARD", "0") == "1"
        self.convstate_l1 = os.environ.get("QWEN35_CONVSTATE_L1", "0") == "1"
        self.deltanet_megafused_l1_shard = os.environ.get("QWEN35_DELTANET_MEGAFUSED_L1_SHARD", "0") == "1"

        # Sharding grid — Blackhole P150 has 130 worker cores; 128 = 8x16 grid.
        # Step-down fallbacks: 8x8 (64 cores), 4x8 (32 cores). See spec fallback ladder.
        self.shard_grid_rows = int(os.environ.get("QWEN35_SHARD_GRID_ROWS", "8"))
        self.shard_grid_cols = int(os.environ.get("QWEN35_SHARD_GRID_COLS", "16"))
```

- [ ] **Step 2: Run the existing test suite to confirm no regression.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS (flags default off → no behaviour change).

- [ ] **Step 3: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/model_config.py
git commit -m "phase 1: add chunk-size constants and feature flags to model config"
```

---

## Task 1.1: MLP — sharded L1 path (spec sub-item 1.1)

Highest-leverage Phase 1 change. Lifts matmul FPU% from 11–15 % to (predicted) 35–60 %.

### Task 1.1.1: Refactor `Qwen35MLP.forward` to support a sharded L1 branch

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py`

- [ ] **Step 1: Run the smoke test to confirm baseline PCC before changes.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS.

- [ ] **Step 2: Replace `forward` with a chunk-size-keyed selector.**

Edit `qwen35_mlp.py`. Replace lines 48–60 with:

```python
    def forward(self, x):
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        if T == 1:
            # Decode path: unchanged
            mc = ttnn.L1_MEMORY_CONFIG
        elif self.args.mlp_l1_shard and T == self.args.deltanet_chunk_size:
            # Phase 1.1: sharded-L1 prefill path, gated on the DeltaNet chunk size.
            # The chunk-size match is a hard invariant: this branch is only safe
            # when called from prefill_layer_chunked which loops on chunk_size=2048.
            return self._forward_sharded(x, ckc)
        elif T <= 512:
            # Pre-Phase-1 short-prefill path
            mc = ttnn.L1_MEMORY_CONFIG
        else:
            # Pre-Phase-1 long-prefill path (DRAM-interleaved, the slow one)
            mc = ttnn.DRAM_MEMORY_CONFIG

        w1_out = ttnn.linear(x, self.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, self.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        output = ttnn.linear(hidden, self.w2, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        return output

    def _forward_sharded(self, x, ckc):
        """Phase 1.1 sharded path. Activations live in L1 BLOCK_SHARDED across
        the configured grid; output is L1-sharded so downstream consumers
        (residual add, norm) inherit the layout without a DRAM round-trip.

        Per-core math (chunk_size=2048, 8x16=128 cores):
          act 2048x4096   bf16 = 16 MB → 125 KB / core
          act 2048x12288  bf16 = 48 MB → 370 KB / core
          weights stay in DRAM (interleaved on this fallback path; see Task 1.1.2 for DRAM-shard upgrade)

        Long-context invariant: this branch is only reached when T equals
        deltanet_chunk_size, so 128 k contexts hit it 64 times with identical
        per-chunk footprint. Do NOT add any seq-length-shaped allocation here.
        """
        grid_rows = self.args.shard_grid_rows
        grid_cols = self.args.shard_grid_cols
        T = self.args.deltanet_chunk_size  # invariant: caller checked T == this
        # Hidden dim and intermediate dim from model config (NOT from x.shape — keeps
        # the function pure on the chunk-size invariant).
        H = self.args.dim
        I = self.args.hidden_dim

        core_grid = ttnn.CoreGrid(y=grid_rows, x=grid_cols)
        act_in_mc = ttnn.create_sharded_memory_config(
            shape=(T, H),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        act_intermediate_mc = ttnn.create_sharded_memory_config(
            shape=(T, I),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        x_sharded = ttnn.to_memory_config(x, act_in_mc)

        w1_out = ttnn.linear(
            x_sharded, self.w1,
            activation="silu",
            compute_kernel_config=ckc,
            memory_config=act_intermediate_mc,
        )
        w3_out = ttnn.linear(
            x_sharded, self.w3,
            compute_kernel_config=ckc,
            memory_config=act_intermediate_mc,
        )
        ttnn.deallocate(x_sharded)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=act_intermediate_mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        output = ttnn.linear(
            hidden, self.w2,
            compute_kernel_config=ckc,
            memory_config=act_in_mc,
        )
        ttnn.deallocate(hidden)
        return output
```

Then add `self.args = args` to `__init__` if not already stored. Look at current `__init__`: it doesn't store args. Add the line right after `self.device = device`:

```python
        self.device = device
        self.args = args
        prefix = f"layers.{layer_num}.mlp"
```

- [ ] **Step 3: Run smoke test with flag OFF (default) — must still pass.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS (flag off → goes through original `T > 512` DRAM branch).

- [ ] **Step 4: Run smoke test with flag ON.**

```bash
QWEN35_MLP_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS.

If it fails with PCC drift: apply fallback ladder step 1 (step down grid). Set `QWEN35_SHARD_GRID_ROWS=8 QWEN35_SHARD_GRID_COLS=8` and re-run.

If it fails with L1 OOM: same — step down grid first.

If it fails with `ttnn.linear` rejecting the sharded output config: jump to fallback ladder step 3 (activations-only L1, weights stay DRAM-interleaved) — drop `memory_config=act_intermediate_mc` from `w1_out`/`w3_out` lines and let ttnn pick interleaved.

- [ ] **Step 5: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py
git commit -m "phase 1.1: add MLP sharded-L1 prefill branch (feature-flagged)"
```

### Task 1.1.2: Switch MLP weights from DRAM-interleaved to DRAM-sharded

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py`

- [ ] **Step 1: Add a DRAM-sharded weight loader.**

Edit `qwen35_mlp.py`. Add a new helper function inside `__init__` after the current `load_weight` function. Replace the existing `def load_weight` block (lines 30–40) with:

```python
        def load_weight(name, dtype=ttnn.bfloat8_b, dram_shard=False):
            """Load 2D weight, transpose to [in, out] for ttnn.linear."""
            t = state_dict[f"{prefix}.{name}"].T.contiguous()
            if dram_shard:
                # DRAM_SHARDED along the output dim — parallel bank reads when
                # the matmul is configured with DRAMShardedProgramConfig.
                # Cache file name differs so we don't reuse an interleaved cache.
                mc = ttnn.MemoryConfig(
                    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    buffer_type=ttnn.BufferType.DRAM,
                )
                cache_suffix = "_dramshard"
            else:
                mc = ttnn.DRAM_MEMORY_CONFIG
                cache_suffix = ""
            return ttnn.as_tensor(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mc,
                cache_file_name=(weight_cache_path / f"{prefix}.{name}{cache_suffix}") if weight_cache_path else None,
            )
```

Then update the three weight loads (lines 44–46) to:

```python
        dram_shard = args.mlp_l1_shard  # tied to the same flag — DRAM-shard is only useful with sharded matmul
        self.w1 = load_weight("gate_proj.weight", dtype=ttnn.bfloat4_b, dram_shard=dram_shard)
        self.w2 = load_weight("down_proj.weight", dram_shard=dram_shard)
        self.w3 = load_weight("up_proj.weight", dtype=ttnn.bfloat4_b, dram_shard=dram_shard)
```

- [ ] **Step 2: Run smoke test with flag OFF.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS (flag off → DRAM-interleaved as before).

- [ ] **Step 3: Run smoke test with flag ON.**

```bash
QWEN35_MLP_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS. If `ttnn.linear` errors with "DRAM-sharded weight requires DRAMShardedProgramConfig", apply fallback ladder step 3 — set `dram_shard=False` on the weight loads and proceed without DRAM-sharded weights (still recovers most of the activation-side win).

- [ ] **Step 4: Re-profile to confirm matmul FPU% lifted.**

```bash
QWEN35_MLP_L1_SHARD=1 TT_METAL_DEVICE_PROFILER=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_4k --capture=no && \
LATEST=$(ls -t generated/profiler/reports | head -1) && \
python3 .claude/skills/profiler-report-analysis/scripts/process_profile.py \
  --csv generated/profiler/reports/$LATEST/ops_perf_results_$LATEST.csv \
  --output-dir generated/profiler/reports/$LATEST/analysis/ \
  --sdpas-per-iteration 1
```

Expected (in the new `summary.json`): `decode_last_iter` is irrelevant; in `prefill.per_op["MatmulDeviceOperation"]`, `kernel_total_ns` should drop by 30–50 % vs baseline (76.78 ms → ~40–55 ms). `avg_fpu_util_pct` should rise into the 35–55 % range.

If FPU% does not rise above 25 %: the DRAM-sharded weight pathway likely isn't engaging. Check the matmul program config — `ttnn.linear` may need `program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(...)` explicitly. If that's required, add it to each `linear` call in `_forward_sharded` and re-profile.

- [ ] **Step 5: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py
git commit -m "phase 1.1: load MLP weights DRAM-sharded when sharded path enabled"
```

### Task 1.1.3: Long-context regression — run `traced_32k` with the flag on

**Files:**
- None (test only).

- [ ] **Step 1: Run `traced_32k` with the MLP flag enabled.**

```bash
QWEN35_MLP_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_32k --capture=no -x
```

Expected: PASS. The chunked-prefill loop calls MLP once per chunk × 32 layers × 16 chunks (32 k / 2048) = 512 MLP invocations, all hitting the sharded branch since `T == deltanet_chunk_size`. PCC must hold; TTFT-per-token must not regress.

- [ ] **Step 2: Run `traced_128k` (or the documented substitute).**

```bash
QWEN35_MLP_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_128k --capture=no -x
```

Expected: PASS.

- [ ] **Step 3: Capture the post-1.1 long-context TTFTs.**

Append to `phase1-baseline.json`:

```json
{
  "post_task_1_1": {
    "traced_4k_ttft_s": <captured_value>,
    "traced_32k_ttft_s": <captured_value>,
    "traced_128k_ttft_s": <captured_value_or_null>,
    "matmul_pct_of_kernel": <captured_value>,
    "matmul_avg_fpu_pct": <captured_value>
  }
}
```

- [ ] **Step 4: Commit.**

```bash
git add docs/superpowers/plans/phase1-baseline.json
git commit -m "phase 1.1: capture post-MLP-shard metrics across all context lengths"
```

### Task 1.1.4: Flip the MLP flag on by default

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py`

- [ ] **Step 1: Change the default.**

Edit `model_config.py`. Change:

```python
        self.mlp_l1_shard = os.environ.get("QWEN35_MLP_L1_SHARD", "0") == "1"
```

to:

```python
        self.mlp_l1_shard = os.environ.get("QWEN35_MLP_L1_SHARD", "1") == "1"
```

- [ ] **Step 2: Run smoke test with no env vars set.**

```bash
unset QWEN35_MLP_L1_SHARD && \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS.

- [ ] **Step 3: Run `traced_4k` with no env vars set; confirm TTFT improved.**

```bash
unset QWEN35_MLP_L1_SHARD && \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_4k --capture=no
```

Expected: TTFT ≤ 3.1 s (≥ 300 ms saved vs 3.4 s baseline).

- [ ] **Step 4: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/model_config.py
git commit -m "phase 1.1: enable MLP sharded-L1 path by default"
```

---

## Task 1.2: DeltaNet `mega_fused_weight` matmul — sharded L1 (spec sub-item 1.2, partial)

The DeltaNet projection in `qwen35_gated_deltanet.py:489` is a single `ttnn.linear` that produces QKV + a + b + g in one shot. Same sharding pattern as MLP.

**Note:** The full-attention QKV (`qwen35_gated_attention.py`) goes through an experimental wrapper with its own `memory_config` plumbing. Deferred to a follow-up plan (Phase 1.2b).

### Task 1.2.1: Add a sharded path for `mega_fused_weight` linear

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`

- [ ] **Step 1: Locate the call site.**

Open `qwen35_gated_deltanet.py:485-489`. Current code:

```python
        mc = ttnn.DRAM_MEMORY_CONFIG if T > 512 else None
        ckc = self.compute_kernel_config

        # ---- 1. Projections (single mega-fused matmul for QKV+a+b+g) ----
        mega_out = ttnn.linear(x, self.mega_fused_weight, memory_config=mc, compute_kernel_config=ckc)
```

- [ ] **Step 2: Add the sharded branch.**

Replace lines 485–489 with:

```python
        # Phase 1.2: gated on chunk-size invariant, NOT raw T.
        if self.args.deltanet_megafused_l1_shard and T == self.args.deltanet_chunk_size:
            grid_rows = self.args.shard_grid_rows
            grid_cols = self.args.shard_grid_cols
            core_grid = ttnn.CoreGrid(y=grid_rows, x=grid_cols)
            in_mc = ttnn.create_sharded_memory_config(
                shape=(T, self.args.dim),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            mega_total_dim = self.mega_qkv_dim + self.mega_a_dim + self.mega_b_dim + self.mega_g_dim
            out_mc = ttnn.create_sharded_memory_config(
                shape=(T, mega_total_dim),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            x_sharded = ttnn.to_memory_config(x, in_mc)
            mc = out_mc  # used by downstream concat/conv ops in this function
            ckc = self.compute_kernel_config
            mega_out = ttnn.linear(x_sharded, self.mega_fused_weight, memory_config=out_mc, compute_kernel_config=ckc)
            ttnn.deallocate(x_sharded)
        else:
            mc = ttnn.DRAM_MEMORY_CONFIG if T > 512 else None
            ckc = self.compute_kernel_config
            mega_out = ttnn.linear(x, self.mega_fused_weight, memory_config=mc, compute_kernel_config=ckc)
```

This requires `self.args` to be stored on the module. Search `qwen35_gated_deltanet.py` for `self.args =`. If not present, add it in `__init__`. Find the `__init__` and add right after `self.device = device`:

```python
        self.args = args
```

Also confirm `self.mega_g_dim` exists; if it's named differently (e.g. `self.mega_gate_dim`), use that name.

- [ ] **Step 3: Run smoke test with flag OFF.**

```bash
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS.

- [ ] **Step 4: Run smoke test with flag ON.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS. The `mega_out[:, :, ...]` slicing on lines 491–500 works on sharded tensors via `ttnn.slice`/`__getitem__`. If slicing on a sharded tensor errors:
- **Fallback ladder step 1**: `mega_out = ttnn.to_memory_config(mega_out, ttnn.DRAM_MEMORY_CONFIG)` immediately after the linear, then proceed with DRAM-resident slicing. Smaller win but still recovers the matmul-side speedup.

- [ ] **Step 5: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py
git commit -m "phase 1.2: shard DeltaNet mega_fused_weight matmul on chunk-size match"
```

### Task 1.2.2: DRAM-shard the `mega_fused_weight` weight tensor

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`

- [ ] **Step 1: Find the weight load.**

Search the file for `self.mega_fused_weight = `. It's a `ttnn.as_tensor` call somewhere in `__init__`.

- [ ] **Step 2: Switch its memory config to DRAM-sharded when the flag is on.**

In the weight-load block, replace the `memory_config=ttnn.DRAM_MEMORY_CONFIG` with:

```python
            memory_config=(
                ttnn.MemoryConfig(
                    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    buffer_type=ttnn.BufferType.DRAM,
                ) if args.deltanet_megafused_l1_shard else ttnn.DRAM_MEMORY_CONFIG
            ),
```

And update the `cache_file_name` to vary the suffix (`_dramshard` when sharded) so caches don't collide with the interleaved variant.

- [ ] **Step 3: Run smoke test with flag ON.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS. Same fallback as Task 1.1.2 step 3 if `ttnn.linear` needs an explicit `MatmulDRAMSharded` program config.

- [ ] **Step 4: Run `traced_32k` and `traced_128k` long-context regression.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k 'traced_32k or traced_128k' --capture=no -x
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py
git commit -m "phase 1.2: DRAM-shard DeltaNet mega_fused_weight when sharded path enabled"
```

### Task 1.2.3: Re-profile, validate, flip flag default

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py`

- [ ] **Step 1: Re-profile.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 TT_METAL_DEVICE_PROFILER=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_4k --capture=no && \
LATEST=$(ls -t generated/profiler/reports | head -1) && \
python3 .claude/skills/profiler-report-analysis/scripts/process_profile.py \
  --csv generated/profiler/reports/$LATEST/ops_perf_results_$LATEST.csv \
  --output-dir generated/profiler/reports/$LATEST/analysis/ \
  --sdpas-per-iteration 1
```

Expected: matmul `pct_of_kernel_time` further reduced; cumulative TTFT ≈ 2.95–3.0 s.

- [ ] **Step 2: Capture metrics into `phase1-baseline.json` under `post_task_1_2`.**

- [ ] **Step 3: Flip the default to `1` if PCC + perf both green.**

Edit `model_config.py`:

```python
        self.deltanet_megafused_l1_shard = os.environ.get("QWEN35_DELTANET_MEGAFUSED_L1_SHARD", "1") == "1"
```

- [ ] **Step 4: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/model_config.py docs/superpowers/plans/phase1-baseline.json
git commit -m "phase 1.2: enable DeltaNet mega_fused sharded path by default"
```

---

## Task 1.3: Conv-state pad/concat path → L1 (spec sub-item 1.3)

Move the rolling-buffer pattern at `qwen35_gated_deltanet.py:503-540` from DRAM to L1.

### Task 1.3.1: Switch conv-state ops to L1 when flag is on

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`

- [ ] **Step 1: Read the current conv-state block (lines 503–540).**

Identified pattern: `mc` is set on line 485 to `DRAM_MEMORY_CONFIG if T > 512 else None`. All ops in lines 505–539 use `memory_config=mc`. There's an explicit `to_memory_config(DRAM_MEMORY_CONFIG)` block at lines 558–562 because "Kernel inputs must be in DRAM (NOC reads)". This means conv-state CAN run in L1 up to the point where it hands off to the GDN kernel, which still requires DRAM.

- [ ] **Step 2: Add the convstate L1 branch.**

Inside the `if self.args.deltanet_megafused_l1_shard and T == self.args.deltanet_chunk_size:` branch from Task 1.2.1, change the comment-stage variable assignment:

```python
            mc = out_mc  # used by downstream concat/conv ops in this function
```

to:

```python
            # Conv-state path (lines 503-540) inherits L1 sharded layout.
            # Forced back to DRAM at lines 558-562 before the GDN kernel call.
            convstate_mc = ttnn.L1_MEMORY_CONFIG if self.args.convstate_l1 else (out_mc if self.args.convstate_l1 else ttnn.DRAM_MEMORY_CONFIG)
            mc = convstate_mc
```

Wait — that's tangled. Replace it with a cleaner version:

```python
            mc = ttnn.L1_MEMORY_CONFIG if self.args.convstate_l1 else ttnn.DRAM_MEMORY_CONFIG
```

This way, once `mega_out` is sliced (potentially with a `to_memory_config(L1_INTERLEAVED)` after slicing — see fallback below if needed), the rest of the conv-state path uses L1 interleaved (simpler than block-sharded for the irregular concat shapes).

If the slicing on the sharded `mega_out` requires an intermediate `to_memory_config`, add it right before the `qkv = mega_out[:, :, ...]` slices (line 491):

```python
            if self.args.convstate_l1:
                mega_out = ttnn.to_memory_config(mega_out, ttnn.L1_MEMORY_CONFIG)
```

- [ ] **Step 3: Run smoke test with all Phase 1 flags ON.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 QWEN35_CONVSTATE_L1=1 \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_pcc_smoke.py --capture=no
```

Expected: PASS. If L1 OOM (the conv path allocates several `2051×8192` and `2080×8192` tensors), apply fallback:
- **Fallback step 1**: skip the redundant Tilize. Locate the line `new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)` (line 524) and the trailing Tilize at the end of conv processing. If layouts already match, drop the Tilize.
- **Fallback step 2**: leave conv-state in DRAM but set `convstate_l1=False`; the rest of Phase 1 is unaffected.

- [ ] **Step 4: Run `traced_32k` and `traced_128k` long-context regression.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 QWEN35_CONVSTATE_L1=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k 'traced_32k or traced_128k' --capture=no -x
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py
git commit -m "phase 1.3: route DeltaNet conv-state path through L1 when flag enabled"
```

### Task 1.3.2: Re-profile and flip flag default

**Files:**
- Modify: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py`

- [ ] **Step 1: Re-profile.**

```bash
QWEN35_MLP_L1_SHARD=1 QWEN35_DELTANET_MEGAFUSED_L1_SHARD=1 QWEN35_CONVSTATE_L1=1 TT_METAL_DEVICE_PROFILER=1 \
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text -k traced_4k --capture=no && \
LATEST=$(ls -t generated/profiler/reports | head -1) && \
python3 .claude/skills/profiler-report-analysis/scripts/process_profile.py \
  --csv generated/profiler/reports/$LATEST/ops_perf_results_$LATEST.csv \
  --output-dir generated/profiler/reports/$LATEST/analysis/ \
  --sdpas-per-iteration 1
```

Expected: TTFT ≈ 2.85 s; `Tilize`+`UntilizeWithUnpadding`+`Slice`+`Concat` combined `pct_of_kernel_time` < 8 %.

- [ ] **Step 2: Capture metrics into `phase1-baseline.json` under `post_task_1_3`.**

- [ ] **Step 3: Flip the default to `1`.**

Edit `model_config.py`:

```python
        self.convstate_l1 = os.environ.get("QWEN35_CONVSTATE_L1", "1") == "1"
```

- [ ] **Step 4: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tt/model_config.py docs/superpowers/plans/phase1-baseline.json
git commit -m "phase 1.3: enable DeltaNet conv-state L1 path by default"
```

---

## Task 1.4: Phase 1 exit gate — final validation

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_phase1_perf.py`

### Task 1.4.1: Write the perf regression test

- [ ] **Step 1: Create the test.**

Create `models/demos/blackhole/qwen3_5_9b/tests/test_prefill_phase1_perf.py`:

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 perf regression test.

Asserts the Phase 1 exit criteria from the design spec:
- Matmul `pct_of_kernel_time` < 10 % in the prefill section
- Matmul `avg_fpu_util_pct` > 35 %
- TTFT for traced_4k <= 2950 ms

Run only after the full Phase 1 stack is enabled (default). Skipped if the
profiler isn't available.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from models.utility_functions import run_for_blackhole

REPO_ROOT = Path(__file__).resolve().parents[5]
PROFILER_DIR = REPO_ROOT / "generated/profiler/reports"
SCRIPT = Path.home() / ".claude/skills/profiler-report-analysis/scripts/process_profile.py"


@run_for_blackhole()
@pytest.mark.timeout(900)
def test_prefill_phase1_exit_criteria(device):
    if not SCRIPT.exists():
        pytest.skip(f"profiler-report-analysis script not found at {SCRIPT}")

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    cmd_run = [
        "pytest",
        str(REPO_ROOT / "models/demos/blackhole/qwen3_5_9b/demo/text_demo.py"),
        "-k", "traced_4k",
        "--capture=no",
    ]
    subprocess.run(cmd_run, cwd=REPO_ROOT, env=env, check=True)

    latest = sorted(PROFILER_DIR.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    csv_path = latest / f"ops_perf_results_{latest.name}.csv"
    out_dir = latest / "analysis"
    subprocess.run(
        ["python3", str(SCRIPT),
         "--csv", str(csv_path),
         "--output-dir", str(out_dir),
         "--sdpas-per-iteration", "1"],
        check=True,
    )

    summary = json.loads((out_dir / "summary.json").read_text())
    matmul = summary["prefill"]["per_op"].get("MatmulDeviceOperation", {})
    pct_kernel = matmul.get("pct_of_kernel_time", 99.0)
    fpu_pct = matmul.get("avg_fpu_util_pct", 0.0)

    assert pct_kernel < 10.0, f"matmul pct_of_kernel_time = {pct_kernel:.2f}% — phase 1 exit demands < 10%"
    assert fpu_pct > 35.0, f"matmul avg_fpu_util_pct = {fpu_pct:.2f}% — phase 1 exit demands > 35%"
```

- [ ] **Step 2: Run the perf regression test with all flags at their defaults (now ON).**

```bash
cd /local/ttuser/atupe/tt-metal && \
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_phase1_perf.py --capture=no
```

Expected: PASS. If FAIL on `pct_of_kernel_time`: re-inspect the profiler output; the per-op breakdown will show whether matmul time was actually saved or if it shifted to glue ops. Apply the relevant sub-item's fallback ladder.

If FAIL on `avg_fpu_util_pct`: the sharded path is engaged but the matmul kernel isn't getting enough work per dispatch — likely the program config isn't using the full compute grid. Inspect `INPUT_0_MEMORY` and `INPUT_1_MEMORY` columns in `prefill.csv`; if they don't show `*_SHARDED`, the `to_memory_config` calls are silently failing.

- [ ] **Step 3: Commit.**

```bash
git add models/demos/blackhole/qwen3_5_9b/tests/test_prefill_phase1_perf.py
git commit -m "phase 1: add exit-criteria perf regression test"
```

### Task 1.4.2: Update the project memory baseline

**Files:**
- Modify: `/local/ttuser/.claude/projects/-local-ttuser-atupe-tt-metal/memory/project_qwen35_perf_baseline.md`

- [ ] **Step 1: Read the existing memory file.**

```bash
cat /local/ttuser/.claude/projects/-local-ttuser-atupe-tt-metal/memory/project_qwen35_perf_baseline.md
```

- [ ] **Step 2: Append the post-Phase-1 baseline.**

Use the `Edit` tool (not Bash) to append a new section under the existing one. The file uses Markdown — add a section like:

```markdown

## Post-Phase-1 (2026-04-29 + N)

After enabling MLP sharded-L1 + DeltaNet mega_fused sharded + conv-state L1:
- `traced_4k`: TTFT = <captured value> s (was 3.4 s)
- Matmul `pct_of_kernel_time`: <captured> % (was 18.2 %)
- Matmul `avg_fpu_util_pct`: <captured> % (was 13 %)
- Long-context: `traced_32k` PASS, `traced_128k` PASS
```

- [ ] **Step 3: No commit (memory file is not in the repo).**

---

## Task 1.5: Phase 1 wrap-up — write the Phase 2 trigger note

**Files:**
- Modify: `docs/superpowers/specs/2026-04-29-qwen35-9b-prefill-optimization-design.md`

- [ ] **Step 1: Append a "Phase 1 results" section to the spec.**

Use the `Edit` tool to append a new section at the end of the spec:

```markdown

## Phase 1 results (filled in 2026-04-29 + N)

- **Cumulative TTFT savings**: `traced_4k` 3.4 s → <captured> s
- **Long-context regression**: `traced_32k` PASS, `traced_128k` PASS
- **Matmul kernel share**: 18.2 % → <captured> %
- **Matmul FPU%**: 13 % → <captured> %
- **Sub-items shipped**: 1.1 (MLP), 1.2 (DeltaNet mega-fused; full-attention QKV deferred to Phase 1.2b plan), 1.3 (conv-state)

**Phase 2 trigger**: re-run the profiler analysis (Step 2 of Task 0.1) on the post-Phase-1 profile. If glue ops (`Reshape`, `Tilize`, `Untilize`, `Slice`, `Concat`, `BinaryNg`) account for ≥ 8 % of the new prefill kernel time, Phase 2 is worth authoring. If they're < 5 %, skip Phase 2 and go straight to Phase 3.
```

- [ ] **Step 2: Commit.**

```bash
git add docs/superpowers/specs/2026-04-29-qwen35-9b-prefill-optimization-design.md
git commit -m "phase 1: record results and Phase 2 trigger condition in design spec"
```

---

## Phase 1 Exit Criteria (must all hold to consider Phase 1 done)

1. ✅ All sub-item commits land on `qwen9b-p150` (or its successor branch).
2. ✅ `test_prefill_pcc_smoke.py::test_prefill_pcc_smoke` PASSes with all Phase 1 flags at their defaults.
3. ✅ `test_prefill_phase1_perf.py::test_prefill_phase1_exit_criteria` PASSes.
4. ✅ `text_demo.py::test_demo_text` PASSes for `traced_4k`, `traced_32k`, and `traced_128k` (or documented substitute).
5. ✅ TTFT for `traced_4k` ≤ 2950 ms (≥ 450 ms saved vs 3.4 s baseline).
6. ✅ `phase1-baseline.json` contains baseline + post-task metrics for 1.1, 1.2, 1.3.

---

## What Phase 1 explicitly does NOT do

- Touch the GDN custom kernel (`gdn_kernel/program_factory.py`) — Phase 3.
- Modify the experimental `gated_attention_forward_ttnn` wrapper — deferred to Phase 1.2b plan.
- Eliminate the GDN output reshape — Phase 3.3.
- Investigate or merge the 2-GDN-calls-per-linear-layer pattern — Phase 3.4.
- Add fused matmul-silu-mul ops — Phase 4.2.

---

## Self-Review Notes

Performed an inline self-review against the spec:

- **Spec coverage**: Phase 0 (0.1, 0.2, 0.3) → Tasks 0.1, 0.2, 0.3, 0.4. Phase 1.1 → Tasks 1.1.1, 1.1.2, 1.1.3, 1.1.4. Phase 1.2 (DeltaNet mega-fused only — full-attn QKV explicitly deferred per spec) → Tasks 1.2.1, 1.2.2, 1.2.3. Phase 1.3 → Tasks 1.3.1, 1.3.2. Phase 1 exit → Tasks 1.4.1, 1.4.2, 1.5.
- **Long-context invariant**: every Task that touches kernels gates the new branch on `T == self.args.deltanet_chunk_size` and includes a `traced_32k`+`traced_128k` regression step.
- **Fallback ladders**: Each sub-item's failure modes (PCC drift, L1 OOM, missing program config) are tied back to the spec's fallback ladder steps. The plan does not invent new fallbacks — it references the spec's ladder by step number.
- **Type consistency**: feature flag names (`mlp_l1_shard`, `convstate_l1`, `deltanet_megafused_l1_shard`) are consistent across `model_config.py`, the env var names, and the call sites that gate on them. Grid constants (`shard_grid_rows`, `shard_grid_cols`) consistent.
- **Placeholder scan**: no "TBD"/"TODO"/"similar to". The "captured value" placeholders in JSON updates are intentional — they're filled in by running the prior step's commands. Each step that produces such a value runs a real command whose output goes there.
- **Phase 2/3/4 deferral is explicit and gated** on actual measured Phase 1 outcomes, per the spec's "bail-out option".
