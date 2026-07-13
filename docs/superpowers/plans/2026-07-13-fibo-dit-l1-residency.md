# FIBO DIT Transformer — L1 Activation Residency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the FIBO DIT transformer's intermediate activations resident in L1 (interleaved) instead of DRAM during the denoise forward, measured via `test_performance_bria_fibo.py`, keeping output bit-exact.

**Architecture:** A single opt-in `activation_memory_config` (a `ttnn.MemoryConfig | None`) carried on every `Module` (default `None` ⇒ today's DRAM behavior). FIBO flips it to `ttnn.L1_MEMORY_CONFIG` after build via a recursive setter; each op reads `self._act_mem_cfg` and passes it to `minimal_matmul` / CCL / SDPA / norm. CCL buffers (in `CCLManager`, not a `Module`) receive the config as a call-time argument from the block that invokes them. Because the default is `None` and only FIBO calls the setter, Flux/Wan/LTX stay byte-identical.

**Tech Stack:** Python, ttnn, tt-metal on a 2×2 Blackhole mesh (sp=2, tp=2, spatial seq 4096).

## Global Constraints

- **Opt-in only:** every new parameter/attribute defaults to `None`; the DRAM code path when `None` must be byte-for-byte the current path. Other DiT models (Flux, Wan, LTX) must be unaffected — verify by the default being off and (Task 4) a Flux spot-check if any shared default is at risk.
- **Bit-exact target:** L1-interleaved does not change the math. `test_fibo_transformer_mesh` PCC must stay at the current baseline (≈0.9953, gate asserts ≥0.99). Any drop is a bug, not a trajectory shift.
- **Keep/revert per site:** enable L1 at a site only if it (1) does not OOM and (2) does not regress denoise it/s. On L1 OOM, that tensor stays DRAM (the "if possible").
- **On-device test discipline (learned):** when a subagent runs an on-device test, it MUST BLOCK until the test process exits — poll the log file with an until-loop (`while kill -0 <pid> 2>/dev/null; do sleep 15; done`), NOT a detached background run. Do not launch a second run while one holds the device (`CHIP_IN_USE_*` lock). Report only after the process exits and the PCC/timing lines are in the log.
- **Do NOT change** matmul block sizes (Phase 1 done, at ceiling) or SDPA chunk sizes (reverted, dead end). This plan changes only *where tensors live*.
- **Run env (all commands):** prefix with `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole` and use `python_env/bin/python`. Run from repo root `/localdev/mstojkovic/tt-metal`.

## File Structure

- `models/tt_dit/layers/module.py` — add `_act_mem_cfg` attribute + recursive `set_activation_memory_config()` on the `Module` base. (Owns the mechanism.)
- `models/tt_dit/layers/linear.py` — `Linear/ColParallelLinear/RowParallelLinear` read `self._act_mem_cfg` and pass `memory_config=` into the 4 minimal-matmul variants + `forward_fused_addcmul`.
- `models/tt_dit/parallel/manager.py` — `all_gather` / `reduce_scatter` / `get_ag_ping_pong_buffer` / `get_rs_ping_pong_buffer` accept an optional `memory_config` (default DRAM), threaded to buffer allocation + the RS output.
- `models/tt_dit/blocks/attention.py` — pass `self._act_mem_cfg` to the CCL calls and (if the op supports it) the SDPA output.
- `models/tt_dit/layers/normalization.py` — `DistributedLayerNorm` passes `self._act_mem_cfg` to its post-all-gather op + stats all-gather.
- `models/tt_dit/models/transformers/transformer_bria_fibo.py` — the single switch: read env `FIBO_ACT_L1`, call `self.set_activation_memory_config(ttnn.L1_MEMORY_CONFIG)` at the end of `__init__`.
- No structural change to `test_performance_bria_fibo.py`; its docstring baselines are refreshed in Task 4.

Blocks (`transformer_block.py`, `transformer_flux1.py`) and `feedforward.py` need **no signature changes** — their sub-modules read `self._act_mem_cfg` directly; the CCL calls they make already go through `self.ccl_manager`, so only the manager call-sites in those files that pass buffers need the `memory_config` kwarg (Task 2).

---

### Task 0: Capture the DRAM baseline

**Files:** none (measurement only).

**Interfaces:**
- Produces: baseline numbers recorded in this plan's Task 4 checklist — transformer PCC, denoise it/s (untraced + traced), whole-pipeline seconds. Every later task compares against these.

- [ ] **Step 1: Record the correctness baseline**

Run (BLOCK until it exits):
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_transformer.py::test_fibo_transformer_mesh -v -s \
  2>&1 | tee /tmp/fibo_l1_baseline_pcc.log
```
Expected: PASS, PCC printed (record the exact value, ≈0.9953).

- [ ] **Step 2: Record the perf baseline**

Run (BLOCK until it exits):
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown -v -s \
  2>&1 | tee /tmp/fibo_l1_baseline_perf.log
```
Expected: PASS. Record from the breakdown log: denoise it/s (untraced + traced) and total pipeline seconds.

- [ ] **Step 3: Record baselines in this plan**

Fill the "Baseline (Task 0)" line in Task 4's table with the measured numbers. No commit (measurement only).

---

### Task 1: Mechanism + matmul outputs → L1 + FIBO switch

**Files:**
- Modify: `models/tt_dit/layers/module.py` (add attribute + setter to `Module`)
- Modify: `models/tt_dit/layers/linear.py` (3 forwards + `forward_fused_addcmul`)
- Modify: `models/tt_dit/models/transformers/transformer_bria_fibo.py` (`__init__` tail: env switch)

**Interfaces:**
- Produces:
  - `Module._act_mem_cfg: ttnn.MemoryConfig | None` (default `None`)
  - `Module.set_activation_memory_config(memory_config: ttnn.MemoryConfig | None) -> None` (recursive over `named_children()`)
  - `linear.py` forwards pass `memory_config=self._act_mem_cfg` to every `minimal_matmul*` call.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Add the attribute + setter to `Module.__init__`/class body**

In `models/tt_dit/layers/module.py`, in `Module.__init__` (after `self.coresident_exclusions = None`, line 39) add:
```python
        self._act_mem_cfg = None  # activation output memory_config; None => op default (DRAM). Opt-in, FIBO-only.
```
And add this method to `Module` (near `named_children`):
```python
    def set_activation_memory_config(self, memory_config) -> None:
        """Recursively set the activation output memory_config on this module and all children.

        Opt-in L1 residency: with the default (never called), _act_mem_cfg stays None and every
        op keeps its current (DRAM) output layout, so other models are unaffected.
        """
        self._act_mem_cfg = memory_config
        for _, child in self.named_children():
            child.set_activation_memory_config(memory_config)
```

- [ ] **Step 2: Thread it into `linear.py` matmul calls**

In `models/tt_dit/layers/linear.py`, pass `memory_config=self._act_mem_cfg` into every minimal-matmul call:
- `Linear.forward` — the `ttnn.experimental.minimal_matmul(...)` at line ~80: add `memory_config=self._act_mem_cfg,`.
- `ColParallelLinear.forward` — three call sites: `all_gather_minimal_matmul_async` (~223), `minimal_matmul_split` (~253), `minimal_matmul` (~266): add `memory_config=self._act_mem_cfg,` to each (the AGMM/split variants accept `memory_config`; if a variant rejects it, leave that one and note it in the report).
- `RowParallelLinear.forward` — `minimal_matmul` (~371): add `memory_config=self._act_mem_cfg,`.
- `RowParallelLinear.forward_fused_addcmul` — change the two hardcoded DRAM configs (`memory_config_mm`, `rs_output_mem_config`, lines ~434-435) to `self._act_mem_cfg or ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)`.

Note: `minimal_matmul(memory_config=None)` inherits in0 (current behavior), so `None` is a no-op. Do NOT pass it to `_apply_activation_fn` (that stays as-is).

- [ ] **Step 3: Add the FIBO switch**

In `models/tt_dit/models/transformers/transformer_bria_fibo.py`, at the END of `BriaFiboTransformer.__init__` (after all submodules are constructed), add:
```python
        import os
        if os.environ.get("FIBO_ACT_L1") == "1":
            self.set_activation_memory_config(ttnn.L1_MEMORY_CONFIG)
```
This is the single A/B switch: unset ⇒ DRAM (baseline); `FIBO_ACT_L1=1` ⇒ L1. Confirm `ttnn` is already imported in the file (it is).

- [ ] **Step 4: Correctness gate — bit-exact PCC with L1 on**

Run (BLOCK until exit):
```bash
FIBO_ACT_L1=1 HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_transformer.py::test_fibo_transformer_mesh -v -s \
  2>&1 | tee /tmp/fibo_l1_task1_pcc.log
```
Expected: PASS, PCC == Task 0 baseline (±0.0000; matmul-output L1 is bit-exact). If PCC drops or it OOMs (`Out of Memory` / allocation failure in the log), STOP and report — the mechanism is mis-wired or L1 doesn't fit; do not proceed.

- [ ] **Step 5: Perf check — L1 on vs baseline**

Run (BLOCK until exit):
```bash
FIBO_ACT_L1=1 HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown -v -s \
  2>&1 | tee /tmp/fibo_l1_task1_perf.log
```
Expected: PASS. Record denoise it/s (untraced + traced) vs Task 0. Report the delta (may be small — matmuls are compute-bound; the big win comes when the stream stays L1 across CCL, Task 2).

- [ ] **Step 6: Commit**

```bash
git add models/tt_dit/layers/module.py models/tt_dit/layers/linear.py \
        models/tt_dit/models/transformers/transformer_bria_fibo.py
git commit -m "perf(fibo-pipeline): opt-in L1 activation memory_config; matmul outputs to L1

Recursive set_activation_memory_config on Module (default None => DRAM, other
models unaffected). FIBO_ACT_L1=1 flips the DIT transformer's matmul outputs to
L1_MEMORY_CONFIG. Bit-exact (PCC unchanged).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: CCL buffers → L1 (keep the stream resident across all-gather / reduce-scatter)

**Files:**
- Modify: `models/tt_dit/parallel/manager.py` (AG/RS methods + buffer allocators)
- Modify: `models/tt_dit/blocks/attention.py` (pass config to its CCL calls)
- Modify: `models/tt_dit/blocks/transformer_block.py`, `models/tt_dit/models/transformers/transformer_flux1.py` (pass config to their `ccl_manager` calls)

**Interfaces:**
- Consumes: `self._act_mem_cfg` (Task 1).
- Produces: `CCLManager.all_gather(..., memory_config=None)`, `.reduce_scatter(..., memory_config=None)`, `.all_gather_persistent_buffer(..., memory_config=None)`, `.get_ag_ping_pong_buffer(..., memory_config=ttnn.DRAM_MEMORY_CONFIG)`, `.get_rs_ping_pong_buffer(..., memory_config=ttnn.DRAM_MEMORY_CONFIG)`. Default preserves DRAM.

- [ ] **Step 1: Thread `memory_config` through the manager**

In `models/tt_dit/parallel/manager.py`:
- `get_ag_ping_pong_buffer(self, shape, dim, mesh_axis, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)` — use `memory_config=memory_config` in the `ttnn.from_torch(...)` at line ~181. Include `memory_config` in the `cache_key` tuple (so DRAM and L1 variants don't collide).
- `get_rs_ping_pong_buffer(...)` — same treatment (find its allocator; add `memory_config` param + cache-key entry).
- `all_gather(self, tensor, /, *, dim, mesh_axis, use_hyperparams, use_persistent_buffer=False, memory_config=None)` — forward `memory_config` (when not None) to the persistent-buffer allocator; if `all_gather_async` accepts an output `memory_config`, pass it too.
- `all_gather_persistent_buffer(self, tensor, /, *, dim, mesh_axis, use_hyperparams=False, memory_config=None)` — forward to `all_gather`.
- `reduce_scatter(self, tensor, /, *, dim, mesh_axis, use_persistent_buffer=False, memory_config=None)` — replace the hardcoded `memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)` at line ~511 with `memory_config=memory_config or ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)`, and forward to `get_rs_ping_pong_buffer`.
- `reduce_scatter_persistent_buffer(...)` — add `memory_config=None`, forward.

- [ ] **Step 2: Pass the config at the block/attention CCL call-sites**

Add `memory_config=self._act_mem_cfg` to each `self.ccl_manager.all_gather_persistent_buffer(...)` / `.reduce_scatter(...)` / `.all_gather(...)` call in:
- `attention.py` — the two `all_gather_persistent_buffer` calls (~325-327, ~331-333). (The ring-SDPA `get_ag_ping_pong_buffer` calls at ~282-287 are SDPA-internal; leave for Task 3.)
- `transformer_block.py` — the pre-attention and pre-FF `all_gather_persistent_buffer` calls, and any `reduce_scatter`.
- `transformer_flux1.py` — the pre-norm/pre-attn `all_gather` calls; `proj_out`'s reduce-scatter goes through `RowParallelLinear` (already covered by Task 1's `forward` — confirm the RS output picks up `self._act_mem_cfg`).
- `RowParallelLinear.forward` (`linear.py`) — pass `memory_config=self._act_mem_cfg` to its `self.ccl_manager.reduce_scatter(...)` at ~385.

- [ ] **Step 3: Correctness gate**

Run the Task 1 Step 4 PCC command with `FIBO_ACT_L1=1`, log to `/tmp/fibo_l1_task2_pcc.log`. Expected: PASS, PCC == baseline. **If `Out of Memory` appears** (persistent ping-pong buffers now sit permanently in L1 — a known risk), that is the "does not fit" outcome: report which buffer/shape OOM'd and proceed to Step 5 to selectively keep only the RS-output-L1 (non-persistent) part, leaving persistent AG buffers in DRAM.

- [ ] **Step 4: Perf check**

Run the Task 1 Step 5 perf command with `FIBO_ACT_L1=1`, log to `/tmp/fibo_l1_task2_perf.log`. Record denoise it/s vs Task 0 and Task 1. This is where the inherit-bucket win should show (the stream now stays L1 across all-gathers).

- [ ] **Step 5: Resolve OOM / regression (if any)**

If Step 3 OOM'd or Step 4 regressed: keep the RS-output-L1 change (cheap, non-persistent) and revert the persistent AG-buffer-L1 to DRAM (pass `memory_config=None` at the AG call-sites) so only the block-boundary residual stream stays L1. Re-run Steps 3-4. Record the final surviving configuration.

- [ ] **Step 6: Commit**

```bash
git add models/tt_dit/parallel/manager.py models/tt_dit/blocks/attention.py \
        models/tt_dit/blocks/transformer_block.py \
        models/tt_dit/models/transformers/transformer_flux1.py \
        models/tt_dit/layers/linear.py
git commit -m "perf(fibo-pipeline): opt-in L1 for CCL buffers (keep DIT stream L1 across CCL)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Norm output + SDPA output → L1

**Files:**
- Modify: `models/tt_dit/layers/normalization.py` (`DistributedLayerNorm`)
- Modify: `models/tt_dit/blocks/attention.py` (SDPA output)

**Interfaces:**
- Consumes: `self._act_mem_cfg`.
- Produces: nothing new consumed downstream (leaf changes).

- [ ] **Step 1: DistributedLayerNorm output → L1**

In `models/tt_dit/layers/normalization.py`, `DistributedLayerNorm.forward` (~240-372): if the `dit_layernorm_post_allgather` op accepts a `memory_config`, pass `memory_config=self._act_mem_cfg`; pass `memory_config=self._act_mem_cfg` to its stats `all_gather_persistent_buffer` (~357). If an op does not accept `memory_config`, leave it (it will inherit) and note it. RMSNorm (`dit_rms_norm_unary_fused`) inherits — leave unless trivially parameterizable.

- [ ] **Step 2: SDPA output → L1 (guarded)**

In `models/tt_dit/blocks/attention.py`: check whether `ttnn.transformer.ring_joint_scaled_dot_product_attention` and `ttnn.transformer.joint_scaled_dot_product_attention` accept an output `memory_config` (inspect the op docstring like `python_env/bin/python -c "import ttnn; print(ttnn.transformer.ring_joint_scaled_dot_product_attention.__doc__)"`). If yes, pass `memory_config=self._act_mem_cfg`. If no, do NOT add a `to_memory_config` copy here (its cost likely exceeds the gain and SDPA CBs are near the L1 ceiling) — record "SDPA output not parameterizable; left DRAM" and move on. Do NOT touch `sdpa_program_config` / chunk sizes.

- [ ] **Step 3: Correctness gate**

Task 1 Step 4 PCC command, `FIBO_ACT_L1=1`, log `/tmp/fibo_l1_task3_pcc.log`. Expected PASS, PCC == baseline. On SDPA-related OOM, revert the SDPA-output change (Step 2) and keep only the norm change.

- [ ] **Step 4: Perf check**

Task 1 Step 5 perf command, `FIBO_ACT_L1=1`, log `/tmp/fibo_l1_task3_perf.log`. Record it/s deltas; keep each sub-change only if neutral-or-better and non-OOM.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/layers/normalization.py models/tt_dit/blocks/attention.py
git commit -m "perf(fibo-pipeline): opt-in L1 for DIT norm (and SDPA output where supported)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Lock survivors + finalize

**Files:**
- Modify: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` (docstring baseline refresh only)
- Possibly modify: any Task 1-3 file to revert a site that OOM'd/regressed.

**Interfaces:**
- Consumes: the recorded per-task numbers.

- [ ] **Step 1: Final A/B measurement**

Run the perf breakdown BOTH ways to lock the end-to-end delta (BLOCK each):
```bash
# baseline (DRAM)
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown -v -s \
  2>&1 | tee /tmp/fibo_l1_final_dram.log
# L1
FIBO_ACT_L1=1 HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown -v -s \
  2>&1 | tee /tmp/fibo_l1_final_l1.log
```
Fill the results table:

| Config | transformer PCC | denoise it/s (untraced) | denoise it/s (traced) | pipeline s |
|---|---|---|---|---|
| Baseline (Task 0) | | | | |
| L1 final | | | | |

- [ ] **Step 2: Decide keep-or-drop the whole feature**

If L1-final is faster (or neutral) and PCC holds: keep, and default the switch appropriately (leave it env-gated so it is explicit and reversible; do NOT force L1 on for all builds unless the win is clear and you note it). If L1-final regresses everywhere and no site helped: report that empirically and leave the mechanism in place but the switch off (still useful for Phase 2 sharding). Either outcome is a valid, reported result.

- [ ] **Step 3: Verify other models unaffected**

Confirm the DRAM path is untouched: with `FIBO_ACT_L1` unset, `test_fibo_transformer_mesh` PCC == Task 0 baseline (already true if you never regressed it). Optionally run one Flux/Wan transformer test if one is quick, to confirm no shared default changed. Report result.

- [ ] **Step 4: Refresh the perf-test docstring baselines**

Update the "Prior baselines" style notes in `test_performance_bria_fibo.py` (and/or add a one-line note) to record the L1-residency it/s so future runs have the reference. Keep it factual.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py
git commit -m "perf(fibo-pipeline): record L1 activation-residency results (denoise Phase 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** matmul L1 (Task 1) ✓; CCL buffers L1 (Task 2) ✓; SDPA + norm L1 (Task 3) ✓; inherit bucket rides Tasks 1-2 for free ✓; opt-in/isolation (Global Constraints + `None` defaults + Task 4 Step 3) ✓; bit-exact PCC gate (every task) ✓; keep/revert-per-site (Tasks 2-4) ✓; measurement via the two named tests ✓; Phase-2 sharding explicitly deferred (spec) ✓; weights/encoder/VAE non-goals (spec) ✓.

**Placeholder scan:** no TBD/TODO; each code step shows the concrete edit; op-signature uncertainties (SDPA/norm `memory_config` support) are handled with an explicit "inspect the docstring, pass if supported else record and skip" instruction rather than a guess.

**Type consistency:** `_act_mem_cfg` / `set_activation_memory_config` / `memory_config` naming is consistent across module.py, linear.py, manager.py, attention.py, normalization.py, and the FIBO switch.
