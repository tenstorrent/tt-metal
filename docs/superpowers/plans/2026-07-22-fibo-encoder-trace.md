# FIBO SmolLM3 Encoder Trace (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the SmolLM3 encoder device forward as a ttnn trace and replay it, staying bit-exact across many sequential encodes — fixing the replay-noise that got the prior attempt reverted.

> **OUTCOME (2026-07-22):** The replay-noise does **not reproduce** on the simplified encoder — Tasks 2
> (root-cause) and 3 (fix) collapsed to "already stable." Verified on the 4×8 Galaxy: trace is bit-exact
> (traced == untraced, PCC 1.0002), stable across 16 isolated replays and 3 full-pipeline generations
> (gen 2/3 == gen 1 at PCC 0.9999999), and **3.58× faster** on the real JSON encode (1021.8 ms →
> 285.6 ms). Delivered: `use_trace` + `Tracer` (Task 1), a corrected replay-stability gate (asserts
> traced-replay == captured baseline; json-vs-HF ≥ 0.99), and a gated `encoder_use_trace` pipeline flag
> (default off). No CCL fix was needed.

**Architecture:** Wrap the existing `SmolLM3TextEncoderWrapper._forward` (already a clean device-in/device-out unit returning one stacked tensor) in a `Tracer`, behind a `use_trace` flag. Investigation-first: reproduce the replay-noise, root-cause the encoder CCLManager ping-pong/semaphore phase desync, then apply the fix, gated by an N-encode replay-stability test.

**Tech Stack:** Python, `ttnn` (traces, CCL all-gather), tt_dit `Tracer` (`utils/tracing.py`), `CCLManager` (`parallel/manager.py`), pytest on a 4×8 Blackhole Galaxy.

## Global Constraints

- **Preserve exactly:** the SP math, weights, and the output contract `prompt_embeds = cat(hidden[-1], hidden[-2], dim=-1)`; the shipped fast readback (`_read_seq_sharded`); the untraced path byte-for-byte.
- **Single bucket only** (1024); pos and neg share one trace. No multi-bucket work.
- **`_forward` stays single-output** (the stacked `ttnn.concat(all_hidden_states, dim=0)`), not a multi-tensor tuple.
- **Inputs to the Tracer are already-SP-sharded device tensors** (host tensors would lose the sharding — `Tracer._tensor_to_device` would `.to(device)` replicated).
- **PCC threshold 0.99** vs HF for every encode in the stability gate.
- The untraced path must remain the default and unchanged until the stability gate is green; `test_fibo_encode_device_profile` and `use_torch=True` always stay untraced.
- **Hardware:** all device tests run on the **4×8 Galaxy** only. `_DEVICE_PARAMS` (with `trace_region_size=200000000`) is required for any traced run; `_PROFILE_DEVICE_PARAMS` (no trace region) must stay untraced.

**Files (all under `models/tt_dit/`):**
- `pipelines/bria_fibo/text_encoder.py` — wrapper: add `use_trace`, the `Tracer`, routing.
- `pipelines/bria_fibo/pipeline_bria_fibo.py` — pass `use_trace` into the wrapper (Task 4).
- `tests/models/bria_fibo/test_performance_bria_fibo.py` — `test_fibo_encode_perf` already builds the pipe; used in Task 4.
- `tests/encoders/smollm3/test_smollm3.py` — add the N-encode replay-stability test.

---

### Task 1: Re-apply the forward Tracer behind `use_trace` + add the replay-stability test (reproduction)

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/text_encoder.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (add `test_fibo_wrapper_encode_replay_stable`)

**Interfaces:**
- Produces: `SmolLM3TextEncoderWrapper.__init__(..., use_trace: bool = False)`; a per-bucket `self._tracers: dict[int, Tracer]`; `encode_prompt` routes the device forward through the tracer when `use_trace`. `_forward(tt_ids, tt_cos, tt_sin) -> ttnn.Tensor` unchanged (single stacked output).
- Consumes: existing `pick_bucket`, `_prep_inputs`, `_read_seq_sharded`; `Tracer` from `models/tt_dit/utils/tracing.py`.

- [ ] **Step 1: Write the failing replay-stability test**

Add to `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (mirrors `test_fibo_wrapper_encode`, but builds with `use_trace=True` and checks EVERY encode, not just the first):

```python
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 200000000}],
    indirect=["device_params"],
)
def test_fibo_wrapper_encode_replay_stable(*, mesh_device):
    """The traced encoder must stay bit-exact across MANY sequential encodes (the reverted bug was
    'noise after the first run'). Capture on the first encode, replay on the rest; every readback
    must match HF at PCC 0.99 -- not just the first."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from models.tt_dit.pipelines.bria_fibo.text_encoder import SmolLM3TextEncoderWrapper

    sp_axis, tp_axis = 1, 0
    pc = EncoderParallelConfig.from_tuples(
        tp=(mesh_device.shape[tp_axis], tp_axis), sp=(mesh_device.shape[sp_axis], sp_axis)
    )
    try:
        ckpt = snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO unavailable: {e}")

    json_path = Path(__file__).resolve().parents[2] / "models" / "bria_fibo" / "fibo_vlm_prompt.json"
    if not json_path.is_file():
        pytest.skip(f"JSON prompt fixture missing: {json_path}")
    json_prompt = json_path.read_text().strip()

    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    wrapper = SmolLM3TextEncoderWrapper(
        ckpt, device=mesh_device, ccl_manager=ccl, parallel_config=pc, pad_buckets=(1024,), use_trace=True
    )

    hf = _load_hf_smollm3()

    def hf_embeds(p):
        ids = wrapper.tokenizer([p if p else " "], add_special_tokens=True, return_tensors="pt").input_ids
        if not p:
            ids = torch.tensor([[128000]])  # empty -> BOT, matches the wrapper
        with torch.no_grad():
            ref = hf.model(input_ids=ids, output_hidden_states=True)
        return torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    # Alternate pos/neg across several "generations": capture on run 1, replay on 2..N.
    prompts = [json_prompt, "", json_prompt, "", json_prompt]
    for i, p in enumerate(prompts):
        embeds, _ = wrapper.encode_prompt(p)
        ref = hf_embeds(p)
        assert list(embeds.shape) == list(ref.shape), f"run {i}: {embeds.shape} != {ref.shape}"
        assert_quality(ref, embeds.float(), pcc=0.99, relative_rmse=0.2)
```

- [ ] **Step 2: Run it to confirm it FAILS (reproduces the noise)**

Run (4×8 Galaxy):
`TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python_env/bin/pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_encode_replay_stable" -v -rA -s`
Expected: **FAIL** — either an `AttributeError` (no `use_trace` arg yet) at first, or after Step 3, the PCC assertion fails on run ≥ 2 (the reverted "noise after first run"). Record which run first drops and its PCC.

- [ ] **Step 3: Add `use_trace` + the per-bucket Tracer to the wrapper**

In `models/tt_dit/pipelines/bria_fibo/text_encoder.py`:

Add the import near the top:
```python
from ...utils.tracing import Tracer
```

Add `use_trace` to `__init__` (after `pad_buckets`) and initialize the tracer dict (place after the encoder is built):
```python
        use_trace: bool = False,
    ) -> None:
        ...
        # Trace the device forward (one trace per padding bucket): it is host-dispatch-bound and the
        # fixed bucket gives it a static shape. Pos+neg (both bucket 1024) share the trace: first encode
        # captures, later encodes replay.
        self._use_trace = use_trace
        self._tracers: dict[int, Tracer] = {}
```

Route `encode_prompt`'s device forward through the tracer. Replace the current
`tt_ids, tt_cos, tt_sin = self._prep_inputs(input_ids, seq_len)` / `stacked = self._forward(...)` with:
```python
        bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
        tt_ids, tt_cos, tt_sin = self._prep_inputs(input_ids, seq_len)
        if self._use_trace:
            tracer = self._tracers.get(bucket)
            if tracer is None:
                tracer = Tracer(self._forward, device=self._device, prep_run=True, clone_prep_inputs=False)
                self._tracers[bucket] = tracer
            stacked = tracer(tt_ids, tt_cos, tt_sin)
        else:
            stacked = self._forward(tt_ids, tt_cos, tt_sin)
```

Leave `_prep_inputs`, `_forward`, `_read_seq_sharded`, and the readback/slice/split unchanged.

- [ ] **Step 4: Run the stability test; expect it to reproduce the replay bug (still RED)**

Run the same command as Step 2. Expected: run 1 (capture) passes at PCC 0.99; a later run FAILS (noise) — this is the documented reverted failure, now reproduced under test. If it unexpectedly PASSES all 5 runs, note that (the simplification may have already removed the cause — see Task 2) and proceed to Task 2 to confirm.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/text_encoder.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "test(fibo-pipeline): re-add encoder forward Tracer + replay-stability gate (reproduces revert)"
```

---

### Task 2: Root-cause the replay-noise (instrument the CCL ping-pong phase)

**Files:**
- Investigation only (temporary instrumentation, reverted before commit); findings recorded in the plan/commit message.

**Interfaces:**
- Consumes: the `test_fibo_wrapper_encode_replay_stable` harness from Task 1; `CCLManager` fields `ag_ping_pong_idx`, `sr_ping_pong_idx`, `np_ping_pong_idx`, `_ping_pong_buffer_indices` (dict), and the semaphore dicts (`ag_ping_pong_semaphores`, etc.).
- Produces: a written diagnosis — which phase state differs between capture and replay ≥2, and whether it is (i) Python ping-pong indices, (ii) the `__init__` allocation-run encode advancing the phase before capture, or (iii) device semaphore values not self-resetting across replays.

- [ ] **Step 1: Snapshot the encoder CCLManager phase at capture vs each replay**

Temporarily add, in `encode_prompt` (guarded by an env flag so it is easy to strip), a log of the manager phase right before the traced forward:
```python
        import os as _os
        if _os.environ.get("FIBO_TRACE_DEBUG"):
            from loguru import logger as _lg
            _lg.info(
                f"[trace-dbg] bucket={bucket} captured={bucket in self._tracers and self._tracers[bucket].trace_captured} "
                f"ag_idx={getattr(self._ccl_manager, 'ag_ping_pong_idx', None)} "
                f"buf_idx={dict(getattr(self._ccl_manager, '_ping_pong_buffer_indices', {}))}"
            )
```
(Store the passed `ccl_manager` as `self._ccl_manager` in `__init__` if not already, for the probe.)

Run the stability test with `FIBO_TRACE_DEBUG=1 -s`. Record the phase (ag_idx, buffer indices) at: the `__init__` allocation-run encode(s), the capture encode, and each replay.

- [ ] **Step 2: Compare against the denoise trace's phasing**

Read how the denoise path stays stable: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` `_traced_step` / `_denoise_traced` / `_ensure_trace`, and the `__init__` notes on why encoder/transformer/VAE each get their **own** CCLManager. Determine what guarantees the denoise trace has that the encoder trace lacks (candidate: the transformer manager is untouched-untraced after capture; the encoder manager is advanced by the `__init__` allocation-run encode before capture, or the semaphores are shared across the pos/neg replays without reset).

- [ ] **Step 3: Write the diagnosis**

Record, in the Task-3 commit message and inline in the design doc's outcome note, the confirmed cause and which fix — (a) phase-reset/isolate the encoder manager, or (b) non-ping-pong traced all-gather — it dictates. Strip the temporary `FIBO_TRACE_DEBUG` probe (keep `self._ccl_manager` only if the fix needs it).

- [ ] **Step 4: No commit** (investigation only). The probe is removed; the diagnosis feeds Task 3.

---

### Task 3: Apply the fix and turn the stability gate green

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/text_encoder.py` (and, only if fix (b), `models/tt_dit/parallel/manager.py` or the encoder's all-gather calls)
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_encode_replay_stable`

**Interfaces:**
- Consumes: the Task-2 diagnosis; the Task-1 tracer scaffolding.
- Produces: a traced encoder whose replay stays PCC ≥ 0.99 across all N encodes.

**Land the fix Task 2 indicated. Both candidates are fully specified below; apply exactly one.**

- [ ] **Step 1 (Fix a — preferred): dedicate + phase-reset the encoder CCLManager around capture**

Ensure the encoder's CCLManager is in a known ping-pong phase when the trace captures, and that nothing untraced advances it afterward. In the wrapper, immediately before creating/capturing the tracer, reset the manager's ping-pong phase:
```python
    def _reset_ccl_phase(self) -> None:
        """Put the encoder CCLManager in a deterministic ping-pong phase so the captured trace's
        baked buffer/semaphore selection matches every replay (execute_trace does not advance the
        Python indices)."""
        m = self._ccl_manager
        for name in ("ag_ping_pong_idx", "rs_ping_pong_idx", "rs_ping_pong_idx_fused",
                     "np_ping_pong_idx", "sr_ping_pong_idx", "exp_ring_ping_pong_idx", "barrier_idx"):
            if hasattr(m, name):
                setattr(m, name, [0, 0])
        if hasattr(m, "_ping_pong_buffer_indices"):
            for k in m._ping_pong_buffer_indices:
                m._ping_pong_buffer_indices[k] = 0
```
Call `self._reset_ccl_phase()` right before the `tracer = Tracer(...)` creation in `encode_prompt` (capture path only, i.e. when `self._tracers.get(bucket) is None`). If Task 2 shows the `__init__` allocation-run encode is what desyncs the phase, additionally give the traced encoder its **own** CCLManager created inside the wrapper (constructed from `self._device`) instead of the shared one, so the allocation run and the trace do not share ping-pong state.

- [ ] **Step 1 (Fix b — fallback, only if a is not deterministic): non-ping-pong traced all-gather**

If phase-reset is insufficient (device semaphores do not self-reset across replays), force the encoder's traced all-gathers to a single fixed buffer + fixed semaphore set. In `SmolLM3Attention.forward` / MLP / the encoder's `all_gather_persistent_buffer` calls, thread a `use_ping_pong=False` (or fixed-index) option so capture and replay always select buffer/semaphore index 0. Add the minimal `CCLManager` support for a fixed-index all-gather if not present. Keep this scoped to the encoder's traced path; the denoise path is unchanged.

- [ ] **Step 2: Run the stability gate — must be GREEN**

Run (4×8 Galaxy):
`TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python_env/bin/pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_encode_replay_stable" -v -rA -s`
Expected: **PASS** — all 5 encodes (capture + 4 replays) at PCC ≥ 0.99. If still failing on fix (a), switch to fix (b) and re-run.

- [ ] **Step 3: Confirm the untraced path is unaffected**

Run: `... python_env/bin/pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_encode" -v` (untraced) → PASS at PCC 0.99 (unchanged), and `test_smollm3_encoder_sp` → PASS.

- [ ] **Step 4: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/text_encoder.py
git commit -m "fix(fibo-pipeline): stable encoder trace replay via <fix a|b> (root cause: <cause>)"
```

---

### Task 4: Perf validation + pipeline integration

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` (pass `use_trace=True` to the wrapper)
- Test: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_perf`; `models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc`

**Interfaces:**
- Consumes: the stable traced wrapper (Task 3).
- Produces: the pipeline building the wrapper with `use_trace=True`; a measured encode wall-clock drop.

- [ ] **Step 1: Wire `use_trace` into the pipeline**

In `pipeline_bria_fibo.py` `__init__`, where `SmolLM3TextEncoderWrapper(...)` is constructed, add `use_trace=True`. (The pipeline's `_DEVICE_PARAMS`-equivalent already provisions `trace_region_size`; the encoder trace uses the encoder's CCLManager, separate from the denoise trace's — confirm capture happens after the `__init__` allocation run, matching Task 2's fix.)

- [ ] **Step 2: Measure encode perf (headline)**

Run (4×8 Galaxy, `-s`):
`TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HUB_OFFLINE=1 python_env/bin/pytest "models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_perf" -k "mesh_device1" -v -s --timeout=1800`
Expected: encode wall-clock drops vs the untraced baseline (record both numbers). Per the design's honest-risk note: if the drop is negligible (encode already prep/readback-bound), record it — the flag still lets us keep or drop the trace on its merits.

- [ ] **Step 3: No denoise / pipeline regression**

Run: `... python_env/bin/pytest "models/tt_dit/tests/models/bria_fibo/test_pipeline.py::test_fibo_pipeline_latent_pcc" -k "mesh_device1" -v --timeout=1800`
Expected: PASS at PCC 0.99 for both `guidance_scale` params — the encoder trace (separate CCLManager) does not disturb the resident denoise trace.

- [ ] **Step 4: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py
git commit -m "perf(fibo-pipeline): enable traced SmolLM3 encoder in the pipeline"
```

---

## Self-Review

**Spec coverage:**
- Phase-split + single-bucket Tracer + single output → Task 1 (phase-split already exists; Task 1 adds the Tracer + flag + single-output confirmed).
- Root-cause the replay-noise (CCL ping-pong phase) → Task 2.
- The fix (property: identical phase every traced forward; candidates a/b) → Task 3, both fully specified.
- Guard flag `use_trace` → Task 1 (add) + Task 4 (enable in pipeline); profile/torch paths stay untraced (default `False`).
- Correctness gate (N sequential encodes, PCC 0.99) → Task 1 test, green in Task 3.
- Verification (encode perf drop; `test_smollm3_encoder_sp`; pipeline latent-PCC) → Tasks 3–4.

**Placeholder scan:** Task 3's fix is investigation-gated by design (both candidates carry complete code); the `<fix a|b>` / `<cause>` in the commit message are filled from Task 2's finding, not code placeholders. No TODO/TBD in executable steps.

**Type consistency:** `use_trace: bool` (Task 1 add → Task 4 set). `_forward(tt_ids, tt_cos, tt_sin) -> ttnn.Tensor` single output throughout. `self._tracers: dict[int, Tracer]` keyed by bucket. `_reset_ccl_phase` (Task 3a) operates on the documented `CCLManager` fields. Test name `test_fibo_wrapper_encode_replay_stable` consistent across Tasks 1/3.
