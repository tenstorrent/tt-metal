# FIBO Encoder Trace (capture/replay) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the SmolLM3 encoder device forward as a ttnn trace and replay it, eliminating the host op-dispatch overhead that dominates encode wall-clock, validated via `test_fibo_encode_perf`.

**Architecture:** `SmolLM3TextEncoderWrapper.encode_prompt` splits into host-prep → traced device forward (wrapped in the existing `models/tt_dit/utils/tracing.py::Tracer`, one per padding bucket) → host readback. The SP causal bias (currently `from_torch`'d inside the forward) is cached in the encoder so it is built during the tracer's `prep_run`, never inside the captured region. Fixed 1024-bucket padding makes the forward a static shape, so one trace serves pos+neg and every run.

**Tech Stack:** ttnn traces (`begin/end_trace_capture`, `execute_trace`), the existing `Tracer` helper, PyTorch, pytest. Target: 4×8 Blackhole Galaxy.

## Global Constraints

- SP=8 (axis 1) × TP=4 (axis 0) whole-mesh encoder; 1024-token bucket; these are already shipped and unchanged.
- Trace replay must be numerically identical to the untraced path (same captured ops); correctness gate `assert_quality(pcc >= 0.99)` vs HF.
- The untraced path (`use_trace=False`, `use_torch=True`) must remain byte-for-byte the current behavior.
- Trace capture requires `trace_region_size > 0` at device open. `_DEVICE_PARAMS` (perf test) sets 200 MB; `_PROFILE_DEVICE_PARAMS` (device-profile test) sets none → that test must run untraced.
- Device test prefix: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest ...`. Reset the mesh with `tt-smi -r` before a run if a prior run left it wedged.

---

### Task 1: Cache the SP causal bias in the encoder

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` — `SmolLM3TextEncoder.__init__` (add cache dict) and `forward` (build-once/reuse).
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — existing `test_smollm3_encoder_sp` must still pass (no numeric regression); add `test_smollm3_sp_bias_cached` (bias built once, reused).

**Interfaces:**
- Consumes: `build_sp_causal_bias(seq_local, sp_factor, *, device, sp_axis)` (existing).
- Produces: `SmolLM3TextEncoder` builds the SP causal bias at most once per local seq length and reuses it; `self._sp_bias_cache: dict[int, ttnn.Tensor]`.

- [ ] **Step 1: Write the failing test**

Add to `test_smollm3.py` (device test; reuses the SP setup):

```python
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_smollm3_sp_bias_cached(*, mesh_device):
    """The SP causal bias is built once per local seq length and reused across forwards."""
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    sp_axis, tp_axis, seq = 1, 0, 512
    sp_factor, tp_factor = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
    hf = _load_hf_smollm3()
    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig.from_tuples(tp=(tp_factor, tp_axis), sp=(sp_factor, sp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(
        tokens, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, sp_axis]
    )
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device, mesh_axes=[None, None, sp_axis, None])

    enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    assert len(enc._sp_bias_cache) == 1
    first = next(iter(enc._sp_bias_cache.values()))
    enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    assert len(enc._sp_bias_cache) == 1
    assert next(iter(enc._sp_bias_cache.values())) is first  # same tensor object reused
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_sp_bias_cached" -v`
Expected: FAIL (`AttributeError: ... _sp_bias_cache`).

- [ ] **Step 3: Write minimal implementation**

In `SmolLM3TextEncoder.__init__`, add after `self._rope_theta = config.rope_theta`:

```python
self._sp_bias_cache: dict[int, ttnn.Tensor] = {}
```

In `SmolLM3TextEncoder.forward`, replace the SP branch:

```python
if self._sp_factor > 1:
    # Per-shard rectangular causal bias is constant for a given local seq length; build it once and
    # reuse. Caching keeps it out of a captured trace's region (it is created during the tracer's
    # prep_run, then only read inside capture/replay).
    padded = seq_len
    attention_bias = self._sp_bias_cache.get(seq_len)
    if attention_bias is None:
        attention_bias = build_sp_causal_bias(
            seq_len, self._sp_factor, device=self._device, sp_axis=self._sp_axis
        )
        self._sp_bias_cache[seq_len] = attention_bias
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_sp_bias_cached" "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: all PASS (cache test + both SP-axis PCC cases still ≥ 0.99).

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "perf(fibo-pipeline): cache SP causal bias in SmolLM3 encoder (trace-safe)"
```

---

### Task 2: Trace the encoder forward in the wrapper

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/text_encoder.py` — import `Tracer`; add `use_trace` ctor flag + `self._tracers`; split `encode_prompt` into `_prep_inputs` / `_forward` / readback.
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — add `test_fibo_wrapper_traced` (traced encode matches HF; capture then replay).

**Interfaces:**
- Consumes: `Tracer(function, *, device, prep_run, clone_prep_inputs)` from `models/tt_dit/utils/tracing.py`; `pick_bucket`; the SP-capable encoder (Task 1).
- Produces: `SmolLM3TextEncoderWrapper(..., use_trace: bool = True)`; `encode_prompt` returns the same `(prompt_embeds, list_hidden_states)` contract, traced when `use_trace and not use_torch`.

- [ ] **Step 1: Write the failing test**

Add to `test_smollm3.py`:

```python
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 90000000}],
    indirect=["device_params"],
)
def test_fibo_wrapper_traced(*, mesh_device):
    """Traced encode_prompt (capture then replay) matches HF; both prompts share one 1024 trace."""
    from models.tt_dit.pipelines.bria_fibo.text_encoder import SmolLM3TextEncoderWrapper

    sp_axis, tp_axis = 1, 0
    pc = EncoderParallelConfig.from_tuples(
        tp=(mesh_device.shape[tp_axis], tp_axis), sp=(mesh_device.shape[sp_axis], sp_axis)
    )
    ckpt = os.environ.get("FIBO_PATH", "briaai/FIBO")
    try:
        from huggingface_hub import snapshot_download

        ckpt = snapshot_download(ckpt, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO unavailable: {e}")

    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    wrapper = SmolLM3TextEncoderWrapper(
        ckpt, device=mesh_device, ccl_manager=ccl, parallel_config=pc, pad_buckets=(1024,), use_trace=True
    )
    hf = _load_hf_smollm3()

    for prompt in ("a luxury sports car", "blurry, low quality"):  # 1st captures, 2nd replays
        with torch.no_grad():
            ref = hf.model(input_ids=hf_tokens := wrapper.tokenizer([prompt], return_tensors="pt").input_ids,
                           output_hidden_states=True)
        ref_embeds = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()
        embeds, hidden = wrapper.encode_prompt(prompt)
        assert list(embeds.shape) == list(ref_embeds.shape)
        assert_quality(ref_embeds, embeds.float(), pcc=0.99, relative_rmse=0.2)
    assert set(wrapper._tracers) == {1024} and wrapper._tracers[1024].trace_captured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_traced" -v`
Expected: FAIL (`SmolLM3TextEncoderWrapper.__init__() got an unexpected keyword argument 'use_trace'`).

- [ ] **Step 3: Write minimal implementation**

In `text_encoder.py`, add the import near the top:

```python
from ...utils.tracing import Tracer
```

In `SmolLM3TextEncoderWrapper.__init__`, add the flag + tracer store (after `self._sp_factor = ...`):

```python
self._use_trace = use_trace and not use_torch
self._tracers: dict[int, Tracer] = {}
```
and add `use_trace: bool = True` to the signature (before `use_torch`).

Replace the device branch of `encode_prompt` (everything after the `use_torch` early-return) with:

```python
        tt_ids, tt_cos, tt_sin, bucket = self._prep_inputs(input_ids, seq_len)

        if self._use_trace:
            tracer = self._tracers.get(bucket)
            if tracer is None:
                tracer = Tracer(self._forward, device=self._device, prep_run=True, clone_prep_inputs=False)
                self._tracers[bucket] = tracer
            outputs = tracer(tt_ids, tt_cos, tt_sin)
        else:
            outputs = self._forward(tt_ids, tt_cos, tt_sin)

        prompt_embeds, all_hidden_states = outputs[0], list(outputs[1:])
        gather = (
            dict(mesh_axes=[None, self._sp_axis, None], composer_device=self._device) if self._sp_factor > 1 else {}
        )
        host_prompt_embeds = tt_tensor.to_torch(prompt_embeds, **gather)[:, :seq_len, :]
        host_hidden_states = [tt_tensor.to_torch(h, **gather)[:, :seq_len, :] for h in all_hidden_states]
        return host_prompt_embeds, host_hidden_states

    def _prep_inputs(self, input_ids, seq_len):
        """Host prep: pad to bucket, build RoPE, move to device (sharded on the SP axis)."""
        bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
        padded_ids = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
        cos, sin = self._encoder.create_rope_tensors(1, bucket)
        if self._sp_factor > 1:
            tt_ids = tt_tensor.from_torch(
                padded_ids, device=self._device, dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, self._sp_axis],
            )
            tt_cos = tt_tensor.from_torch(cos, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
            tt_sin = tt_tensor.from_torch(sin, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        else:
            tt_ids = tt_tensor.from_torch(
                padded_ids, device=self._device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            tt_cos = tt_tensor.from_torch(cos, device=self._device)
            tt_sin = tt_tensor.from_torch(sin, device=self._device)
        return tt_ids, tt_cos, tt_sin, bucket

    def _forward(self, tt_ids, tt_cos, tt_sin):
        """Device forward (the traced unit): returns a flat tuple (prompt_embeds, *hidden_states)."""
        prompt_embeds, all_hidden_states = self._encoder.encode(
            tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin)
        )
        return (prompt_embeds, *all_hidden_states)
```

(Keep the existing block comment about bucket padding / attention_mask=None above `_prep_inputs`'s logic or move it there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_traced" -v`
Expected: PASS (both prompts PCC ≥ 0.99; one 1024 trace captured). If the `Tracer` rejects the 38-tensor tuple output, extend `Tracer`/`_tree_map` to accept `tuple` outputs (it already tree-maps, so this should just work).

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/text_encoder.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "perf(fibo-pipeline): trace the SmolLM3 encoder forward (per-bucket capture/replay)"
```

---

### Task 3: Keep the device-profile untraced, then measure the perf win

**Files:**
- Modify: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` — pass `use_trace=False` in `test_fibo_encode_device_profile`; note the traced default in `test_fibo_encode_perf`'s docstring.

**Interfaces:**
- Consumes: the traced wrapper (Task 2), which the pipeline builds with the default `use_trace=True` (no pipeline change needed).

- [ ] **Step 1: Make the op-profile test untraced**

In `test_fibo_encode_device_profile`, add `use_trace=False` to the `SmolLM3TextEncoderWrapper(...)` construction (it uses `_PROFILE_DEVICE_PARAMS`, which has no trace region, and it wants real per-op timings):

```python
    encoder = SmolLM3TextEncoderWrapper(
        ckpt,
        device=mesh_device,
        ccl_manager=ccl,
        parallel_config=EncoderParallelConfig.from_tuple((mesh_device.shape[1], 1)),
        use_trace=False,
    )
```

- [ ] **Step 2: Update the perf-test docstring**

In `test_fibo_encode_perf`, add a sentence: the encoder forward is captured as a ttnn trace (warmup captures at the 1024 bucket, measured runs replay), so the measured wall-clock reflects trace replay + host readback.

- [ ] **Step 3: Run the encode perf test (4×8) and record the speedup**

Run:
```bash
tt-smi -r >/dev/null 2>&1
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  "models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_perf" \
  -k "mesh_device1" -v -s --timeout=1800
```
Expected: PASS; "FIBO encode perf" wall-clock materially below the ~12.5 s untraced baseline (trace replay removes the host op-dispatch gaps). Record the new number.

- [ ] **Step 4: Verify the op-profile test still runs untraced (optional, Tracy build only)**

If a Tracy build is available, confirm `test_fibo_encode_device_profile` still produces per-op rows (i.e. it did not attempt to trace). Otherwise rely on Step 1's `use_trace=False`.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py
git commit -m "test(fibo-pipeline): encode perf uses the traced encoder; device-profile stays untraced"
```

---

## Self-Review

**Spec coverage:**
- Trace device forward only → Tasks 1–2. ✅
- Reuse `Tracer`, keyed by bucket → Task 2. ✅
- SP bias hoisted out of the traced region (cache) → Task 1. ✅
- `use_trace` guard; profile/torch paths untraced → Tasks 2–3. ✅
- Validate via `test_fibo_encode_perf` + traced-correctness test → Tasks 2–3. ✅
- Pipeline benefits for free (default `use_trace=True`) → Task 3 (no pipeline edit needed). ✅

**Placeholder scan:** none — every step has concrete code/commands. The only conditional is the `Tracer` tuple-output extension in Task 2 Step 4, gated on an observed failure with a specific fix (`_tree_map` already handles tuples, so likely a no-op).

**Type consistency:** `_prep_inputs -> (tt_ids, tt_cos, tt_sin, bucket)`, `_forward -> tuple`, `self._tracers: dict[int, Tracer]`, `use_trace`, `self._sp_bias_cache: dict[int, ttnn.Tensor]` are used consistently across tasks.
