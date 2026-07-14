# FIBO SmolLM3 encoder: tensor-parallel on the mesh

## Goal

Run the Bria FIBO pipeline's SmolLM3 text encoder **tensor-parallel across the mesh**
instead of fully replicated, so on the 4×8 Blackhole Galaxy the encoder shards its
attention/MLP compute across 8 devices (axis 1) rather than replicating on all 32.

## Background

The encoder is currently configured fully replicated:

```python
# pipeline_bria_fibo.py, BriaFiboPipelineConfig.default()
encoder_parallel_config = EncoderParallelConfig.from_tuple((1, tp_axis))  # factor 1 -> replicated
```

At factor 1, `SmolLM3Context` sets `tp_axis=None` and the (already-provided) `ccl_manager`
goes unused. `SmolLM3TextEncoder` already **fully supports** tensor parallelism:

* `SmolLM3Attention` shards Q/K/V/O over `tp_axis`, and `optimal_groups`/`split_factor`
  pad the GQA heads so an arbitrary tp factor works despite only 4 KV heads
  (16 q / 4 kv → tp=8 uses `split_factor=2`, done automatically at weight-load).
* `SmolLM3MLP` / attention all-gather along `tp_axis` via `ccl_manager`.
* Optional FSDP over the other axis exists (`is_fsdp`) but is **not** used here.

This is validated: `tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_full_mesh`
runs the encoder at `factor = mesh_device.shape[tp_axis]`, `tp_axis=1`, on a 2×2 mesh
(seq 128 and 2048), PCC-checked against HF. Scheme is Wan-consistent
(`test_performance_wan.py` uses `EncoderParallelConfig.from_tuple((h_factor, tp_axis))`).

## Chosen approach: scheme A, mesh-general

Shard on **axis 1** (the DiT's `tp_axis`) with **factor = `mesh[1]`**:

* 2×2 → tp=2
* 4×8 → tp=8

Rejected alternatives:

* **Scheme B** (tp=4 on axis 0 + FSDP over axis 1): cleaner head math and weight sharding,
  but the encoder's parallel axis would differ from the transformer's `tp_axis`, and it
  activates the currently-unused `is_fsdp` plumbing. More surface area for no measured need.
* **4×8-only special case** (keep 2×2 replicated): would keep 2×2 byte-identical, but adds a
  mesh-shape conditional in the config for no benefit — tp=2 on 2×2 is already PCC-validated.

## Changes

### 1. Pipeline default — `pipelines/bria_fibo/pipeline_bria_fibo.py`

In `BriaFiboPipelineConfig.default()` (~L106), replace:

```python
encoder_parallel_config = EncoderParallelConfig.from_tuple((1, tp_axis))
```

with:

```python
encoder_parallel_config = EncoderParallelConfig.from_tuple((tp_factor, tp_axis))
```

`tp_factor` (= `mesh[tp_axis]`) and `tp_axis` are already computed just above for the DiT
config. Update the adjacent "fully replicated" comment and the module docstring line
("SmolLM3 text encoder … replicated on the submesh") to describe the tensor-parallel config.
No other pipeline change: `__init__` already passes `ccl_manager=self._ccl_manager` to the
`SmolLM3TextEncoderWrapper`, which is a no-op at factor 1 and simply becomes active at
factor > 1.

### 2. Encoder-only device-profile test — `tests/models/bria_fibo/test_performance_bria_fibo.py`

In `test_fibo_encode_device_profile` (~L517), replace the hardcoded replicated config:

```python
parallel_config=EncoderParallelConfig.from_tuple((1, 1))
```

with the mesh-derived tp config (matching `test_smollm3_encoder_full_mesh`):

```python
parallel_config=EncoderParallelConfig.from_tuple((mesh_device.shape[1], 1))
```

Update the test docstring/comment that describe the encoder as "replicated".

## Behavioral consequences

* **2×2**: encoder goes from replicated (tp=1) to tp=2. PCC-validated by
  `test_smollm3_encoder_full_mesh`; changes 2×2 pipeline behavior (intended, mesh-general).
* **4×8**: encoder shards tp=8 on axis 1, uses the existing `_num_links` (2 on 4×8) CCL path.

## Out of scope

* No FSDP / weight sharding.
* No changes to `SmolLM3TextEncoder` numerics or kernels.
* No new correctness test (existing `test_smollm3_encoder_full_mesh` covers the tp path;
  the FIBO pipeline correctness test exercises the full stack).
* **Running anything** — a sweep is in progress; this spec + implementation is edits only.

## Verification (deferred until the sweep frees the hardware)

* `test_smollm3_encoder_full_mesh` on 4×8 (add/confirm the 4×8 param) — PCC vs HF.
* `test_fibo_encode_device_profile -k mesh_device1` — encode profile on 4×8.
* FIBO pipeline correctness/perf on both meshes — image non-degenerate.
