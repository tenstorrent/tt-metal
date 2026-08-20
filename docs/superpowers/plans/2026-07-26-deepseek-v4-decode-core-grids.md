# DeepSeek V4 Decode Core Grids Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use regular row-wise core ranges for every `matmul_decode` shard grid and restore rectangular shard grids at RMSNorm boundaries.

**Architecture:** Add private grid/configuration helpers to `layers.py`, use them in both decode linear classes, and centralize normalization-boundary resharding in one helper called by weighted and unweighted RMSNorm. Unit tests exercise the helpers without requiring accelerator hardware.

**Tech Stack:** Python, TTNN tensor/memory configuration APIs, pytest.

## Global Constraints

- Preserve tensor shapes, shard shapes, memory layouts, buffer types, orientations, dtypes, and numerical behavior.
- Modify only DeepSeek V4 Flash layer sharding and focused tests.
- Do not alter the user's existing uncommitted changes.

---

### Task 1: Core-grid and normalization-boundary helpers

**Files:**
- Create: `models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py`
- Modify: `models/experimental/deepseek_v4_flash/tt/layers.py:1-20`

**Interfaces:**
- Produces: `_decode_core_range_set(num_cores: int, device) -> ttnn.CoreRangeSet`
- Produces: `_decode_width_sharded_l1_config(height: int, width: int, device, num_cores: int | None = None) -> ttnn.MemoryConfig`
- Produces: `_reshard_to_rectangular_grid(x: ttnn.Tensor) -> ttnn.Tensor`

- [ ] **Step 1: Write failing helper tests**

Use these fixtures and assertions:

```python
class FakeDevice:
    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 8)


class FakeTensor:
    def __init__(self, device, memory_config=None):
        self._device = device
        self._memory_config = memory_config

    def is_sharded(self):
        return self._memory_config is not None

    def memory_config(self):
        return self._memory_config

    def device(self):
        return self._device


def test_decode_core_range_uses_row_wise_non_rectangular_grid():
    device = FakeDevice()
    expected = ttnn.num_cores_to_corerangeset(10, device.compute_with_storage_grid_size(), row_wise=True)
    assert _decode_core_range_set(10, device) == expected
    assert expected != rectangular_core_range_set(10, device)


def test_reshard_to_rectangular_grid_preserves_shard_geometry(monkeypatch):
    device = FakeDevice()
    grid = _decode_core_range_set(10, device)
    source = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tensor = FakeTensor(device, source)
    sentinel = object()
    captured = {}
    monkeypatch.setattr(ttnn, "to_memory_config", lambda x, mc: captured.setdefault("mc", mc) or sentinel)
    result = _reshard_to_rectangular_grid(tensor)
    target = captured["mc"]
    assert result is target
    assert target.shard_spec.grid == rectangular_core_range_set(10, device)
    assert target.shard_spec.shape == source.shard_spec.shape
    assert target.shard_spec.orientation == source.shard_spec.orientation
    assert target.memory_layout == source.memory_layout
    assert target.buffer_type == source.buffer_type
```

- [ ] **Step 2: Run tests and verify the missing-helper failure**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
```

Expected: collection fails because `_decode_core_range_set`, `_decode_width_sharded_l1_config`, and `_reshard_to_rectangular_grid` do not exist.

- [ ] **Step 3: Implement the helpers**

Add:

```python
def _decode_core_range_set(num_cores: int, device) -> ttnn.CoreRangeSet:
    return ttnn.num_cores_to_corerangeset(
        num_cores, device.compute_with_storage_grid_size(), row_wise=True
    )


def _decode_width_sharded_l1_config(
    height: int, width: int, device, num_cores: Optional[int] = None
) -> ttnn.MemoryConfig:
    config = width_sharded_l1_config(height, width, device, num_cores)
    shard_spec = config.shard_spec
    grid = _decode_core_range_set(shard_spec.grid.num_cores(), device)
    return ttnn.MemoryConfig(
        config.memory_layout,
        config.buffer_type,
        ttnn.ShardSpec(grid, shard_spec.shape, shard_spec.orientation),
    )


def _reshard_to_rectangular_grid(x: ttnn.Tensor) -> ttnn.Tensor:
    if not x.is_sharded():
        return x
    memory_config = x.memory_config()
    shard_spec = memory_config.shard_spec
    rectangular_grid = rectangular_core_range_set(shard_spec.grid.num_cores(), x.device())
    if shard_spec.grid == rectangular_grid:
        return x
    target = ttnn.MemoryConfig(
        memory_config.memory_layout,
        memory_config.buffer_type,
        ttnn.ShardSpec(rectangular_grid, shard_spec.shape, shard_spec.orientation),
    )
    return ttnn.to_memory_config(x, target)
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
```

Expected: all helper tests pass.

---

### Task 2: Apply regular grids to all decode matmuls

**Files:**
- Modify: `models/experimental/deepseek_v4_flash/tt/layers.py:45-291`
- Modify: `models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py`

**Interfaces:**
- Consumes: `_decode_core_range_set`
- Consumes: `_decode_width_sharded_l1_config`

- [ ] **Step 1: Add failing source-level configuration tests**

Add direct configuration tests:

```python
def test_linear_decode_input_uses_regular_grid():
    layer = LinearDecode.__new__(LinearDecode)
    layer.device = FakeDevice()
    layer.num_inputA_cores = 10
    config = layer.get_input_memory_config(1, 320)
    assert config.shard_spec.grid == _decode_core_range_set(10, layer.device)


def test_batched_linear_decode_input_uses_regular_grid():
    layer = BatchedLinearDecode.__new__(BatchedLinearDecode)
    layer.device = FakeDevice()
    layer.batch = 2
    layer.K = 320
    layer.num_inputA_cores = 10
    config = layer.get_input_memory_config(1)
    assert config.shard_spec.grid == _decode_core_range_set(10, layer.device)


def test_decode_width_sharded_output_uses_regular_grid():
    device = FakeDevice()
    config = _decode_width_sharded_l1_config(1, 320, device, num_cores=10)
    assert config.shard_spec.grid == _decode_core_range_set(10, device)
```

- [ ] **Step 2: Run tests and verify rectangular-grid assertions fail**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
```

Expected: failures show decode configurations still use `rectangular_core_range_set`.

- [ ] **Step 3: Replace decode grid construction**

In `LinearDecode`, replace rectangular grid creation for weight B, input A, and partial output with `_decode_core_range_set`. Replace the non-partial output call to `width_sharded_l1_config` with `_decode_width_sharded_l1_config`.

In `BatchedLinearDecode`, replace rectangular grid creation for weight B and input A with `_decode_core_range_set`.

- [ ] **Step 4: Run decode-grid tests**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
```

Expected: all decode-grid tests pass.

---

### Task 3: Reshard both RMSNorm paths and verify

**Files:**
- Modify: `models/experimental/deepseek_v4_flash/tt/layers.py:294-331`
- Modify: `models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py`

**Interfaces:**
- Consumes: `_reshard_to_rectangular_grid`

- [ ] **Step 1: Add failing RMSNorm-boundary tests**

Add:

```python
def test_weighted_rms_norm_uses_rectangularized_input(monkeypatch):
    layer = DeepSeekV4RMSNorm.__new__(DeepSeekV4RMSNorm)
    layer.sharded = False
    layer.weight = object()
    layer.eps = 1e-6
    source, rectangularized, output = object(), object(), object()
    monkeypatch.setattr(layers, "_reshard_to_rectangular_grid", lambda x: rectangularized)
    monkeypatch.setattr(ttnn, "rms_norm", lambda x, **kwargs: output if x is rectangularized else None)
    assert layer.forward(source) is output


def test_unweighted_rms_norm_uses_rectangularized_input(monkeypatch):
    source, rectangularized, output = object(), object(), object()
    monkeypatch.setattr(layers, "_reshard_to_rectangular_grid", lambda x: rectangularized)
    monkeypatch.setattr(ttnn, "rms_norm", lambda x, **kwargs: output if x is rectangularized else None)
    assert _rms_norm_unweighted(source, 1e-6) is output
```

- [ ] **Step 2: Run tests and verify RMSNorm receives the original tensor**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
```

Expected: both tests fail because the reshard helper is not called at the normalization boundary.

- [ ] **Step 3: Add normalization-boundary resharding**

In `DeepSeekV4RMSNorm.forward`, preserve the current optional width-sharding policy, then call:

```python
x = _reshard_to_rectangular_grid(x)
return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
```

In `_rms_norm_unweighted`, call:

```python
x = _reshard_to_rectangular_grid(x)
return ttnn.rms_norm(x, epsilon=eps)
```

- [ ] **Step 4: Run focused tests and formatting**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py -v
pre-commit run black --files models/experimental/deepseek_v4_flash/tt/layers.py models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py
pre-commit run isort --files models/experimental/deepseek_v4_flash/tt/layers.py models/experimental/deepseek_v4_flash/tests/test_layers_core_grids.py
```

Expected: tests and formatting hooks pass.

- [ ] **Step 5: Run the focused existing smoke test**

Run:

```bash
pytest models/experimental/deepseek_v4_flash/tests/test_weight_loader.py -v
```

Expected: existing focused tests pass or skip for unavailable hardware/checkpoints without a code failure.
