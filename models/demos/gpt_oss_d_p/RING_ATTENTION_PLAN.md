# Ring Attention for GPT-OSS Prefill

## Problem

The current prefill uses plain `scaled_dot_product_attention` with only local K/V.
With SP=4, each mesh row holds a different token shard (e.g. row 0: tokens 0–6399,
row 1: tokens 6400–12799). Token 6400 on row 1 only attends to tokens 6400–12799 —
it cannot see tokens 0–6399 on row 0. This is incorrect and produces degraded
first-token quality near shard boundaries.

DeepSeek V3 d_p solves this with `ring_joint_scaled_dot_product_attention`, which
fuses a cross-row K/V all-gather inside the attention kernel so every token attends
to the full causal context regardless of which SP row it lives on.

## The Op

`ttnn.transformer.ring_joint_scaled_dot_product_attention` is **not MLA-specific**.
DeepSeek passes compressed MLA tensors; GPT-OSS would pass standard MHA tensors.
The same op handles both by accepting variable K/V shapes.

```
ring_joint_scaled_dot_product_attention(
    input_tensor_q,               # [batch, num_heads_local, seq_local, head_dim]
    input_tensor_k,               # [batch, num_kv_heads_local, seq_local, head_dim]
    input_tensor_v,               # [batch, num_kv_heads_local, seq_local, head_dim]
    joint_tensor_q,               # dummy: same shape with seq=0
    joint_tensor_k,               # dummy: same shape with seq=0
    joint_tensor_v,               # dummy: same shape with seq=0
    persistent_output_buffer_k,   # [batch, num_kv_heads_local, seq_total, head_dim]
    persistent_output_buffer_v,   # [batch, num_kv_heads_local, seq_total, head_dim]
    joint_strategy="rear",
    logical_n=seq_total,          # key: full sequence length, not local shard
    program_config=...,
    compute_kernel_config=...,
    dim=2,
    multi_device_global_semaphore=semaphore_handles,  # 2 GlobalSemaphores
    num_links=num_links,
    cluster_axis=sp_axis,         # 0 for row-SP
    mesh_device=mesh_device,
    topology=topology,
    subdevice_id=worker_sub_device_id,
    ccl_core_grid_offset=(grid_x - 1, 0),  # one column reserved for CCL
    use_column_major_ccl=True,
    is_causal=True,
    scale=head_dim ** -0.5,
) -> (output, joint_out, lse)  # only output is used
```

The `logical_n = seq_total` parameter is what makes cross-row attention correct:
the causal mask is applied against the full sequence so token 6400 attends to
positions 0–6400, not just 6400–12799.

Reference call site: `models/demos/deepseek_v3_d_p/tt/mla/mla.py` lines 781–806.
Reference buffer allocation: `models/demos/deepseek_v3_d_p/tt/tt_ccl.py` lines 138–199.

## What Needs to Change

### 1. `tt/ccl.py` — add ring attention infrastructure

`CCLManager` already has RS and AG semaphores. Needs three additions:

**a) 2 GlobalSemaphores for ring attention double-buffering**
```python
ring_attn_semaphores = [
    ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
    ttnn.create_global_semaphore(mesh_device, ccl_cores, 0),
]
```

**b) SubDevice + SubDeviceId** — the ring op requires a registered worker subdevice:
```python
worker_sub_device = ttnn.SubDevice([ccl_core_range_set])
worker_sub_device_id = ttnn.SubDeviceId(0)
# register with sub_device_manager
```

**c) Core grid offset** — one column of the 8×8 grid is reserved for CCL ring
traffic; compute (Q/K/V matmuls, SDPA) uses the remaining 7 columns:
```python
compute_grid = mesh_device.compute_with_storage_grid_size()  # (8, 8) on BH
ring_attn_ccl_grid_offset = ttnn.CoreCoord(compute_grid.x - 1, 0)  # (7, 0)
```

### 2. `tt/attention/__init__.py` — pre-allocate persistent buffers at init

Each `Attention` layer needs its own set of 5 tensors allocated once at `__init__`
time. They live in DRAM and are overwritten by the ring op on every call.

```python
# In Attention.__init__, when sp_factor > 1:
seq_total = config.max_seq_len  # full sequence (all SP rows combined)
num_kv_heads_local = config.num_kv_heads // mesh_config.tp

# Real buffers — written by the ring op during gather
self.persistent_k = ttnn.as_tensor(
    torch.zeros(1, num_kv_heads_local, seq_total, config.head_dim),
    device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
self.persistent_v = ttnn.as_tensor(
    torch.zeros(1, num_kv_heads_local, seq_total, config.head_dim),
    device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Dummy joint tensors (seq=0) required by "rear" joint strategy
self.joint_q  = ttnn.as_tensor(torch.zeros(1, num_heads_local, 0, config.head_dim), ...)
self.joint_kv = ttnn.as_tensor(torch.zeros(1, num_kv_heads_local, 0, config.head_dim), ...)
self.joint_v  = ttnn.as_tensor(torch.zeros(1, num_kv_heads_local, 0, config.head_dim), ...)
```

`seq_total` is the **full** sequence length across all SP rows — if `max_seq_len = 25600`
and `SP = 4`, then `seq_total = 25600`. Each SP row contributes `6400` tokens and the
persistent buffers hold the full gathered sequence.

### 3. `tt/attention/prefill.py` — replace SDPA call

Current (line ~151):
```python
tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
    tt_q, tt_k, tt_v,
    is_causal=True,
    sliding_window_size=config.sliding_window,
    program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
    compute_kernel_config=program_config.get_compute_kernel_config(),
    attention_sink=weights.sinks,
)
```

Replacement (when `sp_factor > 1`):
```python
sp_factor = mesh_config.get_config(Mode.PREFILL).sp  # after fixing to sp=1, this path
                                                       # only fires when ring is enabled
seq_total = seq_len * sp_factor

tt_sdpa_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
    tt_q, tt_k, tt_v,
    self.joint_q, self.joint_kv, self.joint_v,
    persistent_output_buffer_k=self.persistent_k,
    persistent_output_buffer_v=self.persistent_v,
    joint_strategy="rear",
    logical_n=seq_total,
    program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
    compute_kernel_config=program_config.get_compute_kernel_config(),
    dim=2,
    multi_device_global_semaphore=ccl_manager.ring_attn_semaphores,
    num_links=ccl_manager.num_links,
    cluster_axis=mesh_config.sp_axis,
    mesh_device=mesh_device,
    topology=ccl_manager.topology,
    subdevice_id=ccl_manager.worker_sub_device_id,
    ccl_core_grid_offset=ccl_manager.ring_attn_ccl_grid_offset,
    use_column_major_ccl=True,
    is_causal=True,
    scale=config.scaling,
)
```

Fall back to plain SDPA when `sp_factor == 1` (single-row mesh, no ring needed).

### 4. Thread `sp_factor` and `seq_total` through the call stack

`prefill_forward` in `attention/prefill.py` needs to know:
- `sp_factor` — to compute `logical_n = seq_len * sp_factor`
- The persistent buffers and CCL handles — already available via `self` if allocated
  in `Attention.__init__`

`sp_factor` is available from `mesh_config.get_config(Mode.PREFILL).sp` at forward
time, so no new parameters need to be threaded through callers.

### 5. MeshConfig: restore `sp=4` for prefill

The current workaround sets `prefill=ModeConfig(tp=8, sp=1, ep=1)` to avoid the
broken inner-SP resharding. Once ring attention is wired up, this should be
changed back to the natural `sp=SP_FACTOR` and the `_reshard_for_sequence_parallel`
call in `experts/prefill.py` should be removed (it was designed for a different
architecture pattern and remains incorrect for our SP-sharded inputs).

Files to update:
- `tt/runners/prefill_runner.py` — `build_pipeline` MeshConfig
- `tests/accuracy/first_token_prefill.py`
- `tests/accuracy/layer_activations_prefill.py`

## Open Question: Sliding Window

Several GPT-OSS attention layers have `config.sliding_window` set. The ring op's
sliding window support needs to be verified:
- If supported: pass `sliding_window_size` to ring op directly
- If not supported: fall back to plain SDPA for windowed layers, accepting the
  boundary approximation only for those layers (the majority of context is
  unaffected by sliding window anyway)

Check `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp` for the full
ring op signature to confirm.

## Core Grid Impact

Reserving one column for CCL means compute matmuls (QKV projection, output
projection) see a `7×8` grid instead of `8×8`. The GPT-OSS program configs in
`tt/attention_configs.py` assume the full grid. These will need to be re-tuned
or made grid-aware.

Reference: DeepSeek sets `sdpa_compute_grid = (compute_grid.x - 1, compute_grid.y)`
for all attention matmuls when ring attention is active.

## Tasks and Verification

### Task 1 — Add ring attention infrastructure to `CCLManager`

**Work:** Add 2 GlobalSemaphores, SubDevice/SubDeviceId, and `ring_attn_ccl_grid_offset`
to `tt/ccl.py`.

**Verify:**
- Instantiate `CCLManager` inside a `ttnn.open_mesh_device` block and assert that
  `ccl.ring_attn_semaphores` has length 2, `ccl.worker_sub_device_id` is not None,
  and `ccl.ring_attn_ccl_grid_offset` equals `CoreCoord(7, 0)` on a 4×8 BH mesh.
- No device errors or assertion failures on mesh open/close.

---

### Task 2 — Pre-allocate persistent K/V buffers in `Attention.__init__`

**Work:** Allocate `persistent_k`, `persistent_v`, and three zero-size joint tensors
when the prefill MeshConfig has `sp > 1`.

**Verify:**
- Instantiate one `Attention` layer with a 4×8 mesh and `sp=4`, `max_seq_len=25600`.
- Assert `self.persistent_k.shape == [1, num_kv_heads_local, 25600, head_dim]` and
  `self.persistent_v` matches.
- Assert joint tensors have `shape[-2] == 0`.
- Total DRAM allocation stays within device budget (log and check `ttnn.dump_device_memory_state`).

---

### Task 3 — Replace SDPA call in `prefill.py`

**Work:** Swap `scaled_dot_product_attention` for `ring_joint_scaled_dot_product_attention`
when `sp > 1`; keep the plain SDPA path for `sp == 1`.

**Verify (step a) — SP=1 parity:** Run `first_token_prefill.py` on a 1×8 mesh before
and after the change. The sp=1 path still takes the plain SDPA branch; first token and
logit PCC vs. oracle must be identical to the baseline.

**Verify (step b) — ring op runs without crash:** Run `first_token_prefill.py` on a
4×8 mesh with a short prompt. Confirm no fatal errors and logits are finite
(`torch.isfinite(logit_vec).all()`).

**Verify (step c) — cross-boundary correctness:** Run `layer_activations_prefill.py`
on a 4×8 mesh with `--save-layers` and compare per-layer PCC against the HF oracle.
Expect PCC > 0.99 on the first few layers. Before this change, layers receiving
boundary tokens show noticeably lower PCC than interior layers; after, PCC should
be uniform across all layers.

---

### Task 4 — Resolve sliding window

**Work:** Check `sdpa_nanobind.cpp` for `sliding_window_size` in the ring op signature.
Either pass it through or add a per-layer dispatch (ring vs. plain SDPA).

**Verify:**
- Identify which GPT-OSS layers have `config.sliding_window != None` (check
  `hf_config.layer_types` or `sliding_window` field).
- Run `layer_activations_prefill.py` and confirm the sliding-window layers produce
  finite, non-NaN outputs and have acceptable PCC vs. oracle.
- If a fallback path is used, log which layers use it so it's visible in test output.

---

### Task 5 — Core grid adjustment for matmul program configs

**Work:** Update QKV and output projection program configs in `attention_configs.py`
(and wherever matmul grids are computed) to use `(grid_x - 1, grid_y)` when ring
attention is active.

**Verify:**
- Run the existing attention unit tests in `tests/fused_op_unit_tests/` on a 4×8 mesh.
  No "grid size mismatch" or L1 allocation failures.
- Run `first_token_prefill.py` end-to-end and confirm latency is not significantly
  worse than before (matmul efficiency should be comparable on 7 vs 8 columns).

---

### Task 6 — Restore `sp=4` in MeshConfig and remove `_reshard_for_sequence_parallel`

**Work:** Revert the `prefill=ModeConfig(sp=1)` workaround in the runner and accuracy
scripts. Remove or gate out the `_reshard_for_sequence_parallel` / `apply_sequence_parallel_allgather`
calls in `experts/prefill.py` since they are incorrect for SP-sharded inputs.

**Verify:**
- `PREFILL_MAX_SEQ_LEN=25600` run on 4×8 mesh completes without shape errors.
- Run `layer_activations_prefill.py` with `--max-layers 4` and confirm expert-layer
  PCC matches (or improves over) the `sp=1` baseline. The expert inner SP was
  producing garbage values before; removing it should improve accuracy.
- Run `first_token_prefill.py` with oracle comparison and confirm argmax match or
  improved PCC vs. the `sp=1` workaround run.

---

### Task 7 — End-to-end accuracy sign-off

**Work:** None — this is a validation gate before merging.

**Verify:**
- Run `hf_reference_oracle.py` on at least 3 diverse prompts (short, medium, 5K tokens).
- Run `first_token_prefill.py` for all 3 prompts on the 4×8 mesh.
- All 3 must produce `MATCH` or `IN TOP-5` against the oracle.
- Logit PCC > 0.99 for all prompts.
- Run the 25K standalone prefill (`PREFILL_STANDALONE=1`) end-to-end without errors.

---

## Effort Estimate

| Task | Complexity |
|---|---|
| 1. Ring semaphores + subdevice in `CCLManager` | Small (~20 lines) |
| 2. Persistent buffer allocation in `Attention.__init__` | Small (~30 lines) |
| 3. Replace SDPA call in `prefill.py` | Small (one call site) |
| 4. Sliding window handling | Unknown — check op signature first |
| 5. Core grid adjustment for matmul configs | Medium |
| 6. Restore `sp=4`, remove inner-SP expert resharding | Small |
| 7. End-to-end accuracy sign-off | Validation only |

## Key Reference Files

| File | Purpose |
|---|---|
| `models/demos/deepseek_v3_d_p/tt/mla/mla.py:781` | Ring SDPA call site |
| `models/demos/deepseek_v3_d_p/tt/tt_ccl.py:94` | Semaphore setup |
| `models/demos/deepseek_v3_d_p/tt/tt_ccl.py:138` | Persistent buffer allocation |
| `models/demos/gpt_oss_d_p/tt/attention/prefill.py:151` | SDPA to replace |
| `models/demos/gpt_oss_d_p/tt/attention/__init__.py` | Attention init (add buffers here) |
| `models/demos/gpt_oss_d_p/tt/ccl.py` | CCLManager (add ring infra here) |
| `ttnn/tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` | Ring op test with shapes |
