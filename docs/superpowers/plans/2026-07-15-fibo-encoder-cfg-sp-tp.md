# FIBO Encoder CFG=2 × SP=4 × TP=4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-parallelize the SmolLM3 text encoder in the Bria FIBO pipeline so a single encode uses all 32 devices of the 4×8 Galaxy as CFG=2 × SP=4 × TP=4 (two prompts concurrent, sequence split across tokens), with fixed-bucket input padding starting at 1024.

**Architecture:** Extend `EncoderParallelConfig` with cfg + sp. Add all-gather-K/V sequence parallelism to `SmolLM3Attention` (sequence sharded on `sp_axis`; each attention layer all-gathers K/V over `sp_axis`, runs local causal SDPA with an explicit rectangular causal bias carrying the shard's global offset). The pipeline carves the mesh into two (4,4) encoder submeshes and runs positive/negative on them concurrently. DiT/VAE parallelization is untouched.

**Tech Stack:** ttnn, PyTorch, transformers (SmolLM3), pytest. Target arch: Blackhole / 4×8 Galaxy (tests also run on a 2×2 dev mesh).

## Global Constraints

- SPDX headers on any new file: `# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.` / `# SPDX-License-Identifier: Apache-2.0`.
- Bucket length must satisfy `bucket % (sp_factor * 32) == 0`. First bucket: `1024` (1024/4 = 256, divisible by 32).
- SP is gated on `sp_factor > 1`: with sp=1 the encoder must behave exactly as today (no new CCL, `is_causal` SDPA).
- The `_encode` return contract is unchanged: `(cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states)`.
- DiT (`dit_parallel_config`) and VAE (`vae_parallel_config`) config and code are NOT modified.
- PCC gate for all encoder numerics: `assert_quality(..., pcc=0.99)` vs. HF `SmolLM3ForCausalLM`.
- Device test invocation prefix (4×8): `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest ...`.

---

### Task 1: Extend `EncoderParallelConfig` with cfg + sp

**Files:**
- Modify: `models/tt_dit/parallel/config.py:31-38`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (new `test_encoder_parallel_config_from_tuples`)

**Interfaces:**
- Consumes: `ParallelFactor(factor, mesh_axis)` (existing).
- Produces: `EncoderParallelConfig(tensor_parallel, sequence_parallel=None, cfg_parallel=None)` and classmethod `EncoderParallelConfig.from_tuples(*, tp, sp=None, cfg=None)`. Existing `from_tuple((factor, axis))` stays and still sets only `tensor_parallel`.

- [ ] **Step 1: Write the failing test**

Add to `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`:

```python
def test_encoder_parallel_config_from_tuples():
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor

    pc = EncoderParallelConfig.from_tuples(tp=(4, 1), sp=(4, 0), cfg=(2, 1))
    assert pc.tensor_parallel == ParallelFactor(4, 1)
    assert pc.sequence_parallel == ParallelFactor(4, 0)
    assert pc.cfg_parallel == ParallelFactor(2, 1)

    # back-compat: from_tuple sets only tensor_parallel, leaves sp/cfg None
    legacy = EncoderParallelConfig.from_tuple((8, 1))
    assert legacy.tensor_parallel == ParallelFactor(8, 1)
    assert legacy.sequence_parallel is None
    assert legacy.cfg_parallel is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python_env/bin/python -m pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_encoder_parallel_config_from_tuples -v`
Expected: FAIL (`AttributeError: ... 'cfg_parallel'` / `from_tuples` missing).

- [ ] **Step 3: Write minimal implementation**

Replace `EncoderParallelConfig` in `models/tt_dit/parallel/config.py`:

```python
class EncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor | None = None
    cfg_parallel: ParallelFactor | None = None

    @classmethod
    def from_tuple(cls, tp: tuple[int, int]) -> EncoderParallelConfig:
        return cls(tensor_parallel=ParallelFactor(*tp))

    @classmethod
    def from_tuples(
        cls,
        *,
        tp: tuple[int, int],
        sp: tuple[int, int] | None = None,
        cfg: tuple[int, int] | None = None,
    ) -> EncoderParallelConfig:
        return cls(
            tensor_parallel=ParallelFactor(*tp),
            sequence_parallel=ParallelFactor(*sp) if sp is not None else None,
            cfg_parallel=ParallelFactor(*cfg) if cfg is not None else None,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python_env/bin/python -m pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_encoder_parallel_config_from_tuples -v`
Expected: PASS. (This test needs no device.)

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/parallel/config.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "feat(fibo-pipeline): EncoderParallelConfig gains cfg + sp factors"
```

---

### Task 2: Sequence-parallel attention (all-gather K/V) in the encoder

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` — `SmolLM3Context` (~L104-108), `SmolLM3Attention.__init__`/`forward` (~L183-353), `SmolLM3TextEncoder.__init__`/`forward` (~L402-514), add helper `build_sp_causal_bias`.
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (new `test_smollm3_encoder_sp`, device test on 2×2).

**Interfaces:**
- Consumes: `EncoderParallelConfig` with `sequence_parallel` (Task 1); `CCLManager.all_gather_persistent_buffer(x, dim, mesh_axis, use_hyperparams=True)`; `tt_tensor.from_torch(..., mesh_axes=[...])`; `tt_tensor.to_torch(x, mesh_axes=[...])`.
- Produces: `SmolLM3TextEncoder` that, when `parallel_config.sequence_parallel.factor > 1`, expects `input_ids` and `pos_embeds` sharded along the seq dim over `sp_axis`, and returns hidden states sharded the same way. `build_sp_causal_bias(seq_local, sp_factor, *, device, sp_axis) -> ttnn.Tensor` producing a per-shard `(sp_factor,1,seq_local,seq_local*sp_factor)` bias sharded on `sp_axis`.

- [ ] **Step 1: Write the failing test**

Add to `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (mirror `test_smollm3_encoder_full_mesh`, but SP on axis 0 and inputs sharded on the seq dim). On a 2×2 dev mesh use sp=2 (axis 0), tp=2 (axis 1):

```python
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("seq", [64, 256])  # divisible by sp_factor(2)*32
def test_smollm3_encoder_sp(*, mesh_device, seq):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder
    from models.tt_dit.parallel.config import EncoderParallelConfig

    torch.manual_seed(0)
    sp_axis, tp_axis = 0, 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]
    hf = _load_hf_smollm3()
    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig.from_tuples(tp=(tp_factor, tp_axis), sp=(sp_factor, sp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)  # full-length; sharded on seq below
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, sp_axis])
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    prompt_embeds, _ = enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    out = tt_tensor.to_torch(prompt_embeds, mesh_axes=[None, sp_axis, None], composer_device=mesh_device)
    assert_quality(ref_prompt, out, pcc=0.99, relative_rmse=0.2)
```

(Note: `cos`/`sin` from `create_rope_tensors` are shape `(1,1,seq,head_dim)` — confirm the axis index for the seq dim when setting `mesh_axes`; adjust the `mesh_axes` list to shard the seq dim.)

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: FAIL — either a shape error (encoder doesn't shard the seq dim) or PCC failure (no cross-shard attention yet).

- [ ] **Step 3: Write minimal implementation**

(a) `SmolLM3Context` — add sp fields:

```python
@dataclass
class SmolLM3Context:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None
    sp_axis: int | None = None
    sp_factor: int = 1
```

(b) `SmolLM3TextEncoder.__init__` — read sp from config and put on ctx:

```python
sp = parallel_config.sequence_parallel
sp_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None
sp_factor = sp.factor if sp is not None else 1
ctx = SmolLM3Context(
    device=device,
    tp_axis=tp_axis if tp_factor > 1 else None,
    ccl_manager=ccl_manager,
    fsdp_mesh_axis=fsdp_mesh_axis,
    sp_axis=sp_axis,
    sp_factor=sp_factor,
)
if sp_axis is not None and ccl_manager is None:
    raise ValueError("ccl_manager must be provided if sequence parallelism is used")
```

(c) Add the SP causal-bias helper (module scope, near `prepare_attention_bias`):

```python
def build_sp_causal_bias(
    seq_local: int, sp_factor: int, *, device: ttnn.MeshDevice, sp_axis: int
) -> ttnn.Tensor:
    """Per-shard rectangular causal bias: shard r's query row i (global r*seq_local+i)
    attends key j iff j <= r*seq_local+i. Sharded along sp_axis so each device gets its slice."""
    import torch as _torch

    seq_total = seq_local * sp_factor
    q_idx = _torch.arange(seq_local)
    k_idx = _torch.arange(seq_total)
    masks = []
    for r in range(sp_factor):
        allow = k_idx[None, :] <= (r * seq_local + q_idx)[:, None]  # (seq_local, seq_total)
        bias = _torch.where(allow, 0.0, float("-inf"))
        masks.append(bias)
    stacked = _torch.stack(masks, dim=0).reshape(sp_factor, 1, seq_local, seq_total)
    return tensor.from_torch(
        stacked, device=device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None]
    )
```

(d) `SmolLM3Attention.__init__` — store sp:

```python
self._sp_axis = ctx.sp_axis
self._sp_factor = ctx.sp_factor
```

(e) `SmolLM3Attention.forward` — after RoPE, all-gather K/V and use the SP bias:

```python
cos, sin = pos_embeds
if self._use_rope:
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

if self._sp_factor > 1:
    # gather full-sequence K/V (already RoPE'd) across the sequence-parallel axis
    k = self._ccl_manager.all_gather_persistent_buffer(k, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)
    v = self._ccl_manager.all_gather_persistent_buffer(v, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)

x = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_bias,
    is_causal=attention_bias is None,
    program_config=self._sdpa_program_config(k.shape[2]),
    compute_kernel_config=self._sdpa_compute_kernel_config,
)
```

(f) `SmolLM3TextEncoder.forward` — build the SP bias once and thread it as `attention_bias`; keep the existing `attention_mask` path untouched:

```python
def forward(self, input_ids, *, attention_mask=None, pos_embeds):
    batch_size, seq_local = input_ids.shape

    sp_factor = self.layers[0].self_attn._sp_factor
    sp_axis = self.layers[0].self_attn._sp_axis
    if sp_factor > 1:
        attention_bias = build_sp_causal_bias(
            seq_local, sp_factor, device=self._device, sp_axis=sp_axis
        )
        padded = seq_local
    elif attention_mask is not None:
        # ... existing pad + prepare_attention_bias path (unchanged) ...
    else:
        padded = seq_local
        attention_bias = None

    hidden_states = self.embed_tokens.forward(input_ids)
    all_hidden_states = []
    for layer in self.layers:
        all_hidden_states.append(hidden_states)
        hidden_states = layer.forward(hidden_states, attention_bias=attention_bias, pos_embeds=pos_embeds)
    hidden_states = self.norm.forward(hidden_states)
    all_hidden_states.append(hidden_states)

    if padded != seq_local:
        all_hidden_states = [x[:, :seq_local, :] for x in all_hidden_states]
    return all_hidden_states
```

(Refactor detail: expose `sp_factor`/`sp_axis` on the encoder — e.g. store `self._sp_factor`/`self._sp_axis` in `__init__` — instead of reaching through `self.layers[0].self_attn`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: PASS (PCC ≥ 0.99). Also re-run the existing sp=1 tests to confirm no regression:
`... -m pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v`

If the K/V all-gather on `dim=2` is unsupported by `all_gather_persistent_buffer`, fall back to the async `all_gather` used in `parallel/config.py::vae_all_gather` (cluster_axis=sp_axis, dim=2). Document whichever works.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "feat(fibo-pipeline): all-gather K/V sequence parallelism in SmolLM3 encoder"
```

---

### Task 3: Fixed-bucket padding + SP-aware host I/O in the FIBO wrapper

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/text_encoder.py` — `SmolLM3TextEncoderWrapper.__init__` (add `pad_buckets`), `encode_prompt` (bucket pick + SP shard/gather).
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (new `test_fibo_wrapper_bucket_pick`, pure-python bucket logic extracted to a helper).

**Interfaces:**
- Consumes: `SmolLM3TextEncoder` SP contract (Task 2), `EncoderParallelConfig` (Task 1).
- Produces: `SmolLM3TextEncoderWrapper(checkpoint, *, device, ccl_manager, parallel_config, pad_buckets=(1024,), use_torch=False)`; module helper `pick_bucket(seq_len, buckets, sp_factor) -> int` (raises `ValueError` if no bucket fits or bucket not divisible by `sp_factor*32`).

- [ ] **Step 1: Write the failing test** (pure python, no device)

```python
def test_fibo_wrapper_bucket_pick():
    from models.tt_dit.pipelines.bria_fibo.text_encoder import pick_bucket

    assert pick_bucket(10, (1024,), sp_factor=4) == 1024
    assert pick_bucket(1024, (1024, 2048), sp_factor=4) == 1024
    assert pick_bucket(1025, (1024, 2048), sp_factor=4) == 2048
    import pytest
    with pytest.raises(ValueError):
        pick_bucket(3000, (1024, 2048), sp_factor=4)   # exceeds all buckets
    with pytest.raises(ValueError):
        pick_bucket(10, (1000,), sp_factor=4)          # 1000 % (4*32) != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_bucket_pick" -v`
Expected: FAIL (`ImportError: cannot import name 'pick_bucket'`).

- [ ] **Step 3: Write minimal implementation**

Add to `text_encoder.py`:

```python
def pick_bucket(seq_len: int, buckets, sp_factor: int) -> int:
    """Smallest bucket >= seq_len; validate divisibility by sp_factor*32."""
    for b in sorted(buckets):
        if b % (sp_factor * 32) != 0:
            raise ValueError(f"bucket {b} not divisible by sp_factor*32 = {sp_factor * 32}")
    for b in sorted(buckets):
        if b >= seq_len:
            return b
    raise ValueError(f"prompt seq_len {seq_len} exceeds all buckets {sorted(buckets)}; add a larger bucket")
```

Then update `SmolLM3TextEncoderWrapper.__init__` to accept `pad_buckets=(1024,)`, store `self._pad_buckets`, read `self._sp_axis`/`self._sp_factor` from `parallel_config.sequence_parallel`, and rewrite the device branch of `encode_prompt`:

```python
seq_len = input_ids.shape[1]
bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
padded_ids = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
cos, sin = self._encoder.create_rope_tensors(1, bucket)

if self._sp_factor > 1:
    tt_ids = tt_tensor.from_torch(padded_ids, device=self._device, dtype=ttnn.uint32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, self._sp_axis])
    tt_cos = tt_tensor.from_torch(cos, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
    tt_sin = tt_tensor.from_torch(sin, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
else:
    tt_ids = tt_tensor.from_torch(padded_ids, device=self._device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_cos = tt_tensor.from_torch(cos, device=self._device)
    tt_sin = tt_tensor.from_torch(sin, device=self._device)

prompt_embeds, all_hidden_states = self._encoder.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))

gather = dict(mesh_axes=[None, self._sp_axis, None], composer_device=self._device) if self._sp_factor > 1 else {}
host_prompt_embeds = tt_tensor.to_torch(prompt_embeds, **gather)[:, :seq_len, :]
host_hidden_states = [tt_tensor.to_torch(h, **gather)[:, :seq_len, :] for h in all_hidden_states]
return host_prompt_embeds, host_hidden_states
```

(Confirm the `mesh_axes` seq index for `cos`/`sin` matches `create_rope_tensors`' output rank, as in Task 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_bucket_pick" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/text_encoder.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "feat(fibo-pipeline): fixed-bucket padding + SP host I/O in FIBO text-encoder wrapper"
```

---

### Task 4: Pipeline — encoder submeshes, two encoders, concurrent `_encode`

**Files:**
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py` — `FiboConfig.create` encoder config (~L104-109), `__init__` text-encoder build (~L172-179), `_encode` (~L269-282).
- Test: exercised by Task 5's device test.

**Interfaces:**
- Consumes: `EncoderParallelConfig.from_tuples` (Task 1), SP-capable encoder (Task 2), wrapper with `pad_buckets` (Task 3), `cfg.create_submeshes`, `CCLManager`.
- Produces: `self._text_encoders: list[SmolLM3TextEncoderWrapper]` (len 1 or 2). `_encode` dispatches pos→`[0]`, neg→`[1 if len==2 else 0]`.

- [ ] **Step 1: Build the encoder config + submeshes**

In `FiboConfig.create`, replace the encoder config. On the full mesh axis 1 (size 8): TP=4, CFG=2; SP=`sp_factor` on axis 0:

```python
# Encoder: CFG=2 x SP(axis0) x TP=4(axis1) — its own submesh split, independent of the DiT.
enc_tp_factor = min(4, mesh[tp_axis])
enc_cfg_factor = 2 if mesh[tp_axis] // enc_tp_factor >= 2 else 1
encoder_parallel_config = EncoderParallelConfig.from_tuples(
    tp=(enc_tp_factor, tp_axis),
    sp=(sp_factor, sp_axis),
    cfg=(enc_cfg_factor, tp_axis),
)
```

In `BriaFiboPipeline.__init__`, after the DiT submesh/CCL are built, create encoder submeshes and one encoder per submesh:

```python
from models.tt_dit.parallel.config import DiTParallelConfig as _DiT  # local alias if needed
enc_pc = config.encoder_parallel_config
enc_split = _DiT.from_tuples(
    cfg=(enc_pc.cfg_parallel.factor, enc_pc.cfg_parallel.mesh_axis),
    sp=(enc_pc.sequence_parallel.factor, enc_pc.sequence_parallel.mesh_axis),
    tp=(enc_pc.tensor_parallel.factor, enc_pc.tensor_parallel.mesh_axis),
)
self._encoder_submeshes = create_submeshes(device, enc_split)
logger.info(f"encoder submeshes: {[tuple(s.shape) for s in self._encoder_submeshes]}")

self._text_encoders = []
for sm in self._encoder_submeshes:
    ccl = CCLManager(sm, num_links=config.num_links, topology=config.topology)
    self._text_encoders.append(
        SmolLM3TextEncoderWrapper(
            self._ckpt,
            device=sm,
            ccl_manager=ccl,
            parallel_config=EncoderParallelConfig.from_tuples(
                tp=(enc_pc.tensor_parallel.factor, enc_pc.tensor_parallel.mesh_axis),
                sp=(enc_pc.sequence_parallel.factor, enc_pc.sequence_parallel.mesh_axis),
            ),
            pad_buckets=(1024,),
        )
    )
    ttnn.synchronize_device(sm)
```

(Remove the old single `self._text_encoder = SmolLM3TextEncoderWrapper(...)` build. Confirm the checkpoint name attribute — reuse whatever `__init__` already holds, e.g. `checkpoint`/`self._ckpt`; grep before editing.)

- [ ] **Step 2: Rewrite `_encode` for concurrent CFG**

```python
def _encode(self, prompt: str, negative_prompt: str, *, do_cfg: bool = True) -> tuple:
    encs = self._text_encoders
    cond_embeds, cond_hidden_states = encs[0].encode_prompt(prompt)
    if do_cfg:
        neg_enc = encs[1] if len(encs) > 1 else encs[0]
        uncond_embeds, uncond_hidden_states = neg_enc.encode_prompt(negative_prompt)
    else:
        uncond_embeds, uncond_hidden_states = None, []
    return cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states
```

For real overlap, split `encode_prompt` into enqueue + readback so both submeshes' forwards are in flight before either host readback (see spec §5). Minimal correct first cut: keep `encode_prompt` as-is (submeshes still run their own programs; only the host readbacks serialize). Add a `# TODO(overlap): split enqueue/readback` note and confirm correctness before optimizing.

- [ ] **Step 3: Run the encoder PCC + a smoke build**

Run the SP encoder test (Task 2) still passes, then a pipeline build smoke (Task 5 covers the full run):
`HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py
git commit -m "feat(fibo-pipeline): CFG=2 encoder submeshes + concurrent _encode"
```

---

### Task 5: Wire the layout into `test_fibo_encode_perf` and verify end-to-end

**Files:**
- Modify: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py` — `_num_links` (L88-94), `test_fibo_encode_perf` docstring/comment (L328-386).

**Interfaces:**
- Consumes: updated pipeline (Task 4).
- Produces: `_num_links` returns a valid per-hop link count for (4,4) submeshes.

- [ ] **Step 1: Add (4,4) to `_num_links`**

```python
return {(2, 2): 4, (2, 1): 4, (4, 4): 2, (4, 8): 2}.get(tuple(mesh_device.shape), 1)
```

(Confirm the safe link count for a (4,4) submesh on this platform; 2 mirrors the (4,8) pick. `(2,1)` covers the 2×2 encoder submesh.)

- [ ] **Step 2: Update the test docstring/comment**

Replace the layout description in `test_fibo_encode_perf`'s docstring to state the new CFG=2 × SP=4 × TP=4 encode (positive/negative on two (4,4) submeshes, input padded to the 1024 bucket). No logic change — it already calls `pipe._encode(prompt, negative_prompt, do_cfg=True)`.

If the FIBO JSON prompt tokenizes to > 1024, `pick_bucket` will raise; in that case add `2048` to `pad_buckets` in the pipeline (Task 4 Step 1) and note it. Measure once:

```bash
python_env/bin/python -c "
from transformers import AutoTokenizer
from pathlib import Path
p = Path('models/tt_dit/tests/models/bria_fibo/fibo_vlm_prompt.json').read_text().strip()
tok = AutoTokenizer.from_pretrained('<FIBO ckpt>', subfolder='tokenizer')
print('json prompt tokens:', len(tok(p, add_special_tokens=True).input_ids))
"
```

- [ ] **Step 3: Run the encode perf test (4×8)**

Run:
```bash
HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
  python_env/bin/python -m pytest \
  "models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_perf" \
  -k "mesh_device1" -v -s --timeout=1800
```
Expected: PASS; logs "FIBO encode perf — avg of 3 runs" with the two branches' output shapes. Compare the avg encode seconds to the pre-change baseline (recorded from `git stash`/prior commit) to confirm a speedup.

- [ ] **Step 4: Commit**

```bash
git add models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py
git commit -m "test(fibo-pipeline): encode perf exercises CFG=2 x SP=4 x TP=4 encoder"
```

---

## Self-Review

**Spec coverage:**
- Config extension → Task 1. ✅
- All-gather K/V SP attention → Task 2. ✅
- Fixed-bucket padding (list, start 1024, error on overflow, divisibility) → Task 3. ✅
- Encoder submeshes + two encoders + concurrent `_encode` (main pipeline) → Task 4. ✅
- Test wiring + `_num_links` + perf verification → Task 5. ✅
- PCC validation of SP numerics → Task 2 Step 4. ✅
- DiT/VAE untouched → enforced by Global Constraints; Tasks only touch encoder config/build. ✅
- 2×2 degradation → Task 2 test runs sp=2/tp=2; Task 4 config derives factors from mesh; Task 5 `_num_links` covers (2,1). ✅

**Placeholder scan:** The only deferred item is the CFG host-overlap optimization (spec §5), explicitly scoped as a labeled TODO with a correct first-cut fallback in Task 4 Step 2 — not a plan gap. The FIBO checkpoint string and the exact `mesh_axes` seq-dim index are marked "grep/confirm before editing" because they must be read from the current code, not guessed.

**Type consistency:** `pick_bucket(seq_len, buckets, sp_factor)`, `build_sp_causal_bias(seq_local, sp_factor, *, device, sp_axis)`, `EncoderParallelConfig.from_tuples(tp=, sp=, cfg=)`, and `self._text_encoders` are used consistently across Tasks 1–5.
