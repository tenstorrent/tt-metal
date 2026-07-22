# FIBO SmolLM3 Encoder Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the FIBO SmolLM3 text encoder by adopting the `friedrich/fibo` branch's clean patterns (declarative state conversion, lazy cache-backed checkpoint, slim config) and pruning every code path the full pipeline never runs — while preserving SP×TP parallelism, fused qkv, and the fast readback exactly.

**Architecture:** The encoder stays standalone (no migration onto the shared `encoders/transformer.py`). Weight-key conversion becomes a declarative `StateConversion` table; weights load lazily through `cache.load_model` via a new `SmolLm3Checkpoint`; the config becomes a lean dataclass; the encoder is reduced to the sequence-parallel (`sp_factor >= 2`) path used by the pipeline on both 2×2 Blackhole and 4×8 Galaxy.

**Tech Stack:** Python, PyTorch, `ttnn`, tt_dit `Module` framework, `transformers` (SmolLM3), pytest.

## Global Constraints

- **Preserve exactly (same math / same PCC):** SP×TP parallelism (SP=8×TP=4 on 4×8, SP=2×TP=2 on 2×2), the SP causal-bias cache (`build_sp_causal_bias`, `_sp_bias_cache`), fused `qkv_proj` with `optimal_groups` head-padding, and the fast shard-selective readback (`_read_seq_sharded`).
- **Do not** migrate to the shared `TransformerEncoder`. Keep `model_smollm3.py` standalone.
- **`sp_factor` and `tp_factor` stay parameterized**; `sp_factor` is no longer allowed to be 1.
- **PCC threshold** for all correctness tests stays at the current values (`pcc=0.99`, `relative_rmse` where already used).
- Keep public wrapper constructor signature `SmolLM3TextEncoderWrapper(checkpoint, *, device, ccl_manager, parallel_config, pad_buckets=(1024,))` (minus `use_torch`) so the pipeline call site is unchanged.
- Keep the classmethod name `SmolLM3Config.from_hf_config` (callers depend on it).
- Copyright headers: keep existing `© 2025` / `© 2026` headers already on each file; do not add or renumber years.

**Hardware note:** device tests run on the **4×8 Galaxy only** (per the decision to test just the Galaxy path). Host-only tests (config, rope, bucket, `StateConversion`) run anywhere. Each step marks which is which. The encoder code stays parameterized for 2×2 as well, but no 2×2 test is kept.

---

### Task 1: Slim `config.py` to a dataclass

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/config.py` (full rewrite)
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (remove `test_smollm3_config_defaults`; keep `test_smollm3_config_from_hf`)

**Interfaces:**
- Consumes: a `transformers` SmolLM3 config object (has `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `rms_norm_eps`, `hidden_act`, optional `head_dim`, `no_rope_layers`, `no_rope_layer_interval`, `attention_bias`, `max_position_embeddings`, and rope theta via top-level or `rope_parameters`/`rope_scaling`).
- Produces: `SmolLM3Config` dataclass with attributes `vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, rope_theta, max_position_embeddings, hidden_act, attention_bias, no_rope_layer_interval, no_rope_layers: list[int]` and classmethod `from_hf_config(hf_config) -> SmolLM3Config`. `no_rope_layers[i] == 1` means apply RoPE, `0` means NoPE.

- [ ] **Step 1: Rewrite `config.py`**

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


def _read_rope_theta(hf_config) -> float:
    """rope_theta may be a top-level attr or nested under rope_parameters/rope_scaling (transformers 5.x)."""
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
        theta = params.get("rope_theta")
    if theta is None:
        raise ValueError("SmolLM3 config missing rope_theta (checked top-level and rope_parameters/rope_scaling)")
    return theta


@dataclass
class SmolLM3Config:
    """Configuration for the SmolLM3-3B text encoder (used by Bria FIBO)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536
    hidden_act: str = "silu"
    attention_bias: bool = False
    no_rope_layer_interval: int = 4
    no_rope_layers: list[int] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # HF default: no_rope_layers[i] = int((i + 1) % interval != 0); 1 = apply RoPE, 0 = NoPE.
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0) for i in range(self.num_hidden_layers)
            ]
        else:
            self.no_rope_layers = list(self.no_rope_layers)

    @classmethod
    def from_hf_config(cls, hf_config) -> "SmolLM3Config":
        """Build from a transformers SmolLM3Config (or the .config of a loaded model)."""
        hf_no_rope = getattr(hf_config, "no_rope_layers", None)
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", None),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=_read_rope_theta(hf_config),
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            attention_bias=getattr(hf_config, "attention_bias", False),
            no_rope_layer_interval=getattr(hf_config, "no_rope_layer_interval", 4),
            no_rope_layers=list(hf_no_rope) if hf_no_rope is not None else None,
        )
```

- [ ] **Step 2: Remove the obsolete defaults test**

In `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`, delete `test_smollm3_config_defaults` (lines ~18-33) — no-arg construction is no longer supported. Leave `test_smollm3_config_from_hf` unchanged.

- [ ] **Step 3: Run the kept config test (host-only)**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_config_from_hf -v`
Expected: PASS (or SKIP if the FIBO config is unavailable offline — acceptable, means no regression).

- [ ] **Step 4: Verify no other no-arg construction exists**

Run: `grep -rn "SmolLM3Config(" models/tt_dit --include=*.py | grep -v "from_hf_config" | grep -v "def "`
Expected: no hits constructing `SmolLM3Config()` without HF args (other than the class definition). If any appear, they must switch to `from_hf_config`.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/encoders/smollm3/config.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "refactor(fibo-pipeline): slim SmolLM3Config to a dataclass"
```

---

### Task 2: Declarative top-level `StateConversion`

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` (add `StateConversion` + `STATE_CONVERSION`; replace `SmolLM3TextEncoder._prepare_torch_state` body; drop `pop_substate` import)
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (add host unit test `test_smollm3_state_conversion`)

**Interfaces:**
- Produces: `StateConversion` dataclass with `rename: list[tuple[str, str]]`, `remove: list[str]`, and `convert(state_dict) -> dict`. Module-level constant `STATE_CONVERSION` that (a) strips a leading `model.` from `embed_tokens|layers|norm` keys and (b) removes `lm_head*` and `*rotary_emb*` keys, handling BOTH the full `SmolLM3ForCausalLM` state dict and the inner `SmolLM3Model` state dict with no warnings.
- Consumes (by later tasks): the encoder's top-level `_prepare_torch_state` applies `STATE_CONVERSION` in place; the per-attention fusion hook (`SmolLM3Attention._prepare_torch_state`) is unchanged and still runs after.

- [ ] **Step 1: Write the failing host unit test**

Add to `test_smollm3.py` (imports `StateConversion`, `STATE_CONVERSION` lazily inside the test to avoid device import at collection):

```python
def test_smollm3_state_conversion():
    import torch as _torch

    from models.tt_dit.encoders.smollm3.model_smollm3 import STATE_CONVERSION

    # Full SmolLM3ForCausalLM-style dict: model.* prefix + lm_head + a rotary_emb buffer.
    full = {
        "model.embed_tokens.weight": _torch.zeros(2, 2),
        "model.layers.0.self_attn.q_proj.weight": _torch.zeros(2, 2),
        "model.layers.0.mlp.gate_proj.weight": _torch.zeros(2, 2),
        "model.norm.weight": _torch.zeros(2),
        "model.rotary_emb.inv_freq": _torch.zeros(2),
        "lm_head.weight": _torch.zeros(2, 2),
    }
    out = STATE_CONVERSION.convert(full)
    assert set(out) == {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "norm.weight",
    }

    # Inner SmolLM3Model-style dict (no model. prefix, no lm_head): keys pass through unchanged.
    inner = {
        "embed_tokens.weight": _torch.zeros(2, 2),
        "layers.0.self_attn.q_proj.weight": _torch.zeros(2, 2),
        "norm.weight": _torch.zeros(2),
    }
    out2 = STATE_CONVERSION.convert(inner)
    assert set(out2) == set(inner)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_state_conversion -v`
Expected: FAIL with `ImportError: cannot import name 'STATE_CONVERSION'`.

- [ ] **Step 3: Add `StateConversion` + `STATE_CONVERSION` to `model_smollm3.py`**

Add near the top of `model_smollm3.py` (after the existing imports; add `import re` and `import warnings`, and `from dataclasses import dataclass, field` — `dataclass` is already imported, add `field` if needed; the module already imports `dataclass`):

```python
import re
import warnings
from collections.abc import Mapping, Sequence


@dataclass
class StateConversion:
    """Declarative torch-state-dict key remapping: ordered regex renames, then removes."""

    rename: Sequence[tuple[str, str]] = ()
    remove: Sequence[str] = ()

    def convert(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        in_ = dict(state_dict)
        out: dict[str, torch.Tensor] = {}
        compiled = [(re.compile(p), t) for (p, t) in self.rename]
        removes = [re.compile(p) for p in self.remove]
        for k in list(in_):
            transformed = False
            for pattern, t in compiled:
                new_k, count = pattern.subn(t, k, count=1)
                if count == 1:
                    out[new_k] = in_.pop(k)
                    transformed = True
                    break
            if transformed:
                continue
            for pattern in removes:
                if pattern.search(k):
                    in_.pop(k)
                    transformed = True
                    break
            if not transformed:
                warnings.warn(f"unprocessed key: {k}", stacklevel=2)
        return {**in_, **out}


# Strip a leading `model.` from encoder submodules; drop the LM head and rotary buffers.
# Works on both the full SmolLM3ForCausalLM dict (model.* + lm_head) and the inner
# SmolLM3Model dict (already-stripped keys pass through the rename as no-ops).
STATE_CONVERSION = StateConversion(
    rename=[(r"^(?:model\.)?(embed_tokens|layers|norm)", r"\1")],
    remove=[r"(?:^|\.)lm_head(?:\.|$)", r"(?:^|\.)rotary_emb(?:\.|$)"],
)
```

Note: the rename alternation excludes `rotary_emb`, so `model.rotary_emb.*` is only matched by the remove list (no double-match). Inner-dict keys like `embed_tokens.weight` match the rename and are replaced with themselves (no-op, but counted as processed → no warning).

- [ ] **Step 4: Replace the encoder's top-level `_prepare_torch_state` body**

In `SmolLM3TextEncoder._prepare_torch_state` (currently `model_smollm3.py:490-498`), replace the imperative prefix-strip + `pop_substate` calls with:

```python
    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        converted = STATE_CONVERSION.convert(state)
        state.clear()
        state.update(converted)
```

Remove the now-unused import: delete `from ...utils.substate import pop_substate` (line 21) if `pop_substate` is not used elsewhere in the file (verify with grep in Step 6).

- [ ] **Step 5: Run the host unit test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_state_conversion -v`
Expected: PASS.

- [ ] **Step 6: Verify `pop_substate` is fully removed**

Run: `grep -n "pop_substate" models/tt_dit/encoders/smollm3/model_smollm3.py`
Expected: no hits (import and usage both gone).

- [ ] **Step 7: (Hardware, 4×8) regression — weights still load + PCC holds**

Run: `pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: PASS at `pcc=0.99` — proves the declarative conversion + unchanged fusion produce identical weights. (Skip if no Galaxy available; then rely on the wrapper regression in Task 5.)

- [ ] **Step 8: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "refactor(fibo-pipeline): declarative SmolLM3 top-level state conversion"
```

---

### Task 3: Remove FSDP support

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` (`SmolLM3Context`, `SmolLM3TextEncoder.__init__`, `SmolLM3Mlp.__init__`, `SmolLM3Attention.__init__`)

**Interfaces:**
- Produces: `SmolLM3Context` no longer has `fsdp_mesh_axis`. `SmolLM3TextEncoder.__init__` no longer accepts `is_fsdp`. `ColParallelLinear`/`RowParallelLinear` are constructed without `fsdp_mesh_axis`.
- Rationale: FSDP is unused — no test sets `is_fsdp=True`, the pipeline never sets it.

- [ ] **Step 1: Drop `fsdp_mesh_axis` from `SmolLM3Context`**

In the `@dataclass SmolLM3Context` (lines 123-130), remove the `fsdp_mesh_axis: int | None = None` field. Resulting fields: `device`, `tp_axis`, `ccl_manager`, `sp_axis`, `sp_factor`.

- [ ] **Step 2: Drop `is_fsdp` and `fsdp_mesh_axis` computation in `SmolLM3TextEncoder.__init__`**

In `SmolLM3TextEncoder.__init__` (lines 432-466), remove the `is_fsdp: bool = False` parameter and the block computing `fsdp_mesh_axis` (lines 445-449), and drop `fsdp_mesh_axis=fsdp_mesh_axis` from the `SmolLM3Context(...)` construction. The context construction becomes:

```python
        ctx = SmolLM3Context(
            device=device,
            tp_axis=tp_axis if tp_factor > 1 else None,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
        )
```

- [ ] **Step 3: Drop `fsdp_mesh_axis=` from every Linear in `SmolLM3Mlp` and `SmolLM3Attention`**

In `SmolLM3Mlp.__init__` remove `fsdp_mesh_axis=ctx.fsdp_mesh_axis,` from `gate_proj`, `up_proj`, `down_proj` (lines ~164, 173, 182). In `SmolLM3Attention.__init__` remove `fsdp_mesh_axis=ctx.fsdp_mesh_axis,` from `qkv_proj` and `o_proj` (lines ~238, 247).

- [ ] **Step 4: Verify no FSDP references remain**

Run: `grep -n "fsdp\|is_fsdp" models/tt_dit/encoders/smollm3/model_smollm3.py`
Expected: no hits.

- [ ] **Step 5: (Hardware, 4×8) regression**

Run: `pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" -v`
Expected: PASS at `pcc=0.99` (behavior unchanged; FSDP was never exercised). Skip if no hardware.

- [ ] **Step 6: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py
git commit -m "refactor(fibo-pipeline): drop unused FSDP path from SmolLM3 encoder"
```

---

### Task 4: Prune to the SP-only path (drop non-SP + mask paths)

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` (`SmolLM3TextEncoder.forward`/`.encode`, `SmolLM3Attention.forward`; delete `prepare_attention_bias`; delete `MAX_CHUNK_SIZE`-based non-SP handling)
- Modify: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` (remove `(1,1)` unit tests + masked test; convert `(2,2)` test to SP; drop `attention_mask=` args in kept calls)

**Interfaces:**
- Produces: `SmolLM3TextEncoder.forward(input_ids, *, pos_embeds)` and `.encode(input_ids, *, pos_embeds)` — the `attention_mask` parameter is **removed**. The encoder requires `sp_factor >= 2` (raises if not). `SmolLM3Attention.forward(x, *, attention_bias, pos_embeds)` now always all-gathers K/V on the SP axis and always uses the passed `attention_bias` (never `is_causal`).
- Consumes: `build_sp_causal_bias` and `_sp_bias_cache` (unchanged).

- [ ] **Step 1: Simplify `SmolLM3TextEncoder.forward` to SP-only**

Replace `forward` (lines 519-568) with (drops `attention_mask` param, the `elif attention_mask` branch, the `else` non-SP branch, and the `padded != seq_len` trim since buckets are always SP-shardable and no local padding happens):

```python
    def encode(
        self,
        input_ids: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """Return (prompt_embeds, all_hidden_states) matching the FIBO output contract.

        prompt_embeds = concat(all_hidden_states[-1], all_hidden_states[-2], dim=-1)
        shape: [B, T, 2 * hidden_size]
        """
        all_hidden_states = self.forward(input_ids, pos_embeds=pos_embeds)
        prompt_embeds = ttnn.concat([all_hidden_states[-1], all_hidden_states[-2]], dim=-1)
        return prompt_embeds, all_hidden_states

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        # Sequence-parallel only: ``.shape`` reports the LOCAL (per-shard) sequence length; the full
        # sequence is ``seq_len * sp_factor`` and lives across the sp axis after the K/V all-gather.
        _batch_size, seq_len = input_ids.shape

        # Per-shard rectangular causal bias (global offset baked in), threaded through every layer.
        # Constant for a given local seq length, so build once and cache: keeps the host->device build
        # out of a captured trace (runs during the tracer's prep_run, read-only in capture/replay).
        attention_bias = self._sp_bias_cache.get(seq_len)
        if attention_bias is None:
            attention_bias = build_sp_causal_bias(seq_len, self._sp_factor, device=self._device, sp_axis=self._sp_axis)
            self._sp_bias_cache[seq_len] = attention_bias

        hidden_states = self.embed_tokens.forward(input_ids)

        # HF output_hidden_states convention: append the INPUT to each layer, then the final norm.
        all_hidden_states: list[ttnn.Tensor] = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            hidden_states = layer.forward(hidden_states, attention_bias=attention_bias, pos_embeds=pos_embeds)
        hidden_states = self.norm.forward(hidden_states)
        all_hidden_states.append(hidden_states)
        return all_hidden_states
```

- [ ] **Step 2: Require `sp_factor >= 2` in `SmolLM3TextEncoder.__init__`**

After computing `sp_factor` / setting `self._sp_factor` (near lines 450-466), add:

```python
        if ctx.sp_axis is None or ctx.sp_factor < 2:
            raise ValueError("SmolLM3TextEncoder requires sequence parallelism (sp_factor >= 2)")
```

- [ ] **Step 3: Simplify `SmolLM3Attention.forward` (always SP, always biased)**

In `SmolLM3Attention.forward` (lines 321-370): the K/V all-gather is now unconditional and SDPA always uses the bias. Replace the `if self._sp_factor > 1:` guard and the SDPA call:

```python
        # Sequence-parallel: gather full-sequence K/V (already RoPE'd) across the sp axis so each
        # shard's local Q attends the whole sequence. The rectangular causal bias carries the shard's
        # global offset.
        k = self._ccl_manager.all_gather_persistent_buffer(k, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)
        v = self._ccl_manager.all_gather_persistent_buffer(v, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            is_causal=False,
            program_config=self._sdpa_program_config(k.shape[2]),
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
```

(The `SmolLM3Attention.forward` signature keeps `attention_bias` — it is now always the SP bias, never `None`. The tp>1 all-gathers around `o_proj` are unchanged.)

- [ ] **Step 4: Delete `prepare_attention_bias`**

Remove the `prepare_attention_bias` function (lines 91-101) — it is only used by the deleted mask path. Leave `build_sp_causal_bias` and `MAX_CHUNK_SIZE` (still used by `_sdpa_program_config`).

- [ ] **Step 5: Remove `(1,1)` unit tests and the masked test**

In `test_smollm3.py`, delete these device tests (they exercise deleted single-device / mask paths):
`test_smollm3_mlp`, `test_smollm3_attention`, `test_smollm3_decoder_layer`, `test_smollm3_encoder_all_layers`, `test_smollm3_encode_contract`, `test_smollm3_encoder_masked`.

- [ ] **Step 6: Remove the `(2,2)` test (Galaxy-only testing)**

Delete `test_smollm3_encoder_full_mesh` (lines 352-384). Per the decision to test only the 4×8 Galaxy path, we do not keep or convert a 2×2 test. The encoder code stays parameterized for both meshes (`sp_factor` general), but coverage runs on Galaxy only via `test_smollm3_encoder_sp` / `test_smollm3_sp_bias_cached` / `test_fibo_wrapper_encode`.

- [ ] **Step 7: Drop `attention_mask=` from the kept SP test calls**

In the remaining device tests that call `enc.encode(...)` / `enc.forward(...)` with `attention_mask=None`, remove that argument:
- `test_smollm3_encoder_sp` (line ~427): `enc.encode(tt_ids, pos_embeds=(tt_cos, tt_sin))`.
- `test_smollm3_sp_bias_cached` (lines ~472, 475): both `enc.encode(tt_ids, pos_embeds=(tt_cos, tt_sin))`.

- [ ] **Step 8: Verify deleted symbols have no remaining references**

Run: `grep -n "prepare_attention_bias\|attention_mask" models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py`
Expected: no hits in `model_smollm3.py`; in the test file, only inside `test_fibo_wrapper_encode` if it references `attention_mask` (it does not — it uses the wrapper). If any `enc.forward(..., attention_mask=...)` remain, fix them.

- [ ] **Step 9: (Hardware, 4×8) regression**

Run: `pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_sp" "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_sp_bias_cached" -v`
Expected: PASS at `pcc=0.99` on the 4×8 Galaxy.

- [ ] **Step 10: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "refactor(fibo-pipeline): prune SmolLM3 encoder to the SP-only path"
```

---

### Task 5: Lazy `SmolLm3Checkpoint` + wrapper rewire (drop eager load & `use_torch`)

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py` (add `SmolLm3Checkpoint`; add imports)
- Modify: `models/tt_dit/pipelines/bria_fibo/text_encoder.py` (use `SmolLm3Checkpoint.build`; drop `use_torch`; collapse SP-only prep/readback)

**Interfaces:**
- Consumes: `cache.load_model(model, *, get_torch_state_dict, model_name, subfolder, parallel_config, mesh_shape)` (existing, `models/tt_dit/utils/cache.py`); `SmolLM3Config.from_hf_config` (Task 1); `SmolLM3TextEncoder` (Task 4, SP-only).
- Produces: `SmolLm3Checkpoint(name)` with `.config: SmolLM3Config` and `.build(*, device, parallel_config, ccl_manager=None) -> SmolLM3TextEncoder`. `SmolLM3TextEncoderWrapper.__init__(checkpoint, *, device, ccl_manager, parallel_config, pad_buckets=(1024,))` (no `use_torch`).

- [ ] **Step 1: Add `SmolLm3Checkpoint` to `model_smollm3.py`**

Add imports at the top of `model_smollm3.py`:

```python
import transformers
from ...utils import cache
from .config import SmolLM3Config
```

Add at the end of the file:

```python
class SmolLm3Checkpoint:
    """A SmolLM3 text-encoder checkpoint: reads only config.json up front; loads weights lazily.

    ``build()`` constructs a ``SmolLM3TextEncoder`` and populates it via ``cache.load_model`` — the
    torch weights are fetched only on a cache miss (and, if ``TT_DIT_CACHE_DIR`` is unset, loaded
    directly without writing a cache).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder="text_encoder")
        self.config = SmolLM3Config.from_hf_config(hf_config)

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
    ) -> "SmolLM3TextEncoder":
        model = SmolLM3TextEncoder(
            self.config, device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder="text_encoder",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
        )
        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.AutoModelForCausalLM.from_pretrained(
            self._name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        # Raw full-model state dict; SmolLM3TextEncoder._prepare_torch_state (STATE_CONVERSION) strips
        # `model.` / drops lm_head+rotary_emb, and the per-attention hook fuses qkv, during load.
        return torch_model.state_dict()
```

- [ ] **Step 2: Rewire the wrapper `__init__` (drop eager load + `use_torch`)**

In `models/tt_dit/pipelines/bria_fibo/text_encoder.py`, replace the imports and `__init__` (lines 21-106). New imports (drop `AutoConfig`, `SmolLM3ForCausalLM`, and the direct `SmolLM3Config`/`SmolLM3TextEncoder` imports since the checkpoint owns them; keep `AutoTokenizer`):

```python
from transformers import AutoTokenizer

import ttnn

from ...encoders.smollm3.model_smollm3 import SmolLm3Checkpoint
from ...utils import tensor as tt_tensor
```

New `__init__`:

```python
    def __init__(
        self,
        checkpoint: str,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: "CCLManager | None",
        parallel_config: "EncoderParallelConfig",
        pad_buckets=(1024,),
    ) -> None:
        self._device = device
        self._pad_buckets = tuple(pad_buckets)
        sp = parallel_config.sequence_parallel
        self._sp_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None
        self._sp_factor = sp.factor if (sp is not None) else 1

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self._encoder = SmolLm3Checkpoint(checkpoint).build(
            device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
```

- [ ] **Step 3: Drop the `use_torch` branch in `encode_prompt`**

In `encode_prompt` (lines 125-147), delete the `if self._use_torch:` block (lines 130-134). The method proceeds straight from `seq_len = input_ids.shape[1]` to `tt_ids, tt_cos, tt_sin = self._prep_inputs(...)`. Update the `encode_prompt` call to the encoder to drop `attention_mask` — see Step 4 (it flows through `_forward`).

- [ ] **Step 4: Collapse `_forward` / `_prep_inputs` / `_read_seq_sharded` to SP-only**

In `_forward` (line 186), drop `attention_mask=None`:

```python
        all_hidden_states = self._encoder.forward(tt_ids, pos_embeds=(tt_cos, tt_sin))
```

In `_prep_inputs` (lines 149-176), remove the `else` (non-SP) branch — `self._sp_factor > 1` always holds now:

```python
    def _prep_inputs(self, input_ids: torch.Tensor, seq_len: int) -> tuple:
        """Host prep: pad to a fixed bucket, build RoPE, move to device (sharded on the SP axis)."""
        bucket = pick_bucket(seq_len, self._pad_buckets, self._sp_factor)
        padded_ids = torch.nn.functional.pad(input_ids, (0, bucket - seq_len), value=0)
        cos, sin = self._encoder.create_rope_tensors(1, bucket)
        tt_ids = tt_tensor.from_torch(
            padded_ids,
            device=self._device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, self._sp_axis],
        )
        tt_cos = tt_tensor.from_torch(cos, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        tt_sin = tt_tensor.from_torch(sin, device=self._device, mesh_axes=[None, None, self._sp_axis, None])
        return tt_ids, tt_cos, tt_sin
```

In `_read_seq_sharded` (lines 189-203), remove the `if self._sp_factor <= 1:` early return (lines 197-198) — the SP fast path is the only path.

Delete the now-unused `import torch` usages? `torch` is still used (`torch.full`, `torch.nn.functional.pad`, `torch.cat`, `@torch.no_grad`), so keep `import torch`. Remove `from typing import TYPE_CHECKING, Any` only if `Any` is now unused — `_read_seq_sharded` uses `tuple[Any, ...]`, so keep it.

- [ ] **Step 5: Verify `use_torch` and eager-load are gone**

Run: `grep -n "use_torch\|SmolLM3ForCausalLM\|_torch_encoder\|load_torch_state_dict" models/tt_dit/pipelines/bria_fibo/text_encoder.py`
Expected: no hits.

- [ ] **Step 6: Verify the pipeline still constructs the wrapper correctly**

Run: `grep -n "SmolLM3TextEncoderWrapper(" models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`
Expected: the call passes only `checkpoint, device=, ccl_manager=, parallel_config=` (and optionally `pad_buckets=`), NOT `use_torch=`. If it passes `use_torch=`, remove that argument at the call site.

- [ ] **Step 7: (Hardware, 4×8) end-to-end wrapper regression**

Run: `pytest "models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_fibo_wrapper_encode" -v`
Expected: PASS — wrapper builds via `SmolLm3Checkpoint`, lazy-loads weights, and matches HF at `pcc=0.99`. This is the primary end-to-end gate for the whole refactor.

- [ ] **Step 8: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/pipelines/bria_fibo/text_encoder.py
git commit -m "refactor(fibo-pipeline): lazy SmolLm3Checkpoint + SP-only encoder wrapper"
```

---

### Task 6: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Host-only tests pass anywhere**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -k "config_from_hf or rope_matches_hf or state_conversion or bucket_pick or parallel_config_from_tuples" -v`
Expected: PASS (config test may SKIP offline).

- [ ] **Step 2: Full encoder suite on the 4×8 Galaxy**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v`
Expected: host tests + the `(4,8)` device tests PASS. No test references a removed symbol.

- [ ] **Step 3: Pipeline latent-PCC anchor (4×8)**

Run the FIBO pipeline latent-PCC test the branch already uses (locate it):
`grep -rln "output_type=\"latent\"\|latent.*pcc\|_gather_latent" models/tt_dit/tests/models/bria_fibo/`
Then run that test on the Galaxy. Expected: PASS at the same PCC as before this refactor — proves the encoder change is end-to-end behavior-preserving.

- [ ] **Step 4: Encode perf unchanged**

Run the encode perf test (`test_fibo_encode_perf` or equivalent in `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py`) on the Galaxy. Expected: SP=8×TP=4 encode time within noise of the pre-refactor ~12.5s (fast readback preserved). If materially slower, STOP and investigate (likely the readback or cache path).

- [ ] **Step 5: Final grep for dead references across the encoder + wrapper**

Run: `grep -rn "is_fsdp\|fsdp_mesh_axis\|prepare_attention_bias\|use_torch\|pop_substate\|MAX_CHUNK_SIZE" models/tt_dit/encoders/smollm3/ models/tt_dit/pipelines/bria_fibo/text_encoder.py`
Expected: only `MAX_CHUNK_SIZE` in `model_smollm3.py` (still used by `_sdpa_program_config`); everything else absent.

---

## Self-Review

**Spec coverage:**
- Change 1 (declarative StateConversion) → Task 2. Caveat that fusion stays imperative is honored (attention `_prepare_torch_state` untouched).
- Change 2 (lazy `SmolLm3Checkpoint` + `cache.load_model`) → Task 5.
- Change 3 (slim config) → Task 1.
- Change 4 (prune FSDP / non-SP / mask) → Tasks 3 + 4; `use_torch` drop → Task 5.
- Testing strategy (remove (1,1) + mask tests; remove (2,2) test — Galaxy-only; keep (4,8) SP + pipeline PCC) → Task 4 Steps 5-6, Task 6.
- Verification / definition-of-done → Task 6.

**Placeholder scan:** No TBD/TODO; every code step shows concrete code; commands have expected output.

**Type consistency:** `SmolLM3Config.from_hf_config` name kept across Tasks 1/5. `SmolLm3Checkpoint.build(*, device, parallel_config, ccl_manager)` signature consistent between Task 5 definition and wrapper call. `SmolLM3TextEncoder.forward(input_ids, *, pos_embeds)` / `.encode(...)` signatures consistent between Task 4 (definition) and Task 4/5 callers (tests + wrapper `_forward`). `STATE_CONVERSION` / `StateConversion.convert` names consistent between Tasks 2 and 5.
