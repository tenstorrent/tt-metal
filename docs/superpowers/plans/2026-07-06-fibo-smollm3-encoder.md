# FIBO SmolLM3 Text Encoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tt_dit TTNN encoder that numerically reproduces HuggingFace `SmolLM3ForCausalLM(output_hidden_states=True)`, exposing FIBO's two conditioning outputs (the 4096-dim `concat(hs[-1], hs[-2])` and the full per-layer hidden-state list), PCC-validated against the HF reference on a Blackhole mesh.

**Architecture:** SmolLM3-3B is a plain Llama-style decoder (RMSNorm, SwiGLU, GQA, rotate-half RoPE) with one twist — **NoPE** layers (every 4th layer skips RoPE). We adapt the existing `encoders/qwen25vl` decoder stack (closest match) for the layers/attention/MLP, borrow the all-hidden-states forward shape from `encoders/gemma`, and add the three net-new pieces: NoPE, a plain single-axis RoPE table (θ=5e6), and an HF-exact all-hidden-states output contract. Mesh-native tensor parallelism via tt_dit primitives.

**Tech Stack:** Python, PyTorch (reference), `transformers` (SmolLM3 reference), `ttnn`, tt_dit framework (`models/tt_dit`). Runs on Tenstorrent Blackhole (P150) mesh.

**Spec:** `docs/superpowers/specs/2026-07-06-fibo-smollm3-encoder-design.md`

## Global Constraints

- **Every new `.py` file starts with the SPDX header** (two lines, exactly as in sibling files):
  `# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.` then a blank line then `# SPDX-License-Identifier: Apache-2.0`.
- **Package location:** `models/tt_dit/encoders/smollm3/`. Tests: `models/tt_dit/tests/encoders/smollm3/`.
- **Precision:** bf16 weights/activations throughout. Compute-kernel configs use `math_fidelity=HiFi2` for matmuls and `HiFi4` for RMSNorm/SDPA (mirror qwen25vl/gemma).
- **Parallelism:** gate all CCL (`all_gather`) calls on `parallel_config.tensor_parallel.factor > 1` (gemma style) so `(1,1)` dev runs need no fabric/CCL.
- **SmolLM3-3B config (verified):** `vocab_size=128256`, `hidden_size=2048`, `intermediate_size=11008`, `num_hidden_layers=36`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=128`, `rms_norm_eps=1e-6`, `rope_theta=5000000.0`, `max_position_embeddings=65536`, `no_rope_layer_interval=4`, `hidden_act="silu"`, `attention_bias=False`. NoPE (skip RoPE) on layers where `(i+1) % 4 == 0` → layers 3,7,11,…,35.
- **Reference checkpoint:** `briaai/FIBO`, subfolder `text_encoder` (gated — requires `huggingface-cli login`). Tests read `FIBO_PATH` env (default `"briaai/FIBO"`) and `pytest.skip` if unavailable.
- **PCC gate:** per-hidden-state PCC ≥ 0.99 (bf16), via `models.tt_dit.utils.check.assert_quality`.
- **Templates to copy from (read these first):** `models/tt_dit/encoders/qwen25vl/model_qwen25vl.py` (decoder stack), `models/tt_dit/encoders/gemma/model_gemma.py` (all-hidden-states forward), `models/tt_dit/tests/encoders/qwen25vl/test_qwen25vl.py` and `.../gemma/test_gemma_encoder_all_layers.py` (test patterns).
- **The all-hidden-states convention is the subtlest correctness point.** HF `output_hidden_states` = `[embed, out(L0), …, out(L_{N-2}), finalnorm(out(L_{N-1}))]`, length `N+1`. Our encoder MUST reproduce this exactly by appending the **input to** each layer, then the final norm (NOT gemma's append-after-each-layer, which yields `N+2` and a different `hs[-2]`). FIBO reads `hs[-1]` and `hs[-2]` from this HF tuple.

---

## File Structure

- `models/tt_dit/encoders/smollm3/__init__.py` — package marker (empty + SPDX).
- `models/tt_dit/encoders/smollm3/config.py` — `SmolLM3Config` (values + derived `no_rope_layers`).
- `models/tt_dit/encoders/smollm3/model_smollm3.py` — `SmolLM3Context`, `SmolLM3RmsNorm`, `SmolLM3Mlp`, `SmolLM3Attention`, `SmolLM3DecoderLayer`, `SmolLM3TextEncoder`, plus helpers `create_rope_tensors`, `_apply_rope`, `_rotate_half`, `optimal_groups`, `_pad`, `prepare_attention_bias`, and the `encode_prompt` FIBO-contract helper.
- `models/tt_dit/tests/encoders/smollm3/__init__.py` — empty + SPDX.
- `models/tt_dit/tests/encoders/smollm3/test_smollm3.py` — all tests (config, rope, mlp, attention, layer, encoder, contract).
- `models/tt_dit/models/BriaFibo.md` — model doc (created in the final task; extended by later sub-projects).

---

### Task 1: Package skeleton + `SmolLM3Config`

**Files:**
- Create: `models/tt_dit/encoders/smollm3/__init__.py`
- Create: `models/tt_dit/encoders/smollm3/config.py`
- Create: `models/tt_dit/tests/encoders/smollm3/__init__.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3Config` with attributes listed in Global Constraints plus `no_rope_layers: list[int]` (length `num_hidden_layers`, `1`=apply RoPE, `0`=NoPE) and a classmethod `from_hf_config(hf_config)`.

- [ ] **Step 1: Write the failing test**

In `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.tt_dit.encoders.smollm3.config import SmolLM3Config


def test_smollm3_config_defaults():
    c = SmolLM3Config()
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 4
    assert c.head_dim == 128
    assert c.num_hidden_layers == 36
    assert c.intermediate_size == 11008
    assert c.rope_theta == 5000000.0
    assert c.rms_norm_eps == 1e-6
    assert c.vocab_size == 128256
    assert c.attention_bias is False
    # NoPE on every 4th layer (0-indexed 3,7,...,35); 1 = apply rope, 0 = NoPE
    assert len(c.no_rope_layers) == 36
    assert c.no_rope_layers[0] == 1 and c.no_rope_layers[3] == 0 and c.no_rope_layers[7] == 0
    assert sum(c.no_rope_layers) == 27  # 36 - 9 NoPE layers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_config_defaults -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (config not created yet).

- [ ] **Step 3: Write the implementation**

`models/tt_dit/encoders/smollm3/__init__.py` and `models/tt_dit/tests/encoders/smollm3/__init__.py`: SPDX header only.

`models/tt_dit/encoders/smollm3/config.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class SmolLM3Config:
    """Configuration for the SmolLM3-3B text encoder (used by Bria FIBO)."""

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 5000000.0,
        max_position_embeddings: int = 65536,
        hidden_act: str = "silu",
        attention_bias: bool = False,
        no_rope_layer_interval: int = 4,
        no_rope_layers: list[int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.no_rope_layer_interval = no_rope_layer_interval
        # HF default: no_rope_layers[i] = int((i + 1) % interval != 0); 1 = apply RoPE, 0 = NoPE.
        if no_rope_layers is not None:
            self.no_rope_layers = list(no_rope_layers)
        else:
            self.no_rope_layers = [int((i + 1) % no_rope_layer_interval != 0) for i in range(num_hidden_layers)]

    @classmethod
    def from_hf_config(cls, hf_config) -> "SmolLM3Config":
        """Build from a transformers SmolLM3Config (or the .config of a loaded model)."""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", None),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            max_position_embeddings=hf_config.max_position_embeddings,
            hidden_act=hf_config.hidden_act,
            attention_bias=getattr(hf_config, "attention_bias", False),
            no_rope_layers=list(getattr(hf_config, "no_rope_layers", None)) if getattr(hf_config, "no_rope_layers", None) is not None else None,
            no_rope_layer_interval=getattr(hf_config, "no_rope_layer_interval", 4),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_config_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/encoders/smollm3/ models/tt_dit/tests/encoders/smollm3/
git commit -m "feat(smollm3): package skeleton + SmolLM3Config"
```

---

### Task 2: Plain RoPE tables (`create_rope_tensors`) — host unit test

**Files:**
- Create: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `create_rope_tensors(batch_size: int, sequence_length: int, head_dim: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]` returning `(cos, sin)` each shaped `(batch, 1, seq, head_dim)` (full-width, non-interleaved). Matches HF `SmolLM3RotaryEmbedding` (attention_scaling=1.0).
- Produces: `_rotate_half(x)` and `_apply_rope(x, cos, sin)` (ttnn), used by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`:
```python
import torch


def test_smollm3_rope_matches_hf():
    from models.tt_dit.encoders.smollm3.model_smollm3 import create_rope_tensors

    head_dim, rope_theta, batch, seq = 128, 5000000.0, 2, 40
    cos, sin = create_rope_tensors(batch, seq, head_dim, rope_theta)
    assert cos.shape == (batch, 1, seq, head_dim)
    assert sin.shape == (batch, 1, seq, head_dim)

    # HF reference: inv_freq then emb=cat(freqs,freqs); cos/sin over (seq, head_dim)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    pos = torch.arange(seq).float()
    freqs = torch.outer(pos, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    ref_cos, ref_sin = emb.cos(), emb.sin()

    torch.testing.assert_close(cos[0, 0], ref_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sin[0, 0], ref_sin, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_rope_matches_hf -v`
Expected: FAIL with `ImportError` (module/function missing).

- [ ] **Step 3: Write the implementation**

Create `models/tt_dit/encoders/smollm3/model_smollm3.py` with the SPDX header, imports, and these helpers (the rest of the file is added in later tasks):
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.substate import pop_substate, rename_substate
from .config import SmolLM3Config

MAX_CHUNK_SIZE = 128


def create_rope_tensors(
    batch_size: int, sequence_length: int, head_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain single-axis RoPE tables matching HF SmolLM3RotaryEmbedding (attention_scaling=1.0).

    Returns (cos, sin) each shaped (batch, 1, seq, head_dim), full-width (non-interleaved).
    """
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)  # (B, seq)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)  # (B, hd/2, 1)
    position_ids_expanded = position_ids[:, None, :].float()  # (B, 1, seq)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # (B, seq, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, seq, hd)
    cos = emb.cos().unsqueeze(1)  # (B, 1, seq, hd)
    sin = emb.sin().unsqueeze(1)
    return cos, sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return x * cos + _rotate_half(x) * sin
```

Then copy these three helpers **verbatim** from `models/tt_dit/encoders/qwen25vl/model_qwen25vl.py` into this file (they are model-agnostic): `optimal_groups` (lines 487-514), `_pad` (517-521), and `prepare_attention_bias` (524-534).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_rope_matches_hf -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/tt_dit/encoders/smollm3/model_smollm3.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py
git commit -m "feat(smollm3): plain RoPE tables + rope/attention helpers"
```

---

### Task 3: `SmolLM3RmsNorm` + `SmolLM3Mlp` — device parity vs HF submodules

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3Context` (dataclass: `device`, `tp_axis`, `ccl_manager`, `fsdp_mesh_axis=None`).
- Produces: `SmolLM3RmsNorm(size, *, eps, ctx)` — plain RMSNorm (no offset), `forward(x)`.
- Produces: `SmolLM3Mlp(hidden_size, intermediate_size, hidden_act, ctx)` — SwiGLU, no bias; `forward(x)`; child modules named `gate_proj`, `up_proj`, `down_proj` (match HF).

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`:
```python
import os
import pytest
import ttnn
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as tt_tensor
from models.tt_dit.utils.check import assert_quality

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_hf_smollm3():
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(FIBO_PATH, subfolder="text_encoder", torch_dtype=torch.float32)
    except Exception as e:  # gated / offline
        pytest.skip(f"FIBO text_encoder unavailable: {e}")
    return model.eval()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_mlp(*, mesh_device):
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3Context, SmolLM3Mlp

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    hf_mlp = hf.model.layers[0].mlp
    cfg = hf.config
    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)

    mlp = SmolLM3Mlp(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act, ctx)
    mlp.load_torch_state_dict(hf_mlp.state_dict())

    x = torch.randn(1, 128, cfg.hidden_size)
    with torch.no_grad():
        ref = hf_mlp(x)
    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_out = mlp.forward(tt_x)
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_mlp -v`
Expected: FAIL with `ImportError` (`SmolLM3Mlp` not defined).

- [ ] **Step 3: Write the implementation**

Add to `model_smollm3.py`. `SmolLM3Context` — copy `Qwen25VlContext` (qwen lines 29-34) renamed:
```python
@dataclass
class SmolLM3Context:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None
```

`SmolLM3RmsNorm` — copy `Qwen25VlRmsNorm` (qwen lines 461-472) renamed (unchanged logic — plain RMSNorm, no offset, HiFi4):
```python
class SmolLM3RmsNorm(RMSNorm):
    def __init__(self, size: int, *, eps: float, ctx: SmolLM3Context) -> None:
        super().__init__(size, norm_eps=eps, bias=False, mesh_device=ctx.device)
        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self._compute_kernel_config)
```

`SmolLM3Mlp` — copy `Qwen25VlMlp` (qwen lines 401-458) renamed, with **one change**: gate the final all-gather on `factor > 1` instead of `tp_axis is not None`. Concretely the class body is identical to qwen's except the constructor also stores `self._tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1`, and `forward` ends with:
```python
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.down_proj(x)
        if self._tp_factor > 1:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        return x
```
(Keep qwen's constructor: `gate_proj`/`up_proj` = `ColParallelLinear(..., bias=False, ...)`, `down_proj` = `RowParallelLinear(..., bias=False, ...)`, `act_fn = ttnn.silu`, and the `hidden_act != "silu"` guard.) SmolLM3's MLP keys (`gate_proj`, `up_proj`, `down_proj`) match the child names, so **no `_prepare_torch_state` override is needed**.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_mlp -v`
Expected: PASS (PCC ≥ 0.99). If the checkpoint is gated/unavailable, it SKIPS — run `huggingface-cli login` and accept the FIBO license first.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): RmsNorm + SwiGLU MLP with device parity test"
```

---

### Task 4: `SmolLM3Attention` (GQA, fused QKV, RoPE, NoPE) — device parity vs HF attention

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3Attention(*, hidden_size, num_heads, num_key_value_heads, use_rope: bool, ctx)`; `forward(x, *, attention_bias, pos_embeds) -> ttnn.Tensor`. Child linear named `qkv_proj` (fused, no bias) and `o_proj` (no bias). Merges HF `q_proj/k_proj/v_proj` → `qkv_proj` in `_prepare_torch_state`. When `use_rope` is False, RoPE is skipped (NoPE).

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`. This mirrors the proven qwen attention test (`tests/encoders/qwen25vl/test_qwen25vl.py:48-116`), adapted for SmolLM3's plain RoPE and NoPE:
```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
@pytest.mark.parametrize("use_rope", [pytest.param(True, id="rope"), pytest.param(False, id="nope")])
def test_smollm3_attention(*, mesh_device, use_rope):
    from models.tt_dit.encoders.smollm3.model_smollm3 import (
        SmolLM3Attention,
        SmolLM3Context,
        create_rope_tensors,
    )

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    cfg = hf.config
    seq = 128

    # Pick a reference layer whose HF use_rope matches, then force it to be safe.
    hf_attn = hf.model.layers[0].self_attn
    hf_attn.use_rope = use_rope

    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)
    attn = SmolLM3Attention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        use_rope=use_rope,
        ctx=ctx,
    )
    attn.load_torch_state_dict(hf_attn.state_dict())

    x = torch.randn(1, seq, cfg.hidden_size)
    cos, sin = create_rope_tensors(1, seq, cfg.head_dim, cfg.rope_theta)

    # HF reference: pure-causal (all real tokens), so device is_causal path matches.
    with torch.no_grad():
        ref, _ = hf_attn(
            x,
            position_embeddings=(cos[:, 0], sin[:, 0]),  # HF expects (B, seq, head_dim)
            attention_mask=None,
        )

    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device)
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device)
    tt_out = attn.forward(tt_x, attention_bias=None, pos_embeds=(tt_cos, tt_sin))
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99, relative_rmse=0.2)
```
> Note: if HF's `SmolLM3Attention` rejects `attention_mask=None` in your transformers version, build a boolean lower-triangular mask `torch.tril(torch.ones(seq, seq)).bool()[None, None]` and pass it (all tokens real → equivalent to causal); the device side still uses `is_causal` (bias=None).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_attention -v`
Expected: FAIL with `ImportError` (`SmolLM3Attention` not defined).

- [ ] **Step 3: Write the implementation**

Add `SmolLM3Attention` to `model_smollm3.py` by copying `Qwen25VlAttention` (qwen lines 226-397) renamed, with exactly these changes:

1. Add `use_rope: bool` to `__init__` params (keyword-only) and store `self._use_rope = use_rope`.
2. Set the fused QKV projection **without bias** (SmolLM3 has `attention_bias=False`): change the `self.qkv_proj = ColParallelLinear(...)` call to pass `bias=False`.
3. In `_prepare_torch_state`, keep the weight-merge block (qwen 317-320) and the `o_proj` reshape block (327-341), but **delete the bias-merge block** (qwen 322-325) — SmolLM3 has no q/k/v bias.
4. In `forward`, guard RoPE on `self._use_rope`:
```python
        cos, sin = pos_embeds
        if self._use_rope:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
```
5. Gate the two post-projection all-gathers on `self._tp_factor > 1` instead of `self._tp_axis is not None` (qwen already stores `self._tp_factor`).

Keep everything else identical: the `optimal_groups`/`nlp_create_qkv_heads` GQA path, `_sdpa_program_config`, `_sdpa_compute_kernel_config` (HiFi4), `scaled_dot_product_attention(is_causal=attention_bias is None, attn_mask=attention_bias, ...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_attention -v`
Expected: PASS for both `rope` and `nope` params (PCC ≥ 0.99). The `nope` case failing while `rope` passes ⇒ the NoPE guard is wrong; the reverse ⇒ the RoPE table/apply is wrong.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): GQA attention with fused QKV, RoPE, and NoPE"
```

---

### Task 5: `SmolLM3DecoderLayer` — device parity vs HF (1-layer model)

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3DecoderLayer(*, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, hidden_act, rms_norm_eps, use_rope, ctx)`; `forward(x, *, attention_bias, pos_embeds) -> ttnn.Tensor`. Child modules named `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm` (match HF).

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`. Uses a truncated 1-layer HF model as the oracle (stable `output_hidden_states` API):
```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_decoder_layer(*, mesh_device):
    from models.tt_dit.encoders.smollm3.model_smollm3 import (
        SmolLM3Context,
        SmolLM3DecoderLayer,
        create_rope_tensors,
    )

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    cfg = hf.config
    seq = 128
    hf_layer = hf.model.layers[0]  # layer 0 is a RoPE layer
    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)

    layer = SmolLM3DecoderLayer(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        rms_norm_eps=cfg.rms_norm_eps,
        use_rope=bool(cfg.no_rope_layers[0]) if getattr(cfg, "no_rope_layers", None) else True,
        ctx=ctx,
    )
    layer.load_torch_state_dict(hf_layer.state_dict())

    x = torch.randn(1, seq, cfg.hidden_size)
    cos, sin = create_rope_tensors(1, seq, cfg.head_dim, cfg.rope_theta)
    with torch.no_grad():
        ref = hf_layer(x, position_embeddings=(cos[:, 0], sin[:, 0]), attention_mask=None)
        ref = ref[0] if isinstance(ref, tuple) else ref

    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_out = layer.forward(
        tt_x,
        attention_bias=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99, relative_rmse=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_decoder_layer -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Add `SmolLM3DecoderLayer` by copying `Qwen25VlDecoderLayer` (qwen lines 181-222) renamed, with these changes:
1. Add `use_rope: bool` to `__init__` and pass it through to `SmolLM3Attention(..., use_rope=use_rope, ...)`.
2. Use `SmolLM3Attention`, `SmolLM3Mlp`, `SmolLM3RmsNorm` (the renamed classes).
The `forward` is unchanged (pre-norm attn + residual, pre-norm mlp + residual). Child names (`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`) match HF, so **no `_prepare_torch_state` override**.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_decoder_layer -v`
Expected: PASS (PCC ≥ 0.99).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): decoder layer with device parity test"
```

---

### Task 6: `SmolLM3TextEncoder` — HF-exact all-hidden-states + per-layer PCC gate

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3TextEncoder(config: SmolLM3Config, *, device, parallel_config, ccl_manager, is_fsdp=False)`; `forward(input_ids, *, attention_mask=None, pos_embeds) -> list[ttnn.Tensor]` returning the HF-ordered hidden-state list `[embed, out(L0), …, out(L_{N-2}), finalnorm(out(L_{N-1}))]` (length `N+1`); and `create_rope_tensors(batch, seq) -> (cos, sin)`. Child modules: `embed_tokens`, `layers`, `norm`.

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`. Truncated-depth per-layer PCC (fast, tp=1) — mirrors the gemma all-layers test, but compares **index-for-index** because our list already matches HF:
```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_encoder_all_layers(*, mesh_device):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    n_layers = int(os.environ.get("N_LAYERS", "6"))
    seq = 128
    hf = _load_hf_smollm3()
    hf.model.layers = hf.model.layers[:n_layers]
    hf.config.num_hidden_layers = n_layers

    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_hs = [h.float() for h in ref.hidden_states]  # length n_layers + 1

    cfg = SmolLM3Config.from_hf_config(hf.config)
    cfg.num_hidden_layers = n_layers
    cfg.no_rope_layers = cfg.no_rope_layers[:n_layers]

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    hs = enc.forward(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    assert len(hs) == len(ref_hs), f"got {len(hs)} states, expected {len(ref_hs)}"
    for i, (r, d) in enumerate(zip(ref_hs, hs)):
        assert_quality(r, tt_tensor.to_torch(d), pcc=0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_all_layers -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Add `SmolLM3TextEncoder` to `model_smollm3.py`, adapting `Qwen25VlTextEncoder` (qwen 38-177) and `GemmaEncoder.forward` (gemma 468-533). Constructor takes a `SmolLM3Config` (not loose kwargs):
```python
class SmolLM3TextEncoder(Module):
    def __init__(
        self,
        config: SmolLM3Config,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        tp_factor = parallel_config.tensor_parallel.factor
        fsdp_mesh_axis = None
        if is_fsdp and tp_factor > 1:
            other = 1 - tp_axis
            if device.shape[other] > 1:
                fsdp_mesh_axis = other
        ctx = SmolLM3Context(
            device=device,
            tp_axis=tp_axis if tp_factor > 1 else None,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )
        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            raise ValueError("ccl_manager must be provided if tensor parallelism is used")

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device=device)
        self.layers = ModuleList(
            SmolLM3DecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                use_rope=bool(config.no_rope_layers[i]),
                ctx=ctx,
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = SmolLM3RmsNorm(config.hidden_size, eps=config.rms_norm_eps, ctx=ctx)

        self._config = config
        self._device = device
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF SmolLM3ForCausalLM prefix is "model."; SmolLM3Model state has no prefix.
        # Accept either by stripping "model." if present, then drop lm_head.
        prefix = "model."
        for k in list(state):
            if k.startswith(prefix):
                state[k[len(prefix):]] = state.pop(k)
        pop_substate(state, "lm_head")
        rename_substate(state, "rotary_emb", "")  # HF keeps a buffer-only rotary_emb submodule; drop it
        pop_substate(state, "rotary_emb")

    def create_rope_tensors(self, batch_size: int, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return create_rope_tensors(batch_size, sequence_length, self._head_dim, self._rope_theta)

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            padded = -(-seq_len // 32) * 32 if seq_len < MAX_CHUNK_SIZE else -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE
            input_ids = ttnn.pad(input_ids, [(0, padded - seq_len)], value=0)
            pos_embeds = tuple(ttnn.pad(x, [(0, 0), (0, 0), (0, padded - seq_len), (0, 0)], value=0) for x in pos_embeds)
            attention_mask = ttnn.pad(attention_mask, [(0, padded - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            padded = seq_len
            attention_bias = None

        hidden_states = self.embed_tokens.forward(input_ids)

        # HF output_hidden_states convention: append the INPUT to each layer, then the final norm.
        all_hidden_states: list[ttnn.Tensor] = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            hidden_states = layer.forward(hidden_states, attention_bias=attention_bias, pos_embeds=pos_embeds)
        hidden_states = self.norm.forward(hidden_states)
        all_hidden_states.append(hidden_states)

        if padded != seq_len:
            all_hidden_states = [x[:, :seq_len, :] for x in all_hidden_states]
        return all_hidden_states
```
> Note on `_prepare_torch_state`: the test passes `hf.model.state_dict()` (no `model.` prefix, no `lm_head`), so the strip/pop are no-ops there; they make the loader robust when handed a full `SmolLM3ForCausalLM.state_dict()`. If HF stores no `rotary_emb.*` keys, the `pop_substate` is a harmless no-op — keep it. Load with `strict=False` in Task 7's full-model path if any buffer keys remain.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_all_layers -v`
Expected: PASS — all `n_layers + 1` states at PCC ≥ 0.99. A single failing index localizes the first divergent layer (NoPE layers 3,7 are within the default N_LAYERS=6 only if you raise it; run with `N_LAYERS=8` once to cover a NoPE layer: `N_LAYERS=8 pytest ...::test_smollm3_encoder_all_layers -v`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): full text encoder with HF-exact all-hidden-states"
```

---

### Task 7: `encode_prompt` — FIBO output contract (prompt_embeds + per-layer list)

**Files:**
- Modify: `models/tt_dit/encoders/smollm3/model_smollm3.py`
- Test: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`

**Interfaces:**
- Produces: `SmolLM3TextEncoder.encode(input_ids, *, attention_mask=None, pos_embeds) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]` returning `(prompt_embeds, all_hidden_states)` where `prompt_embeds = concat(hs[-1], hs[-2])` along the feature dim (shape `[B, T, 2*hidden]` = 4096).

- [ ] **Step 1: Write the failing test**

Append to `test_smollm3.py`:
```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_encode_contract(*, mesh_device):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    n_layers = int(os.environ.get("N_LAYERS", "6"))
    seq = 128
    hf = _load_hf_smollm3()
    hf.model.layers = hf.model.layers[:n_layers]
    hf.config.num_hidden_layers = n_layers
    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    cfg.num_hidden_layers = n_layers
    cfg.no_rope_layers = cfg.no_rope_layers[:n_layers]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    prompt_embeds, hs = enc.encode(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    out = tt_tensor.to_torch(prompt_embeds)
    assert out.shape[-1] == 2 * cfg.hidden_size
    assert_quality(ref_prompt, out, pcc=0.99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encode_contract -v`
Expected: FAIL with `AttributeError` (`encode` not defined).

- [ ] **Step 3: Write the implementation**

Add to `SmolLM3TextEncoder`:
```python
    def encode(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        all_hidden_states = self.forward(input_ids, attention_mask=attention_mask, pos_embeds=pos_embeds)
        prompt_embeds = ttnn.concat([all_hidden_states[-1], all_hidden_states[-2]], dim=-1)
        return prompt_embeds, all_hidden_states
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encode_contract -v`
Expected: PASS. `out.shape[-1] == 4096`, PCC ≥ 0.99.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): encode() FIBO prompt-embeds contract"
```

---

### Task 8: Full 36-layer validation on (2,2) Blackhole mesh + model doc

**Files:**
- Modify: `models/tt_dit/tests/encoders/smollm3/test_smollm3.py`
- Create: `models/tt_dit/models/BriaFibo.md`

**Interfaces:** none new — this task validates the full model at target parallelism and documents it.

- [ ] **Step 1: Write the full-depth, multi-chip test**

Append to `test_smollm3.py`. Runs all 36 layers at tp on a 2×2 mesh with the fabric config, masked (padded) and unmasked, near the 3000-token limit:
```python
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}], indirect=["device_params"])
@pytest.mark.parametrize("seq", [128, 2048])
def test_smollm3_encoder_full_mesh(*, mesh_device, seq):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    tp_axis = 1
    hf = _load_hf_smollm3()
    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    prompt_embeds, _ = enc.encode(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    assert_quality(ref_prompt, tt_tensor.to_torch(prompt_embeds), pcc=0.99)
```

- [ ] **Step 2: Run it (full model, real weights, 2×2)**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py::test_smollm3_encoder_full_mesh -v`
Expected: PASS at PCC ≥ 0.99 for both seq lengths. If bf16 drift over 36 layers dips below 0.99 at `seq=2048`, first confirm the shorter-seq case passes, then (only if needed) document the measured floor in the test and the spec's risk section rather than silently lowering it.

- [ ] **Step 3: Run the whole suite once**

Run: `pytest models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v`
Expected: all tests PASS (or SKIP if weights absent). Capture the output.

- [ ] **Step 4: Write the model doc**

Create `models/tt_dit/models/BriaFibo.md` documenting: the 4-sub-project decomposition (link the spec), the SmolLM3 encoder (config, NoPE, all-hidden-states contract, files), how to run its tests (the exact `pytest` commands above, incl. `FIBO_PATH`/`N_LAYERS` env vars and `huggingface-cli login`), and the measured PCC. Mark the transformer/VAE/pipeline sub-projects as TODO.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(smollm3): full-mesh (2x2) validation + BriaFibo model doc"
```

---

## Self-Review

**Spec coverage:** encoder module (Tasks 3-6) ✓; config incl. NoPE (Task 1) ✓; plain θ=5e6 RoPE (Task 2) ✓; NoPE guard (Task 4) ✓; HF-exact all-hidden-states contract (Task 6) ✓; `prompt_embeds` 4096 concat (Task 7) ✓; weight remap `model.`/`lm_head` (Task 6) ✓; mesh-native TP + (2,2) validation (Task 8) ✓; PCC ≥ 0.99 gates on real weights, short + long prompts, NoPE-layer coverage (Tasks 6, 8) ✓; model doc (Task 8) ✓. Open items from the spec (gated config check, 37-vs-57 mapping) are correctly deferred (config check is the `_load_hf_smollm3` skip + `from_hf_config`; 37-vs-57 is a sub-project-2 concern).

**Placeholder scan:** no TBD/TODO in code steps; all steps carry complete code or exact copy-from-template diffs with line numbers, plus concrete run commands and expected results.

**Type consistency:** `SmolLM3Context`, `SmolLM3RmsNorm`, `SmolLM3Mlp`, `SmolLM3Attention(use_rope=)`, `SmolLM3DecoderLayer(use_rope=)`, `SmolLM3TextEncoder(config, device=, parallel_config=, ccl_manager=)`, `create_rope_tensors(batch, seq, head_dim, rope_theta)`, `encode()→(prompt_embeds, list)` are used consistently across tasks; child module names (`embed_tokens`, `layers`, `norm`, `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`, `qkv_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) match HF keys so the recursive loader resolves them.
