# Qwen3.5‑9B Generator Interface — Implementation Plan (Plan 1: interface conformance)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Qwen35Model` drivable by the stock `tt_transformers.Generator` for **decode**, replace the bespoke vLLM wrapper with a local `Generator` subclass, and route **all** prefill through one dispatch helper to the model's existing prefill paths — without regressing long‑context generation.

**Architecture:** The model gains the `Generator` **decode** contract methods (`prepare_inputs_decode`, `prepare_decode_inputs_host`, `ttnn_decode_forward`, `process_output_decode`, `switch_mode`) built by reshaping the existing `_forward_decode` / `decode_paged` bodies. A local `Qwen35ForCausalLM(Generator)` wrapper drives decode via the stock `Generator`; **all prefill (short and long) is model‑owned** via the existing `prefill_paged` (non‑traced) / `prefill_traced_chunked` (traced), selected by a small dispatch helper. GDN recurrent state and the 8 attention KV caches stay model‑bound; the `kv_cache` contract param is accepted but unused.

> **Decision update (during execution, supersedes spec D3):** prefill is **entirely model‑owned**, not split at 2048 via Generator. Reason: the framework fills the paged KV cache *inside* `ttnn_prefill_forward`, whereas Qwen fills it in `prefill_paged`'s post‑forward housekeeping (`model.py:1029‑1055`) + `reset_state`; routing short prefill through Generator would skip that (decode reads an empty cache) and adds Generator's padding/`get_last_token` alignment subtleties. Making all prefill model‑owned is simpler, lower‑risk, matches gemma4, and still yields a clean `Generator`‑subclass wrapper + Generator‑driven decode/trace/sampling. Consequence: the prefill contract methods (`prepare_inputs_prefill`, `ttnn_prefill_forward`, `process_output_prefill`), the `max_prefill_chunk_size` tweak, and the device `pack_rope` helper are **removed** from this plan; Tasks 3/4/5/6 below reflect this.

**Tech Stack:** Python, ttnn, tt‑metal, `models/tt_transformers`, pytest. Target HW: Blackhole P150 (single device).

**Spec:** `docs/superpowers/specs/2026-06-01-qwen9b-generator-conformance-design.md`

---

## Conventions for this plan

- **NO GIT COMMITS.** Per the project's local‑only rule, the usual "commit" step is replaced by a **Validation checkpoint** (run the named test/command, confirm expected output, do not `git add`/`git commit`/`git push`).
- **Device required.** All `pytest` steps run on a Blackhole P150 with the Qwen3.5‑9B weights available; `HF_MODEL` points at the checkpoint (env already set in `demo/text_demo.py:32`). Use `n_layers=4` for fast iteration where a step says so.
- **Reuse, don't re‑paste.** Where a step says "lift the body of X (`file:lines`)", copy that exact code and apply only the input/return changes shown. The device‑op bodies already exist and are correct; do not rewrite them.
- **Out of scope (Plan 2):** the building‑block reuse swaps (MLP/LM‑head/embedding/RMSNorm/RoPE) — spec §7. **Deferred:** Tier‑B paged GDN state for continuous batching — spec §1.

> **Prerequisite (applied during execution, user‑approved):** the stock `Generator` can't be imported under Qwen's required transformers 5.9.0 — `models/common/llama_models.py:12` imported `AutoModelForVision2Seq`, removed in transformers 5.x. Applied a backward‑compatible shim there (`try: AutoModelForVision2Seq except ImportError: AutoModelForImageTextToText as AutoModelForVision2Seq`). This is the ONE sanctioned edit to a shared framework file; verified `import Generator`/`import Qwen35ForCausalLM` both succeed afterward. Worth upstreaming. See memory `qwen9b-generator-transformers5-shim`.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `tt/model.py` | `Qwen35Model` — gains the **decode** contract methods; keeps all prefill + paged methods; loses decode‑trace methods (after P2) | Modify |
| `tt/generator_interface.py` | Small helpers: `pack_rope_host`/`unpack_rope` for the decode rope payload; the model‑owned `prefill_dispatch` | **Create** |
| `tt/qwen35_vllm.py` | `Qwen35ForCausalLM(Generator)` replacing `TTQwen35ForCausalLM(nn.Module)` | Rewrite |
| `demo/text_demo.py` | Decode driven by `Generator`; prefill via the model dispatch helper; scaffolding unchanged | Modify |
| `tests/unit/test_generator_contract.py` | New: rope roundtrip + Generator‑driven decode equivalence vs baseline | **Create** |
| `tests/unit/test_decode_trace_equivalence.py` | New: Generator traced decode == old `decode_traced_paged` token stream (the prove‑then‑delete gate) | **Create** |

---

## Task 0: Baseline logits oracle (P0)

Capture current decode + short‑prefill logits so later "behavior‑preserving" claims are testable.

**Files:**
- Create: `tests/unit/test_generator_contract.py`

- [ ] **Step 1: Write the oracle‑capture test (records, asserts nothing yet)**

```python
# tests/unit/test_generator_contract.py
import os
import pytest
import torch
import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
FIXTURE = "models/demos/blackhole/qwen3_5_9b/tests/fixtures/generator_baseline.pt"

def _build(device, n_layers=4):
    return Qwen35Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, n_layers=n_layers)

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_capture_baseline(device):
    device.enable_program_cache()
    model = _build(device)
    BLOCK, NBLK = 64, 32
    model.allocate_kv_caches([NBLK, model.args.n_kv_heads, BLOCK, model.args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)

    prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)  # T=16, short
    pf_logits = ttnn.to_torch(model.prefill_paged(prompt, page_table)).squeeze().float()
    next_tok = int(pf_logits.argmax())

    dec = []
    tok = torch.tensor([[next_tok]], dtype=torch.long)
    for pos in range(16, 20):
        dl = ttnn.to_torch(model.decode_paged(tok, current_pos=pos, page_table=page_table)).squeeze().float()
        dec.append(dl)
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)

    torch.save({"prefill_logits": pf_logits, "decode_logits": torch.stack(dec)}, FIXTURE)
    assert pf_logits.shape[-1] == model.args.vocab_size
```

- [ ] **Step 2: Run to capture the fixture**

Run: `pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py::test_capture_baseline -v -s`
Expected: PASS; `tests/fixtures/generator_baseline.pt` written.

- [ ] **Step 3: Validation checkpoint (no commit)**

Confirm the fixture file exists and is non‑empty: `ls -la models/demos/blackhole/qwen3_5_9b/tests/fixtures/generator_baseline.pt`. Do not commit.

---

## Task 1: Add Generator attributes + `switch_mode` (P1)

`Generator` reads `model.mesh_device`, `model.configuration.max_seq_len`, `model.sampling`, `model.sampling_dp`, and calls `model.switch_mode(...)`.

**Files:**
- Modify: `tt/model.py` (`Qwen35Model.__init__`, add `switch_mode`)

- [ ] **Step 1: Add the attributes in `__init__`**

In `Qwen35Model.__init__` (after `self.device = mesh_device`, `tt/model.py:32`), add:

```python
        self.mesh_device = mesh_device          # Generator reads model.mesh_device
        self.configuration = args               # Generator reads model.configuration.max_seq_len
        self.sampling = None                    # host sampling only (no on-device sampler)
        self.sampling_dp = 1
        self._supports_on_device_sampling = False
```

- [ ] **Step 2: Add `switch_mode` (no prefetcher → no‑op)**

Add a method to `Qwen35Model`:

```python
    def switch_mode(self, mode):
        """Generator calls this on mode change; Qwen has no prefetcher, so no-op."""
        return None
```

- [ ] **Step 3: Validation checkpoint (no commit)**

Run: `python -c "import ast,sys; ast.parse(open('models/demos/blackhole/qwen3_5_9b/tt/model.py').read()); print('ok')"`
Expected: `ok`. Do not commit.

---

## Task 2: Rope payload helper + decode contract (P1)

`Generator` carries one tensor in the rope slot (`copy_host_to_device` can't take a nested `(cos,sin)` tuple — `common.py:557‑562`). Pack cos+sin into one host tensor; unpack inside the forward (spec D8).

**Files:**
- Create: `tt/generator_interface.py`
- Modify: `tt/model.py` (add `prepare_decode_inputs_host`, `prepare_inputs_decode`, `ttnn_decode_forward`, `process_output_decode`)
- Test: `tests/unit/test_generator_contract.py`

- [ ] **Step 1: Write the rope‑pack roundtrip test**

```python
# append to tests/unit/test_generator_contract.py
from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import pack_rope, unpack_rope

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_pack_roundtrip(device):
    cos = torch.randn(1, 1, 1, 64)
    sin = torch.randn(1, 1, 1, 64)
    cos_tt = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    packed = pack_rope(cos_tt, sin_tt)
    c2, s2 = unpack_rope(packed)
    assert tuple(c2.shape) == tuple(cos_tt.shape)
    assert tuple(s2.shape) == tuple(sin_tt.shape)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest .../tests/unit/test_generator_contract.py::test_rope_pack_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: ...generator_interface`.

- [ ] **Step 3: Implement `pack_rope`/`unpack_rope`**

```python
# tt/generator_interface.py
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the Generator contract: pack the (cos,sin) rope pair into one
tensor (copy_host_to_device cannot carry a nested tuple), and the shared
short/long prefill dispatch (so the wrapper and demo define the seam once)."""
import torch
import ttnn

def pack_rope(cos, sin):
    """Concat cos,sin along dim 0 -> one tensor for the Generator's rope slot.
    DEVICE path (prefill): cos,sin are device ttnn tensors (from rope.get_rot_mats)."""
    return ttnn.concat([cos, sin], dim=0)

def pack_rope_host(cos_host, sin_host):
    """HOST path (decode): cos,sin are HOST ttnn tensors (from rope.get_cos_sin_host),
    so ttnn.concat (a device op) can't be used — pack via torch instead."""
    cos_t = ttnn.to_torch(cos_host)
    sin_t = ttnn.to_torch(sin_host)
    packed = torch.cat([cos_t, sin_t], dim=0)
    return ttnn.from_torch(packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

def unpack_rope(packed):
    """Split a pack_rope()/pack_rope_host() tensor back into (cos, sin). Works on a
    device tensor (slicing dim 0); called inside ttnn_decode_forward / ttnn_prefill_forward."""
    n = packed.shape[0] // 2
    return packed[0:n], packed[n : 2 * n]
```

Note: `pack_rope` (device `ttnn.concat`) is used by the prefill path (Task 3, where cos/sin are device tensors) and exercised by the roundtrip test below. `pack_rope_host` (torch) is used by decode's `prepare_decode_inputs_host` (Step 5), because `get_cos_sin_host` returns HOST tensors `[1,1,rope_head_dim]` TILE.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest .../tests/unit/test_generator_contract.py::test_rope_pack_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Add `prepare_decode_inputs_host` to `Qwen35Model`**

Returns host ttnn tensors in the order `ttnn_decode_forward` consumes them: `(tokens, current_pos, rope_packed, page_table)`. Build from the host‑prep half of `decode_paged` (`tt/model.py:1074‑1087`) + `rope.get_cos_sin_host`.

```python
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import pack_rope_host
        import torch
        B = tokens.shape[0]
        tokens_tt = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = current_pos[0].item() if isinstance(current_pos, torch.Tensor) else int(current_pos)
        cos_host, sin_host = self.rope.get_cos_sin_host(pos)        # HOST ttnn tensors [1,1,rope_head_dim]
        rope_packed = pack_rope_host(cos_host, sin_host)           # torch-based (host); device concat can't run on host
        cur_pos_tt = ttnn.from_torch(
            torch.full((B,), pos, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None else None
        )
        return tokens_tt, cur_pos_tt, rope_packed, page_table_tt
```

- [ ] **Step 6: Add `prepare_inputs_decode` (host → device)**

```python
    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        from models.tt_transformers.tt.common import copy_host_to_device
        host = self.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        return copy_host_to_device(host, mesh_device=self.mesh_device)
```

- [ ] **Step 7: Add `ttnn_decode_forward` (wraps existing `_forward_decode`)**

`_forward_decode(token_ids_buf, cos, sin, cur_pos_tensor, page_table)` (`tt/model.py:326`) is already device‑only and trace‑safe. Wrap it: unpack the rope payload, ignore `kv_cache` (model‑bound state, spec D7), return `(logits, None)`.

```python
    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None,
        kv_cache=None, sampling_on_device=False, capture_sampling_trace=False, **kwargs,
    ):
        from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import unpack_rope
        cos, sin = unpack_rope(rot_mat_idxs)
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
        return logits, None    # host sampling: no log_probs / on-device tokens
```

- [ ] **Step 8: Add `process_output_decode` (mirror reference single‑device branch, `tt_transformers/tt/model.py:594‑609`)**

```python
    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        out = ttnn.to_torch(tt_out).float()
        return out[:, :, :B, : self.args.vocab_size].view(B, S, -1)
```

- [ ] **Step 9: Validation checkpoint (no commit)**

Run: `pytest .../tests/unit/test_generator_contract.py::test_rope_pack_roundtrip -v` → PASS; syntax‑check `model.py` as in Task 1 Step 3. Do not commit.

---

## Task 3: Finalize rope helpers — remove dead device `pack_rope`, retarget the roundtrip test (P1)

Under the model‑owns‑all‑prefill decision, the device `pack_rope` is dead (only the dropped prefill path used it). Decode uses `pack_rope_host` (host) + `unpack_rope` (device slice). Remove `pack_rope` and point the roundtrip test at the real decode path. **No prefill contract methods are added** (`prepare_inputs_prefill` / `ttnn_prefill_forward` / `process_output_prefill` are NOT created), and **no `max_prefill_chunk_size` change** is made.

**Files:**
- Modify: `tt/generator_interface.py` (delete `pack_rope`)
- Modify: `tests/unit/test_generator_contract.py` (retarget `test_rope_pack_roundtrip`)

- [ ] **Step 1: Delete the dead `pack_rope` function**

In `tt/generator_interface.py` remove the `pack_rope(cos, sin)` function entirely. Keep `pack_rope_host` and `unpack_rope`.

- [ ] **Step 2: Retarget the roundtrip test to the host‑pack decode path**

Replace the existing `test_rope_pack_roundtrip` in `tests/unit/test_generator_contract.py` (and its import) with:

```python
from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import pack_rope_host, unpack_rope

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_pack_roundtrip(device):
    # Mirrors the decode flow: pack on host, copy to device, unpack on device.
    cos = torch.randn(1, 1, 64)
    sin = torch.randn(1, 1, 64)
    cos_host = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_host = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    packed = ttnn.to_device(pack_rope_host(cos_host, sin_host), device)
    c2, s2 = unpack_rope(packed)
    assert tuple(c2.shape) == (1, 1, 64)
    assert tuple(s2.shape) == (1, 1, 64)
```

- [ ] **Step 3: Validation checkpoint (no commit)**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py::test_rope_pack_roundtrip -v` → PASS; syntax‑check `tt/generator_interface.py`. Do not commit.

---

## Task 4: Local `Qwen35ForCausalLM(Generator)` wrapper (P1)

**Files:**
- Rewrite: `tt/qwen35_vllm.py`

- [ ] **Step 1: Rewrite the wrapper as a `Generator` subclass**

Replaces `TTQwen35ForCausalLM(nn.Module)` and drops the three vLLM protocol stubs.

```python
# tt/qwen35_vllm.py
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM wrapper for Qwen3.5-9B: a thin tt_transformers Generator subclass."""
import os
import math
import torch
import ttnn
from models.tt_transformers.tt.generator import Generator
from models.demos.blackhole.qwen3_5_9b.tt.common import create_tt_model
from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prefill_dispatch

_PREFILL_WARMUP_CHUNK = 2048
_BLOCK_SIZE = 64

class Qwen35ForCausalLM(Generator):
    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len,
                              tt_data_parallel=1, optimizations=None, **kwargs):
        name_or_path = hf_config._name_or_path
        if name_or_path and not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download
            name_or_path = snapshot_download(name_or_path)
        args, model, _ = create_tt_model(mesh_device, max_batch_size, max_seq_len, hf_model=name_or_path)
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        logits = prefill_dispatch(self.model[0], tokens, page_table, prompt_lens,
                                  use_trace=kwargs.get("enable_trace", False))
        logits = ttnn.to_torch(logits) if isinstance(logits, ttnn.Tensor) else logits
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, rope_deltas

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *a, **k):
        if getattr(self, "_warmed_prefill", False):
            return
        self._warmed_prefill = True
        num_blocks = math.ceil(4096 / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        if enable_trace:
            self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, *a, **k):
        if not enable_trace:
            return
        dummy = torch.zeros(1, 1, dtype=torch.long)
        start = torch.zeros(1, dtype=torch.int64)
        pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.decode_forward(dummy, start, page_table=pt, kv_cache=kv_cache, enable_trace=True, read_from_device=False)
```

- [ ] **Step 2: Validation checkpoint (no commit)**

Run: `python -c "import ast; ast.parse(open('models/demos/blackhole/qwen3_5_9b/tt/qwen35_vllm.py').read()); print('ok')"`
Expected: `ok` (note: `prefill_dispatch` is added in Task 5; until then this import will fail at runtime, which is fine). Do not commit.

---

## Task 5: Shared prefill dispatch helper (P1/P3)

**Files:**
- Modify: `tt/generator_interface.py` (add `prefill_dispatch`)

- [ ] **Step 1: Add `prefill_dispatch`**

All prefill is **model‑owned** (decision update above): traced → the chunk‑outer trace; non‑traced → `prefill_paged`. `prefill_paged` already branches short(concat)/long(chunked) internally and does the post‑forward paged‑cache fill + GDN finalize that decode depends on; `prefill_traced_chunked`'s eager tail does the same via the paged path. So both leave the model decode‑ready for any length. No Generator involvement, no length threshold.

```python
# append to tt/generator_interface.py

def prefill_dispatch(model, tokens, page_table, prompt_lens, use_trace):
    """All prefill is model-owned. traced -> chunk-outer trace; non-traced -> paged.
    Both fill the paged KV cache + finalize GDN state, so decode continues correctly."""
    T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
    if use_trace:
        return model.prefill_traced_chunked(tokens, page_table, actual_len=T)
    return model.prefill_paged(tokens, page_table)
```

- [ ] **Step 2: Validation checkpoint (no commit)**

Syntax‑check `tt/generator_interface.py`. Do not commit.

---

## Task 6: Contract test — Generator drives decode (P2)

Proves the decode contract methods (Task 2) are wired correctly and behavior‑preserving vs the baseline (Task 0). Prefill is model‑owned (same path as the baseline), so the test runs prefill via the model to establish KV + GDN state, then drives **decode** through the stock `Generator` (non‑traced — isolates the contract from the trace risk, which Task 7 covers).

**Files:**
- Modify: `tests/unit/test_generator_contract.py`

- [ ] **Step 1: Write the non‑traced decode equivalence test**

```python
# append to tests/unit/test_generator_contract.py
from models.tt_transformers.tt.generator import Generator

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generator_decode_matches_baseline(device):
    device.enable_program_cache()
    base = torch.load(FIXTURE)
    model = _build(device)
    args = model.args
    BLOCK, NBLK = 64, 32
    model.allocate_kv_caches([NBLK, args.n_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], device)

    # Prefill is model-owned (identical path to the baseline) — establishes KV + GDN state.
    prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)
    pf = ttnn.to_torch(model.prefill_paged(prompt, page_table)).squeeze().float()
    assert torch.allclose(pf, base["prefill_logits"], atol=1e-2, rtol=1e-2), "prefill drifted from baseline"
    next_tok = int(pf.argmax())

    # Decode is Generator-driven — the new contract path under test.
    tok = torch.tensor([[next_tok]], dtype=torch.long)
    for i, pos in enumerate(range(16, 20)):
        out = gen.decode_forward(tok, torch.tensor([pos]), page_table=page_table, kv_cache=None,
                                 enable_trace=False, read_from_device=True)
        dl = out[0].squeeze().float() if isinstance(out, tuple) else out.squeeze().float()
        assert torch.allclose(dl, base["decode_logits"][i], atol=1e-2, rtol=1e-2), f"decode step {i} drifted"
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)
```

- [ ] **Step 2: Run; debug until it passes**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py::test_generator_decode_matches_baseline -v -s`
Expected: PASS. If decode drifts/shapes mismatch, fix the Task 2 decode contract — common culprits: rope unpack dim, `current_pos` tensor shape `[B]`, the `prepare_decode_inputs_host` tuple order vs `ttnn_decode_forward` positional params, or `process_output_decode` slicing. Note Generator's `decode_forward` host path may return `(logits, log_probs)` — the test already handles tuple vs tensor.

- [ ] **Step 3: Validation checkpoint (no commit)**

Re‑run the full file: `python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py -v`. All PASS. Do not commit.

---

## Task 7: Traced decode equivalence gate, then delete old decode‑trace (P2)

Per the new‑before‑delete rule: prove Generator‑traced decode equals the old `decode_traced_paged` token stream. **This task does NOT delete anything** — the demo still calls the old methods until Task 8 migrates it, so the actual deletion is deferred to Task 9 (after the demo no longer references them). Task 7 is purely the proof gate.

**Files:**
- Create: `tests/unit/test_decode_trace_equivalence.py`

- [ ] **Step 1: Write the traced‑vs‑old token‑stream test**

```python
# tests/unit/test_decode_trace_equivalence.py
import pytest, torch, ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model
from models.tt_transformers.tt.generator import Generator

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]

@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_traced_decode_matches_old(device):
    device.enable_program_cache()
    NBLK, BLOCK = 32, 64

    def gen_tokens(use_generator):
        model = Qwen35Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, n_layers=4)
        model.allocate_kv_caches([NBLK, model.args.n_kv_heads, BLOCK, model.args.head_dim], ttnn.bfloat16, batch_size=1)
        pt = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)
        prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)
        logits = ttnn.to_torch(model.prefill_paged(prompt, pt)).squeeze().float()
        tok = int(logits.argmax()); out = [tok]
        if use_generator:
            g = Generator([model], [model.args], device)
            t = torch.tensor([[tok]], dtype=torch.long)
            for pos in range(16, 24):
                r = g.decode_forward(t, torch.tensor([pos]), page_table=pt, kv_cache=None,
                                     enable_trace=True, read_from_device=True)
                dl = (r[0] if isinstance(r, tuple) else r).squeeze().float()
                tok = int(dl.argmax()); out.append(tok); t = torch.tensor([[tok]], dtype=torch.long)
        else:
            model.capture_decode_trace_paged(device, pt)
            t = torch.tensor([[tok]], dtype=torch.long)
            for pos in range(16, 24):
                dl = ttnn.to_torch(model.decode_traced_paged(t, pos, pt)).squeeze().float()
                tok = int(dl.argmax()); out.append(tok); t = torch.tensor([[tok]], dtype=torch.long)
        return out

    assert gen_tokens(use_generator=True) == gen_tokens(use_generator=False)
```

- [ ] **Step 2: Run the gate**

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_decode_trace_equivalence.py -v -s`
Expected: PASS (identical token streams).
**If FAIL (R1 — Generator's trace doesn't carry the model‑bound GDN state):** STOP. Report it. Do NOT change scope. The old methods remain the working fallback; the demo can keep using them. A GDN‑state hook into Generator's trace would be needed — escalate that decision rather than improvising.

- [ ] **Step 3: Validation checkpoint (no commit)**

Confirm the equivalence test PASSES. No deletions in this task — the decode‑trace methods are removed in Task 9, after Task 8 stops referencing them. Do not commit.

---

## Task 8: Migrate the demo to Generator + dispatch helper (P3)

Keep all scaffolding; swap only the generation core. Long‑context coverage must not regress.

**Files:**
- Modify: `demo/text_demo.py` (`_run_traced_generation`, `_run_paged_generation`)

- [ ] **Step 1: Build a Generator and route decode through it (traced path)**

In `_run_traced_generation` (`demo/text_demo.py:261`), after `model.allocate_kv_caches(...)`, construct the Generator and replace the decode loop. Prefill keeps using the model's chunk‑outer trace via the shared helper:

```python
    from models.tt_transformers.tt.generator import Generator
    from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prefill_dispatch
    gen = Generator([model], [model.args], device)
    # Prefill is model-owned (all lengths): the existing trace-capture +
    # prefill_traced_chunked logic in this function is unchanged. Equivalently it
    # can be expressed as: prefill_dispatch(model, token_ids, page_table, torch.tensor([T]), use_trace=True)
    # Decode loop (now Generator-driven):
    tok = torch.tensor([[next_token]], dtype=torch.long)
    for i in range(max_generated_tokens - 1):
        out = gen.decode_forward(tok, torch.tensor([current_pos]), page_table=page_table,
                                 kv_cache=None, enable_trace=True, read_from_device=True)
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        next_token = int(dl.argmax()); current_pos += 1
        if next_token == tokenizer.eos_token_id: break
        generated.append(next_token)
```

- [ ] **Step 2: Route the non‑traced demo path through the helper**

In `_run_paged_generation` (`demo/text_demo.py:358`), replace the direct `model.prefill_paged` + `model.decode_paged` calls with `prefill_dispatch(model, token_ids, page_table, torch.tensor([T]), use_trace=False)` for prefill and `gen.decode_forward(..., enable_trace=False)` for decode (same shape as Step 1).

- [ ] **Step 3: Run moderate e2e cases (both decode modes)**

⚠️ `pytest -k` is a SUBSTRING match: `-k "traced_128"` ALSO runs `traced_128k` (~100k tokens, very slow — can hang the device, needs `tt-smi -r`). Use collision‑free moderate ids:

Run: `export HF_MODEL=/local/ttuser/atupe/Qwen9b && python_env/bin/pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced_4k or paged_4k"`
Expected: PASS; coherent generated text; perf assertions hold. (`traced_4k` exercises traced decode via Generator; `paged_4k` exercises non‑traced decode via Generator. Neither id substring‑collides with a longer case.)

- [ ] **Step 4: Long‑context no‑regression (deferred, run explicitly)**

The long cases (`traced_8k/16k/32k/64k/128k`) are a deliberate slow regression — run them EXPLICITLY by full id, not via a short filter, and not as part of the Task 8 gate:
`python_env/bin/pytest "models/demos/blackhole/qwen3_5_9b/demo/text_demo.py::test_demo_text[traced_8k]" -v -s`
The chunk‑outer prefill path is untouched and Generator decode is already proven for long context in the unit gate, so this is low‑risk confirmation. If the device hangs, recover with `tt-smi -r`.

- [ ] **Step 5: Validation checkpoint (no commit)**

Confirm Steps 3–4 PASS. Do not commit.

---

## Task 9: Retire the now‑redundant trace methods (P5 cleanup)

Deletes both the **decode‑trace** methods (proven redundant in Task 7, no longer referenced after the Task 8 demo migration) and the **legacy whole‑sequence bucket prefill‑trace** methods (superseded by chunk‑outer `prefill_traced_chunked` / `prefill_paged`). All deletions are **by name** (line numbers in the original plan are stale — Tasks 1–6 shifted them). Do this only after Tasks 6–8 are green.

**Files:**
- Modify: `tt/model.py`

- [ ] **Step 1: Confirm no remaining callers (anywhere)**

Run: `grep -rn "capture_decode_trace_paged\|decode_traced_paged\|capture_prefill_trace_paged\|prefill_traced_paged" models/demos/blackhole/qwen3_5_9b/`
Expected: matches ONLY in `tt/model.py` (the definitions) — no demo/wrapper/test callers. If anything else still references them, STOP — Task 8 didn't fully migrate; fix that first.

- [ ] **Step 2: Delete the four methods (by name) + their now‑dead `__init__` buffers**

Delete these methods from `Qwen35Model`: `capture_decode_trace_paged`, `decode_traced_paged`, `capture_prefill_trace_paged`, `prefill_traced_paged`. KEEP `capture_prefill_trace_chunked`, `prefill_traced_chunked`, `_forward_prefill_chunk`, `_forward_prefill_chunk_eager`, `_forward_decode`, `prefill_paged`, `decode_paged`, `prefill_layer_chunked`, `reset_state`, `allocate_kv_caches`, `set_paged_kv_caches`, `_fill_paged_cache_from_prefill`, `_reset_dn_state_inplace`, `_init_dn_zero_buffers`.

For each `__init__` buffer attribute, GREP its name across the post‑deletion file and remove ONLY those referenced nowhere else: candidates from the decode‑trace methods are `_trace_id`, `_trace_token_ids`, `_trace_cos`, `_trace_sin`, `_trace_cur_pos`, `_trace_page_table`, `_trace_output`, `_prev_page_table`; from the bucket prefill‑trace, `_prefill_trace_inputs`, `_prefill_trace_id`, `_prefill_trace_logits`, `_prefill_bucket_size`. **Do NOT remove** `_chunked_*`, `_chunk_*`, `_dn_zero_recurrent`, `_dn_zero_conv`, `_deltanet_external_states`, `_paged_kv_caches`, `_attention_layer_indices` without grep‑confirming they're unused — the chunk‑outer trace + decode still use several of these. `_save_deltanet_states`/`_restore_deltanet_states`: grep; if only the deleted decode‑trace used them, delete; otherwise keep.

After deleting, syntax‑check and re‑import the model (`python_env/bin/python -c "import ast; ast.parse(open('.../tt/model.py').read()); print('ok')"`).

- [ ] **Step 3: Validation checkpoint (no commit)**

Re‑run the unit tests (`test_generator_contract.py`, `test_decode_trace_equivalence.py`) + the demo `traced_8k` and `traced_128k` cases → all PASS. `grep` confirms the four methods are gone from `tt/model.py`. Do not commit.

---

## Self‑review notes

- **Spec coverage (post Option‑B):** decode contract + attrs → Tasks 1–2; rope helpers → Tasks 2–3; model‑owned prefill dispatch → Task 5; wrapper → Task 4; decode‑trace ownership moved to Generator → proven in Task 7; demo migration → Task 8; cleanup (delete redundant trace methods) → Task 9; testing → Tasks 0,6,7 + demo regression in 8. §7 reuse (P4) and Tier‑B deferred to Plan 2 / out of scope. NOTE: the spec's "Generator drives short prefill" (D3) is superseded — all prefill is model‑owned (see Decision update at top).
- **R1 (GDN state under Generator decode trace)** is gated by Task 7 (prove‑only); on failure, STOP/escalate (don't improvise a framework hook). Deletion of old decode‑trace methods deferred to Task 9 (after Task 8 stops referencing them).
- **Type/name consistency:** `pack_rope_host`/`unpack_rope` (decode rope) and `prefill_dispatch(model, tokens, page_table, prompt_lens, use_trace)` used identically across Tasks 2,3,4,5,8; the decode host‑prep 4‑tuple `(tokens, current_pos, rope_packed, page_table)` matches `ttnn_decode_forward`'s positional params; `process_output_decode` slices the 3D `[B,1,vocab]` decode logits.
```
