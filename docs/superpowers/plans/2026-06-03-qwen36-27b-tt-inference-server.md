# Qwen3.6-27B (text-only) tt-inference-server Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the existing `models/demos/qwen3_6_galaxy_v2/` Qwen3.6-27B text model through tt-inference-server's vLLM OpenAI API on BH Galaxy (32× P150).

**Architecture:** Rewire the model's `generator_vllm.py` to construct the *local* v2 `TtTransformer` (the way `text_demo_qwen36.py` does) instead of the stale Llama copy; load weights via raw safetensors (the checkpoint's `qwen3_5` arch is not in any public transformers, so `AutoModelForCausalLM` cannot be used). Register the TT class and a thin `qwen3_5` config in the vLLM plugin so vLLM's `AutoConfig` can parse the checkpoint. Add a YAML catalog entry + ImplSpec for the `BLACKHOLE_GALAXY` device.

**Tech Stack:** Python, TTNN, vLLM (tt-vllm-plugin), transformers 4.53.0 (NO bump), tt-inference-server `run.py` workflow.

**Spec:** `docs/superpowers/specs/2026-06-03-qwen36-27b-tt-inference-server-design.md`

**Verified facts driving this plan:**
- Authoritative demo construction: `models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py:153-171` (`TtQwen36ModelArgs` + local `TtTransformer`).
- Weights load via raw safetensors: `text_demo_qwen36.py:127-134` (`_load_full_state_dict`).
- `qwen3_5` model_type absent from transformers 4.53.0/4.57.1/4.57.6/5.10.1 (verified by wheel inspection).
- Checkpoint text params: `vocab_size=248320`, `hidden_size=5120`, `num_hidden_layers=64`, `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`, `max_position_embeddings=262144`, `intermediate_size=17408`.
- vLLM arch resolution prepends `"TT"`: `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/platform.py:91-94`.
- Catalog is YAML: `tt-inference-server/workflows/model_specs/prod/llm.yaml`; ImplSpec dict in `tt-inference-server/workflows/model_spec.py:271-272`.
- `DeviceTypes.BLACKHOLE_GALAXY` exists: `tt-inference-server/workflows/workflow_types.py:80`.

**Hardware note:** Tasks 1–4 are authored/tested off-device (import, config-parse, catalog-load). Task 5 requires the BH Galaxy (32-chip). Device-gated steps are marked **[DEVICE]**.

---

### Task 1: Rewrite `generator_vllm.py` to the local v2 model path

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` (full rewrite)
- Test: `models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py` (create)

- [ ] **Step 1: Write the failing import/structure test** (no device needed)

Create `models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Off-device structural checks for the qwen3.6 vLLM generator wrapper."""
import importlib
import inspect


def test_generator_vllm_uses_local_v2_modules():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    src = inspect.getsource(mod)
    # Must NOT import the parent llama3_70b_galaxy model/config classes.
    assert "llama3_70b_galaxy.tt.llama_model" not in src
    assert "llama3_70b_galaxy.tt.model_config" not in src
    assert "llama3_70b_galaxy.tt.qwen_model_config" not in src
    # Must use the local v2 qwen36 args + local transformer.
    assert "qwen3_6_galaxy_v2.tt.qwen36_model_config" in src
    assert "qwen3_6_galaxy_v2.tt.llama_model" in src
    assert "qwen3_6_galaxy_v2.tt.generator" in src


def test_serving_class_exists_and_has_vllm_api():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    cls = getattr(mod, "Qwen3_5ForConditionalGeneration")
    assert hasattr(cls, "initialize_vllm_model")
    assert hasattr(cls, "allocate_kv_cache")
    # Must NOT call AutoModelForCausalLM (broken for this checkpoint).
    assert "AutoModelForCausalLM" not in inspect.getsource(mod)


def test_no_prefetcher_perf_mode_kwarg():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    src = inspect.getsource(mod)
    assert "enable_prefetcher_performance_mode=True" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate && pytest models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py -v`
Expected: FAIL (current file imports `llama3_70b_galaxy`, has no `Qwen3_5ForConditionalGeneration`).

- [ ] **Step 3: Rewrite `generator_vllm.py`**

Replace the entire file `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` with:

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for Qwen3.6-27B (text-only) on BH Galaxy.

Mirrors the construction in demo/text_demo_qwen36.py: local v2 TtTransformer +
TtQwen36ModelArgs. Weights are loaded from raw safetensors because the
checkpoint's `qwen3_5` architecture is not in any public transformers release,
so AutoModelForCausalLM cannot load it.
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_st
from tqdm import tqdm

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs


def _resolve_ckpt_dir() -> Path:
    """Local checkpoint dir. The server sets HF_MODEL to a local symlink dir."""
    hf_model = os.getenv("HF_MODEL", "Qwen/Qwen3.6-27B")
    p = Path(hf_model)
    if p.is_dir():
        return p
    # Fall back to a resolved HF snapshot.
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model))


def _load_full_state_dict(ckpt_dir: Path) -> dict:
    """Load the raw HF state dict (model.language_model.* keys) from safetensors."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        sd = {}
        for fn in files:
            sd.update(load_st(str(ckpt_dir / fn)))
        return sd
    return load_st(str(ckpt_dir / "model.safetensors"))


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    """Paged KV cache allocation for qwen3.6 (n_kv_heads padded to 8)."""
    kv_dtype = ttnn.bfloat8_b if os.getenv("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16
    submesh_devices = [model.mesh_device]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_dtype,
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
            ]
            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def initialize_vllm_text_transformer_qwen36(
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
):
    instruct = "instruct" in str(getattr(hf_config, "_name_or_path", "")).lower()
    args = TtQwen36ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None:
        args.n_layers = n_layers

    # Raw safetensors load — AutoModelForCausalLM cannot parse `qwen3_5`.
    ckpt_dir = _resolve_ckpt_dir()
    state_dict = _load_full_state_dict(ckpt_dir)

    weight_cache_path = args.weight_cache_path(dtype)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    tt_model = TtTransformer(
        args=args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        use_paged_kv_cache=True,
        mode="prefill",
    )
    return tt_model, args


class Qwen3_5ForConditionalGeneration(Generator):
    """Text-only vLLM serving class for Qwen3.6-27B. Name matches the HF arch
    so platform.py's `TT` prefix resolves to this class."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        tt_model, model_args = initialize_vllm_text_transformer_qwen36(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)
```

- [ ] **Step 4: Run the structural test to verify it passes**

Run: `pytest models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify Generator ctor signature matches** (guard against arg mismatch)

Run: `python -c "import inspect; from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator; print(inspect.signature(Generator.__init__))"`
Expected: shows `(self, model, model_args, mesh_device, tokenizer=None, formatter=None)` — confirms `cls(tt_model, model_args, mesh_device)` is correct. If it differs, fix the `cls(...)` call and re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py models/demos/qwen3_6_galaxy_v2/tests/test_generator_vllm_import.py
git commit -m "Qwen3.6-27B: rewire generator_vllm to local v2 model + raw-safetensors load"
```

---

### Task 2: Register the `qwen3_5` config and the TT model in the vLLM plugin

**Files:**
- Create: `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py`
- Modify: `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/__init__.py`
- Test: `tt-inference-server/tt-vllm-plugin/tests/test_qwen3_5_config.py` (create)

- [ ] **Step 1: Write the failing config-parse test**

Create `tt-inference-server/tt-vllm-plugin/tests/test_qwen3_5_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from transformers import AutoConfig


CKPT = os.getenv("QWEN36_CKPT_DIR", "")


def test_qwen3_5_config_registered():
    # Importing the plugin registers the config.
    import tt_vllm_plugin  # noqa: F401
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "qwen3_5" in CONFIG_MAPPING


@pytest.mark.skipif(not CKPT, reason="set QWEN36_CKPT_DIR to a local Qwen3.6-27B dir")
def test_autoconfig_parses_checkpoint():
    import tt_vllm_plugin  # noqa: F401

    cfg = AutoConfig.from_pretrained(CKPT)
    assert cfg.num_hidden_layers == 64
    assert cfg.hidden_size == 5120
    assert cfg.num_attention_heads == 24
    assert cfg.vocab_size == 248320
    assert cfg.max_position_embeddings == 262144
    assert cfg.architectures == ["Qwen3_5ForConditionalGeneration"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tt-inference-server/tt-vllm-plugin && python -m pytest tests/test_qwen3_5_config.py::test_qwen3_5_config_registered -v`
Expected: FAIL ("qwen3_5" not in CONFIG_MAPPING).

- [ ] **Step 3: Write the config class**

Create `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Thin `qwen3_5` config so vLLM's AutoConfig can parse the Qwen3.6-27B
checkpoint. The TT model reads config.json itself; this only needs to surface
the text-tower fields vLLM uses for scheduling + KV-cache sizing.

`qwen3_5` is absent from public transformers (verified 4.53.0/4.57.x/5.x), so
we register it here at plugin import.
"""
from transformers import PretrainedConfig


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, **kwargs):
        # The checkpoint nests text params under `text_config`. Promote them so
        # vLLM's standard attribute reads (num_hidden_layers, hidden_size, ...)
        # resolve on the top-level config.
        text_config = kwargs.get("text_config", {}) or {}
        for key in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
        ):
            if key in text_config and key not in kwargs:
                kwargs[key] = text_config[key]
        super().__init__(**kwargs)
        # Keep the conditional-generation arch name so platform.py resolves
        # TTQwen3_5ForConditionalGeneration.
        if not getattr(self, "architectures", None):
            self.architectures = ["Qwen3_5ForConditionalGeneration"]


def register_qwen3_5_config():
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "qwen3_5" not in CONFIG_MAPPING:
        AutoConfig.register("qwen3_5", Qwen3_5Config)
```

- [ ] **Step 4: Call the registration from the plugin `__init__`**

In `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/__init__.py`, at the top of `register_models()` (right after `from vllm import ModelRegistry`), add:

```python
    # Register the qwen3_5 config so vLLM's AutoConfig can parse Qwen3.6-27B
    # (qwen3_5 is not in public transformers). Must happen before model load.
    try:
        from tt_vllm_plugin.qwen3_5_config import register_qwen3_5_config

        register_qwen3_5_config()
        print("Registered qwen3_5 config")
    except Exception as e:
        import logging

        logging.warning(f"Failed to register qwen3_5 config: {e}")

    # Register TT Qwen3.6-27B (galaxy v2, text-only)
    try:
        ModelRegistry.register_model(
            "TTQwen3_5ForConditionalGeneration",
            "models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_5ForConditionalGeneration",
        )
        print("Registered TT Qwen3.6-27B")
    except Exception as e:
        import logging

        logging.warning(f"Failed to register TTQwen3_5ForConditionalGeneration: {e}")
```

- [ ] **Step 5: Run the registration test to verify it passes**

Run: `cd tt-inference-server/tt-vllm-plugin && python -m pytest tests/test_qwen3_5_config.py::test_qwen3_5_config_registered -v`
Expected: PASS.

- [ ] **Step 6: Run the checkpoint-parse test** (if a local dir is available)

Run: `QWEN36_CKPT_DIR=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 python -m pytest tests/test_qwen3_5_config.py::test_autoconfig_parses_checkpoint -v`
Expected: PASS. If `max_position_embeddings`/heads assertions fail, the field names in `text_config` differ — adjust the promotion list in `qwen3_5_config.py` and re-run.

- [ ] **Step 7: Commit**

```bash
git add tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py \
        tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/__init__.py \
        tt-inference-server/tt-vllm-plugin/tests/test_qwen3_5_config.py
git commit -m "tt-vllm-plugin: register qwen3_5 config + TT Qwen3.6-27B model"
```

---

### Task 3: Add the ImplSpec and YAML catalog entry (BH Galaxy)

**Files:**
- Modify: `tt-inference-server/workflows/model_spec.py` (ImplSpec + IMPL dict)
- Modify: `tt-inference-server/workflows/model_specs/prod/llm.yaml`
- Modify: `tt-inference-server/workflows/model_specs/dev/llm.yaml`
- Test: `tt-inference-server/tests/test_qwen36_model_spec.py` (create)

- [ ] **Step 1: Write the failing catalog-load test**

Create `tt-inference-server/tests/test_qwen36_model_spec.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import os
os.environ.setdefault("MODEL_SPECS_ENV", "dev")

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes


def _find_qwen36_spec():
    for spec in MODEL_SPECS.values():
        if getattr(spec, "hf_model_repo", "") == "Qwen/Qwen3.6-27B" \
                and spec.device == DeviceTypes.BLACKHOLE_GALAXY:
            return spec
    return None


def test_qwen36_spec_present_for_blackhole_galaxy():
    spec = _find_qwen36_spec()
    assert spec is not None, "Qwen3.6-27B BLACKHOLE_GALAXY spec not found"
    assert spec.max_context == 262144
    assert spec.impl.code_path == "models/demos/qwen3_6_galaxy_v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tt-inference-server && MODEL_SPECS_ENV=dev python -m pytest tests/test_qwen36_model_spec.py -v`
Expected: FAIL (spec not found).

- [ ] **Step 3: Add the ImplSpec**

In `tt-inference-server/workflows/model_spec.py`, after the `qwen3_32b_galaxy_impl = ImplSpec(...)` block (ends ~line 230), add:

```python
qwen3_6_galaxy_impl = ImplSpec(
    impl_id="qwen3_6_galaxy",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/qwen3_6_galaxy_v2",
)
```

(Match the exact field names used by the sibling `qwen3_32b_galaxy_impl`; read lines 220–231 first and mirror them.)

- [ ] **Step 4: Register it in the IMPL dict**

In the impl registry dict (~line 271, where `"qwen3_32b_galaxy": qwen3_32b_galaxy_impl,` is), add:

```python
    "qwen3_6_galaxy": qwen3_6_galaxy_impl,
```

- [ ] **Step 5: Add the YAML catalog entry (dev first)**

Append to `tt-inference-server/workflows/model_specs/dev/llm.yaml`:

```yaml
- weights:
    - Qwen/Qwen3.6-27B
  impl: qwen3_6_galaxy
  version: "0.1.0"
  tt_metal_commit: fa021238662
  inference_engine: VLLM
  model_type: LLM
  supported_modalities:
    - text
  device_model_specs:
    - device: BLACKHOLE_GALAXY
      max_concurrency: 32  # 8 * 4
      max_context: 262144  # 256 * 1024
      default_impl: true
      env_vars:
        VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1
      vllm_args:
        trust_remote_code: true
  status: EXPERIMENTAL
  has_builtin_warmup: true
```

(Validate keys against an existing galaxy entry — `prod/llm.yaml:227-262`. If `override_tt_config` knobs like `dispatch_core_axis`/`fabric_config` are required for BH galaxy bring-up, copy them from the demo's run config in Task 5 and add here.)

- [ ] **Step 6: Run the catalog-load test to verify it passes**

Run: `cd tt-inference-server && MODEL_SPECS_ENV=dev python -m pytest tests/test_qwen36_model_spec.py -v`
Expected: PASS. If `DeviceTypes.from_string("BLACKHOLE_GALAXY")` errors, confirm the YAML device string the loader expects (read `DeviceTypes.from_string` in `workflow_types.py`) and use that exact token.

- [ ] **Step 7: Mirror the entry into prod and regenerate release JSON**

Copy the same YAML block into `tt-inference-server/workflows/model_specs/prod/llm.yaml`, then regenerate:

Run: `cd tt-inference-server && python -c "from scripts.build_docker_images import generate_model_specs_json; print(generate_model_specs_json())"`
Expected: prints the regenerated `release_model_spec.json` path with no error.

- [ ] **Step 8: Commit**

```bash
git add tt-inference-server/workflows/model_spec.py \
        tt-inference-server/workflows/model_specs/dev/llm.yaml \
        tt-inference-server/workflows/model_specs/prod/llm.yaml \
        tt-inference-server/release_model_spec.json \
        tt-inference-server/tests/test_qwen36_model_spec.py
git commit -m "tt-inference-server: add Qwen3.6-27B BH-galaxy model spec + impl"
```

---

### Task 4: Verify Docker / run-workflow wiring (no new Dockerfile expected)

**Files:**
- Read: `tt-inference-server/vllm-tt-metal/multihost_entrypoint.sh`
- Read: `tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py:343-366` (`register_tt_models`)
- Possibly modify: `tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py` (impl_id mapping)

- [ ] **Step 1: Confirm `impl_id` env wiring**

Run: `grep -n "impl_id\|TT_LLAMA_TEXT_VER\|TT_QWEN3_TEXT_VER\|register_tt_models" tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py`
Expected: `register_tt_models(impl_id)` is called with the spec's `impl.impl_id`. Our model registers via the plugin (Task 2), so `register_tt_models` needs **no** new branch unless it gates registration by impl_id. If it does, add an `elif impl_id == "qwen3_6_galaxy":` no-op/pass branch so the plugin registration is the source of truth.

- [ ] **Step 2: Confirm the multihost entrypoint covers 32-chip BH galaxy**

Run: `grep -n "galaxy\|32\|mesh\|MESH\|num.*dev\|DEVICE" tt-inference-server/vllm-tt-metal/multihost_entrypoint.sh`
Expected: the same entrypoint serves the parent `llama3_70b_galaxy` 32-device galaxy. Document whether any BH-specific env (e.g. `FABRIC`, `dispatch_core_axis`) must be set; if so, fold into the YAML `override_tt_config`/`env_vars` from Task 3 Step 5.

- [ ] **Step 3: Commit (only if a change was needed)**

```bash
git add tt-inference-server/vllm-tt-metal/src/run_vllm_api_server.py
git commit -m "tt-inference-server: wire qwen3_6_galaxy impl_id into vllm launcher"
```

If no change was needed, record the finding in the BRINGUP_LOG note (Task 5) and skip the commit.

---

### Task 5: **[DEVICE]** Bring up the server, validate accuracy, document

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/BRINGUP_LOG.md`
- Modify: `models/demos/qwen3_6_galaxy_v2/README.md`

- [ ] **Step 1: [DEVICE] Reset the device and start the server**

```bash
tt-smi -r
cd tt-inference-server
export TT_METAL_HOME=/home/tt-admin/ssinghal/qwen36/new/tt-metal
MODEL_SPECS_ENV=dev python run.py --model Qwen/Qwen3.6-27B --workflow server \
    --local-server --tt-device tt-galaxy-bh --skip-system-sw-validation
```
Expected: server reaches "Application startup complete" / listens on `:8000` with no device hang. If `AutoConfig` errors on `qwen3_5`, the plugin config registration (Task 2) did not load — confirm the plugin is installed in the server env (`pip show tt-vllm-plugin`).

- [ ] **Step 2: [DEVICE] Smoke test one completion**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16}'
```
Expected: JSON with a coherent completion mentioning "Paris".

- [ ] **Step 3: [DEVICE] Accuracy parity vs the demo**

Run the demo on a fixed prompt set and the server on the same set; compare. Use the demo's existing targets at `models/demos/qwen3_6_galaxy_v2/demo/text_demo_targets.json`.
Expected: server first-token / answer accuracy within 2–3 pp of the demo. Record both numbers.

- [ ] **Step 4: [DEVICE] Long-context stress (256k bar)**

Send one request near `max_context` (e.g. 200k-token prompt) and confirm no hang / garbled output.
Expected: coherent output; if it garbles, cap `max_context` in the YAML (Task 3) to the validated length and note it.

- [ ] **Step 5: [DEVICE] Confirm no reset needed between requests**

Send 5 sequential requests; confirm none require `tt-smi -r`.

- [ ] **Step 6: Update BRINGUP_LOG and README**

In `models/demos/qwen3_6_galaxy_v2/BRINGUP_LOG.md`, append a dated entry: server start command, accuracy numbers (demo vs server), long-context result, and the Task-4 Docker/entrypoint finding. Mark a `V2-server` stage DONE.

In `models/demos/qwen3_6_galaxy_v2/README.md`, add a "Server" section with the `run.py` start command, the smoke `curl`, and known limitations (text-only; BH galaxy only; transformers stays 4.53.0 via plugin config registration).

- [ ] **Step 7: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/BRINGUP_LOG.md models/demos/qwen3_6_galaxy_v2/README.md
git commit -m "Qwen3.6-27B: document tt-inference-server bring-up (server results + commands)"
```

---

## Self-Review Notes

- **Spec coverage:** Component 1 → Task 1; Component 2 (2a config + 2b model reg) → Task 2; Component 3 (model_spec + catalog) → Task 3; Component 4 (Docker/workflow) → Task 4; Component 5 (validation + docs) → Task 5. Transformers no-bump decision is enforced by Task 2 (config registration) and asserted in Task 1 (no `AutoModelForCausalLM`).
- **Names are consistent across tasks:** serving class `Qwen3_5ForConditionalGeneration`; registered arch `TTQwen3_5ForConditionalGeneration`; config `Qwen3_5Config` / `register_qwen3_5_config()`; impl_id `qwen3_6_galaxy`; code_path `models/demos/qwen3_6_galaxy_v2`; `max_context = 262144`.
- **Known fix-up points flagged inline** (not placeholders): Generator ctor signature check (T1.S5), text_config field-name promotion (T2.S6), `DeviceTypes.from_string` token (T3.S6), BH override_tt_config knobs (T3.S5/T4.S2).
