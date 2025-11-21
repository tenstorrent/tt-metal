# TTTv2 Design Proposal

## Problem Analysis

The core issue with TTTv1 (lives in models/tt_transformers/ directory) is the **N×M×P explosion**:
- N models × M platforms × P tests = exponential complexity
- Every change requires validating all combinations
- Adding model #11 requires testing against models 1-10

This problem gets worse with each newly added model
Our motivations for TTTv2 is to address this scaling problem and enalbe scaling to 100+ models

## Goals

Single developer can add a new LLM model without tribal knowledge

TT-Transformers (TTT) code as a library
- Modular
- Composable
- Readable
- Reproducible
- Maintainable and releasable

TTTv1 is good, first achievement of the goals for 10+ models; now TTTv2 must do better to get to 100+ models. TTTv2 should be a collection of building blocks that models consume, not a framework that controls models.

## Proposed Architecture

### Zen of TTTv2 Architecture

- Library, not Framework
- Balance between code and codegen
- If-else on static conditions in forward() is bad
- Lazy is better than proactive in loading weights
- Unit tests are better than end-to-end tests

### Core Library Structure

```bash
models/common/testing/    # Testing utilities
├── auto_compose.py
├── distribute_as.py
├── metrics.py
└── validation_tools.py

models/common/modules/
├── attention/
├── ffn/
├── embeddings/
├── normalization/
├── decoder_layer.py   # Pre-built decoder patterns
├── encoder_layer.py   # Pre-built encoder patterns
└── cross_attention.py # Cross-attention patterns

models/common/integration/   # Adapters to external frameworks and shared demo/runtime helpers
├── demo_integration.py      # Shared device/runtime helpers used by demos and adapters
└── vllm_integration.py      # vLLM-specific integration helpers (TTTv2-aware)

models/common/tests/      # Unit tests for real-world configurations
├── test_vllm_integration.py
├── attention/
├── ffn/
├── testing/
└── integration/

models/common/examples/      # Examples of how to use TTTv2 building blocks and patterns
└── demo.py            # standard way to building a demo of a model using TTTv2 building blocks and patterns
```

TTNN is the only tt-metal internal library that TTTv2 depends on.

**Key Points:**
- No model-specific code in TTTv2
- Models are free to organize as they wish
- Models live in separate location -- for now models/experimental/tt_transformers_v2/
- Models pip install specific TTTv2 versions -- for now add a commit hash as the last known good version for each model

### Testing

In bringing up new models on TT devices, it is common practice to compare against reference models in terms of key metrics such as PCC, max abs error, etc. Though following essentially the same pattern, model developers often write their own testing tools to debug or measure new models. Such time could be saved by carefully engineered, thoroughly tested testing tools. The time saving multiplies with the number of helped developers. Our goal is to use these tools within the models team to speed up model debugging and bring-up. We are also going to use these tools in building out TTTv2.

The testing tools include `compare_to_torch` and `compare_to_ttnn` python decorators. The [README.md](../README.md) file explains the basic usage of them and `models/common/tests/test_validation_tools.py` shows detailed examples of how they can be used. We also added a comprehensive example in `models/experimental/tt_transformers_v2/ds_r1_qwen.py` that shows how the testing tools could be used in debugging a complete ML model.

We added extensive unit tests on all the testing tools, covering multiple degrees of freedom -- device topology, dtypes, tensor layouts, and basic tensor shapes. We leave the extensive coverage of different tensor shapes to the future -- as planned, we will collect key tensor shapes from production models to include in the unit tests.

### Modules

Model developpers can implement their models using a spectrum of components. On one end of the spectrum, they can use only TTNN ops to implement their models. On the other end of the spectrum, they can use TTTv2 modules to implement their models.

The TTTv2 modules are designed to provide more optimized implementations, including:
- Functional and performant transformers-specific kernels
- Configurations for TTNN ops and transformer-specific kernels within each module, e.g., memory_config, program_config, sharding specs, etc.

#### Module specialization

By model specification, we mean the model architecture specification, including the dataflow and the module specification. We can also view a reference model code a form of model specification, e.g., HuggingFace model code.

Model specification to model implementation has a gap. For example, on TT device, we typically use different implementation for prefill and decode. Such details could be considered as part of the TTTv2 model implementation. TTTv2 modules should provide both prefill_forward and decode_forward specialization out of the box.

Based on the configurations of the contained TTNN ops, TTTv2 modules will need to handle additional axis of specialization, e.g., sharding based on model hidden size, etc.

#### Design options of module interfaces

TTTv2 allows the model developers to use any model specification format they want. The model specification format could be from huggingface, onnx, custom, etc. Implementing a model based a model specification format means mapping the model specification format to either the TTTv2 module interfaces or a graph of TTNN ops.

Therefore, for TTTv2, we must design module interfaces that are agnostic to the model specification formats. We can build adapters to map popular model specification formats (e.g, HuggingFace) to the TTTv2 module interfaces. The model developers should always has the option to map a full-custom model specification to TTTv2 module interfaces.

We show example adapters for HuggingFace in the following sections. Custom model specification formats can be implemented by following the same pattern.

##### Huggingface adapter

To keep TTTv2 model-agnostic while still making it easy to reuse existing HuggingFace (HF) model code, we can introduce a thin “adapter” layer, for example:

- TTTv2 attention modules stay HF-agnostic and expose simple, device-aware interfaces.
- HF-specific adapters live with the model code and are responsible for:
  - Translating HF configs into TTTv2 configs.
  - Reshaping tensors, handling RoPE, masks, kv-cache, etc.
  - Preserving the exact HF `forward(...)` signature and return type so the rest of the HF model remains unchanged.
- Adapters use **composition** (wrap a TTTv2 module) rather than inheritance from HF attention classes. At the model level we may still use inheritance from HF model classes purely to swap modules or customize `from_pretrained`.

Because HF attention modules do **not** share a single unified `forward` signature across architectures (BERT vs GPT-2 vs LLaMA vs Deepseek, encoder vs cross-attention, etc.), we cannot enforce a single global adapter signature. Instead we think in terms of *per-architecture* (or even per-model) adapter interfaces.

In practice, the adapter is responsible for mirroring the HF-facing behavior (projections, RoPE, cache plumbing, layout quirks) and then delegating the core attention computation to `ttt_core` via composition.

As a concrete example, consider a DeepseekV3-style HF attention module and its TTTv2-backed adapter:

```python
import torch
from torch import nn
from typing import Optional, Tuple


class DeepseekV3Attention(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...


class DeepseekV3AttentionAdapter(nn.Module): # instead of nn.Module, maybe we could use LightweightModule from models/common/lightweightmodule.py
    """
    Wraps a TTTv2 attention core and exposes the exact same
    forward(...) signature as DeepseekV3Attention so it can be
    used as a drop-in replacement inside the HF model.
    """

    def __init__(
        self,
        ttt_core: "TTTv2AttentionCore",
    ):
        super().__init__()
        self.ttt_core = ttt_core
        ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
```

Once such an adapter exists, we can reuse the recursive module replacement patterns from `transformers` (commonly used for quantization) to swap HF attention blocks with TTTv2-backed adapters. Crucially, we perform replacement at **higher-level semantic modules** (e.g., `DeepseekV3Attention`) rather than at low-level layers like `nn.Linear`, which often need more model-specific context.

```python
def replace_deepseek_attention_with_ttt(
    model: nn.Module,
    build_ttt_core: callable,
) -> nn.Module:
    """
    Recursively traverse the HF model and replace each DeepseekV3Attention
    with a DeepseekV3AttentionAdapter that wraps a TTTv2 attention core.
    """
    for name, module in list(model.named_children()):
        # Recurse first
        replace_deepseek_attention_with_ttt(module, build_ttt_core)

        # Replace higher-level HF attention blocks, not leaf nn.Linear layers
        if isinstance(module, DeepseekV3Attention):
            ttt_core = build_ttt_core(module)  # construct TTTv2 core from HF config/weights
            setattr(model, name, DeepseekV3AttentionAdapter(ttt_core=ttt_core))

    return model
```

This pattern keeps:
- TTTv2 modules completely HF-agnostic.
- HuggingFace integration localized to thin adapters and a small amount of traversal code.
- `from_pretrained` and HF utilities (generation, caching, etc.) working unchanged, as long as the adapter preserves the original `forward(...)` contract.


#### Module configuration

TTTv2 uses a module-centric approach where each module provides its own hardware configuration defaults. TTTv1 code already contains good default configurations for each module. They are typically in formats of lambda functions that take in the module specification and return the hardware configuration. TTTv2 should reuse these default configurations but refactor them for each module to be more module-specific.

todo)){ add example code for module configuration

A concrete pattern:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    # Device-facing knobs with module-specific defaults
    memory_config: MemoryConfig = default_mem_cfg()  # per-device tuned default
    program_config: ProgramConfig = default_prog_cfg()
    shard_layout: Optional[ShardLayout] = None       # optional override


class AttentionCore:
    def __init__(self, spec: AttentionSpec, config: AttentionConfig | None = None):
        self.config = config or AttentionConfig()
        self.spec = spec

    def forward(...):
        # TTNN ops always consume the per-module config (defaults or caller overrides)
        ...


# Caller override example
attn = AttentionCore(
    spec=AttentionSpec(...),
    config=AttentionConfig(
        memory_config=prefer_dram_for_prefill(),
        shard_layout=ring_2x2(),
    ),
)
```
}


#### Extension

TTTv2 is a library and allows the model developers to freely pick what they want to use from TTTv2.

How does TTTv2 grow to support new modules?

TTTv2 treats extension primarily as a pure-Python, constructor-based pattern rather than a string/registry-driven one. New modules implement small, stable interfaces (for example, `AttentionCore`, `MLPCore`) and are passed as instances or classes into higher-level patterns such as `DecoderLayer`. Model code always imports and refers to symbols directly, which keeps IDE navigation, refactoring, and type checking working naturally.

At the same time, TTTv2 stays tight by keeping a small curated set of implementations in `tttv2.core.*`, while allowing more experimental or model-specific implementations to live in `tttv2.contrib.*` or external repositories:

- **Core interfaces**: TTTv2 defines minimal abstract base classes / protocols for each kind of module (attention, MLP, normalization, etc.). Any implementation that satisfies these interfaces can be used in TTTv2 patterns.
- **Constructor injection**: Higher-level blocks (e.g., `DecoderLayer`) accept module instances or classes via their constructors, so model authors can freely mix TTTv2 modules with their own implementations:

  ```python
  layer = DecoderLayer(
      attn_core=MyFancyAttention(...),  # implements AttentionCore
      mlp_core=SwiGLU(...),
  )
  ```

- **Contrib and external modules**: TTTv2 keeps `core` small and stable, and uses `contrib` (and external repos) for additional modules. An optional, metadata-only registration API (e.g., `register_candidate(name="my_flash_v3", kind="attention", ...)`) lets external modules self-describe for discovery and potential promotion into `tttv2.contrib`, without changing how they are used in model code.

This approach lets TTTv2 grow organically with new modules and community contributions, while preserving a tight, well-defined core API and a good day-to-day developer experience.

### Tests

TTTv2 has extensive unit tests for the testing tools and the modules. We will add more unit tests for the modules in the future.

#### vLLM integration

TTTv2 itself stays unaware of vLLM. Integration with vLLM happens in a thin
adapter layer that depends on both vLLM and TTTv2 but keeps the core modules
TTNN‑only.

- The vLLM tech report describes the required interface for TT models:
  `initialize_vllm_model`, `allocate_kv_cache`, `prefill_forward`,
  `decode_forward`, and (for multi‑modal models) additional cross‑attention
  hooks and input processors. See
  [tech_reports/LLMs/vLLM_integration.md](../../tech_reports/LLMs/vLLM_integration.md)
  for the full details.
- We split “integration” into two pieces under `models/common/integration/`:
  - `demo_integration.py`: shared, vLLM‑agnostic helpers used by demos and
    adapters (device bring‑up, common prefill/decode helpers, optional
    validation wiring, paged KV cache helpers, etc.).
  - `vllm_integration.py`: vLLM‑specific adapters that implement the vLLM
    interfaces by delegating to TTTv2 modules and the helpers in
    `demo_integration.py`.
- `models/common/modules/**` remains TTNN‑only and model‑agnostic (no `torch`,
  no HuggingFace, no vLLM), while `models/common/integration/**` is allowed to
  depend on external serving stacks and model configs in order to wrap TTTv2
  modules behind higher‑level interfaces.

Concretely, a TTTv2‑based model that wants vLLM support:

1. Builds its core blocks (attention, MLP, etc.) from TTTv2 modules.
2. Implements a model‑local demo script under
   `models/experimental/tt_transformers_v2/<model>/demo.py` that:
   - Uses `models/common/integration/demo_integration.py` for device bring‑up
     and common runtime helpers.
   - Wires in model‑specific pieces such as weight loading, tokenizer, and
     prompt construction.
3. Adds a vLLM integration class under
   `models/experimental/tt_transformers_v2/<model>/vllm.py` (or the existing
   `models/tt_transformers/tt/generator_vllm.py` path) that:
   - Conforms to the vLLM interfaces.
   - Reuses the same prefill/decode logic and KV cache helpers from the demo
     path (via `demo_integration.py`) instead of re‑implementing them.

Recommended workflow:
- First, bring up the model in a standalone demo (following the `demo.py`
  pattern: device → model → prompt → generate → optional validation report).
  This is the easiest place to debug TTTv2 modules, weight loading, and
  accuracy.
- Once the demo path is stable, add the vLLM integration class as a thin
  adapter that reuses the same prefill/decode logic, KV cache helpers, and
  model construction, instead of re‑implementing these paths specifically for
  vLLM.


### Examples

`models/experimental/tt_transformers_v2/ds_r1_qwen.py` is the current reference for what a
standard `demo.py` should look like when using TTNN and (optionally) TTTv2 modules.
The high‑level pattern for `models/common/examples/demo.py` is:

1. **Open a mesh device**
   - Select a mesh shape based on the target system (e.g., N150, N300, T3K, etc.).
   - Fall back to a reasonable default when only a single device is available.
   - Optionally set TTNN’s default device so testing utilities can infer it.

2. **Configure validation (optional)**
   - Obtain the global registry via `get_validation_registry()` from `models.common.validation_tools`.
   - Enable or disable validation for the run (e.g., `registry.enabled = True` for debug / accuracy runs).
   - Individual components (RMSNorm, attention, MLP, logits head, etc.) are decorated with `@compare_to_torch`
     or `@compare_to_ttnn`, so calls during generation automatically record accuracy and timing metrics.

3. **Build the model**
   - Construct a model that runs on the opened mesh device. In `ds_r1_qwen.py` this is a pure‑TTNN Qwen
     implementation, but in TTTv2‑style demos the model would typically compose TTTv2 modules such as
     attention, MLP, and normalization blocks (for example, using a `DecoderLayer` that is parameterized
     by TTTv2 attention / MLP cores).
   - Weight loading and any HF adapters live with the model code, not inside TTTv2.

4. **Prepare inputs and generate**
   - Tokenize or otherwise prepare a prompt (e.g., via a HF tokenizer’s chat template).
   - Run a short generation loop on device (prefill + decode), using the model’s `forward` / `generate`
     entry point.

5. **Report validation results (optional)**
   - If validation was enabled, query the registry and, if non‑empty, call `registry.print_report(verbose=True)`.
   - The report summarizes pass/fail status for each decorated function, per‑metric values (e.g.,
     max abs error, PCC), and implementation vs. reference execution times, and derives an average
     speedup. This makes the demo a one‑shot view of both functional behavior (the generated text)
     and numerical/performance parity versus the chosen reference.

6. **Clean up**
   - Close the mesh device (`ttnn.close_mesh_device(...)`) and release any other resources.

In this pattern, the only “model‑specific” logic inside `demo.py` is how weights are loaded and how the
prompt is constructed. The surrounding scaffolding (device lifecycle, validation registry usage, and
end‑of‑run reporting) is reusable across models. TTTv2 modules fit naturally into this structure: the
“Build the model” step simply swaps a hand‑rolled TTNN implementation for a composition of TTTv2
building blocks configured for the target hardware.

The validation pieces in this flow are strictly optional: for quick bring‑up, profiling, or customer
demos where reference comparisons are not needed, model authors can simply skip configuring the
validation registry and omit the report step, and the rest of the `demo.py` template still applies.

We treat `models/common/examples/demo.py` as a **template** that model developers can reuse: they can
either copy it next to their model (e.g., under `models/experimental/tt_transformers_v2/<model>/demo.py`)
and fill in `build_model` / `generate`, or import its helper functions (device bring‑up, validation
configuration, reporting) into a model‑specific demo script. This keeps the bring‑up experience
consistent across models while preserving full freedom in how each model is structured.

For models that also integrate with vLLM, we follow the same pattern at the serving layer: a
`models/experimental/tt_transformers_v2/<model>/vllm.py` file (or an entry in
`models/tt_transformers/tt/generator_vllm.py`) implements the vLLM interfaces by delegating to the
same model construction and runtime helpers used by `demo.py` (via `models/common/integration/demo_integration.py`
and `models/common/integration/vllm_integration.py`). This keeps the demo and vLLM paths aligned and
minimizes duplicated prefill/decode logic.

Conceptually, `demo.py` and `vllm.py` share the same core but optimize for different goals:
- `demo.py` is a developer‑facing bring‑up and debugging tool and also a **demonstration vehicle** for
  close‑to‑hardware performance. It is where we most easily inspect correctness, enable validation, and
  measure raw TTNN/TTTv2 performance before any serving overheads.
- `vllm.py` is a production‑facing adapter that implements vLLM’s interfaces and focuses on end‑to‑end
  serving behavior (continuous batching, padding rules, tracing, memory limits, etc.). The demo path
  serves as a useful **baseline** for understanding performance trade‑offs introduced by vLLM (e.g.,
  batching, KV cache policies, orchestration costs).

### Public API Guidance and Top-Level Imports

For TTTv2, we care more about **clear, ergonomic imports** than about strictly hiding internal modules. The library should feel like a normal Python module inside this repo where:
- common building blocks are easy to import from a single façade module, and
- deeper modules remain importable for power users, but are clearly treated as internal.

Instead of a separate pip package, TTTv2 lives under `models.common` and exposes a façade module `models.common.tt_transformers`:

```python
# models/common/tt_transformers.py
"""TTTv2: transformer building blocks inside models.common."""

from dataclasses import dataclass

# Re-export key modules/classes from models.common.modules
from .modules.codegen_attention import TTTv2AttentionCodeGen, HardwareConfig, AttentionConfig
# from .modules.<other_module> import MultiHeadAttention, DecoderLayer, ...
from .testing import to_torch_auto_compose


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    label: str | None = None  # e.g. "dev", "rc1"

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.label}" if self.label else base


VERSION = Version(2, 0, 0, "dev")
__version__ = str(VERSION)

__all__ = [
    "TTTv2AttentionCodeGen",
    "HardwareConfig",
    "AttentionConfig",
    # "MultiHeadAttention",
    # "DecoderLayer",
    "VERSION",
    "__version__",
    "to_torch_auto_compose",
]
```

Usage in model code:

```python
from models.common import tt_transformers

core = tt_transformers.TTTv2AttentionCodeGen(
    hw_config=tt_transformers.HardwareConfig(...),
    attn_config=tt_transformers.AttentionConfig(...),
)

print(tt_transformers.__version__)  # "2.0.0-dev"
```

Individual modules under `models.common.modules` can still use `__all__` to highlight their public surface, but the main guidance for users is: “import TTTv2 from `models.common.tt_transformers`.”

#### API Surface Benefits

Using `__all__` and curated re-exports provides:

1. **Clear contracts**
   - Everything in `__all__` is public/recommended API, documented, intended to be stable across minor versions, and tested.
   - Everything not in `__all__` is considered an internal implementation detail that can change without notice and is not documented for external use.

2. **Tooling support**
   - IDEs and linters respect `__all__`, so `from models.common.tt_transformers import *` only imports the recommended API surface.
   - Documentation generators can be configured to only show objects listed in `__all__`.

### Key Design Points

#### Dependency

TTTv2 is designed to be a library and allows the model developers to freely pick what they want to use from TTTv2. Therefore, TTTv2 should not have any dependencies on the model code. We should also limit the dependencies that TTTv2 has. TTNN is a dependency of TTTv2 and we will work hard to make the only dependency of TTTv2 (besides python standard libraries).

Concretely, we enforce the following dependency boundaries:
- `models/common/modules/**` must not depend on PyTorch (`torch`, `torch.nn`, etc.). These modules only depend on TTNN and Python standard libraries.
  - Enforce with a simple CI check (e.g., rg "import torch" models/common/modules must be empty).
- `models/common/testing/**` is allowed to depend on PyTorch for reference implementations, `compare_to_torch`, and HF-based validation.
- Any PyTorch- or HF-facing adapters live alongside model code (e.g., `models/experimental/tt_transformers_v2/`) or in a dedicated integration layer, not in `models/common/modules/`.

#### Weight loading and caching
todo)){ work on this section; we want be careful not to violate the dependency boundaries as specified above.
consider: For things like codegen_attention.py that currently mix nn.Module / torch into generation, either:
- Move the torch-based class generation into a models/common/integration/torch/ (or similar) area, or
- Make the core generator produce torch-free code and have a thin torch wrapper in that integration layer.

Tensor Cache Strategy
    - Weight caches: Created at config time when converting from reference model --> allow skipping reference model loading if the weight cache is already available
      - weight cache are generated from the reference model's weight and we want a "lazy" approach to loading reference model weights -- if TTNN weight cache is already available, we can skip loading the reference model weights
    - Activation caches: Pre-allocated at compile time when shapes are known
    - Both use hardware-optimized layouts and dtypes --> need checks to make sure the available caches are compatible with the layout/dtype configurations
}

we need a map from model to what TTT module expects
TTTv2 should rely on model author to provide a clear mapping instead of what TTTv1 is doing
example: tt_dit, gpt-oss to see if there is any good ideas out there
module should take torch.Tensor (ttnn.Tensor on host?)
more transparently supports separation between weight conversion and weight loading and from_torch.
some modules need state to be able to avoid passing in a torch.Tensor during construction time.

#### Tests

- Unit Tests for the modules: we need to provide comprehensive coverage across all modules and all configurations.
  - configurations include: tensor shapes, data types, device topologies, sharding strategies, etc.

- Testing metrics should include accuracy and performance.
  - may add memory occupancy in future


#### Versioning & Compatibility

**Version Guarantee Matrix:**
```
TTTv2 2.0.x → TTTv2 2.0.y: Binary compatible (patch) -- no kernel changes
TTTv2 2.0.x → TTTv2 2.1.x: Source compatible (minor) -- kernel implementations may change but the interface should not change
TTTv2 2.x.x → TTTv2 3.x.x: Migration required (major) -- Interface changes or new features
```

**Model Pinning:**
```bash
# models/experimental/tt_transformers_v2/llama3/requirements.txt
tt_transformers==2.1.*  # Pin to minor version
# we may want to use commit hash of tt-metal for now to pin the dependency on tt_transformers
```

We may want to implement a model CI tool that checks the experimental models against the TTTv2 version.

#### TTNN ops configuration

- Direct TTNN Op Configuration
    - No abstraction over TTNN ops initially (pragmatic approach)
    - Each module has device-specific defaults for its TTNN ops
    - Users can override individual ops when needed
- In the future, maybe we could identify common patterns of TTNN ops and create a library of common patterns that can be used to configure the TTNN ops.

#### Configuration observability
todo)) {
From the sake of sane debugging of the integrated model, need to be able to dump configuration that are relevant to integration (such as vLLM integration and tt-metal integration):
Stemming from debugging P0 issues:
people saying that their run of the model does not work
we need a significant amount of time to figure out which side the bug is on -- tt-inference-server, vLLM, or tt-metal.
no confidence in if we have the same config.
core question: how do we quickly figure out where the bug is? Is it tt-metal only? Is it a combo of the multiple parts? For example:
tt-metal has fixed page size for example
vLLM has more flexible configs
What if we have two simple JSON configs?
one is the complete config that tt-inference-server/vLLM is using
the other one is the one that tt-metal is using
What config to dump?
maybe we could just add some code in model's prefill or decode function to dump the config -- on the tt-metal side
we also should be careful with some difference in kv_cache_shape
vLLM configure the kv_cache_shape internally and pass it to generator_vllm.py
So, we should print this config our from our model code.
}

#### H. Code generation

We treat codegen as an optional specialization emitter backed by a curated config registry, not as the only way to get runnable code:

- **Default path (checked-in outputs):** For supported shapes/topologies (e.g., 1d T3K, 2d Galaxy), we check in prebuilt specializations. Most developers read/modify concrete code; no generator required for day-to-day edits.
- **Config registry:** A small, declarative mapping `(shape, topology, ModuleConfig fingerprint) → specialization`. The registry is reviewed and linted to avoid combinatorial explosion (pick one sharding family per target; start with 1d, add 2d as a separate family).
- **Runtime fingerprints:** On first use, modules hash `(ModuleConfig, shape bucket, topology, version)` and look for a matching specialization. Hits use the checked-in path. Misses emit a clear suggestion: “no specialization for fp=XYZ; generate or fall back.”
- **Escape hatch generator:** The mini-compiler only runs when adding a new fingerprint (new shape/topology/config). Templates stay tiny and torch-free (see `codegen/TTTv2_BIDIRECTIONAL_CODE_GENERATION.py`, `models/tt_transformers_v2_prototype_2/example_attention_codegen.py`). Outputs are deterministic and then checked in.
- **Isolation / blast radius:** Specializations are per-fingerprint; changes in “master” code don’t force retesting all models. If a specialization misbehaves (fabric/prefetch/trace issues), bisecting is contained to the small generated artifact.
- **Compile-only modes:** Env vars like `TT_METAL_NULL_KERNELS` and `TT_METAL_KERNELS_EARLY_RETURN` let us compile/dispatch without executing kernels, useful for timing and debugging while firmware adds full compile-only support.

##### Why codegen helps

- Determinism vs drift: one source maps `(shape, topology, config) -> kernel choices`, avoiding scattered hand-tweaks that silently enlarge the surface area.
- Auditability: the generator/registry is a decision log—“shape X → config Y because Z”—so we can regenerate, diff, or roll back specific choice points.
- Repeatable specialization: adding a new shape/topology is one deterministic specialization instead of edits across many files, reducing blast radius.
- Compact interfaces: thin templates force a small, typed config API; new knobs are deliberate additions, not ad hoc kwargs threaded through call sites.

##### Zen of TTTv2 code generation

- Codegen is an opt-in specialization emitter, not a mandatory layer; the product code is the checked-in specialization.
- Curate a lean registry of `(shape, topology, ModuleConfig) -> specialization`; constrain sharding families (start 1d, add 2d separately) to avoid combinatorial sprawl.
- Keep templates tiny, declarative, and torch-free; move complexity into data tables and explicit config knobs.
- Runtime fingerprints connect running models to specializations; misses are signals to add a new entry, not to mutate “master” code.
- Prefer readability for common targets (prebuilt outputs) and fall back to the generator only when a new shape/topology/config appears.

[forward function](codegen/TTTv2_BIDIRECTIONAL_CODE_GENERATION.py) at line 348 can be a free function that can be used to generate the source code for the class. It is currently used to generate the source code for the class. This free function is like a template! Future TTNN ops could be hidden behind a free function as well that serves as a template for the source code generation, which gives us a chance to adjust the APIs and have room for working around TTNN limitations.

This approach could be combined with the metaclass approach to generate the source code for the class. Or we could make it simpler by specializing a compile_function for each specialized forward function. Such compile_function could generate the specialized forward function that contains concretized configurations based on the tensor shape and hardware configurations
- look at this file models/tt_transformers_v2_prototype_2/example_attention_codegen.py for example code

I have an idea: there is that env var that will skip execution of all kernels! So, we could have a compile time option to skip execution of all kernels even today!!!
- `TT_METAL_NULL_KERNELS`
- `TT_METAL_KERNELS_EARLY_RETURN`: https://github.com/tenstorrent/tt-metal/pull/19016
> We'd like to be able to determine dispatch time if kernels aren't running. We have a TT_METAL_NULL_KERNELS flag, but it's targeted towards unit tests and optimizes out the kernel body.

So, `TT_METAL_KERNELS_EARLY_RETURN` may be able to allow us to compile all the kernels without having to run them! I am asking John Bauman to see if this idea would work. The firmware team is working on proper compile support.

### Migration

We will not provide migration tools. We will migrate select models to TTTv2.

The rest of the models should migrate to TTTv2 by their owners. We will provide migration guides for once we have migrated select models ourselves.

We could also try to make a AGENTS.md to help people use AI agents to migrate their models to TTTv2.

## Success Metrics

- Time to add new model: 1 model per day per developer (with AI agents)
- Testing: tt-metal should contain no model tests! Only unit tests for the modules and the testing tools.
    - the accuracy and performance of the modules should compose into those of the models
- Model performance: should show on a plot that using TTTv2 can achieve much better performance than using TTNN ops directly;
  - (stretch goal) ideally, we should also show the performance on a roofline plot

## Example: Adding a New Model

- Developer (and AI agents) uses TTNN ops directly with TTTv2 testing tools

- Developer (and AI agents) uses TTTv2 modules to implement the model

## Future Design Options
  Good sources for inspiration:
  - https://docs.google.com/document/d/1U_SSgQF_sX8n58s4aAtUYPmU17zwS6HCBOageyFB3P4/edit?tab=t.0#heading=h.esiriihac5fj
  - https://docs.google.com/document/d/113TpLUoOiUOkXyHRvfHAuVICtk6IoaRqaG9CGo9PgUk/edit?tab=t.0
  - https://docs.google.com/document/d/1KmaawWqk4nZ_cBQoc8LBE6EYbXwjjD8A8FBA7vTen_g/edit?tab=t.0
  - https://docs.google.com/document/d/1ZSLFTBloSjAWCjV1lN1Wjap3lBVfNY87XK22KnCJo6o/edit?tab=t.0#heading=h.av5yl7o68evz

  models/experimental/tt_dit
  models/tt_cnn
  models/demos/gpt_oss
  models/demos/deepseek_v3

  #### Move models out of TTTv2
  Move models out of TTTv2 to separate repos.

  #### Stress testing
  Running a model for hours and hours. TTTv2 should provide API for this?

  #### Composable Model performance from module performance data
  Based on unit tests of modules, we should be able to use module performance data to compose model performance expectations.

  #### Community Governance - Clear process for accepting new patterns/modules

  Refer to [this doc for details](models/tt_transformers_v2_prototype_2/docs/community_governance_design.md)

  Community governance is critical for TTTv2's success in supporting 100+ models. Key considerations:

  1. **API Stability vs Innovation**
     - Strict semantic versioning with experimental namespaces
     - Modules can graduate from experimental → contrib → core
     - Breaking changes only allowed in major versions

  2. **Quality Control Through Tiers**
     ```python
     class ModuleTier(Enum):
         CORE = "core"              # TTT team maintained
         CONTRIB = "contrib"        # Community reviewed
         COMMUNITY = "community"    # Community maintained
         EXPERIMENTAL = "experimental"  # No guarantees
     ```

  3. **Namespacing for Scale**
     ```python
     # Hierarchical names prevent conflicts
     attention="stanford-nlp/attention/efficient/linear-v2"
     attention="deepmind/attention/perceiver-ar"
     ```

  4. **Automated Governance**
     - Performance regression detection
     - Dependency conflict checking
     - Abandoned module detection
     - Compatibility matrix generation

  5. **Clear Decision Process**
     - Technical Steering Committee for core changes
     - Module maintainers for contrib
     - Automated merge for experimental (if tests pass)

  6. **Certification Levels**
     - Functionally tested
     - Performance verified
     - Hardware validated
     - Production certified

  #### interface to netmap or other higher level model interfaces
netmap: https://docs.google.com/document/d/1KmaawWqk4nZ_cBQoc8LBE6EYbXwjjD8A8FBA7vTen_g/edit?tab=t.0

netmap wants to build compute graph with ttnn ops. The design in this doc can apply to ttnn ops as well! The other contribution that netmap has is that it can help people create sharding specs but I think programmatical ways such as the CuTe DSL is preferrable in scaling scenarioes.

Other ideas in netmap like the use of ttnn trace tool to trace HF models is also interesting and also adoptable by TTTv2.

TTTv2 as a library provides the flexibility to choose different higher level model interfaces --> we can build one too!

#### refactor hardware configs of TTNN ops for each module

After we get things working with the current design, we should spend some time to study and refactor the hardware configs of TTNN ops for each module. Maybe there is a smaller number of patterns for config than there are models! This could be a good thing to create an abstraction layer for hardware configs -- patching up the only leak in the abstract interface in this design.

For exampl, instead of specifying config for a TTNN matmul op, we could say RingMatmulConfig, BcastMatMul, etc.

#### Code Coverage in Testing

**Challenge**: How do we ensure that all building blocks are adequately tested across different configurations used by models?

**Requirements**:
- Track which building block configurations are tested by models
- Identify untested configurations or parameter combinations
- Generate coverage reports showing:
  - Which modules have test coverage
  - Which parameter ranges are tested
  - Which combinations are missing

**Open Questions**:
- Should we track coverage at the module level or parameter level?
- How to aggregate coverage across all models using TTTv2?
- Should coverage influence CI/CD decisions?

**Potential Approach**:
```python
# Automatic coverage tracking in TestSuite
with TestSuite(model, track_coverage=True) as suite:
    suite.test(model.attention).expect(latency_ms=5.0)
    # Automatically records that MultiHeadAttention with
    # hidden_dim=4096, num_heads=32 was tested

# Generate coverage report
coverage_report = suite.get_coverage_report()
# Shows: attention.MultiHeadAttention tested with:
#   - hidden_dim: [4096, 8192] (missing: < 2048)
#   - num_heads: [32, 64] (missing: 8, 16)
```
