# CLAUDE.md — TTTv2 Implementation Guide

**Your Mission**: Implement TTTv2 as specified in `models/tt_transformers_v2_prototype_2/docs/TTTv2_design.md`. This is a library refactor from TTTv1 to enable scaling from 10→100+ models.

---

## Phase 0: Understanding (DO THIS FIRST)

Before writing ANY code, systematically understand the design:

### 0.1 Read the Design Doc End-to-End
```bash
# Open and read thoroughly:
models/tt_transformers_v2_prototype_2/docs/TTTv2_design.md
```
Please also read `models/tt_transformers_v2_prototype_2/docs/*` for reference. Please ignore `models/tt_transformers_v2_prototype_2/docs/model_spec/assets/*`.

**Key takeaways to internalize:**
- **Library, not Framework** — TTTv2 provides building blocks; models compose them
- **Core owns ZERO models** — models live externally and pin to TTTv2 versions
- **N×M×P → O(1)** — adding model #101 should NOT require retesting models 1-100
- **Three-tier testing**: (1) generic unit tests with real-world configs, (2) composition tests (if patterns exist), (3) model-driven suites
- **Patterns are optional** — Models can use building_blocks directly; patterns provide convenience but aren't mandatory

### 0.2 Map the Design to Existing Code

**Use `codebase_search` aggressively to understand:**

1. **Where is TTTv1?**
   - Query: "Where is the current tt_transformers implementation structure?"
   - Expected: `models/tt_transformers/tt/` contains Transformer, TransformerBlock, Attention, etc.; `models/tt_transformers/tests/` contains tests for tt_transformers; `models/tt_transformers/demo` contains demo models.

2. **What building blocks exist in TTTv1?**
   - Query: "How does attention work in tt_transformers?"
   - Query: "How are FFN/MLP layers implemented in tt_transformers?"
   - Query: "Where is RMSNorm and LayerNorm implemented?"
   - Query: "How are embeddings handled in tt_transformers?"

3. **How does TTTv1 handle hardware config?**
   - Query: "How are TTNN operations configured in tt_transformers?"
   - Query: "Where are device-specific configurations set?"

4. **What are the prototype attempts?**
   - Scan: `models/tt_transformers_v2_prototype_2/`; do not look at `models/tt_transformers_v2_prototype_1/`
   - Note what's already been tried; learn from it but don't be constrained by it

**Checkpoint**: Before proceeding, write a 5-bullet summary of what you learned in your response. If you can't, search more.

---

## Phase 1: Architecture Setup ✅ COMPLETED

### 1.1 Create Directory Structure ✅

**Target layout (UPDATED with actual implementation):**
```
tt_transformers_v2/
  src/
    building_blocks/
      attention/           # Fractal API: each complex module gets subdirectory
        __init__.py
        mha.py            # Multi-head attention
        gqa.py            # Grouped-query attention
        flash.py          # Flash attention
        sliding.py        # Sliding window attention
      ffn/                # Feed-forward networks
        __init__.py
        mlp.py            # Standard MLP
        gated_mlp.py      # Gated MLP (SwiGLU, GeGLU)
        moe.py            # Mixture of Experts
      normalization/      # Normalization layers
        __init__.py
        rmsnorm.py        # RMS normalization
        layernorm.py      # Layer normalization
      embeddings/         # Embedding layers
        __init__.py
        token.py          # Token embeddings
        position.py       # Position embeddings
        rotary.py         # Rotary embeddings (RoPE)
      ccl/                # Collective Communication Layer
        __init__.py
        manager.py        # Shared semaphore management
        all_reduce.py     # AllReduceSpec, AllReduceImplConfig, all_reduce_forward
        all_gather.py     # AllGatherSpec, AllGatherImplConfig, all_gather_forward
        distributed_norm.py # DistributedRMSNormSpec (not DistributedNormSpec)
      lm_head/            # Language model head (follows fractal pattern)
        __init__.py
        lm_head.py        # LMHeadSpec, LMHeadImplConfig, etc.
      __init__.py         # Re-exports for backward compatibility
    patterns/              # OPTIONAL — focus on building_blocks first
      decoder_layer.py
      encoder_layer.py
      causal_lm.py
      __init__.py
    testing/
      suite.py
      utils.py
      __init__.py
    __init__.py           # PUBLIC API — strict exports only
                          # Exception: advanced users may need TTNN interface directly
  tests/                  # NO __init__.py files in test directories
    building_blocks/      # Test structure mirrors building blocks structure
      attention/
        test_mha.py
        test_gqa.py
        test_flash.py
        test_sliding.py
      ffn/
        test_mlp.py
        test_gated_mlp.py
        test_moe.py
      normalization/
        test_rmsnorm.py
        test_layernorm.py
      embeddings/
        test_token.py
        test_position.py
        test_rotary.py
      ccl/
        test_manager.py
        test_all_reduce.py
        test_all_gather.py
        test_distributed_norm.py  # Tests DistributedRMSNorm from ccl module
      lm_head/
        test_lm_head.py
    patterns/              # only if patterns/ is implemented
      test_decoder_layer.py
      test_causal_lm.py
    testing/
      test_suite.py
  pyproject.toml          ✅
  README.md
models/                   # OUTSIDE core library
  llama3/                 # Minimal reference (optional)
    model_factory.py      # Reference adapter (optional, non-core)
```

**Adjustments made during implementation:**
1. **Subdirectory structure**: Building blocks are organized into subdirectories for better organization (attention/, ffn/, normalization/, embeddings/, ccl/)
2. **Additional building blocks**: Added three missing components from TTTv1:
   - CCL (Collective Communication Layer) - for distributed operations
   - LM Head - for language model output projection
   - Distributed Normalization - wrapper for multi-device normalization
3. **Multiple implementations**: Each building block category has multiple implementations (e.g., MHA/GQA/Flash for attention)
4. **Fractal API design**: Each building block file contains its own Spec and ImplConfig classes, following the fractal pattern
   - Example: all_reduce.py contains AllReduceSpec and AllReduceImplConfig
   - This ensures consistent API across all building blocks

**Execute:** ✅
- Created all directories
- Added `__init__.py` to every package
- Wrote `pyproject.toml` with minimal deps: `ttnn`, `torch`, `pytest`

**Verification:** ✅
```bash
cd tt_transformers_v2
python -c "import src; print(src.__file__)"  # Works
pytest --collect-only  # Finds test structure
```

### 1.2 Define API Conventions in Code

**Every building block MUST have this structure:**

**Important**: Building blocks are functional/stateless. Patterns (if implemented) follow the same "fractal API design" — they accept `ModelSpec` and `ImplConfig` just like building blocks do. Also, need to look at `models/tt_transformers/tt/` for reference because one of the goals is to implement TTTv1 with the new APIs in TTTv2.

**NOTE: Building blocks now follow fractal subdirectory structure. Complex modules like attention, ffn, etc. get their own subdirectories with multiple files.**

```python
# building_blocks/attention/mha.py (TEMPLATE - Multi-Head Attention)

from dataclasses import dataclass
from typing import Optional, Literal
import ttnn
import torch

@dataclass
class AttentionSpec:
    """
    Mathematical specification only — no implementation details.
    This is the "ModelSpecification" referenced in the design doc.
    """
    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None  # GQA support
    head_dim: Optional[int] = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

    def validate(self):
        """Validate spec constraints (e.g., loaded tensor shapes match)."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

@dataclass
class AttentionImplConfig:
    """
    TTNN-specific implementation choices (ImplementationConfig in design doc).
    Generated from Spec + device + execution strategy (prefill/decode).
    """
    qkv_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat16
    compute_kernel_config: Optional[dict] = None
    use_flash_attention: bool = True
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    # ... other TTNN-specific settings

    # From design: "Module Spec → Execution Strategy → Implementation Config → Module Instantiation → Tensor Caches"
    # Execution Strategy determines prefill/decode split
    # ImplConfig contains TTNN op configs per strategy

def get_default_impl_config(
    spec: AttentionSpec,
    device: str,
    mode: Literal["prefill", "decode"],  # This is the "Execution Strategy"
    strategy: str = "default"  # Could be "performance", "memory", etc.
) -> AttentionImplConfig:
    """
    Return sane defaults for this device & mode.

    Args:
        spec: Mathematical specification
        device: Target device (e.g., "N150", "T3000")
        mode: Execution strategy mode (prefill or decode)
        strategy: Optional performance strategy hint
    """
    # Example:
    if device.startswith("N150"):
        return AttentionImplConfig(
            use_flash_attention=True if mode == "prefill" else False,
            # ... device-specific tuning
        )
    elif device.startswith("T3000"):
        # ... different defaults
        pass
    return AttentionImplConfig()

def prefill_forward(
    hidden_states: ttnn.Tensor,
    spec: AttentionSpec,
    impl_config: AttentionImplConfig,
    position_ids: Optional[ttnn.Tensor] = None,
    attention_mask: Optional[ttnn.Tensor] = None,
    **kwargs
) -> tuple[ttnn.Tensor, dict]:
    """
    Prefill mode: process entire sequence at once.

    Returns:
        output: (batch, seq_len, hidden_dim)
        cache: dict with 'k' and 'v' tensors for decode
    """
    raise NotImplementedError("Refactor from TTTv1")

def decode_forward(
    hidden_states: ttnn.Tensor,
    spec: AttentionSpec,
    impl_config: AttentionImplConfig,
    cache: dict,
    position_ids: Optional[ttnn.Tensor] = None,
    **kwargs
) -> ttnn.Tensor:
    """
    Decode mode: single token, use cached K/V.

    Returns:
        output: (batch, 1, hidden_dim)
    """
    raise NotImplementedError("Refactor from TTTv1")
```

**Replicate this pattern for all building blocks in their respective subdirectories:**
- `attention/`: mha.py, gqa.py, flash.py, sliding.py
- `ffn/`: mlp.py, gated_mlp.py, moe.py
- `normalization/`: rmsnorm.py, layernorm.py
- `embeddings/`: token.py, position.py, rotary.py
- `ccl/`: manager.py, all_reduce.py, all_gather.py, distributed_norm.py
- `lm_head/`: lm_head.py

---

## Phase 2: Refactoring TTTv1 → TTTv2

### 2.1 Building Blocks Implementation Order

**Start with simplest first:**

1. **Normalization** (`normalization/rmsnorm.py`, `normalization/layernorm.py`)
   - RMSNorm is in `models/common/rmsnorm.py` — refactor it
   - Add `get_default_impl_config` for fused vs. unfused TTNN variants
   - Write unit tests with small tensors (e.g., [1, 128, 4096])

2. **Embeddings** (`embeddings/token.py`, `embeddings/position.py`, `embeddings/rotary.py`)
   - Extract from `models/tt_transformers/tt/embedding.py`
   - Handle token embeddings and optional scaling
   - Test with vocab_size=32000, hidden_dim=4096

3. **FFN** (`ffn/mlp.py`, `ffn/gated_mlp.py`, `ffn/moe.py`)
   - SwiGLU is common in Llama/Mistral/Qwen
   - Extract MLP logic from `models/tt_transformers/tt/` decoder blocks
   - Separate prefill/decode if matmul configs differ

4. **Attention** (`attention/mha.py`, `attention/gqa.py`, etc.) — MOST COMPLEX
   - Multi-head (MHA), Grouped-query (GQA), Flash variants
   - RoPE application
   - Sliding window support
   - **Search TTTv1 thoroughly:**
     - Query: "How is attention implemented in tt_transformers decoder?"
     - Query: "Where is RoPE applied in tt_transformers?"
     - Query: "How are KV caches managed?"
   - Split into `prefill_forward` and `decode_forward`

### 2.2 Refactoring Checklist (per module)

For each building block you refactor:

- [ ] Read TTTv1 implementation via `codebase_search` + `grep` + `read_file`
- [ ] Identify all TTNN ops used (e.g., `ttnn.matmul`, `ttnn.softmax`)
- [ ] Extract mathematical logic into pure function
- [ ] Move TTNN configs into `ImplConfig`
- [ ] Write `get_default_impl_config` with device-specific branches
- [ ] Add shape validation in `Spec.__post_init__`
- [ ] Write unit test with:
  - Small tensor (e.g., [2, 128, 4096])
  - Golden reference (PyTorch or HF)
  - Assert `torch.allclose(tt_output.to_torch(), golden, atol=1e-2, rtol=1e-3)`
- [ ] Run test: `pytest tests/building_blocks/<module>/ -v` (e.g., `pytest tests/building_blocks/attention/ -v`)

### 2.3 Testing Strategy for Building Blocks

**Unit tests should:**
- Cover multiple shapes (small, medium, large)
- Test both prefill and decode modes
- Compare against PyTorch golden reference
- Include perf markers (not hard gates initially):
  ```python
  @pytest.mark.perf(latency_ms=5.0, percentile=0.9)
  def test_attention_prefill_perf():
      # ... measure time
  ```

**Where to find golden references:**
- HuggingFace Transformers: `transformers.models.llama.modeling_llama.LlamaAttention`
- PyTorch implementations in `models/demos/*/reference/`

---

## Phase 3: Patterns (Compose Building Blocks) — OPTIONAL

**Note from design doc:** "It is unclear how much value `patterns` could provide as a model owner could build their model code using `building_blocks` directly. It will become clear as we develop TTTv2 in more details. For now, `patterns` are optional."

**Prioritize building_blocks first.** Only implement patterns if:
1. Building blocks are solid and tested
2. You see clear reuse opportunities (e.g., multiple models use identical decoder layer structure)
3. Time permits

### 3.1 Decoder Layer (if implementing patterns)

**File:** `patterns/decoder_layer.py`

**Fractal API design:** Patterns follow the same API structure as building blocks — they accept `Spec` and `ImplConfig`.

```python
from dataclasses import dataclass
from src.building_blocks import (
    AttentionSpec, AttentionImplConfig, prefill_forward as attn_prefill,
    FFNSpec, FFNImplConfig, ffn_forward,
    RMSNormSpec, RMSNormImplConfig, rmsnorm_forward
)

@dataclass
class DecoderLayerSpec:
    attention: AttentionSpec
    ffn: FFNSpec
    input_norm: RMSNormSpec
    post_attn_norm: RMSNormSpec

@dataclass
class DecoderLayerImplConfig:
    attention_prefill: AttentionImplConfig
    attention_decode: AttentionImplConfig
    ffn_prefill: FFNImplConfig
    ffn_decode: FFNImplConfig
    input_norm: RMSNormImplConfig
    post_attn_norm: RMSNormImplConfig

def get_default_impl_config(spec: DecoderLayerSpec, device: str) -> DecoderLayerImplConfig:
    # Get defaults for each sub-block
    return DecoderLayerImplConfig(
        attention_prefill=get_default_impl_config(spec.attention, device, "prefill"),
        attention_decode=get_default_impl_config(spec.attention, device, "decode"),
        # ...
    )

def prefill_forward(hidden_states, spec, impl_config, **kwargs):
    # x = norm(hidden_states)
    # attn_out, cache = attention(x)
    # hidden_states = hidden_states + attn_out
    # x = norm(hidden_states)
    # ffn_out = ffn(x)
    # return hidden_states + ffn_out, cache
    raise NotImplementedError

def decode_forward(hidden_states, spec, impl_config, cache, **kwargs):
    # Similar but with decode variants
    raise NotImplementedError
```

### 3.2 CausalLM (Full Model) — if implementing patterns

**File:** `patterns/causal_lm.py`

Compose:
- Token embedding
- N × decoder layers
- Final norm
- LM head

Expose `prefill_forward(tokens) -> logits, caches` and `decode_forward(token, caches) -> logits`.

**Test:** Build a tiny 2-layer model with dim=512, run 10-token prefill + 5 decode steps.

**Alternative:** Model owners can compose building blocks directly without patterns. Patterns just provide convenience.

---

## Phase 4: Testing Utilities

### 4.1 Fluent TestSuite

**File:** `testing/suite.py`

Design from `TTTv2_design.md` examples:

```python
class TestSuite:
    def __init__(self, model, seed=None):
        self.model = model
        self.cases = []
        self.seed = seed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_test(self, fn):
        """Add a test case for a specific function/module."""
        return TestCase(fn, self)

    def run(self):
        """Execute all test cases."""
        for case in self.cases:
            case.execute()

class TestCase:
    def __init__(self, fn, suite):
        self.fn = fn
        self.suite = suite
        self.inputs = None
        self.ref = None
        self.atol = None
        self.rtol = None
        self.pcc_threshold = None

    def with_inputs(self, trace_fn=None, **explicit):
        """Capture inputs from trace or use explicit tensors."""
        if trace_fn:
            # Intercept fn calls during trace_fn() to capture args
            pass
        else:
            self.inputs = explicit
        return self

    def expect_correctness(self, ref, atol=1e-3, rtol=1e-3):
        self.ref = ref
        self.atol = atol
        self.rtol = rtol
        return self

    def expect_pcc(self, threshold, ref=None):
        self.pcc_threshold = threshold
        if ref:
            self.ref = ref
        return self

    def expect_perf(self, latency_ms, percentile=0.9):
        # Store perf expectation
        return self

    def execute(self):
        # Run self.fn with self.inputs
        # Compare against self.ref
        # Assert thresholds
        raise NotImplementedError
```

**Test:** Write `tests/testing/test_suite.py` that uses TestSuite on a dummy function.

---

## Phase 5: Reference Adapter (Optional, Non-Core)

**Important:** `models/` and `ModelFactory` are NOT part of TTTv2 core. They're reference examples. The design doc explicitly states: "we do not control the interface of `models` and do not consider them as part of the TTTv2 APIs."

**Architecture:** Each model subdirectory (e.g., `models/llama3/`, `models/qwen/`, `models/mistral/`) has its own `model_factory.py`. Model owners are free to structure their adapters however they want.

**File:** `models/llama3/model_factory.py` (example for Llama-3)

Key responsibilities per design doc:
1. Load model from HF endpoint
2. Construct `ModelSpec` from HF config (map HF layers to TTTv2 modules, raise error if no mapping)
3. Register reference model for each mapped TTTv2 module instance
4. Construct default `ImplConfig`
5. Load/generate TT tensor caches (with optional lazy mode)
6. Support checkpoint loading with `validate()` to check tensor shapes

```python
from src.building_blocks import AttentionSpec, FFNSpec, RMSNormSpec, EmbeddingSpec
from transformers import AutoConfig, AutoModelForCausalLM

class ModelFactory:
    @staticmethod
    def from_huggingface(
        model_id: str,
        device: str,
        checkpoint_path: Optional[str] = None,
        validate: bool = True,
        **overrides
    ):
        hf_config = AutoConfig.from_pretrained(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id)  # Reference model

        # Map HF config → TTTv2 specs
        attn_spec = AttentionSpec(
            hidden_dim=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=getattr(hf_config, 'num_key_value_heads', None),
            # ...
        )
        # ... build full model spec
        # ... register hf_model.model.layers[i].self_attn as reference for layer i attention

        # If checkpoint_path provided, load and optionally validate
        if checkpoint_path:
            # load tensors
            if validate:
                model_spec.validate()  # Check tensor shapes match spec

        # Build model with default ImplConfig or use overrides
        # model = build_model(model_spec, device)  # Could use patterns or direct composition
        return model
```

**Test:**
```python
from models.llama3.model_factory import ModelFactory
model = ModelFactory.from_huggingface("meta-llama/Llama-3-8b", device="cpu")
# Should instantiate without error
```

**Note:** Each model directory is independent. `models/qwen/model_factory.py`, `models/mistral/model_factory.py`, etc. can have completely different interfaces. The only requirement is that they use TTTv2 building blocks internally.

---

## Phase 6: Integration & Cleanup

### 6.1 Public API Enforcement

**In `src/__init__.py` (or wherever the package root is):**

```python
# Expose ONLY what's documented
from src.building_blocks import (
    AttentionSpec,
    AttentionImplConfig,
    FFNSpec,
    FFNImplConfig,
    RMSNormSpec,
    RMSNormImplConfig,
    EmbeddingSpec,
    EmbeddingImplConfig,
    # ... forward functions
)

# Patterns are optional
# from src.patterns import (
#     DecoderLayerSpec,
#     CausalLM,
# )

from src.testing import TestSuite

__all__ = [
    "AttentionSpec",
    "AttentionImplConfig",
    "FFNSpec",
    "FFNImplConfig",
    # ...
    "TestSuite",
]

# Note from design doc: "There is one exception: advanced users of TTTv2 might need
# to deal with TTNN interface directly for now"
# So it's OK if TTNN types leak through ImplConfig
```

**Verify:** `import src; dir(src)` should show ONLY public symbols (though TTNN may be accessible for advanced users).

### 6.2 Run Full Test Suite

```bash
pytest tests/ -v --tb=short
```

**Fix all failures before declaring a milestone done.**

### 6.3 Write README

Create `tt_transformers_v2/README.md` with:
- Quick install
- 5-line usage example showing building_blocks usage
- Emphasize: "Library, not Framework"
- Clarify: patterns are optional, models live outside core
- Link to design doc
- API reference (or link to docs)

---

## Workflow Guidelines

### How to Use Claude Code Tools

**For exploration:**
- `codebase_search`: Ask questions like "How does X work?" or "Where is Y implemented?"
- `grep`: Search for exact symbols/strings (e.g., `grep -r "class Attention"`)
- `read_file`: Once you know the file, read it in full

**For implementation:**
- `search_replace`: Edit existing files (preferred over `write`)
- `write`: Only for new files
- `run_terminal_cmd`: Run pytest, linters, etc.

**For validation:**
- `read_lints`: Check for Python errors after edits
- Always run tests after each module: `pytest tests/building_blocks/test_X.py -v`

### Code Quality Standards

- **Type hints**: All public functions must have type annotations
- **Docstrings**: Every public function/class needs docstring with Args/Returns
- **No magic numbers**: Use named constants or config fields
- **Early returns**: Avoid deep nesting; fail fast with clear errors
- **Stateless**: Prefer pure functions; where state is needed (caches), pass explicitly

### Anti-Patterns to Avoid

❌ **Don't:** Put model-specific code in core library
✅ **Do:** Keep core generic; models override via ImplConfig

❌ **Don't:** Hard-code device configs
✅ **Do:** Use `get_default_impl_config(device, mode)`

❌ **Don't:** Mix prefill/decode in one function with if/else
✅ **Do:** Separate functions with different TTNN configs

❌ **Don't:** Broad `try/except` to hide errors
✅ **Do:** Let errors propagate; validate inputs explicitly

---

## Milestones & Checkpoints

### Milestone 0: Understanding ✓
- [x] Read `TTTv2_design.md` thoroughly
- [x] Searched TTTv1 codebase for attention, ffn, norm, embeddings
- [x] Written 5-bullet summary of TTTv1 architecture
- [x] Identified key refactor targets

### Milestone 1: Skeleton ✓
- [x] Created directory structure
- [x] Added all `__init__.py` files
- [x] Wrote `pyproject.toml`
- [x] Imports work: `python -c "import tt_transformers_v2"`

### Milestone 2: Building Block Stubs ✓
- [x] `normalization.py` with Spec/ImplConfig/forward stubs + `validate()` method
- [x] `embeddings.py` with Spec/ImplConfig/forward stubs + `validate()` method
- [x] `ffn.py` with Spec/ImplConfig/prefill/decode stubs + `validate()` method
- [x] `attention.py` with Spec/ImplConfig/prefill/decode stubs + `validate()` method
- [x] Unit test files created (even if just imports)
- [x] All Specs follow naming: `AttentionSpec`, `FFNSpec`, etc. (ModelSpecification in design)
- [x] All ImplConfigs follow naming: `AttentionImplConfig`, `FFNImplConfig`, etc. (ImplementationConfig in design)

### Milestone 2.5: Additional Building Blocks (Added during Phase 1) ✓
- [x] `ccl/` directory with fractal pattern (manager.py, all_reduce.py, all_gather.py, distributed_norm.py)
- [x] `lm_head/` directory with lm_head.py following fractal pattern
- [x] Distributed norm functionality in `ccl/distributed_norm.py` (DistributedRMSNormSpec)
- [x] All follow fractal API design pattern
- [x] Test files created for new building blocks in matching subdirectories
- [x] All exports added to `__init__.py` files
- [x] Removed all `__init__.py` files from test directories

### Milestone 3: Normalization Implementation ✓
- [ ] Refactored RMSNorm from TTTv1
- [ ] Wrote unit tests against PyTorch reference
- [ ] Tests pass: `pytest tests/building_blocks/normalization/ -v`

### Milestone 4: Embeddings Implementation ✓
- [ ] Refactored embeddings from TTTv1
- [ ] Tests pass

### Milestone 5: FFN Implementation ✓
- [ ] Refactored FFN/SwiGLU from TTTv1
- [ ] Separate prefill/decode if needed
- [ ] Tests pass

### Milestone 6: Attention Implementation ✓
- [ ] Refactored attention with RoPE
- [ ] Handle MHA, GQA variants
- [ ] Prefill/decode separation
- [ ] Cache management
- [ ] Tests pass (this is the hardest one)

### Milestone 7: Decoder Layer Pattern ✓ (OPTIONAL)
- [ ] **SKIP if prioritizing building_blocks** — patterns are optional per design doc
- [ ] Composed attention + ffn + norms
- [ ] Follows fractal API design (same Spec/ImplConfig structure as building blocks)
- [ ] Tests pass

### Milestone 8: CausalLM Pattern ✓ (OPTIONAL)
- [ ] **SKIP if prioritizing building_blocks** — model owners can compose building_blocks directly
- [ ] Full model composition
- [ ] Prefill + decode forwards work
- [ ] Smoke test with 2-layer tiny model

### Milestone 9: TestSuite Utilities ✓
- [ ] Fluent API works
- [ ] Can capture inputs from trace
- [ ] Can compare against reference
- [ ] Tests pass

### Milestone 10: Reference Adapter (Optional, Non-Core) ✓
- [ ] **Remember: models/ is NOT part of TTTv2 core**
- [ ] Each model has its own `model_factory.py` (e.g., `models/llama3/model_factory.py`, `models/qwen/model_factory.py`)
- [ ] `ModelFactory.from_huggingface` works with all 6 responsibilities (load HF, construct spec, register refs, construct ImplConfig, load caches, support checkpoint validation)
- [ ] Smoke test instantiates Llama-3 spec
- [ ] Reference models registered for test suite integration
- [ ] Model directories are independent; different interfaces are OK as long as they use TTTv2 building blocks

### Milestone 11: Polish ✓
- [ ] Public API locked down in `__init__.py`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] README written
- [ ] Linter clean

---

## Decision Log (Update as You Go)

Use this section to record key decisions:

**Example:**
- **Decision**: Use separate `prefill_forward`/`decode_forward` instead of mode parameter
  - **Rationale**: Different TTNN configs; clearer API; avoids if/else branches
  - **Date**: 2025-10-07

**Implemented Decisions:**
- **Decision**: Adopted fractal API design with subdirectory structure for building blocks
  - **Rationale**: Better organization, each complex module gets its own namespace, clearer file separation
  - **Date**: 2025-10-08
  - **Changes**:
    - attention/ contains mha.py, gqa.py, flash.py, sliding.py
    - ffn/ contains mlp.py, gated_mlp.py, moe.py
    - normalization/ contains rmsnorm.py, layernorm.py
    - embeddings/ contains token.py, position.py, rotary.py
    - ccl/ contains manager.py, all_reduce.py, all_gather.py, distributed_norm.py
    - lm_head/ contains lm_head.py

- **Decision**: Removed `__init__.py` files from all test directories
  - **Rationale**: Pytest best practice - test directories work better without __init__.py files
  - **Date**: 2025-10-08

- **Decision**: Test structure mirrors building blocks subdirectory structure
  - **Rationale**: Consistency, easy to find corresponding tests for each module
  - **Date**: 2025-10-08

- **Decision**: No separate distributed_norm.py at root - lives in ccl/distributed_norm.py
  - **Rationale**: Distributed norm is a CCL operation, makes sense to group with other collective ops
  - **Date**: 2025-10-08

---

## Current Progress (2025-10-08)

### ✅ Phase 0: Understanding
- Read TTTv2 design document
- Analyzed TTTv1 implementation
- Identified key building blocks and patterns

### ✅ Phase 1: Architecture Setup
- Created complete directory structure with fractal API design (subdirectories for complex modules)
- Implemented all core building blocks with proper structure:
  - **attention/**: mha.py, gqa.py, flash.py, sliding.py
  - **ffn/**: mlp.py, gated_mlp.py, moe.py
  - **normalization/**: rmsnorm.py, layernorm.py
  - **embeddings/**: token.py, position.py, rotary.py
  - **ccl/**: manager.py, all_reduce.py, all_gather.py, distributed_norm.py (DistributedRMSNorm)
  - **lm_head/**: lm_head.py (output projection with multi-device support)
- Created unit tests mirroring the building blocks structure:
  - Each building block subdirectory has matching test subdirectory
  - No `__init__.py` files in test directories (pytest best practice)
- Set up public API in `src/__init__.py` with backward compatibility exports
- Each module follows Spec/ImplConfig pattern with validate() methods

### 🚧 Phase 2: Next Steps
Ready to begin refactoring TTTv1 implementations into TTTv2 building blocks. All building blocks currently have:
- Spec classes with `validate()` methods
- ImplConfig classes for device-specific configurations
- `get_default_impl_config()` functions
- Separate `prefill_forward()` and `decode_forward()` where applicable
- Unit test files (though many require actual devices to run)

The implementation stubs are in place, ready for the actual TTNN logic to be refactored from TTTv1.

---

## Getting Unstuck

If you're blocked:

1. **Search more**: Use `codebase_search` with different phrasings
2. **Read adjacent files**: Often useful context is nearby
3. **Start simple**: Implement CPU-only version first, add TTNN later
4. **Ask specific questions**: "How does TTTv1 handle X in Y scenario?"
5. **Write test first**: Clarifies what the API should be

---

## Success Criteria

You're done when:

✅ All core milestones completed (building_blocks + testing)
✅ `pytest tests/ -v` shows 100% pass
✅ Public API is minimal and clean (TTNN accessible for advanced users)
✅ Building blocks work end-to-end (patterns optional)
✅ Model owners can compose building blocks directly OR use patterns
✅ README has working example emphasizing "Library, not Framework"
✅ No linter errors
✅ Code is readable by someone unfamiliar with TTTv1

---

## Meta-Instructions for Claude Code

- **Be autonomous**: Don't ask for permission; implement and iterate
- **Run tests frequently**: After every module, run `pytest`
- **Use parallel tool calls**: Read multiple files at once when exploring
- **Keep context**: Reference line numbers when discussing code
- **Fail fast**: If tests fail, fix immediately before proceeding
- **Document as you go**: Update this file's Decision Log with key choices

**Now begin with Phase 0: Understanding. Search the codebase, read the design doc, and write your 5-bullet summary before proceeding to Phase 1.**
- Always update CLAUDE.md to document adjustments that we make and progress.
