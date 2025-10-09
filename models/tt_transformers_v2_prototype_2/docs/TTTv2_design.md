# Introduction
We are working on productionize `models/tt_transformers`.

We refer to tt_transformers as TTT in this document. The current version of TTT is called TTTv1 and we are working on TTTv2.

# Motivations
Scaling problem to support 100+ models

Currently one change in TTTv1 has many consequences:
- Adding a new model or modifying an existing model's performance requires retesting every combination of all other models and hardware that we support
- Each contributor faces a Models x Platforms x Tests problem

This problem gets worse with each newly added model

# Goals
Single developer can add a new LLM model without tribal knowledge

TT-Transformers (TTT) code as a library
- Modular
- Composable
- Readable
- Reproducible
- Maintainable and releasable

TTTv1 is good, first achievement of the goals for 10+ models; now TTTv2 must do better to get to 100+ models

# Approach
![[Pasted image 20251007092349.png]]shows high level architecture of TTTv2.

## Key Design Principles

### Tightened Scope
- TTTv2 has a clearly defined boundary where only blocks overlapping with the TTTv2 circle are considered core parts of the project
- Model implementations themselves are **not** core parts of TTTv2, allowing for better separation of concerns

### Model Implementation Strategy
Model implementations (Llama/Qwen, Mistral, Gemma3, etc.) follow these principles:
- Not owned or maintained by TTTv2 team
- Implemented on basis of a specific release of TTTv2
- Can overwrite defaults to customize behavior
- Upgrades to TTTv2 (e.g., 2.0 → 2.1 following semantic versioning) do not require model implementations to upgrade

### Architecture Flow
The architecture follows a clear flow of information (as shown by arrows in the diagram):

1. **Model Conversion Layer**: Model implementations convert model-specific formats to TTTv2 format through copy/modify operations
2. **Standard Interfaces**:
   - Standard demos provide entry points
   - Standard vLLM generator handles model execution
   - TT HW config APIs manage hardware configurations
3. **ML Model Configuration**: Provides ML model config APIs
4. **Core TTT Modules**: depends on TTNN, these are the fundamental building blocks; minimize other dependencies
5. **Testing Infrastructure**: Unit tests for all key instances of TTT modules ensure reliability
6. **Debug Support**: Debug mode allows for detailed inspection and troubleshooting
7. **Performance Monitoring**: APIs to select unit tests to run, with accuracy and performance expectations for each module

This modular approach enables developers to add new LLM models without requiring deep tribal knowledge, while maintaining clear boundaries between the core library and model-specific implementations.

## Usage

Model implementations should:
- Import TTTv2 modules as needed
- Override defaults where necessary
- Implement model-specific logic separately
- Pin to specific TTTv2 version for stability

## Version Compatibility

TTTv2 follows semantic versioning:
- Major version changes (2.0 → 3.0): Breaking API changes
- Minor version changes (2.0 → 2.1): Backwards compatible features
- Patch version changes (2.0.0 → 2.0.1): Bug fixes only

# TTTv2 High-level Design

> “Library, **not** Framework” – make adding the *N + 1-th* model O(1), not O(N).

## 1. Executive Summary
TT-Transformers v1 (TTTv1) successfully ran 10+ LLMs on Tenstorrent hardware but hit an **N × M × P** scaling wall:

* **N** models × **M** hardware targets × **P** test suites → exponential validation cost
* Model-specific logic leaked into core; small edits caused cross-model regressions
* Contributors needed deep tribal knowledge of kernels, tiling, and firmware quirks

**TTTv2** turns the monolith into a **pure Python library of building blocks** that model adapters consume.  Core owns *no* models; models live in their own repos and pin to semantically versioned TTTv2 releases.  This collapses the test matrix and lets a single engineer bring up a new model in < 1 week.

## 2. Goals & Non-Goals
### Goals
1. **Scalability** – Support **100+** models without core changes.
2. **Developer UX** – “pip install tt-transformers-v2 ; import attention, ffn …”
3. **Low-level configuration** – modules know their own TTNN configs and also allows detailed overrides
4. **Stability** – Public API stable within a minor release; breaking changes only in majors.
5. **Testability** – Components tested in isolation; models compose test suites with zero boiler-plate.

### Non-Goals
* Auto-migration of TTTv1 models (manual templates will be provided).
* Owning a “universal model zoo” inside core.
* Building graph DSL – we piggy-back on TTNN ops directly for now

## 3. Key Design Principles
1. **Tight Scope** – Only code inside the library circle (`src/`) is “core”.  Nothing model-specific lives here.
2. **Composable Building Blocks** – `attention/`, `ffn/`, `embeddings/`, `normalization/` are atomic, functional code.
3. **Patterns over Classes** – Pre-built decoder/encoder layers live in `patterns/`, not bespoke monoliths.
4. **Module-Centric Implementation Configuration** – Each module provides `get_default_impl_config(device, mode)` with sane TTNN defaults; callers may override.
5. **Strict Public API** – Controlled only via top-level `__init__.py`/`__all__`; internal packages are hidden; There is one exception: advanced users of TTTv2 might need to deal with TTNN interface directly for now
6. **Three-Tier Testing** – (1) generic unit tests with real-world configuration, (2), and (3) model-driven test suites.
7. **Semantic Versioning**

## 4. Repository Layout
```text
.
├── tt_transformers_v2/               # Core library – ZERO model code
│   ├── src/                          # library source
│   │   ├── building_blocks/          # attention, ffn, norm, embeddings, etc.
│   │   ├── patterns/                 # (Optional) decoder_layer, encoder_layer, etc.
│   │   └── testing/                  # TestSuite, performance/accuracy test utilities
│   └── tests/                        # Unit/composition tests for core library
│
└── models/                           # Reference models (llama3/, mistral/) & adapters
    ├── deepseek/
│   │   └── ModelFactory.py           # Model-specific adapter (optional, non-core)
    ├── llama3/
    ├── qwen/
    └── mistral/
```

What separates `building_blocks` from `patterns`? The main difference is testing. `building_blocks` are smaller, unit-tested modules that are used to build `patterns`. Such unit tests are meant to provide wide coverage of model specifications and implementation configurations. `patterns` will be tested as well but the focus is on testing the successful composition of `building_blocks` and thus only limited sets of specifications and configurations are needed.

It is unclear how much value `patterns` could provide as a model owner could build their model code using `building_blocks` directly. It will become clear as we develop TTTv2 in more details. For now, `patterns` are optional.

## 5. Usage examples

### Code Example for existing models in TTTv1

For such models, TTTv1 contains code to implement all the functions/methods in TTTv2. So, the main work here is to refactor TTTv1 code. There are mainly these destinations for the refactored code:
- TTTv2 modules such as `attention`, `ffn`, `normalization`, `patterns` with interfaces for `ModelSpecification` and `ImplementationConfig`
- Reference models such as `llama3`, `qwen`,and `mistral` with adapters such as `ModelFactory`

We want to keep the code as clean as possible, leveraging refactored TTTv1 code as much as possible. However, this is only to demonstrate an upper bound of the cleanness of the code. We will not require the model implementation to be as clean as the code example below.

```python
# HuggingFace → TTTv2 adapter usage (existing model sources)
from transformers import AutoConfig
from models.llama3.model_factory import ModelFactory # ModelFactory imports TTTv2 library and uses its APIs to build the model

###############################################
# Disclaimer: TTTv2 does not own ModelFactory #
###############################################

# Build a model using ModelFactory
# - load model from HF endpoint
# - construct ModelSpecification from the hf model (using hf_config, layer names and types, etc.)
#   - map hf model layers to TTTv2 patterns and modules --> raise Error when a mapping is not found
#   - register reference model for each instance of mapped TTTv2 patterns and modules
# - construct default ImplementationConfig from mapped modules
# - load TT tensor caches, or load model checkpoint and generate TT tensor caches (option to enable lazy mode)
# NOTE: each model owner may choose to manually write a separate adaptor code for their model or inherit/compose ModelFactory into their own ModelFactory
model = ModelFactory.from_huggingface("meta-llama/Llama-3-8b", device="N150")

# Run a simple forward
with open("input_prompts.json", "r") as f:
    user_input = json.load(f)
out = model(user_input)

# Generate a number of tokens
# What does HF provide here as functions on model? --> it would be great to at least have the same API as HF's models
# Furthermore, maybe we could just use HF model code to drive the TTTv2 modules? --> need more investigation
# - Maybe we can just allow people to inherit/compose an HF model into our model --> need more investigation
out_tokens = model.generate(user_input, max_new_tokens=200)

# (Optional) Unit-test a few modules with fluent TestSuite
from tt_transformers_v2.testing import TestSuite

with TestSuite(model) as suite:
	# how to connect with Pytest
    # programmatically specify which block to test
	import functools
	sample_run = functools.partial(model.generate, user_input)

	# NOTE: reference models were registered for each TTTv2 during model construction
    suite.add_case(model.layers[0].attention.forward_prefill)
    .with_inputs(trace_fn=sample_run) # `trace_fn` will generate real
    .expect_pcc(pcc=0.99, atol=1e-3, rtol=1e-3).expect_perf(latency_ms=5.0)

    suite.add_case(model.layers[0].ffn.forward_decode)
    .with_inputs(trace_fn=sample_run)
    .expect_pcc(pcc=0.99, atol=1e-3, rtol=1e-3).expect_perf(latency_ms=3.0)

    # run the tests
	# - check all the test cases are properly setup, e.g., is there reference model? is there inputs?
	# - run `trace_fn` to generate inputs for all test cases and `suite` will capture the inputs for registerd test cases
	# - run the test cases and check against the expected accuracy/performance
    suite.run()
```

Note that `models` (thus `models.ModelFactory`) is not core part of the TTTv2 library. It is a place for reference models and adapters and we could consider moving it in the future to, for example, a `model_zoo` repo. Thus, we do not control the interface of `models` and do not consider them as part of the TTTv2 APIs. Developers are welcome to build their own model instance in whichever way they prefer. As long as they use the TTTv2 APIs, they can take advantage of the TTTv2 core features.

### Code Example for use of TTTv2 APIs
Model using TTTv2 APIs (model spec, implementation config) and modules (patterns, building blocks). The example is meant to show a scenario when a new model is being worked on. Another way to see this example is that this could be what it looks like under the hood of `ModelFactory.from_huggingface` for an model.

```python
# ModelSpec + ImplConfig (no external adapters)
from tt_transformers_v2.patterns import CausalLM
from tt_transformers_v2.building_blocks import AttentionSpec
from .models import ModelFactory

# 1) Specify model parameters
# Let's say the new model is similar to Llama-3-8b; we could first load the model spec from HuggingFace and then apply patches to the model spec for the new model; Once our tests are passing, we can refactor this piece of code.
model_spec = ModelFactory.from_huggingface("meta-llama/Llama-3-8b", checkpoint_path="path/to/new/model/safetensors", validate=False)
attn_spec = AttentionSpec(
    # just an example -- not aiming for completeness
    hidden_dim=4096,
    num_heads=32,
)
for layer_spec in model_spec.layers:
    layer_spec.attention.model_spec = attn_spec
	layer_spec.attention.load_checkpoint()
model_spec.validate() # checks loaded tensor shapes against specficified shapes

# Make model with default implementation config used under the hood
model = CausalLM(model_spec, device="N150")

# Overriding the default implementation config
# model spec and implementation config are designed to have the same architecture and the same names for the composing module instances. Thus, we can use the same attribute chain to narrow both `model_spec` and `model.impl_config` down to the same module instance.
model.impl_config.layers[0].attention.qkv_proj.dtype = ttnn.bfloat16

# By default, model will implement Prefill/Decode specialization
import torch
import ttnn
prompt_ids = ttnn.from_torch(torch.tensor([1, 2, 3, 4])) # TTTv2 could provide an API to help construct input tensor to `model`
prefill_out = model.prefill_forward(prompt_ids)
next_token_out = model.decode_forward(prefill_out, cache_position=4)
```

There are APIs to override the default execution strategy and implementation config. To construct a model implementation, the following steps are taken:
```
Module Spec → Execution Strategy → Implementation Config → Module Instantiation → Tensor Caches
    ↓                 ↓                    ↓                       ↓                   ↓
(math only)    (prefill/decode)      (TTNN op configs)       (model impl)      (weight/activation)
```
For example:
```python
impl_config = CausalLM.ImplConfig(model_spec, device="N150")
impl_config.strategy = "performance"
impl_config.layers[0].attention.activation = "gelu"
model = CausalLM(model_spec, impl_config=impl_config) # will generate/load the tensor caches
```

TTTv2 also provides utility functions that works with the library components to build test suites. For example:
```python
from tt_transformers_v2.testing import TestSuite
from transformers import AutoModelForCausalLM
with TestSuite(model, seed=1234) as suite: # Context manager approach
    # 1) Explicit inputs example
    suite.add_test(model.layers[0].attention_prefill)
    .with_inputs({
        # parameter names of function under test are used as keys with values being direct inputs to the function
        "hidden_states": ttnn.from_torch(torch.randn(2, 512, 4096, device="cpu")),
        "attention_mask": None,
    })
	# the next line of code using AutoModelForCasualLM maybe incorrect but the idea should be the same
    .expect_correctness(ref=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b").model.layers[0].self_attn, atol=1e-3, rtol=1e-3)
    .expect_perf(latency_ms=5.0, percentile=0.9)

    # 2) Capture from a real forward (no manual tensors) example
    def sample_run():
        toks = tokenizer("hello world")
        prefill_out = model.prefill_forward(toks)
        decode_out = []
        for i in range(16):
            decode_out.append(model.decode_forward(prefill_out, cache_position=len(toks) + i))
        return decode_out

    golden_func = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b").model.layers[3].ffn

    suite.add_test(model.layers[3].ffn_decode)
    .with_inputs(trace_fn=sample_run)
    .expect_pcc(0.99, ref=golden_func).expect_perf(latency_ms=3.0, percentile=0.9)

    suite.run()  # run trace_fn to capture inputs, runs all cases, fails fast on regression
```


## Notes
### Building Blocks (`src/building_blocks/*`)
* **Attention** – MHA, GQA, Flash, SlidingWindow, RoPE utils
* **FFN** – MLP, SwiGLU, GeGLU, MoE
* **Normalization** – LayerNorm, RMSNorm, fused TTNN RMSNorm
* **Embeddings** – Token & positional

We would like to create **stateless + functional** modules; but it may take several revisions to get there.

### Patterns (`src/patterns/*`)
Patterns are intermediate-level architecture pieces that are commonly used in ML model designs. They compose building blocks together into a compute graph. For example, `decoder` and `encoder` are two patterns.

One of the design point is to create fractal API design for both patterns and building blocks. From the APIs' view, patterns and building blocks should look almost identical. For example, a pattern, just like a building block, will accept both `ModelSpecification` and `ImplementationConfig` in their construction function.

### Quality Strategy
| Layer                | Owner        | Test Location            | CI Frequency                                 |
| -------------------- | ------------ | ------------------------ | -------------------------------------------- |
| Unit (ops)           | Core         | `tests/building_blocks/` | Merge gate                                   |
| Patterns (real dims) | Core         | `tests/patterns/`        | select ones in Merge gate and all in nightly |
| Model                | Model owners | external repos           | models CI                                    |

Performance & memory budgets are asserted in tests – regressions block merge.

### Versioning & Compatibility
```
2.0.x → 2.0.y    patch: binary-compatible
2.0.x → 2.1.x    minor: source-compatible
2.x.x → 3.0.0    major: breaking
```
Models pin `tt_transformers_v2` to a specific version in their own `requirements.txt`.
