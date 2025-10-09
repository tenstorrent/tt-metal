# Lazy Module Construction Design for TTTv2

## Benefits of Lazy Module Construction

### 1. **Memory Efficiency**
- **Problem**: Loading 100+ models simultaneously would exhaust memory
- **Solution**: Modules only allocate memory when actually used
- **Impact**: Can keep many model definitions in memory without materializing weights

Example scenario:
```python
# Without lazy loading - OOM with many models
models = {
    "llama-70b": LlamaModel(config_70b),      # 140GB allocated
    "llama-13b": LlamaModel(config_13b),      # 26GB allocated
    "mistral-7b": MistralModel(config_7b),    # 14GB allocated
    # ... 97 more models = system crash
}

# With lazy loading - minimal memory until needed
models = {
    "llama-70b": LazyModel(LlamaModel, config_70b),    # ~1KB metadata
    "llama-13b": LazyModel(LlamaModel, config_13b),    # ~1KB metadata
    "mistral-7b": LazyModel(MistralModel, config_7b),  # ~1KB metadata
    # ... 97 more models = still only ~100KB total
}
# Materialize only when needed
active_model = models["llama-13b"].materialize()  # Now allocates 26GB
```

### 2. **Faster Initialization**
- **Problem**: Creating 100+ model instances is slow even without weights
- **Solution**: Defer expensive operations until needed
- **Impact**: Application startup time reduced from minutes to seconds

### 3. **Dynamic Resource Management**
- **Problem**: Different deployment scenarios need different resource allocations
- **Solution**: Make resource decisions at materialization time
- **Impact**: Same code works for edge devices and data centers

### 4. **Configuration Flexibility**
- **Problem**: Hardware configs might change between definition and execution
- **Solution**: Capture intent, execute with final context
- **Impact**: Better separation of "what" vs "how"

### 5. **Testing and Development Speed**
- **Problem**: Unit tests that instantiate many models are slow
- **Solution**: Tests can verify configurations without materialization
- **Impact**: 10-100x faster test suites

## Typical Lazy Loading Techniques

### 1. **Proxy Pattern**
Create a lightweight proxy that defers to the real object when needed:

```python
class LazyProxy:
    def __init__(self, target_class, *args, **kwargs):
        self._target_class = target_class
        self._args = args
        self._kwargs = kwargs
        self._instance = None

    def _materialize(self):
        if self._instance is None:
            self._instance = self._target_class(*self._args, **self._kwargs)
        return self._instance

    def __getattr__(self, name):
        return getattr(self._materialize(), name)
```

### 2. **Factory Pattern with Delayed Execution**
Store construction logic without executing it:

```python
class ModuleFactory:
    def __init__(self):
        self._builders = {}

    def register(self, name: str, builder: Callable):
        self._builders[name] = builder

    def create(self, name: str, lazy=True):
        if lazy:
            return lambda: self._builders[name]()
        return self._builders[name]()
```

### 3. **Descriptor Protocol**
Use Python descriptors for automatic lazy loading:

```python
class LazyDescriptor:
    def __init__(self, loader_func):
        self.loader_func = loader_func
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Load and cache the value
        value = self.loader_func()
        setattr(instance, self.name, value)
        return value
```

### 4. **Promise/Future Pattern**
Computation that will happen later:

```python
class LazyModule:
    def __init__(self, module_spec):
        self.spec = module_spec
        self._future = None

    def materialize_async(self):
        if self._future is None:
            self._future = asyncio.create_task(self._build_module())
        return self._future

    async def _build_module(self):
        # Expensive module construction
        pass
```

### 5. **Copy-on-Write (COW)**
Share resources until modification is needed:

```python
class COWModule:
    def __init__(self, shared_params):
        self._shared_params = shared_params
        self._own_params = None

    def get_params(self):
        return self._own_params or self._shared_params

    def modify_params(self):
        if self._own_params is None:
            self._own_params = copy.deepcopy(self._shared_params)
        return self._own_params
```

## Design Considerations for TTTv2

### 1. **Granularity Levels**

```python
# Fine-grained: Individual layers
attention = LazyModule(MultiHeadAttention, config={...})
ffn = LazyModule(SwiGLU, config={...})

# Coarse-grained: Entire models
model = LazyModel(LlamaModel, config={...})

# Hybrid: Lazy models with lazy layers
model = LazyModel(
    layers=[
        LazyModule(Attention, layer_config[i])
        for i in range(num_layers)
    ]
)
```

### 2. **Materialization Triggers**

```python
# Explicit materialization
module = lazy_module.materialize(device="cuda:0")

# Implicit on first use
output = lazy_module.forward(input)  # Triggers materialization

# Batch materialization
LazyModule.materialize_all([module1, module2, module3])

# Conditional materialization
if memory_available() > module.estimated_memory():
    module.materialize()
```

### 3. **Resource Hints**

```python
# Provide hints for resource planning
lazy_module = LazyModule(
    MultiHeadAttention,
    config={...},
    resource_hints={
        "memory_mb": 512,
        "compute_flops": 1e12,
        "preferred_device": "cuda",
        "can_share_weights_with": ["encoder_attn_0"]
    }
)
```

### 4. **Serialization Support**

```python
# Save lazy module without materializing
lazy_module.save_spec("model_spec.json")

# Load as lazy
lazy_module = LazyModule.from_spec("model_spec.json")

# Partial materialization for checkpointing
lazy_module.materialize_weights_only()
```

## Implementation Strategies

### Strategy 1: **Metadata-First Approach**
```python
class LazyModule:
    def __init__(self, module_class, config):
        self.module_class = module_class
        self.config = config
        self.metadata = self._extract_metadata()

    def _extract_metadata(self):
        # Extract info without creating module
        return {
            "param_count": calculate_params(self.module_class, self.config),
            "memory_estimate": estimate_memory(self.module_class, self.config),
            "compute_estimate": estimate_compute(self.module_class, self.config)
        }
```

### Strategy 2: **Builder Pattern**
```python
class ModuleBuilder:
    def __init__(self):
        self.specs = []

    def add_attention(self, **kwargs):
        self.specs.append(("attention", kwargs))
        return self

    def add_ffn(self, **kwargs):
        self.specs.append(("ffn", kwargs))
        return self

    def build(self, lazy=True):
        if lazy:
            return LazyModel(self.specs)
        return self._materialize_all()
```

### Strategy 3: **Graph-Based Construction**
```python
class ComputeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_lazy_node(self, name, module_spec):
        self.nodes[name] = LazyNode(module_spec)

    def materialize_subgraph(self, start_nodes):
        # Only materialize needed nodes
        required = self._find_dependencies(start_nodes)
        for node in required:
            node.materialize()
```

## Best Practices

1. **Clear Materialization Semantics**
   - Make it obvious when materialization happens
   - Avoid surprising implicit materializations
   - Provide explicit control over timing

2. **Resource Estimation**
   - Provide accurate memory estimates before materialization
   - Allow querying resource requirements
   - Enable resource-aware scheduling

3. **Error Handling**
   - Fail fast on invalid configurations
   - Provide clear errors at definition time when possible
   - Handle materialization failures gracefully

4. **Debugging Support**
   - Allow inspection of lazy modules
   - Provide visualization of unmaterialized graphs
   - Support step-by-step materialization

5. **Performance Monitoring**
   - Track materialization time
   - Monitor memory allocation patterns
   - Profile lazy vs eager performance
