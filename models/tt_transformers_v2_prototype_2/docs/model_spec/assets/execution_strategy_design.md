# TTTv2 Execution Strategy Design: Bridging Architecture and Implementation

## Core Problem

The gap between model specification (what to compute) and execution strategies (how to compute) needs a clean abstraction that:
1. Makes prefill/decode specialization the default for TT hardware
2. Allows for future execution strategies without major refactoring
3. Maintains the module-level abstraction

## Design Solution: Execution Strategy Pattern

### Layer 1: Module Specification (What)
```python
@dataclass(frozen=True)
class AttentionSpec:
    """Pure mathematical specification"""
    hidden_dim: int
    num_heads: int
    # No execution details!
```

### Layer 2: Execution Strategy (How)
```python
from abc import ABC, abstractmethod
from enum import Enum

class ExecutionMode(Enum):
    """Current execution context"""
    PREFILL = "prefill"
    DECODE = "decode"
    UNIFIED = "unified"  # For models that don't specialize

class ExecutionStrategy(ABC):
    """Base class for all execution strategies"""

    @abstractmethod
    def get_execution_modes(self) -> List[ExecutionMode]:
        """Which execution modes this strategy supports"""
        pass

    @abstractmethod
    def specialize_module(self, module_spec: ModuleSpec, device: str) -> Dict[ExecutionMode, Any]:
        """Create specialized implementations for each mode"""
        pass

    @abstractmethod
    def select_implementation(self, mode: ExecutionMode) -> Any:
        """Select appropriate implementation for current mode"""
        pass
```

### Default Strategy: Prefill/Decode Specialization

```python
class PrefillDecodeStrategy(ExecutionStrategy):
    """
    Default TTTv2 strategy optimized for TT hardware.
    Creates separate implementations for prefill and decode.
    """

    def get_execution_modes(self) -> List[ExecutionMode]:
        return [ExecutionMode.PREFILL, ExecutionMode.DECODE]

    def specialize_module(self, module_spec: AttentionSpec, device: str) -> Dict[ExecutionMode, Any]:
        """Create specialized prefill and decode implementations"""

        implementations = {}

        # Prefill: Optimized for processing many tokens at once
        implementations[ExecutionMode.PREFILL] = self._create_prefill_attention(
            module_spec, device
        )

        # Decode: Optimized for single token generation with KV cache
        implementations[ExecutionMode.DECODE] = self._create_decode_attention(
            module_spec, device
        )

        return implementations

    def _create_prefill_attention(self, spec: AttentionSpec, device: str):
        """Prefill-optimized attention"""
        return PrefillAttention(
            hidden_dim=spec.hidden_dim,
            num_heads=spec.num_heads,
            device=device,
            # Prefill-specific optimizations
            use_flash_attention=True,
            enable_causal_mask=True,
            kv_cache_enabled=False  # No cache during prefill
        )

    def _create_decode_attention(self, spec: AttentionSpec, device: str):
        """Decode-optimized attention with KV caching"""
        return DecodeAttention(
            hidden_dim=spec.hidden_dim,
            num_heads=spec.num_heads,
            device=device,
            # Decode-specific optimizations
            use_paged_kv_cache=True,
            cache_dtype="bfloat16",
            single_token_mode=True
        )
```

### Module Implementation with Strategy

```python
class MultiHeadAttention(BaseModule):
    """
    Attention module that uses execution strategies.
    Default is prefill/decode, but extensible to other strategies.
    """

    def __init__(self,
                 spec: AttentionSpec,
                 device: str = None,
                 execution_strategy: ExecutionStrategy = None):

        self.spec = spec
        self.device = device

        # Use default strategy if not provided
        if execution_strategy is None:
            execution_strategy = PrefillDecodeStrategy()

        self.execution_strategy = execution_strategy
        self.current_mode = ExecutionMode.PREFILL  # Default mode

        # Let strategy create specialized implementations
        self.implementations = execution_strategy.specialize_module(spec, device)

        # Validate we have all required implementations
        self._validate_implementations()

    def set_mode(self, mode: ExecutionMode):
        """Switch execution mode (e.g., from prefill to decode)"""
        if mode not in self.implementations:
            raise ValueError(f"Mode {mode} not supported by current strategy")
        self.current_mode = mode

    def forward(self, *args, **kwargs):
        """Forward pass using current execution mode"""
        # Select implementation based on current mode
        impl = self.implementations[self.current_mode]
        return impl.forward(*args, **kwargs)

    # Convenience methods that make prefill/decode feel native
    def prefill_forward(self, *args, **kwargs):
        """Direct access to prefill implementation"""
        if ExecutionMode.PREFILL not in self.implementations:
            raise RuntimeError("Prefill mode not available")
        return self.implementations[ExecutionMode.PREFILL].forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        """Direct access to decode implementation"""
        if ExecutionMode.DECODE not in self.implementations:
            raise RuntimeError("Decode mode not available")
        return self.implementations[ExecutionMode.DECODE].forward(*args, **kwargs)
```

## Future Strategy Examples

### 1. Unified Strategy (No Specialization)

```python
class UnifiedStrategy(ExecutionStrategy):
    """Single implementation for all modes - simpler but less optimized"""

    def get_execution_modes(self) -> List[ExecutionMode]:
        return [ExecutionMode.UNIFIED]

    def specialize_module(self, module_spec: ModuleSpec, device: str):
        # Just one implementation
        return {
            ExecutionMode.UNIFIED: StandardAttention(module_spec, device)
        }
```

### 2. Multi-Mode Strategy

```python
class MultiModeStrategy(ExecutionStrategy):
    """
    Future strategy with more execution modes.
    Example: Different strategies for different sequence lengths.
    """

    def get_execution_modes(self) -> List[ExecutionMode]:
        return [
            ExecutionMode.PREFILL,
            ExecutionMode.DECODE,
            ExecutionMode.LONG_CONTEXT,  # New mode for 100k+ tokens
            ExecutionMode.SPECULATIVE,   # For speculative decoding
        ]

    def specialize_module(self, module_spec: ModuleSpec, device: str):
        implementations = {}

        # Standard prefill/decode
        implementations[ExecutionMode.PREFILL] = PrefillAttention(...)
        implementations[ExecutionMode.DECODE] = DecodeAttention(...)

        # New: Long context with different algorithm
        implementations[ExecutionMode.LONG_CONTEXT] = StreamingAttention(
            module_spec,
            chunk_size=1024,
            use_landmark_attention=True
        )

        # New: Speculative decoding variant
        implementations[ExecutionMode.SPECULATIVE] = SpeculativeAttention(
            module_spec,
            draft_model_compatible=True
        )

        return implementations
```

### 3. Dynamic Strategy Selection

```python
class DynamicStrategy(ExecutionStrategy):
    """Selects strategy based on runtime conditions"""

    def __init__(self):
        self.strategies = {
            'memory_constrained': MemoryEfficientStrategy(),
            'latency_optimized': LatencyOptimizedStrategy(),
            'balanced': PrefillDecodeStrategy()
        }

    def select_strategy(self, context: RuntimeContext) -> ExecutionStrategy:
        """Dynamically select strategy based on context"""
        if context.available_memory < 1_000_000_000:  # 1GB
            return self.strategies['memory_constrained']
        elif context.batch_size == 1 and context.latency_critical:
            return self.strategies['latency_optimized']
        else:
            return self.strategies['balanced']
```

## Making TTTv2 Forward-Compatible

### 1. Strategy Registry

```python
class StrategyRegistry:
    """Global registry for execution strategies"""
    _strategies = {
        'prefill_decode': PrefillDecodeStrategy,
        'unified': UnifiedStrategy,
        'multi_mode': MultiModeStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: Type[ExecutionStrategy]):
        """Register new strategies without modifying core code"""
        cls._strategies[name] = strategy_class

    @classmethod
    def get(cls, name: str) -> ExecutionStrategy:
        return cls._strategies[name]()

# Future users can register custom strategies
StrategyRegistry.register('my_custom_strategy', MyCustomStrategy)
```

### 2. Configuration-Driven Strategy Selection

```python
@dataclass
class ModuleConfig:
    """Extended configuration with strategy selection"""
    spec: ModuleSpec
    device: str
    execution_strategy: str = 'prefill_decode'  # Default
    strategy_config: Dict[str, Any] = None

def create_module(config: ModuleConfig):
    """Factory that respects strategy configuration"""
    strategy_class = StrategyRegistry.get(config.execution_strategy)
    strategy = strategy_class(**(config.strategy_config or {}))

    return MultiHeadAttention(
        spec=config.spec,
        device=config.device,
        execution_strategy=strategy
    )
```

### 3. Gradual Migration Path

```python
class BackwardCompatibleAttention(MultiHeadAttention):
    """
    Maintains old API while using new strategy system internally.
    Allows gradual migration.
    """

    def __init__(self, hidden_dim: int, num_heads: int, device: str = None):
        # Create spec from old-style parameters
        spec = AttentionSpec(hidden_dim=hidden_dim, num_heads=num_heads)

        # Use default prefill/decode strategy
        super().__init__(spec, device, PrefillDecodeStrategy())

    # Old API still works
    def forward(self, x, mode='prefill'):
        if mode == 'prefill':
            self.set_mode(ExecutionMode.PREFILL)
        elif mode == 'decode':
            self.set_mode(ExecutionMode.DECODE)

        return super().forward(x)
```

## Benefits of This Design

### 1. **Prefill/Decode as First-Class Default**
```python
# Natural API for TT hardware optimization
attn = MultiHeadAttention(spec)
output = attn.prefill_forward(tokens)  # Clear intent
output = attn.decode_forward(token)    # Optimized path
```

### 2. **Future-Proof Architecture**
- New strategies can be added without changing module interfaces
- Existing code continues to work with new strategies
- Strategy selection can be configuration-driven

### 3. **Clean Separation of Concerns**
- Module specs remain pure (mathematical only)
- Execution strategies encapsulate "how"
- Hardware configs remain separate

### 4. **Progressive Disclosure**
```python
# Beginner: Just works with defaults
attn = MultiHeadAttention(spec)

# Intermediate: Explicit mode control
attn.set_mode(ExecutionMode.DECODE)

# Advanced: Custom strategies
attn = MultiHeadAttention(spec, execution_strategy=MyCustomStrategy())

# Expert: Dynamic strategy switching
attn.switch_strategy(new_strategy)
```

## Example: Adding New Execution Strategy in Future

```python
# Future: Someone wants to add continuous batching support
class ContinuousBatchingStrategy(ExecutionStrategy):
    """New strategy for continuous batching inference"""

    def get_execution_modes(self):
        return [
            ExecutionMode.PREFILL,
            ExecutionMode.DECODE,
            ExecutionMode.CONTINUOUS_BATCH
        ]

    def specialize_module(self, spec, device):
        # Reuse existing prefill/decode
        base_strategy = PrefillDecodeStrategy()
        implementations = base_strategy.specialize_module(spec, device)

        # Add new mode
        implementations[ExecutionMode.CONTINUOUS_BATCH] = ContinuousBatchAttention(
            spec,
            enable_dynamic_batching=True,
            enable_iteration_scheduling=True
        )

        return implementations

# Register and use without changing TTTv2 core
StrategyRegistry.register('continuous_batching', ContinuousBatchingStrategy)

# Users can now use it
attn = MultiHeadAttention(
    spec,
    execution_strategy=StrategyRegistry.get('continuous_batching')
)
```

## Conclusion

This design achieves the balance you're looking for:
1. **Prefill/decode is the default** and feels native to TTTv2
2. **Clean abstraction** between spec and execution
3. **Extensible** for future strategies without major refactoring
4. **Module-level** strategy application maintains your design philosophy

The key insight is that execution strategies are a separate concern from both module specification (what) and hardware configuration (performance tuning), allowing each to evolve independently.
