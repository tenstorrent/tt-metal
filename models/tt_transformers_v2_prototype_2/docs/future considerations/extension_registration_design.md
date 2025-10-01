# TTTv2 Extension Registration Design

## The Ambiguity Problem

The current design has conflicting messages about extensions:
- Line 213: "TTTv2 provides extension mechanisms to register new modules"
- Lines 563-570: Shows direct instantiation without registration

This creates confusion: Do users need to register custom modules or not?

## Proposed Solution: Dual-Purpose System

### Core Principle: "Use Without Registration, Register for Discovery"

Custom modules can be used immediately through inheritance (no registration required), but registration provides additional benefits for discovery, documentation, and ecosystem integration.

## Design Components

### 1. Base Classes for Extension (No Registration Required)

```python
# tt_transformers_v2/attention/base.py
class BaseAttention(TTTModule):
    """Base class for all attention mechanisms.

    Inherit from this to create custom attention without registration.
    """
    def __init__(self, hidden_dim: int, num_heads: int, device=None):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device

    @abstractmethod
    def compute_attention(self, q, k, v, mask=None):
        """Override this method in your implementation"""
        pass

    def forward(self, x, mask=None):
        # Standard attention flow with calls to compute_attention
        q, k, v = self.compute_qkv(x)
        return self.compute_attention(q, k, v, mask)

# Usage without registration - works immediately!
class MyCustomAttention(BaseAttention):
    def compute_attention(self, q, k, v, mask=None):
        # Custom implementation
        return my_special_attention_logic(q, k, v)

# Direct usage
custom_attn = MyCustomAttention(hidden_dim=4096, num_heads=32)
output = custom_attn(input_tensor)
```

### 2. Optional Registry for Discovery and Documentation

```python
# tt_transformers_v2/registry.py
from typing import Type, Dict, Any, List, Optional
from dataclasses import dataclass
import inspect


@dataclass
class ModuleMetadata:
    """Metadata for registered modules"""
    name: str
    module_class: Type
    category: str  # "attention", "ffn", "norm", etc.
    description: str
    author: str
    version: str
    hardware_support: List[str]  # ["cpu", "cuda", "ttnn"]
    tags: List[str]  # ["experimental", "production", "sparse", etc.]
    config_schema: Optional[Dict[str, Any]] = None
    performance_hints: Optional[Dict[str, Any]] = None


class ModuleRegistry:
    """
    Optional registry for module discovery and documentation.

    Registration provides:
    - Discoverability through search/list APIs
    - Automatic documentation generation
    - Config validation
    - Performance recommendations
    - Community sharing
    """

    _registry: Dict[str, ModuleMetadata] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls,
                 name: str,
                 category: str,
                 description: str = "",
                 author: str = "",
                 version: str = "1.0.0",
                 hardware_support: List[str] = None,
                 tags: List[str] = None,
                 alias: Optional[str] = None):
        """
        Decorator to register a module for discovery.

        Registration is OPTIONAL - modules work without it!
        """
        def decorator(module_class: Type):
            # Extract config schema from __init__ signature
            config_schema = cls._extract_config_schema(module_class)

            metadata = ModuleMetadata(
                name=name,
                module_class=module_class,
                category=category,
                description=description or module_class.__doc__ or "",
                author=author,
                version=version,
                hardware_support=hardware_support or ["cpu", "cuda", "ttnn"],
                tags=tags or [],
                config_schema=config_schema
            )

            cls._registry[name] = metadata

            # Register alias if provided
            if alias:
                cls._aliases[alias] = name

            # Add registry info to the class
            module_class._registry_metadata = metadata

            return module_class

        return decorator

    @classmethod
    def _extract_config_schema(cls, module_class: Type) -> Dict[str, Any]:
        """Extract configuration schema from class __init__"""
        sig = inspect.signature(module_class.__init__)
        schema = {}

        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'device']:
                continue

            schema[param_name] = {
                'type': param.annotation if param.annotation != param.empty else Any,
                'default': param.default if param.default != param.empty else None,
                'required': param.default == param.empty
            }

        return schema

    @classmethod
    def list_modules(cls,
                    category: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    hardware: Optional[str] = None) -> List[ModuleMetadata]:
        """List all registered modules with optional filtering"""
        modules = list(cls._registry.values())

        if category:
            modules = [m for m in modules if m.category == category]

        if tags:
            modules = [m for m in modules if any(tag in m.tags for tag in tags)]

        if hardware:
            modules = [m for m in modules if hardware in m.hardware_support]

        return modules

    @classmethod
    def get_module(cls, name: str) -> Optional[Type]:
        """Get a module class by name or alias"""
        # Check aliases
        if name in cls._aliases:
            name = cls._aliases[name]

        metadata = cls._registry.get(name)
        return metadata.module_class if metadata else None

    @classmethod
    def search(cls, query: str) -> List[ModuleMetadata]:
        """Search modules by name, description, or tags"""
        query_lower = query.lower()
        results = []

        for metadata in cls._registry.values():
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag for tag in metadata.tags)):
                results.append(metadata)

        return results

    @classmethod
    def validate_config(cls, module_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against module schema"""
        metadata = cls._registry.get(module_name)
        if not metadata or not metadata.config_schema:
            return []

        errors = []
        schema = metadata.config_schema

        # Check required fields
        for field, field_schema in schema.items():
            if field_schema['required'] and field not in config:
                errors.append(f"Missing required field: {field}")

        # Check types (simplified)
        for field, value in config.items():
            if field in schema:
                expected_type = schema[field]['type']
                if expected_type != Any and not isinstance(value, expected_type):
                    errors.append(f"Field {field} should be {expected_type}, got {type(value)}")

        return errors

    @classmethod
    def generate_docs(cls) -> str:
        """Generate documentation for all registered modules"""
        docs = ["# TTTv2 Registered Modules\n"]

        categories = {}
        for metadata in cls._registry.values():
            if metadata.category not in categories:
                categories[metadata.category] = []
            categories[metadata.category].append(metadata)

        for category, modules in sorted(categories.items()):
            docs.append(f"\n## {category.title()}\n")

            for module in sorted(modules, key=lambda m: m.name):
                docs.append(f"### {module.name}\n")
                docs.append(f"**Author**: {module.author}\n")
                docs.append(f"**Version**: {module.version}\n")
                docs.append(f"**Hardware**: {', '.join(module.hardware_support)}\n")

                if module.tags:
                    docs.append(f"**Tags**: {', '.join(module.tags)}\n")

                docs.append(f"\n{module.description}\n")

                if module.config_schema:
                    docs.append("\n**Configuration**:\n```python\n")
                    for field, schema in module.config_schema.items():
                        req = "required" if schema['required'] else "optional"
                        default = f" = {schema['default']}" if schema['default'] else ""
                        docs.append(f"{field}: {schema['type'].__name__} ({req}){default}\n")
                    docs.append("```\n")

        return "\n".join(docs)


# Convenience function for registration
def register_module(name: str, category: str, **kwargs):
    """Convenience function for registering modules"""
    return ModuleRegistry.register(name, category, **kwargs)
```

### 3. Usage Examples

#### Example 1: Using Custom Module Without Registration

```python
# my_custom_modules.py
from tt_transformers_v2.attention import BaseAttention
import torch

class FlashAttentionV3(BaseAttention):
    """My custom Flash Attention implementation - no registration needed!"""

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int = 2048, device=None):
        super().__init__(hidden_dim, num_heads, device)
        self.window_size = window_size

    def compute_attention(self, q, k, v, mask=None):
        # Custom flash attention logic
        return flash_attention_v3_kernel(q, k, v, self.window_size, mask)

# Use immediately without any registration
model = TransformerModel(
    attention_cls=FlashAttentionV3,  # Just pass the class!
    attention_config={"hidden_dim": 4096, "num_heads": 32, "window_size": 4096}
)
```

#### Example 2: Registering for Discovery and Documentation

```python
# my_custom_modules.py
from tt_transformers_v2.registry import register_module

@register_module(
    name="flash-attention-v3",
    category="attention",
    description="Optimized Flash Attention V3 with sliding window support",
    author="Research Team",
    version="3.0.0",
    hardware_support=["cuda", "ttnn"],
    tags=["production", "optimized", "sliding-window"],
    alias="flash3"  # Short alias
)
class FlashAttentionV3(BaseAttention):
    """Flash Attention V3 with improved memory efficiency.

    This implementation provides:
    - 2x faster than V2 on A100 GPUs
    - Sliding window support for long sequences
    - Optimized for TTNN hardware
    """

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int = 2048, device=None):
        super().__init__(hidden_dim, num_heads, device)
        self.window_size = window_size

    def compute_attention(self, q, k, v, mask=None):
        return flash_attention_v3_kernel(q, k, v, self.window_size, mask)

# Now users can discover it
from tt_transformers_v2.registry import ModuleRegistry

# List all attention modules
attentions = ModuleRegistry.list_modules(category="attention")
for attn in attentions:
    print(f"{attn.name}: {attn.description}")

# Search for sliding window attention
results = ModuleRegistry.search("sliding window")

# Get module by name or alias
FlashAttnClass = ModuleRegistry.get_module("flash3")  # Using alias
attn = FlashAttnClass(hidden_dim=4096, num_heads=32)

# Validate configuration
errors = ModuleRegistry.validate_config("flash-attention-v3", {
    "hidden_dim": 4096,
    "num_heads": 32,
    # "window_size" is optional with default
})
```

#### Example 3: Community Module Sharing

```python
# community_modules.py
from tt_transformers_v2.registry import register_module
from tt_transformers_v2.ffn import BaseFFN

@register_module(
    name="mixture-of-depths-ffn",
    category="ffn",
    description="Mixture of Depths FFN - dynamically skips computation for some tokens",
    author="Community Contributor",
    version="0.1.0",
    hardware_support=["cuda", "ttnn"],
    tags=["experimental", "efficiency", "mod"],
    performance_hints={
        "speedup": "1.5-2x on long sequences",
        "memory": "30% reduction"
    }
)
class MixtureOfDepthsFFN(BaseFFN):
    """Dynamically routes tokens through FFN based on importance."""

    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 routing_threshold: float = 0.5, device=None):
        super().__init__(hidden_dim, intermediate_dim, device)
        self.routing_threshold = routing_threshold
        self.router = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Route only important tokens through FFN
        importance = torch.sigmoid(self.router(x))
        mask = importance > self.routing_threshold

        # Process only selected tokens
        output = x.clone()
        if mask.any():
            selected = x[mask]
            processed = super().forward(selected)
            output[mask] = processed

        return output

# Generate documentation for all community modules
docs = ModuleRegistry.generate_docs()
# Outputs formatted markdown with all registered modules
```

### 4. Integration with Model Factory

```python
# tt_transformers_v2/interfaces/model_factory.py
class ModelFactory:
    @staticmethod
    def from_config(config: Dict[str, Any], device=None):
        """Build model from configuration, supporting both registered and unregistered modules"""

        modules = {}

        for layer_config in config['layers']:
            module_ref = layer_config['module']

            # Check if it's a string reference to registered module
            if isinstance(module_ref, str):
                module_class = ModuleRegistry.get_module(module_ref)
                if not module_class:
                    raise ValueError(f"Unknown module: {module_ref}")
            # Or a direct class reference
            elif isinstance(module_ref, type):
                module_class = module_ref
            else:
                raise ValueError(f"Invalid module reference: {module_ref}")

            # Validate config if module is registered
            if hasattr(module_class, '_registry_metadata'):
                errors = ModuleRegistry.validate_config(
                    module_class._registry_metadata.name,
                    layer_config.get('config', {})
                )
                if errors:
                    raise ValueError(f"Config errors: {errors}")

            # Create module instance
            module = module_class(**layer_config.get('config', {}), device=device)
            modules[layer_config['name']] = module

        return Model(modules)

# Usage with both registered and unregistered modules
config = {
    'layers': [
        {
            'name': 'attention_0',
            'module': 'flash-attention-v3',  # Registered module by name
            'config': {'hidden_dim': 4096, 'num_heads': 32}
        },
        {
            'name': 'attention_1',
            'module': MyUnregisteredAttention,  # Direct class reference
            'config': {'hidden_dim': 4096, 'num_heads': 32}
        }
    ]
}
```

### 5. Module Discovery CLI

```python
# tt_transformers_v2/cli.py
import click
from tt_transformers_v2.registry import ModuleRegistry

@click.group()
def cli():
    """TTTv2 command line interface"""
    pass

@cli.command()
@click.option('--category', help='Filter by category')
@click.option('--tag', help='Filter by tag')
def list_modules(category, tag):
    """List available TTTv2 modules"""
    modules = ModuleRegistry.list_modules(
        category=category,
        tags=[tag] if tag else None
    )

    for module in modules:
        click.echo(f"{module.name} ({module.version}) - {module.description}")
        if module.tags:
            click.echo(f"  Tags: {', '.join(module.tags)}")

@cli.command()
@click.argument('query')
def search(query):
    """Search for modules"""
    results = ModuleRegistry.search(query)

    for module in results:
        click.echo(f"{module.name} - {module.description}")

@cli.command()
@click.argument('module_name')
def info(module_name):
    """Show detailed information about a module"""
    module = ModuleRegistry.get_module(module_name)
    if not module:
        click.echo(f"Module '{module_name}' not found")
        return

    metadata = module._registry_metadata
    click.echo(f"Name: {metadata.name}")
    click.echo(f"Category: {metadata.category}")
    click.echo(f"Author: {metadata.author}")
    click.echo(f"Version: {metadata.version}")
    click.echo(f"Description: {metadata.description}")
    click.echo(f"Hardware: {', '.join(metadata.hardware_support)}")

    if metadata.config_schema:
        click.echo("\nConfiguration:")
        for field, schema in metadata.config_schema.items():
            click.echo(f"  {field}: {schema}")

# Usage:
# $ ttt list-modules --category attention
# $ ttt search "flash attention"
# $ ttt info flash-attention-v3
```

## Benefits of This Design

1. **No Barriers to Entry**: Users can create and use custom modules immediately without registration
2. **Progressive Enhancement**: Registration adds value (discovery, validation, docs) without being required
3. **Clear Separation**: Core functionality (inheritance) vs ecosystem features (registry)
4. **Community Building**: Registry enables sharing and discovery of community modules
5. **Tooling Support**: CLI, documentation generation, and IDE integration via registry
6. **Backward Compatible**: Existing code using direct instantiation continues to work

## Summary

The registration system is **completely optional**. Users can:
- Use custom modules immediately via inheritance (no registration)
- Register modules to gain additional benefits (discovery, validation, documentation)
- Mix registered and unregistered modules in the same model
- Share modules with the community through the registry

This design resolves the ambiguity by making it clear that registration is a value-add feature, not a requirement for basic functionality.
