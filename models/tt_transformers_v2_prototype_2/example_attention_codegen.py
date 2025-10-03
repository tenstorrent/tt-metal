"""
TTTv2 Attention Module with Code Generation

This example demonstrates:
1. How a high-level attention module can generate specialized implementations
2. Template-based code generation for different hardware configurations
3. Compile-time optimization based on tensor shapes and hardware constraints
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import ttnn


@dataclass
class HardwareConfig:
    """Hardware configuration that affects code generation"""

    device_name: str
    grid_size: Tuple[int, int]
    l1_memory_size: int  # in bytes
    supports_flash_attention: bool
    fp32_accumulation: bool
    max_seq_len: int
    shard_strategy: str  # "block", "height", "width"


@dataclass
class AttentionConfig:
    """Attention-specific configuration"""

    hidden_size: int
    num_heads: int
    head_dim: int
    use_rotary_embeddings: bool
    use_sliding_window: bool
    window_size: Optional[int]
    dropout: float

    @property
    def total_dim(self) -> int:
        return self.num_heads * self.head_dim


def forward_qkv_fused(x, *, mod_spec, hw_config, tensor_cache):
    qkv = ttnn.linear(
        x,
        tensor_cache.qkv_linear.weight,
        bias=tensor_cache.qkv_linear.bias,
        compute_kernel_config=hw_config.qkv_linear.compute_kernel_config,
        program_config=hw_config.qkv_linear.program_config,
        memory_config=hw_config.qkv_linear.memory_config,
        dtype=mod_spec.qkv_linear.dtype,
    )

    batch_size, seq_len, _ = x.shape
    qkv = ttnn.reshape(
        qkv,
        [
            batch_size,
            seq_len,
            3,
            mod_spec.num_heads,
            mod_spec.head_dim,
        ],
    )
    query_states = qkv[:, :, 0, :, :]
    key_states = qkv[:, :, 1, :, :]
    value_states = qkv[:, :, 2, :, :]

    return query_states, key_states, value_states


def forward_qkv_unfused(x, *, mod_spec, hw_config, tensor_cache):
    query_states = ttnn.linear(
        x,
        tensor_cache.q_linear.weight,
        bias=tensor_cache.q_linear.bias,
        compute_kernel_config=hw_config.q_linear.compute_kernel_config,
        program_config=hw_config.q_linear.program_config,
        memory_config=hw_config.q_linear.memory_config,
        dtype=mod_spec.q_linear.dtype,
    )
    key_states = ttnn.linear(
        x,
        tensor_cache.k_linear.weight,
        bias=tensor_cache.k_linear.bias,
        compute_kernel_config=hw_config.k_linear.compute_kernel_config,
        program_config=hw_config.k_linear.program_config,
        memory_config=hw_config.k_linear.memory_config,
        dtype=mod_spec.k_linear.dtype,
    )
    value_states = ttnn.linear(
        x,
        tensor_cache.v_linear.weight,
        bias=tensor_cache.v_linear.bias,
        compute_kernel_config=hw_config.v_linear.compute_kernel_config,
        program_config=hw_config.v_linear.program_config,
        memory_config=hw_config.v_linear.memory_config,
        dtype=mod_spec.v_linear.dtype,
    )

    batch_size, seq_len, _ = x.shape
    target_shape = [
        batch_size,
        seq_len,
        mod_spec.num_heads,
        mod_spec.head_dim,
    ]
    query_states = ttnn.reshape(query_states, target_shape)
    key_states = ttnn.reshape(key_states, target_shape)
    value_states = ttnn.reshape(value_states, target_shape)

    return query_states, key_states, value_states


def generate_optimized_source(func_name: str, args: list, func_def: ast.FunctionDef) -> str:
    """Generate complete optimized source code"""

    # Extract function body
    body_lines = ast.unparse(func_def).split("\n")[1:]  # Skip def line
    body = "\n".join(body_lines)

    # Add the forward method with original function body
    source_lines = [
        [
            f'    def {func_name}({", ".join(args)}):',
            f'        """Generated from introspected function"""',
        ]
    ]

    # Indent and add function body
    for line in body.split("\n"):
        if line.strip():
            source_lines.append(f"        {line}")
        else:
            source_lines.append("")

    return "\n".join(source_lines)


def function_to_source(func: Callable, class_name: str = "GeneratedClass") -> str:
    """
    Convert a function to optimized source code by:
    1. Extracting the original source
    2. Analyzing for optimization opportunities
    3. Generating enhanced source with context
    """

    # Get original source
    original_source = inspect.getsource(func)

    # Parse AST for analysis
    tree = ast.parse(original_source)

    # Extract function details
    func_def = tree.body[0]
    func_name = func_def.name
    args = [arg.arg for arg in func_def.args.args]

    # Analyze function body for TTNN operations
    # ttnn_ops = find_ttnn_operations(func_def)

    # Generate optimized source with full context
    source = generate_optimized_source(func_name, args, func_def, ttnn_ops, class_name)

    return source


class TTTv2AttentionCodeGen:
    """
    Generates optimized attention implementations based on configuration.
    This is the core of the code generation approach.
    """

    def __init__(self, hw_config: HardwareConfig, attn_config: AttentionConfig):
        self.hw_config = hw_config
        self.attn_config = attn_config
        self._validate_config()

    def _validate_config(self):
        """Validate that configuration is feasible for hardware"""
        # Check memory constraints
        seq_len = self.hw_config.max_seq_len
        batch_per_core = 1  # simplified
        memory_needed = (
            batch_per_core * seq_len * self.attn_config.total_dim * 2  # Q, K
            + batch_per_core * seq_len * seq_len * 2  # attention scores
        )

        if memory_needed > self.hw_config.l1_memory_size:
            raise ValueError(f"Configuration requires {memory_needed} bytes but L1 has {self.hw_config.l1_memory_size}")

    def generate_forward_function(self) -> str:
        """
        Generate specialized forward function based on configuration.
        This is the template-based approach mentioned in the design.
        """

        # Determine optimal implementation strategy
        use_flash = (
            self.hw_config.supports_flash_attention
            and self.attn_config.hidden_size >= 1024
            and not self.attn_config.use_sliding_window
        )

        # Build function source code
        lines = [
            "def forward(self, hidden_states, attention_mask=None, position_ids=None):",
            '    """',
            f"    Optimized attention forward for {self.hw_config.device_name}",
            f"    Hidden size: {self.attn_config.hidden_size}",
            f"    Num heads: {self.attn_config.num_heads}",
            f'    Implementation: {"flash" if use_flash else "standard"}',
            '    """',
            "    import ttnn",
            "    import torch",
            "",
        ]

        # Add shape extraction
        lines.extend(
            [
                "    batch_size, seq_len, _ = hidden_states.shape",
                "",
            ]
        )

        # Configure memory and compute based on hardware
        lines.extend(self._generate_hardware_config())

        # Generate Q, K, V projections
        lines.extend(self._generate_qkv_projections())

        # Generate attention computation
        if use_flash:
            lines.extend(self._generate_flash_attention())
        else:
            lines.extend(self._generate_standard_attention())

        # Output projection
        lines.extend(self._generate_output_projection())

        return "\n".join(lines)

    def _generate_hardware_config(self) -> list:
        """Generate hardware-specific configuration setup"""
        lines = [
            "    # Hardware-specific configuration",
            "    compute_kernel_config = ttnn.WormholeComputeKernelConfig(",
            f"        math_fidelity=ttnn.MathFidelity.{'HiFi4' if self.hw_config.fp32_accumulation else 'HiFi2'},",
            f"        fp32_dest_acc_en={self.hw_config.fp32_accumulation},",
            "        packer_l1_acc=True",
            "    )",
            "",
        ]

        # Memory configuration based on shard strategy
        if self.hw_config.shard_strategy == "block":
            lines.extend(
                [
                    "    memory_config = ttnn.MemoryConfig(",
                    "        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,",
                    "        buffer_type=ttnn.BufferType.L1",
                    "    )",
                ]
            )
        elif self.hw_config.shard_strategy == "height":
            lines.extend(
                [
                    "    memory_config = ttnn.MemoryConfig(",
                    "        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,",
                    "        buffer_type=ttnn.BufferType.L1",
                    "    )",
                ]
            )
        else:
            lines.extend(
                [
                    "    memory_config = ttnn.MemoryConfig(",
                    "        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,",
                    "        buffer_type=ttnn.BufferType.DRAM",
                    "    )",
                ]
            )

        lines.append("")
        return lines

    def _generate_qkv_projections(self) -> list:
        """Generate Q, K, V projection code"""
        lines = [
            "    # Q, K, V projections",
            "    # Optimized for head dimension and memory layout",
        ]

        # Determine if we can fuse QKV projection
        can_fuse = (
            self.attn_config.head_dim % 32 == 0
            and self.hw_config.l1_memory_size > self.attn_config.total_dim * 3 * 2048
        )

        if can_fuse:
            lines.extend(
                [
                    "    qkv = ttnn.linear(",
                    "        hidden_states,",
                    "        self.qkv_weight,",
                    "        bias=self.qkv_bias if hasattr(self, 'qkv_bias') else None,",
                    "        compute_kernel_config=compute_kernel_config,",
                    "        memory_config=memory_config,",
                    "        dtype=ttnn.bfloat16",
                    "    )",
                    "",
                    f"    # Reshape to separate Q, K, V: [batch, seq, 3, num_heads, head_dim]",
                    f"    qkv = ttnn.reshape(qkv, [batch_size, seq_len, 3, {self.attn_config.num_heads}, {self.attn_config.head_dim}])",
                    "    query_states = qkv[:, :, 0, :, :]",
                    "    key_states = qkv[:, :, 1, :, :]",
                    "    value_states = qkv[:, :, 2, :, :]",
                ]
            )
        else:
            # Separate projections
            lines.extend(
                [
                    "    query_states = ttnn.linear(",
                    "        hidden_states, self.q_weight, bias=self.q_bias,",
                    "        compute_kernel_config=compute_kernel_config,",
                    "        memory_config=memory_config,",
                    "        dtype=ttnn.bfloat16",
                    "    )",
                    "    key_states = ttnn.linear(",
                    "        hidden_states, self.k_weight, bias=self.k_bias,",
                    "        compute_kernel_config=compute_kernel_config,",
                    "        memory_config=memory_config,",
                    "        dtype=ttnn.bfloat16",
                    "    )",
                    "    value_states = ttnn.linear(",
                    "        hidden_states, self.v_weight, bias=self.v_bias,",
                    "        compute_kernel_config=compute_kernel_config,",
                    "        memory_config=memory_config,",
                    "        dtype=ttnn.bfloat16",
                    "    )",
                    "",
                    "    # Reshape to [batch, seq, num_heads, head_dim]",
                    f"    query_states = ttnn.reshape(query_states, [batch_size, seq_len, {self.attn_config.num_heads}, {self.attn_config.head_dim}])",
                    f"    key_states = ttnn.reshape(key_states, [batch_size, seq_len, {self.attn_config.num_heads}, {self.attn_config.head_dim}])",
                    f"    value_states = ttnn.reshape(value_states, [batch_size, seq_len, {self.attn_config.num_heads}, {self.attn_config.head_dim}])",
                ]
            )

        lines.append("")

        # Add rotary embeddings if configured
        if self.attn_config.use_rotary_embeddings:
            lines.extend(
                [
                    "    # Apply rotary position embeddings",
                    "    if position_ids is not None:",
                    "        query_states, key_states = ttnn.apply_rotary_embeddings(",
                    "            query_states, key_states, self.rotary_emb, position_ids",
                    "        )",
                    "",
                ]
            )

        return lines

    def _generate_flash_attention(self) -> list:
        """Generate flash attention implementation"""
        lines = [
            "    # Flash attention implementation",
            "    # Optimized for long sequences and memory efficiency",
            "    attention_output = ttnn.flash_attention(",
            "        query_states,",
            "        key_states,",
            "        value_states,",
            "        is_causal=True,",
            "        attention_mask=attention_mask,",
            f"        dropout_p={self.attn_config.dropout if self.training else 0.0},",
            "        compute_kernel_config=compute_kernel_config,",
            "        memory_config=memory_config",
            "    )",
            "",
        ]
        return lines

    def _generate_standard_attention(self) -> list:
        """Generate standard attention implementation"""
        lines = [
            "    # Standard attention implementation",
            "    # Transpose for matmul: [batch, num_heads, seq, head_dim]",
            "    query_states = ttnn.transpose(query_states, 1, 2)",
            "    key_states = ttnn.transpose(key_states, 1, 2)",
            "    value_states = ttnn.transpose(value_states, 1, 2)",
            "",
        ]

        # Attention scores computation
        lines.extend(
            [
                "    # Compute attention scores",
                f"    scale = 1.0 / torch.sqrt(torch.tensor({self.attn_config.head_dim}, dtype=torch.float32))",
                "    attention_scores = ttnn.matmul(",
                "        query_states,",
                "        ttnn.transpose(key_states, -2, -1),",
                "        compute_kernel_config=compute_kernel_config,",
                "        memory_config=memory_config",
                "    )",
                "    attention_scores = attention_scores * scale",
                "",
            ]
        )

        # Add sliding window mask if configured
        if self.attn_config.use_sliding_window:
            lines.extend(
                [
                    f"    # Apply sliding window mask (window_size={self.attn_config.window_size})",
                    "    if attention_mask is None:",
                    f"        attention_mask = ttnn.create_sliding_window_mask(seq_len, {self.attn_config.window_size}, device=hidden_states.device)",
                    "    attention_scores = attention_scores + attention_mask",
                    "",
                ]
            )
        elif "attention_mask is not None":
            lines.extend(
                [
                    "    # Apply attention mask",
                    "    if attention_mask is not None:",
                    "        attention_scores = attention_scores + attention_mask",
                    "",
                ]
            )

        # Softmax and attention
        lines.extend(
            [
                "    # Softmax and apply attention to values",
                "    attention_probs = ttnn.softmax(attention_scores, dim=-1)",
                "",
                f"    if self.training and {self.attn_config.dropout} > 0:",
                f"        attention_probs = ttnn.dropout(attention_probs, p={self.attn_config.dropout})",
                "",
                "    attention_output = ttnn.matmul(",
                "        attention_probs,",
                "        value_states,",
                "        compute_kernel_config=compute_kernel_config,",
                "        memory_config=memory_config",
                "    )",
                "",
                "    # Transpose back: [batch, seq, num_heads, head_dim]",
                "    attention_output = ttnn.transpose(attention_output, 1, 2)",
                "",
            ]
        )

        return lines

    def _generate_output_projection(self) -> list:
        """Generate output projection code"""
        lines = [
            "    # Reshape and project output",
            f"    attention_output = ttnn.reshape(attention_output, [batch_size, seq_len, {self.attn_config.total_dim}])",
            "",
            "    output = ttnn.linear(",
            "        attention_output,",
            "        self.o_weight,",
            "        bias=self.o_bias if hasattr(self, 'o_bias') else None,",
            "        compute_kernel_config=compute_kernel_config,",
            "        memory_config=memory_config,",
            "        dtype=ttnn.bfloat16",
            "    )",
            "",
            "    return output",
        ]
        return lines

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward"""

        # Build complete class
        class_lines = [
            f"class TTTv2Attention_{self.hw_config.device_name.replace('-', '_')}(nn.Module):",
            f'    """',
            f"    Auto-generated attention module for {self.hw_config.device_name}",
            f"    Configuration:",
            f"      - Hidden size: {self.attn_config.hidden_size}",
            f"      - Num heads: {self.attn_config.num_heads}",
            f"      - Head dim: {self.attn_config.head_dim}",
            f"      - Grid size: {self.hw_config.grid_size}",
            f"      - Shard strategy: {self.hw_config.shard_strategy}",
            f'    """',
            f"",
            f"    def __init__(self, device):",
            f"        super().__init__()",
            f"        self.device = device",
            f"        self.hidden_size = {self.attn_config.hidden_size}",
            f"        self.num_heads = {self.attn_config.num_heads}",
            f"        self.head_dim = {self.attn_config.head_dim}",
            f"        ",
            f"        # Initialize weights",
            f"        self._init_weights()",
            f"",
            f"    def _init_weights(self):",
            f'        """Initialize projection weights"""',
            f"        import torch",
            f"        import ttnn",
            f"",
        ]

        # Add weight initialization based on fusion strategy
        can_fuse = (
            self.attn_config.head_dim % 32 == 0
            and self.hw_config.l1_memory_size > self.attn_config.total_dim * 3 * 2048
        )

        if can_fuse:
            class_lines.extend(
                [
                    f"        # Fused QKV weights",
                    f"        self.qkv_weight = ttnn.create_weight(",
                    f"            shape=[{self.attn_config.hidden_size}, {self.attn_config.total_dim * 3}],",
                    f"            dtype=ttnn.bfloat16,",
                    f"            device=self.device",
                    f"        )",
                    f"        self.qkv_bias = ttnn.create_bias(",
                    f"            shape=[{self.attn_config.total_dim * 3}],",
                    f"            dtype=ttnn.bfloat16,",
                    f"            device=self.device",
                    f"        )",
                ]
            )
        else:
            class_lines.extend(
                [
                    f"        # Separate Q, K, V weights",
                    f"        self.q_weight = ttnn.create_weight(",
                    f"            shape=[{self.attn_config.hidden_size}, {self.attn_config.total_dim}],",
                    f"            dtype=ttnn.bfloat16,",
                    f"            device=self.device",
                    f"        )",
                    f"        self.k_weight = ttnn.create_weight(",
                    f"            shape=[{self.attn_config.hidden_size}, {self.attn_config.total_dim}],",
                    f"            dtype=ttnn.bfloat16,",
                    f"            device=self.device",
                    f"        )",
                    f"        self.v_weight = ttnn.create_weight(",
                    f"            shape=[{self.attn_config.hidden_size}, {self.attn_config.total_dim}],",
                    f"            dtype=ttnn.bfloat16,",
                    f"            device=self.device",
                    f"        )",
                    f"        # Add biases",
                    f"        self.q_bias = ttnn.create_bias([{self.attn_config.total_dim}], dtype=ttnn.bfloat16, device=self.device)",
                    f"        self.k_bias = ttnn.create_bias([{self.attn_config.total_dim}], dtype=ttnn.bfloat16, device=self.device)",
                    f"        self.v_bias = ttnn.create_bias([{self.attn_config.total_dim}], dtype=ttnn.bfloat16, device=self.device)",
                ]
            )

        class_lines.extend(
            [
                f"",
                f"        # Output projection",
                f"        self.o_weight = ttnn.create_weight(",
                f"            shape=[{self.attn_config.total_dim}, {self.attn_config.hidden_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"        self.o_bias = ttnn.create_bias([{self.attn_config.hidden_size}], dtype=ttnn.bfloat16, device=self.device)",
                f"",
            ]
        )

        if self.attn_config.use_rotary_embeddings:
            class_lines.extend(
                [
                    f"        # Rotary embeddings",
                    f"        from ..embeddings import RotaryEmbedding",
                    f"        self.rotary_emb = RotaryEmbedding({self.attn_config.head_dim}, max_seq_len={self.hw_config.max_seq_len})",
                    f"",
                ]
            )

        # Add training flag
        class_lines.extend(
            [
                f"        self.training = False",
                f"",
            ]
        )

        # Add the forward function (with proper indentation)
        # Get the forward function
        forward_src = self.generate_forward_function()
        forward_lines = forward_src.split("\n")
        for line in forward_lines:
            if line.strip():
                class_lines.append(f"    {line}")
            else:
                class_lines.append("")

        return "\n".join(class_lines)


# This is the main API to create a attention module instance based on attn_config (model spec) and hw_config
def Attention(
    attn_config: AttentionConfig,
    *,
    hw_config: Optional[HardwareConfig] = None,
    # todo)) gen_format should be an enum
    gen_format: str = "class",
    # todo)) save_source should be a object that provides filename and etc
    save_source: bool = False,
    filename: str = "",
) -> Tuple[type, str]:
    """
    Main API to compile an attention module for specific hardware and configuration.

    Returns:
        - Compiled module class or pure function
          - class is good for debugging and testing as it can hold addtional metadata
          - function is good for production as it is more lightweight
        - Generated source code
    """

    # Create code generator
    codegen = TTTv2AttentionCodeGen(attn_config, hw_config)

    # Generate source code
    if gen_format == "class":
        source_code = codegen.generate_module_class()
    elif gen_format == "function":
        source_code = codegen.generate_forward_function()
    else:
        raise ValueError(f"Invalid generation format: {gen_format}")

    # Save source if requested
    if save_source:
        with open(filename, "w") as f:
            f.write("# Auto-generated by TTTv2 CodeGen\n")
            f.write("import torch\n")
            f.write("import torch.nn as nn\n")
            f.write("import ttnn\n\n")
            f.write(source_code)

    # Compile the source into a class
    namespace = {
        "nn": type("nn", (), {"Module": object}),  # Mock for demo
        "torch": type("torch", (), {}),
        "ttnn": type("ttnn", (), {}),
    }

    exec(source_code, namespace)

    # Find the generated class
    module_class = None
    for name, obj in namespace.items():
        if name.startswith("TTTv2Attention_"):
            module_class = obj
            break

    return module_class


# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Generate attention for Wormhole with specific config
    print("=== Example 1: Llama-style Attention for Wormhole ===")

    hw_config = HardwareConfig(
        device_name="wormhole_b0",
        grid_size=(8, 7),
        l1_memory_size=1024 * 1024,  # 1MB
        supports_flash_attention=True,
        fp32_accumulation=True,
        max_seq_len=2048,
        shard_strategy="block",
    )

    attn_config = AttentionConfig(
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        use_rotary_embeddings=True,
        use_sliding_window=False,
        window_size=None,
        dropout=0.1,
    )

    module_class = Attention(
        attn_config,
        hw_config=hw_config,
        gen_format="function",
        save_source=True,
        filename=f"attention_{hw_config.device_name}_{attn_config.hidden_size}_{attn_config.num_heads}.py",
    )

    print(f"Generated class: {module_class}")
    print("\nGenerated forward function preview:")
    print("-" * 60)
    # Show first 30 lines of forward function
    forward_start = module_class.source.find("def forward")
    forward_lines = module_class.source[forward_start:].split("\n")[:30]
    print("\n".join(forward_lines))
    print("... [truncated]\n")
