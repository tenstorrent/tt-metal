"""
TTTv2 Extension Examples

This file demonstrates various ways to extend TTTv2, from simple custom modules
to complex community contributions.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tt_transformers_v2.attention import BaseAttention

# =============================================================================
# EXAMPLE 1: Simple Custom Module (No Registration)
# =============================================================================


class LinearAttention(BaseAttention):
    """
    Simple linear attention - no registration needed!
    This can be used immediately in any model.
    """

    def __init__(self, hidden_dim: int, num_heads: int, device=None):
        super().__init__(hidden_dim, num_heads, device)
        self.temperature = (hidden_dim // num_heads) ** 0.5

    def compute_attention(self, q, k, v, mask=None):
        # Linear attention: softmax(Q)softmax(K)^T V
        # Shape: [batch, heads, seq_len, head_dim]

        # Apply feature map
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # Compute KV first (linear complexity)
        # [batch, heads, head_dim, head_dim]
        kv = torch.einsum("bhsd,bhse->bhde", k, v)

        # Apply Q
        # [batch, heads, seq_len, head_dim]
        out = torch.einsum("bhsd,bhde->bhse", q, kv)

        # Normalize
        k_sum = k.sum(dim=-2, keepdim=True)
        out = out / (torch.einsum("bhsd,bhd->bhs", q, k_sum).unsqueeze(-1) + 1e-6)

        return out


# Direct usage - no registration needed!
linear_attn = LinearAttention(hidden_dim=1024, num_heads=8)
output = linear_attn(input_tensor)


# =============================================================================
# EXAMPLE 2: Production-Ready Module with Registration
# =============================================================================

from tt_transformers_v2.registry import register_module


@register_module(
    name="rotary-linear-attention",
    category="attention",
    description="Linear attention with RoPE embeddings and optimized kernels",
    author="TTT Research Team",
    version="1.0.0",
    hardware_support=["cuda", "ttnn"],
    tags=["production", "efficient", "long-context"],
    alias="rola",  # Short name for CLI
)
class RotaryLinearAttention(BaseAttention):
    """
    Production-ready linear attention with rotary embeddings.

    Features:
    - O(n) complexity for long sequences
    - Built-in rotary position embeddings
    - Optimized CUDA and TTNN kernels
    - Supports up to 1M context length
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 1_000_000,
        rope_theta: float = 10_000.0,
        feature_map: str = "elu",
        device=None,
    ):
        super().__init__(hidden_dim, num_heads, device)
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.feature_map = feature_map

        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        # Register optimized kernels if available
        self._register_kernels()

    def _register_kernels(self):
        """Register hardware-specific optimized kernels"""
        if self.device.type == "cuda":
            try:
                import rotary_linear_attention_cuda

                self.optimized_kernel = rotary_linear_attention_cuda.forward
            except ImportError:
                self.optimized_kernel = None
        elif self.device.type == "ttnn":
            # TTNN specific kernel
            self.optimized_kernel = ttnn.ops.rotary_linear_attention
        else:
            self.optimized_kernel = None

    def compute_attention(self, q, k, v, mask=None):
        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k)

        # Use optimized kernel if available
        if self.optimized_kernel is not None:
            return self.optimized_kernel(q, k, v, self.feature_map)

        # Fallback to standard implementation
        return self._compute_attention_fallback(q, k, v, mask)

    def _compute_attention_fallback(self, q, k, v, mask=None):
        # Standard linear attention with feature maps
        if self.feature_map == "elu":
            q = torch.nn.functional.elu(q) + 1
            k = torch.nn.functional.elu(k) + 1
        elif self.feature_map == "relu":
            q = torch.nn.functional.relu(q)
            k = torch.nn.functional.relu(k)

        # ... rest of linear attention logic
        return output

    @classmethod
    def from_pretrained(cls, model_id: str, device=None):
        """Load pretrained rotary linear attention weights"""
        # Load from model hub
        config = load_config(model_id)
        model = cls(**config, device=device)
        model.load_state_dict(load_weights(model_id))
        return model


# =============================================================================
# EXAMPLE 3: Complex Extension - Mixture of Experts FFN
# =============================================================================

from tt_transformers_v2.ffn import BaseFFN
from tt_transformers_v2.registry import register_module


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""

    hidden_dim: int
    expert_dim: int
    num_experts: int
    num_experts_per_token: int
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    device: Optional[str] = None


@register_module(
    name="mixture-of-experts-ffn",
    category="ffn",
    description="Sparse MoE FFN with load balancing and auxiliary losses",
    author="Community",
    version="2.0.0",
    hardware_support=["cuda", "ttnn"],
    tags=["sparse", "moe", "efficient"],
    performance_hints={
        "memory": "Scales with num_experts",
        "compute": "Constant per token (num_experts_per_token)",
        "recommended_num_experts": "8-64 for typical models",
    },
)
class MixtureOfExpertsFFN(BaseFFN):
    """
    Mixture of Experts FFN layer.

    Each token is routed to top-k experts based on a learned router.
    Includes load balancing and auxiliary losses for stable training.
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config.hidden_dim, config.expert_dim, config.device)
        self.config = config

        # Router network
        self.router = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.expert_dim),
                    nn.ReLU(),
                    nn.Linear(config.expert_dim, config.hidden_dim),
                )
                for _ in range(config.num_experts)
            ]
        )

        # For load balancing
        self.register_buffer("expert_counts", torch.zeros(config.num_experts))

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # Compute router logits
        router_logits = self.router(x)  # [batch, seq, num_experts]

        # Select top-k experts per token
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = router_probs.topk(self.config.num_experts_per_token, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Route tokens to experts
        output = torch.zeros_like(x)

        for i in range(self.config.num_experts_per_token):
            # Get expert index for each token
            expert_idx = top_k_indices[..., i]  # [batch, seq]

            # Process each expert
            for e in range(self.config.num_experts):
                # Find tokens routed to this expert
                mask = expert_idx == e

                if mask.any():
                    # Extract tokens for this expert
                    expert_input = x[mask]

                    # Apply expert
                    expert_output = self.experts[e](expert_input)

                    # Weight by routing probability
                    weight = top_k_probs[..., i][mask].unsqueeze(-1)
                    output[mask] += weight * expert_output

                    # Track load balancing
                    self.expert_counts[e] += mask.sum().float()

        # Compute auxiliary losses
        self.aux_loss = self._compute_aux_losses(router_logits, top_k_indices)

        return output

    def _compute_aux_losses(self, router_logits, expert_indices):
        """Compute load balancing and router z-loss"""
        # Load balancing loss
        expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.config.num_experts).float()
        ideal_count = expert_indices.numel() / self.config.num_experts
        load_balance_loss = ((expert_counts - ideal_count) ** 2).mean()

        # Router z-loss (encourages router to be decisive)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).mean()

        total_aux_loss = (
            self.config.router_aux_loss_coef * load_balance_loss + self.config.router_z_loss_coef * router_z_loss
        )

        return total_aux_loss

    def get_expert_statistics(self):
        """Get statistics about expert utilization"""
        total_tokens = self.expert_counts.sum()
        if total_tokens == 0:
            return {}

        return {
            "expert_usage": (self.expert_counts / total_tokens).tolist(),
            "load_balance_score": 1.0 - self.expert_counts.std() / self.expert_counts.mean(),
            "total_tokens_routed": total_tokens.item(),
        }


# =============================================================================
# EXAMPLE 4: Hardware-Specific Extension
# =============================================================================

from tt_transformers_v2.normalization import BaseNorm

import ttnn  # TensTorrent NN library


@register_module(
    name="ttnn-fused-rmsnorm",
    category="normalization",
    description="Fused RMSNorm optimized for TensTorrent hardware",
    author="TT Team",
    version="1.0.0",
    hardware_support=["ttnn"],  # TTNN only!
    tags=["hardware-optimized", "fused", "ttnn"],
)
class TTNNFusedRMSNorm(BaseNorm):
    """
    RMSNorm with fused operations for TensTorrent hardware.

    Fuses:
    - Square root and division
    - Elementwise multiplication with learnable scale
    - Optional residual addition
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        fuse_residual: bool = True,
        device="ttnn",
    ):
        super().__init__(hidden_dim, device)
        self.eps = eps
        self.fuse_residual = fuse_residual

        if elementwise_affine:
            self.weight = ttnn.Parameter(ttnn.ones([1, 1, hidden_dim], device=device))
        else:
            self.register_parameter("weight", None)

        # Pre-compile the fused kernel
        self._compile_kernel()

    def _compile_kernel(self):
        """Compile the fused RMSNorm kernel for TTNN"""

        # Define the fused operation graph
        @ttnn.compile
        def fused_rmsnorm_kernel(x, weight, eps, residual=None):
            # Compute RMS
            variance = ttnn.mean(ttnn.square(x), dim=-1, keepdim=True)
            rms = ttnn.rsqrt(variance + eps)

            # Normalize
            x_norm = x * rms

            # Apply weight
            if weight is not None:
                x_norm = x_norm * weight

            # Add residual if requested
            if residual is not None:
                x_norm = x_norm + residual

            return x_norm

        self.kernel = fused_rmsnorm_kernel

    def forward(self, x, residual=None):
        if not self.fuse_residual:
            residual = None

        return self.kernel(x, self.weight, self.eps, residual)

    @staticmethod
    def benchmark(hidden_dims=[1024, 2048, 4096, 8192]):
        """Benchmark fused vs standard RMSNorm"""
        results = {}

        for dim in hidden_dims:
            # Create inputs
            x = ttnn.randn([32, 512, dim], device="ttnn")

            # Standard RMSNorm
            standard_norm = RMSNorm(dim, device="ttnn")
            t0 = time.time()
            _ = standard_norm(x)
            ttnn.synchronize()
            standard_time = time.time() - t0

            # Fused RMSNorm
            fused_norm = TTNNFusedRMSNorm(dim, device="ttnn")
            t0 = time.time()
            _ = fused_norm(x)
            ttnn.synchronize()
            fused_time = time.time() - t0

            results[dim] = {
                "standard_ms": standard_time * 1000,
                "fused_ms": fused_time * 1000,
                "speedup": standard_time / fused_time,
            }

        return results


# =============================================================================
# EXAMPLE 5: Using Extensions in Models
# =============================================================================


def build_custom_model():
    """Example of building a model with custom extensions"""

    from tt_transformers_v2 import ModelBuilder

    # Method 1: Direct usage (no registration needed)
    model = (
        ModelBuilder("my_model")
        .add_embedding(vocab_size=50000, hidden_dim=1024)
        .add_layer(
            attention_cls=LinearAttention,  # Our custom class
            attention_config={"hidden_dim": 1024, "num_heads": 8},
            ffn_cls=BaseFFN,  # Standard FFN
            ffn_config={"hidden_dim": 1024, "intermediate_dim": 4096},
        )
        .build()
    )

    # Method 2: Using registered modules by name
    model = (
        ModelBuilder("my_model")
        .add_embedding(vocab_size=50000, hidden_dim=1024)
        .add_layer(
            attention="rotary-linear-attention",  # Registered name
            attention_config={"hidden_dim": 1024, "num_heads": 8},
            ffn="mixture-of-experts-ffn",  # Registered MoE
            ffn_config=MoEConfig(hidden_dim=1024, expert_dim=4096, num_experts=8, num_experts_per_token=2),
        )
        .build()
    )

    # Method 3: Mix and match
    model = (
        ModelBuilder("my_model")
        .add_embedding(vocab_size=50000, hidden_dim=1024)
        .add_layer(
            attention="rola",  # Using alias
            attention_config={"hidden_dim": 1024, "num_heads": 8},
            ffn=MyCustomFFN,  # Unregistered custom class
            ffn_config={"hidden_dim": 1024},
            norm="ttnn-fused-rmsnorm" if device == "ttnn" else RMSNorm,
        )
        .repeat_layers(24)
        .build()
    )

    return model


# =============================================================================
# EXAMPLE 6: Community Module Hub Integration
# =============================================================================


class ModuleHub:
    """
    Integration with community module hub for sharing extensions.
    """

    @staticmethod
    def publish(module_class, hub_token: str):
        """Publish a module to the community hub"""
        if not hasattr(module_class, "_registry_metadata"):
            raise ValueError("Module must be registered before publishing")

        metadata = module_class._registry_metadata

        # Package module
        package = {
            "metadata": metadata,
            "source_code": inspect.getsource(module_class),
            "requirements": extract_requirements(module_class),
            "tests": extract_tests(module_class),
        }

        # Upload to hub
        response = hub_api.upload(package, token=hub_token)
        print(f"Published {metadata.name} to hub: {response['url']}")

    @staticmethod
    def install(module_id: str):
        """Install a module from the hub"""
        # Download module
        package = hub_api.download(module_id)

        # Verify and install
        verify_package_safety(package)
        install_requirements(package["requirements"])

        # Register module
        exec(package["source_code"], globals())
        print(f"Installed {module_id}")

    @staticmethod
    def search(query: str, tags: List[str] = None):
        """Search the hub for modules"""
        results = hub_api.search(query, tags=tags)

        for result in results:
            print(f"{result['name']} by {result['author']}")
            print(f"  {result['description']}")
            print(f"  Downloads: {result['downloads']}")
            print(f"  Install: ttt hub install {result['id']}")


# Usage:
# $ ttt hub publish my_attention_module --token $HUB_TOKEN
# $ ttt hub search "efficient attention"
# $ ttt hub install community/linear-attention-v2
