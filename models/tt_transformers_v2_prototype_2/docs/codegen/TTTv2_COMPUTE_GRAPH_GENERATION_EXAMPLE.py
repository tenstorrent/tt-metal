"""
TTTv2 Compute Graph Generation Example

This demonstrates how TTTv2 modules can generate compute graphs
that can be optimized, analyzed, and compiled for specific hardware.
"""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class OpType(Enum):
    """Supported operation types in compute graph"""

    LINEAR = "linear"
    MATMUL = "matmul"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    ADD = "add"
    SPLIT = "split"
    CONCAT = "concat"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"


@dataclass
class TensorInfo:
    """Information about a tensor in the graph"""

    shape: List[int]
    dtype: str = "bfloat16"
    memory_layout: str = "TILE_LAYOUT"
    sharding: Optional[Dict[str, Any]] = None


@dataclass
class ComputeNode:
    """Node in the compute graph"""

    id: str
    name: str
    op_type: OpType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    tensor_info: Optional[TensorInfo] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.op_type.value}_{uuid.uuid4().hex[:8]}"


@dataclass
class ComputeGraph:
    """Compute graph representation"""

    name: str
    nodes: Dict[str, ComputeNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    input_tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    output_tensors: Dict[str, TensorInfo] = field(default_factory=dict)

    def add_node(self, node: ComputeNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, from_node: str, to_node: str):
        """Add an edge between nodes"""
        self.edges.append((from_node, to_node))

    def get_node(self, node_id: str) -> Optional[ComputeNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order"""
        # Build adjacency list
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for from_node, to_node in self.edges:
            adj[from_node].append(to_node)
            in_degree[to_node] += 1

        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


class TTTGraphBuilder:
    """Builder for creating compute graphs from TTT modules"""

    def __init__(self, hardware_profile: Dict[str, Any]):
        self.hw_profile = hardware_profile
        self.current_graph = None

    def create_graph(self, name: str) -> ComputeGraph:
        """Create a new compute graph"""
        self.current_graph = ComputeGraph(name=name)
        return self.current_graph

    def add_linear(
        self,
        name: str,
        input_tensor: str,
        weight_shape: Tuple[int, int],
        bias: bool = True,
        activation: Optional[str] = None,
    ) -> str:
        """Add optimized linear layer to graph"""

        # Calculate optimal configuration
        config = self._optimize_linear_config(weight_shape)

        # Create linear node
        linear_node = ComputeNode(
            id=f"linear_{name}",
            name=name,
            op_type=OpType.LINEAR,
            inputs=[input_tensor, f"{name}_weight"],
            outputs=[f"{name}_output"],
            config={
                "weight_shape": weight_shape,
                "bias": bias,
                "program_config": config["program_config"],
                "compute_kernel_config": config["kernel_config"],
                "memory_config": config["memory_config"],
            },
        )

        if bias:
            linear_node.inputs.append(f"{name}_bias")

        node_id = self.current_graph.add_node(linear_node)

        # Add activation if specified
        if activation:
            act_node = ComputeNode(
                id=f"activation_{name}",
                name=f"{name}_activation",
                op_type=OpType.ACTIVATION,
                inputs=[f"{name}_output"],
                outputs=[f"{name}_activated"],
                config={"type": activation, "fused": config.get("can_fuse_activation", False)},
            )
            act_id = self.current_graph.add_node(act_node)
            self.current_graph.add_edge(node_id, act_id)
            return act_id

        return node_id

    def add_attention(
        self, name: str, hidden_dim: int, num_heads: int, seq_length: int, use_flash: bool = True
    ) -> ComputeGraph:
        """Create subgraph for attention layer"""

        attn_graph = ComputeGraph(name=f"{name}_attention")

        # Determine if we can use optimized attention
        can_use_flash = use_flash and self._can_use_flash_attention(seq_length, hidden_dim)

        if can_use_flash:
            # Single optimized flash attention node
            flash_node = ComputeNode(
                id=f"flash_attn_{name}",
                name=f"{name}_flash_attention",
                op_type=OpType.ATTENTION,
                inputs=["query", "key", "value"],
                outputs=["attention_output"],
                config={
                    "type": "flash_attention",
                    "num_heads": num_heads,
                    "hidden_dim": hidden_dim,
                    "causal": True,
                    "program_config": self._get_flash_attention_config(hidden_dim, num_heads),
                },
            )
            attn_graph.add_node(flash_node)
        else:
            # Decomposed attention with individual operations
            # Q @ K^T
            qk_matmul = ComputeNode(
                id=f"qk_matmul_{name}",
                name=f"{name}_qk_scores",
                op_type=OpType.MATMUL,
                inputs=["query", "key_transposed"],
                outputs=["scores"],
                config=self._optimize_matmul_config((seq_length, hidden_dim), (hidden_dim, seq_length)),
            )
            attn_graph.add_node(qk_matmul)

            # Softmax (handled as activation)
            softmax_node = ComputeNode(
                id=f"softmax_{name}",
                name=f"{name}_softmax",
                op_type=OpType.ACTIVATION,
                inputs=["scores"],
                outputs=["probs"],
                config={"type": "softmax", "dim": -1},
            )
            attn_graph.add_node(softmax_node)
            attn_graph.add_edge(qk_matmul.id, softmax_node.id)

            # Probs @ V
            pv_matmul = ComputeNode(
                id=f"pv_matmul_{name}",
                name=f"{name}_weighted_values",
                op_type=OpType.MATMUL,
                inputs=["probs", "value"],
                outputs=["attention_output"],
                config=self._optimize_matmul_config((seq_length, seq_length), (seq_length, hidden_dim)),
            )
            attn_graph.add_node(pv_matmul)
            attn_graph.add_edge(softmax_node.id, pv_matmul.id)

        return attn_graph

    def _optimize_linear_config(self, weight_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate optimal configuration for linear layer"""
        m, n = weight_shape

        # Simplified optimization based on hardware
        grid_size = self.hw_profile.get("compute_grid", (8, 7))
        cores_x = min(grid_size[0], (m + 127) // 128)
        cores_y = min(grid_size[1], (n + 127) // 128)

        return {
            "program_config": {
                "type": "MatmulMultiCoreReuseMultiCast",
                "grid_size": (cores_x, cores_y),
                "in0_block_w": 2,
                "out_subblock_h": 1,
                "out_subblock_w": 4,
                "per_core_M": (m + cores_x - 1) // cores_x,
                "per_core_N": (n + cores_y - 1) // cores_y,
            },
            "kernel_config": {"math_fidelity": "HiFi4", "fp32_dest_acc": True, "packer_l1_acc": True},
            "memory_config": {"memory_layout": "BLOCK_SHARDED", "buffer_type": "L1"},
            "can_fuse_activation": m * n < 1024 * 1024,  # Simple heuristic
        }

    def _optimize_matmul_config(self, a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate optimal configuration for matmul"""
        m, k1 = a_shape
        k2, n = b_shape
        assert k1 == k2, f"Shape mismatch: {k1} != {k2}"

        # Similar to linear but with different heuristics
        return self._optimize_linear_config((m, n))

    def _can_use_flash_attention(self, seq_length: int, hidden_dim: int) -> bool:
        """Check if flash attention is beneficial"""
        # Simplified heuristic
        return seq_length >= 128 and hidden_dim % 128 == 0 and self.hw_profile.get("supports_flash_attention", True)

    def _get_flash_attention_config(self, hidden_dim: int, num_heads: int) -> Dict[str, Any]:
        """Get optimized flash attention configuration"""
        return {
            "block_size": min(512, hidden_dim),
            "num_loops": 1,
            "use_causal_mask": True,
            "compute_kernel_config": {"math_fidelity": "HiFi4", "fp32_dest_acc": True},
        }


class GraphOptimizer:
    """Optimize compute graphs for specific hardware"""

    def __init__(self, hardware_profile: Dict[str, Any]):
        self.hw_profile = hardware_profile

    def optimize(self, graph: ComputeGraph) -> ComputeGraph:
        """Apply optimization passes to the graph"""

        # Apply various optimization passes
        graph = self._fuse_operations(graph)
        graph = self._optimize_memory_layout(graph)
        graph = self._schedule_operations(graph)

        return graph

    def _fuse_operations(self, graph: ComputeGraph) -> ComputeGraph:
        """Fuse compatible operations"""
        # Example: Fuse linear + activation when possible
        fused_graph = ComputeGraph(name=f"{graph.name}_fused")

        for node_id in graph.topological_sort():
            node = graph.get_node(node_id)

            # Check if this is a linear node followed by activation
            if node.op_type == OpType.LINEAR:
                # Find connected activation
                next_nodes = [graph.get_node(to_node) for from_node, to_node in graph.edges if from_node == node_id]

                activation_node = None
                for next_node in next_nodes:
                    if next_node and next_node.op_type == OpType.ACTIVATION:
                        activation_node = next_node
                        break

                if activation_node and node.config.get("can_fuse_activation", False):
                    # Create fused node
                    fused_node = ComputeNode(
                        id=f"fused_{node.id}_{activation_node.id}",
                        name=f"{node.name}_fused",
                        op_type=OpType.LINEAR,
                        inputs=node.inputs,
                        outputs=activation_node.outputs,
                        config={**node.config, "fused_activation": activation_node.config["type"]},
                    )
                    fused_graph.add_node(fused_node)
                else:
                    fused_graph.add_node(node)
            else:
                # Skip if already fused
                if node.op_type != OpType.ACTIVATION or not any(
                    n for n in fused_graph.nodes.values() if node.id in str(n.id)
                ):
                    fused_graph.add_node(node)

        return fused_graph

    def _optimize_memory_layout(self, graph: ComputeGraph) -> ComputeGraph:
        """Optimize tensor memory layouts"""
        # Analyze data flow and optimize layouts
        # (Simplified implementation)
        return graph

    def _schedule_operations(self, graph: ComputeGraph) -> ComputeGraph:
        """Schedule operations for optimal hardware utilization"""
        # Reorder operations for better pipelining
        # (Simplified implementation)
        return graph


class GraphCompiler:
    """Compile compute graph to executable format"""

    def __init__(self, hardware_profile: Dict[str, Any]):
        self.hw_profile = hardware_profile
        self.optimizer = GraphOptimizer(hardware_profile)

    def compile(self, graph: ComputeGraph, output_format: str = "ttnn") -> Union[str, Dict]:
        """Compile graph to target format"""

        # Optimize the graph
        optimized_graph = self.optimizer.optimize(graph)

        if output_format == "ttnn":
            return self._compile_to_ttnn(optimized_graph)
        elif output_format == "json":
            return self._compile_to_json(optimized_graph)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _compile_to_ttnn(self, graph: ComputeGraph) -> str:
        """Generate TTNN code from graph"""

        code = f'''"""
Auto-generated TTNN implementation from compute graph: {graph.name}
"""

import ttnn
import torch
from typing import Dict, Optional


class {graph.name.replace("-", "_")}:
    def __init__(self, device):
        self.device = device
        self.ops = {{}}
        self._initialize_ops()

    def _initialize_ops(self):
        """Initialize all operations with optimal configurations"""
'''

        # Generate initialization for each node
        for node_id in graph.topological_sort():
            node = graph.get_node(node_id)
            code += self._generate_node_init(node)

        # Generate forward method
        code += '''

    def forward(self, inputs: Dict[str, ttnn.Tensor]) -> Dict[str, ttnn.Tensor]:
        """Execute the compute graph"""
        tensors = inputs.copy()
'''

        # Generate execution for each node
        for node_id in graph.topological_sort():
            node = graph.get_node(node_id)
            code += self._generate_node_execution(node)

        code += """
        return tensors
"""

        return code

    def _generate_node_init(self, node: ComputeNode) -> str:
        """Generate initialization code for a node"""
        if node.op_type == OpType.LINEAR:
            return f"""
        self.ops["{node.id}"] = {{
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.{node.config['kernel_config']['math_fidelity']},
                fp32_dest_acc_en={node.config['kernel_config']['fp32_dest_acc']},
                packer_l1_acc={node.config['kernel_config']['packer_l1_acc']}
            ),
            "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size={node.config['program_config']['grid_size']},
                in0_block_w={node.config['program_config']['in0_block_w']},
                out_subblock_h={node.config['program_config']['out_subblock_h']},
                out_subblock_w={node.config['program_config']['out_subblock_w']},
                per_core_M={node.config['program_config']['per_core_M']},
                per_core_N={node.config['program_config']['per_core_N']},
                fused_activation={node.config.get('fused_activation', 'None')}
            )
        }}
"""
        return ""

    def _generate_node_execution(self, node: ComputeNode) -> str:
        """Generate execution code for a node"""
        if node.op_type == OpType.LINEAR:
            weight_input = node.inputs[1]
            bias_input = node.inputs[2] if len(node.inputs) > 2 else "None"
            return f"""
        tensors["{node.outputs[0]}"] = ttnn.linear(
            tensors["{node.inputs[0]}"],
            tensors["{weight_input}"],
            bias={"tensors['" + bias_input + "']" if bias_input != "None" else "None"},
            compute_kernel_config=self.ops["{node.id}"]["compute_kernel_config"],
            program_config=self.ops["{node.id}"]["program_config"]
        )
"""
        elif node.op_type == OpType.ATTENTION:
            return f"""
        tensors["{node.outputs[0]}"] = ttnn.transformer.flash_attention(
            tensors["{node.inputs[0]}"],  # query
            tensors["{node.inputs[1]}"],  # key
            tensors["{node.inputs[2]}"],  # value
            is_causal={node.config.get('causal', True)},
            **self.ops["{node.id}"]
        )
"""
        return ""

    def _compile_to_json(self, graph: ComputeGraph) -> Dict:
        """Export graph as JSON for analysis or other tools"""
        return {
            "name": graph.name,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.op_type.value,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "config": node.config,
                }
                for node in graph.nodes.values()
            ],
            "edges": graph.edges,
            "optimization_metadata": {
                "hardware": self.hw_profile["device_name"],
                "optimizations_applied": ["fusion", "memory_layout", "scheduling"],
            },
        }


# Example usage
if __name__ == "__main__":
    # Hardware profile
    hw_profile = {
        "device_name": "wormhole_b0",
        "compute_grid": (8, 7),
        "supports_flash_attention": True,
        "l1_memory_per_core": 1024 * 1024,
    }

    # Build a transformer layer graph
    builder = TTTGraphBuilder(hw_profile)
    graph = builder.create_graph("transformer_layer")

    # Add attention subgraph
    attn_graph = builder.add_attention(
        name="self_attention", hidden_dim=4096, num_heads=32, seq_length=2048, use_flash=True
    )

    # Add FFN
    builder.add_linear("ffn_up", "attention_output", (4096, 11008), activation="gelu")
    builder.add_linear("ffn_down", "ffn_up_activated", (11008, 4096))

    # Compile the graph
    compiler = GraphCompiler(hw_profile)

    # Generate TTNN code
    ttnn_code = compiler.compile(graph, output_format="ttnn")
    print("Generated TTNN code:")
    print("=" * 80)
    print(ttnn_code[:1500] + "\n... [truncated] ...")

    # Export as JSON
    json_graph = compiler.compile(graph, output_format="json")
    print("\n\nGraph structure (JSON):")
    print("=" * 80)
    print(json.dumps(json_graph, indent=2)[:1000] + "\n... [truncated] ...")
