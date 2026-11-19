# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 DRAM-sharded native linear-attention projection experiment.

This changes ordered-scan arithmetic to GEMM and therefore requires real-weight
PCC/state/layer34 qualification. It neither installs itself nor changes normal
decoder defaults. All weight conversion/configuration is construction-time.
"""

from types import MethodType, SimpleNamespace


def _input_weights(state_dict, packed):
    """Unpermuted HF coefficients, retaining the TP4 head and padding contract."""
    import torch

    q, k, v = state_dict["linear_attn.in_proj_qkv.weight"].split((2048, 2048, 4096), dim=0)
    z = state_dict["linear_attn.in_proj_z.weight"]
    a, b = (state_dict[f"linear_attn.in_proj_{name}.weight"] for name in ("a", "b"))
    state_parts, output_parts = [], []
    for rank in range(4):
        keys = slice(rank * 512, (rank + 1) * 512)
        values = slice(rank * 1024, (rank + 1) * 1024)
        scalars = slice(rank * 8, (rank + 1) * 8)
        state_parts.append(
            torch.cat(
                (
                    k[keys],
                    v[values],
                    torch.nn.functional.pad(a[scalars], (0, 0, 0, 24)),
                    torch.nn.functional.pad(b[scalars], (0, 0, 0, 24)),
                ),
                dim=0,
            ).T
        )
        output_parts.append(torch.cat((q[keys], z[values]), dim=0).T)
    if packed:
        return {
            "attention_combined": torch.cat(
                [torch.cat((state, output), dim=1) for state, output in zip(state_parts, output_parts)], dim=1
            )
        }
    return {
        "attention_packed": torch.cat(state_parts, dim=1),
        "output_only_packed": torch.cat(output_parts, dim=1),
    }


def install(dec, state_dict, geometry=None, fidelity="HiFi4"):
    """Install experimental singleton projections on this decoder instance.

    ``geometry`` accepts the existing builder's ``default`` and per-role dicts:
    ``attention_combined`` (or split ``attention_packed``/``output_only_packed``)
    and ``attention_out``. Legal readers are 1/2/3, K blocks divide K/32, and
    input core counts divide K/32. Baseline geometry: input 32 cores (matching
    native replicated norm's 32x64 shards), output 8 cores (matching the existing
    local-SiLU 32x128 shards), three readers/bank and K-block 4. Output partials
    remain FP32 until the existing TP4 collective.
    """
    import torch

    import ttnn

    from .optimized_decoder import OptimizedDecoder

    if dec.kind != "linear_attention":
        return dec
    if fidelity not in ("HiFi4", "HiFi2", "LoFi"):
        raise ValueError("unsupported native projection fidelity")
    compute = ttnn.init_device_compute_kernel_config(
        dec.mesh_device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    if hasattr(dec, "mesh_native_projection_probe"):
        raise ValueError("native DRAM projection probe already installed")
    if tuple(dec.mesh_device.shape) != (1, 4):
        raise ValueError("native projection probe requires TP4 on the target 1x4 mesh")
    supplied = {} if geometry is None else geometry
    if not isinstance(supplied, dict) or any(not isinstance(v, dict) for v in supplied.values()):
        raise ValueError("geometry must map role names to builder-option dictionaries")
    raw = _input_weights(state_dict, dec.pack_linear)
    expected = (
        {"attention_combined": 12544} if dec.pack_linear else {"attention_packed": 6400, "output_only_packed": 6144}
    )
    for key, width in expected.items():
        if tuple(raw[key].shape) != (2048, width):
            raise ValueError(f"unexpected global packed projection shape for {key}: {tuple(raw[key].shape)}")
    output = state_dict["linear_attn.out_proj.weight"].T
    if tuple(output.shape) != (4096, 2048):
        raise ValueError("expected unpermuted full output weight [4096,2048]")
    raw["linear_attn.out_proj.weight"] = output
    allowed_roles = {"default", "attention_out", *expected}
    if set(supplied) - allowed_roles:
        raise ValueError(f"unknown geometry roles: {sorted(set(supplied) - allowed_roles)}")

    # Reuse the validated local builder without changing the decoder's existing
    # shared/full program configs or geometry dictionary. New BF16 projection
    # weights are experimental additional residency until the parent promotes
    # and budgets/replaces the superseded representation.
    defaults = dict(readers=3, block_w=4, per_core_n=2, bank_tile_multiple=1)
    selected = {}
    for key in raw:
        role = "attention_out" if key == "linear_attn.out_proj.weight" else key
        spec = dict(defaults, input_cores=8 if role == "attention_out" else 32)
        spec.update(supplied.get("default", {}))
        spec.update(supplied.get(role, {}))
        selected[role] = spec
    builder = SimpleNamespace(mesh_device=dec.mesh_device, projection_geometry=selected, effective_geometry={})
    projections = {}
    for key, weight in raw.items():
        role = "attention_out" if key == "linear_attn.out_proj.weight" else key
        interleaved = ttnn.from_torch(
            weight.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dec.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(dec.mesh_device, dim=0 if role == "attention_out" else 1),
        )
        projections[key] = OptimizedDecoder._make_dram_projection(builder, interleaved, role)
    original = dec.linear

    def linear(
        self, x, key, dtype=ttnn.bfloat16, *, projection_k=512, ordered_out=False, terminal_scan=False, **kwargs
    ):
        if x.shape[-2] != 1 or projection_k != 512 or key not in projections:
            return original(
                x, key, dtype, projection_k=projection_k, ordered_out=ordered_out, terminal_scan=terminal_scan, **kwargs
            )
        is_output = key == "linear_attn.out_proj.weight"
        if not is_output and dtype != ttnn.bfloat16:
            raise ValueError("native input projection requires BF16 output")
        weight, input_memory, output_memory, program = projections[key]
        result = ttnn.linear(
            ttnn.to_memory_config(x, input_memory),
            weight,
            dtype=ttnn.float32 if is_output else ttnn.bfloat16,
            memory_config=output_memory,
            program_config=program,
            compute_kernel_config=compute,
        )
        # Consumers currently slice the packed input or convert output for CCL;
        # preserve their interleaved contract and charge the conversion here.
        return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)

    dec.mesh_native_projection_probe = projections
    dec.linear = MethodType(linear, dec)
    dec.precision_policy["mesh_native_projection_probe"] = {
        "experimental": True,
        "weight_dtype": "BF16",
        "input_output_dtype": "BF16",
        "output_partial_dtype": "FP32",
        "compute": f"{fidelity} FP32 accumulation; approximation disabled",
        "effective_geometry": builder.effective_geometry,
        "packing": "unpermuted TP4 state 1600 + QZ 1536" if dec.pack_linear else "unpermuted TP4 separate state/QZ",
        "scope": "singleton projection_k 512; other paths call original method",
        "extra_weight_bytes_per_rank": sum(value.numel() * 2 // 4 for value in raw.values()),
        "extra_weight_padded_payload_bytes_per_rank": sum(
            spec["k"] * ((spec["n"] + unit - 1) // unit * unit) * 2
            for spec in builder.effective_geometry.values()
            for unit in (32 * spec["bank_tile_multiple"] * dec.mesh_device.dram_grid_size().x,)
        ),
    }
    return dec
