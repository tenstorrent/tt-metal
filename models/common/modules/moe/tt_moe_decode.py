# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configurable Mixture-of-Experts decode block (`TTMoEDecode`).

A single forward step of a decode-time MoE layer on a 2D device mesh, wrapping the
ttnn `all_to_all_dispatch_metadata` → `moe_compute` → `deepseek_moe_fast_reduce_nc_fused`
→ reduce-scatter pipeline. All op kwargs (memory configs, cluster topology, splits,
shared-expert plumbing) are driven by `TTMoEDecodeConfig`, which derives sane defaults
from a minimal YAML per model. This module's job is just to wire the configured pieces
together and expose a `forward(x, scores, indices)` interface.

Pipeline overview (one decode step):
    1. `all_to_all_dispatch_metadata`: route each token to its `select_experts_k` chosen
       routed experts (cross-cluster send) plus all shared experts (local broadcast).
    2. `moe_compute`: per-device per-token-slot matmul `x @ w0`, `x @ w1`, activation
       (SiLU/SWIGLU/GELU), `intermediate @ w2`, optionally with bias.
    3. `deepseek_moe_fast_reduce_nc_fused`: score-weighted combine of the per-expert
       outputs back to per-token results, with a fixed scalar `shared_expert_scale`
       applied to shared-expert contributions.
    4. Reduce-scatter across the replicated mesh axis to produce per-device output
       chunks of width `hidden_size / num_replicated`.

Weight ownership: routed experts are sharded across the dispatch axis; shared experts
are replicated. Weights upload as `bfloat4_b` (with bf16 intermediate tiles for bias).

The two private classes `_TTMoEDecodeExpertState` and `_TTMoEDecodeBuffers` exist to
keep weight/mapping init and per-iteration scratch separated from the forward logic.
"""

from __future__ import annotations

import torch
from loguru import logger
from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
    map_shared_experts,
)

import ttnn
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig


def _tt_to_torch_dtype(tt_dtype):
    """Map a ttnn dtype to the closest host torch dtype for buffer allocation.

    Only the dtypes this module actually uses are handled — `bfloat8_b` falls back to
    `torch.bfloat16` since torch has no native 8-bit float; the host tensor is just a
    placeholder that gets reinterpreted at upload time.
    """
    if tt_dtype == ttnn.bfloat16 or tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    if tt_dtype == ttnn.float32:
        return torch.float32
    if tt_dtype == ttnn.uint16:
        return torch.uint16
    raise ValueError(f"Unsupported tt dtype: {tt_dtype}")


class _TTMoEDecodeExpertState:
    """Owns per-layer routed + shared expert weights and the global expert-mapping table.

    Holds three uploaded ttnn tensors after init:
    - `tt_expert_mapping`: `[num_devices, num_experts]` lookup of which linearized mesh
      coord owns each expert, replicated to every device. Used by both `dispatch` and
      `fast_reduce` ops.
    - `tt_w0_w1`: interleaved-and-tile-reordered w0/w1 weights for `moe_compute`'s
      consumption, sharded across mesh devices along the expert dim. `bfloat4_b`.
    - `tt_w2`: same idea, but with the ring-rotated N-tile layout w2 needs. `bfloat4_b`.

    Bias support: when `has_bias=True`, biases are folded into the same prepared weight
    tensors via `prepare_w0_w1_tensor_with_bias` / `prepare_w2_tensor_with_bias` (one
    extra K/N tile each). Bias + shared experts together is not supported (raises).
    """

    def _load_weights(self):
        # TODO (AM) eventually support loading weights from path
        pass

    @staticmethod
    def _validate(
        torch_w0: "torch.Tensor",
        torch_w1: "torch.Tensor",
        torch_w2: "torch.Tensor",
        mesh_device: ttnn.MeshDevice,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
        has_bias: bool,
        num_routed_experts: int,
        expert_mapping: list[int],
        num_shared_experts: int,
        shared_expert_ids_to_devices: dict[int, list[int]] | None,
        shared_id_to_torch_w0: dict[int, "torch.Tensor"] | None,
        shared_id_to_torch_w1: dict[int, "torch.Tensor"] | None,
        shared_id_to_torch_w2: dict[int, "torch.Tensor"] | None,
        torch_b0: "torch.Tensor" | None,
        torch_b1: "torch.Tensor" | None,
        torch_b2: "torch.Tensor" | None,
    ) -> None:
        """Fail fast on (weights, biases, shared experts) ↔ config mismatch.

        Cross-checks tensor shapes against each other (w0 is the source of truth for
        `L`, `H`, `N`) and against the config-derived counts. Catches the common
        misconfigurations that otherwise surface as opaque shape errors deep inside
        the bf4 preparers or kernels.
        """
        # --- mesh / topology sanity ---
        if cluster_axis not in (0, 1):
            raise ValueError(f"cluster_axis must be 0 or 1, got {cluster_axis}")
        num_devices = mesh_device.get_num_devices()
        if mesh_shape[0] * mesh_shape[1] != num_devices:
            raise ValueError(
                f"mesh_shape {mesh_shape} (= {mesh_shape[0] * mesh_shape[1]} devices) does "
                f"not match mesh_device.get_num_devices() = {num_devices}"
            )
        if num_routed_experts % num_devices != 0:
            raise ValueError(
                f"num_routed_experts ({num_routed_experts}) must be divisible by num_devices ({num_devices})"
            )

        # --- routed weight shape cross-checks (w0 is the source of truth for L, H, N) ---
        if torch_w0.ndim != 4:
            raise ValueError(f"torch_w0 must be 4D [L, E, H, N], got shape {tuple(torch_w0.shape)}")
        L, E, H, N = torch_w0.shape
        if E != num_routed_experts:
            raise ValueError(f"torch_w0.shape[1] = {E} does not match num_routed_experts ({num_routed_experts})")
        if tuple(torch_w1.shape) != (L, E, H, N):
            raise ValueError(f"torch_w1 shape {tuple(torch_w1.shape)} must match torch_w0 shape ({L}, {E}, {H}, {N})")
        if tuple(torch_w2.shape) != (L, E, N, H):
            raise ValueError(f"torch_w2 shape {tuple(torch_w2.shape)} must be (L, E, N, H) = ({L}, {E}, {N}, {H})")

        # --- expert_mapping ---
        if len(expert_mapping) != num_routed_experts:
            raise ValueError(
                f"expert_mapping length {len(expert_mapping)} != num_routed_experts ({num_routed_experts})"
            )
        bad = [(e, d) for e, d in enumerate(expert_mapping) if not (0 <= d < num_devices)]
        if bad:
            raise ValueError(f"expert_mapping has out-of-range device ids (0..{num_devices - 1}): {bad[:5]}")

        # --- bias presence and shapes ---
        bias_tensors = (torch_b0, torch_b1, torch_b2)
        any_bias = any(b is not None for b in bias_tensors)
        all_bias = all(b is not None for b in bias_tensors)
        if has_bias and not all_bias:
            missing = [name for name, b in zip(("b0", "b1", "b2"), bias_tensors) if b is None]
            raise ValueError(f"has_bias=True but {missing} not provided")
        if not has_bias and any_bias:
            raise ValueError("has_bias=False but one or more of torch_b0/b1/b2 was provided")
        if has_bias:
            if tuple(torch_b0.shape) != (L, E, N) or tuple(torch_b1.shape) != (L, E, N):
                raise ValueError(
                    f"b0/b1 must be (L, E, N) = ({L}, {E}, {N}); got "
                    f"{tuple(torch_b0.shape)} and {tuple(torch_b1.shape)}"
                )
            if tuple(torch_b2.shape) != (L, E, H):
                raise ValueError(f"b2 must be (L, E, H) = ({L}, {E}, {H}); got {tuple(torch_b2.shape)}")

        # --- shared experts ---
        if num_shared_experts > 0:
            if shared_expert_ids_to_devices is None:
                raise ValueError(f"num_shared_experts={num_shared_experts} but shared_expert_ids_to_devices is None")
            if len(shared_expert_ids_to_devices) != num_shared_experts:
                raise ValueError(
                    f"shared_expert_ids_to_devices has {len(shared_expert_ids_to_devices)} entries "
                    f"but num_shared_experts={num_shared_experts}"
                )

            shared_experts_per_device = [0] * num_devices
            for edl in shared_expert_ids_to_devices.values():
                for d in edl:
                    shared_experts_per_device[d] += 1
            if len(set(shared_experts_per_device)) > 1 or 0 in shared_experts_per_device:
                raise ValueError(
                    "Every device, should have the same number of, and at least 1 shared expert:"
                    f" {shared_experts_per_device=}"
                )

            expected_ids = set(range(num_routed_experts, num_routed_experts + num_shared_experts))
            if set(shared_expert_ids_to_devices.keys()) != expected_ids:
                raise ValueError(
                    f"shared expert ids must be contiguous after routed: expected {sorted(expected_ids)}, "
                    f"got {sorted(shared_expert_ids_to_devices.keys())}"
                )
            shared_dicts = (shared_id_to_torch_w0, shared_id_to_torch_w1, shared_id_to_torch_w2)
            if any(d is None for d in shared_dicts):
                raise ValueError("num_shared_experts > 0 but shared_id_to_torch_w0/w1/w2 not all provided")
            for name, d in zip(("w0", "w1", "w2"), shared_dicts):
                if set(d.keys()) != expected_ids:
                    raise ValueError(
                        f"shared_id_to_torch_{name} keys {sorted(d.keys())} != "
                        f"shared expert ids {sorted(expected_ids)}"
                    )
                expected_shape = (L, 1, N, H) if name == "w2" else (L, 1, H, N)
                for sid, t in d.items():
                    if tuple(t.shape) != expected_shape:
                        raise ValueError(
                            f"shared_id_to_torch_{name}[{sid}] shape {tuple(t.shape)} != expected {expected_shape}"
                        )
            if has_bias:
                # Mirrors the runtime check in __init__; surface it here too so it fails
                # before any device upload.
                raise NotImplementedError("bias + shared experts is not yet supported")
        else:
            if shared_expert_ids_to_devices:
                raise ValueError(
                    f"num_shared_experts=0 but shared_expert_ids_to_devices is non-empty: "
                    f"{shared_expert_ids_to_devices}"
                )
            if any(d is not None for d in (shared_id_to_torch_w0, shared_id_to_torch_w1, shared_id_to_torch_w2)):
                raise ValueError("num_shared_experts=0 but one or more shared_id_to_torch_w* dicts were provided")

    @staticmethod
    def _init_expert_mapping(
        torch_expert_mapping: "torch.Tensor",
        shared_expert_ids_to_devices: dict[int, list[int]] | None,
        mesh_device: ttnn.MeshDevice,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
    ):
        """Build and upload the `[num_devices, num_experts]` expert-mapping table.

        `mapping[d, e]` = linearized mesh coord of the device that owns expert `e`.
        Routed experts have the same value across all source rows `d` (ownership is
        global), so we just `repeat` the 1D input. Shared experts let different source
        devices pick different replicas based on cluster distance — `map_shared_experts`
        rewrites those columns to the nearest replica per source row.

        Matches the "new format" used by `test_all_to_all_dispatch_metadata_6U.py` and
        `gen_expert_mapping` in `test_moe_compute_6U.py`. Replicated to every device
        because both dispatch and fast-reduce need the full lookup locally.
        """
        if torch_expert_mapping.ndim != 1:
            raise ValueError(
                f"expected 1D expert_mapping (length=num_experts), got shape {tuple(torch_expert_mapping.shape)}"
            )
        num_devices = mesh_device.get_num_devices()
        mapping_2d = torch_expert_mapping.to(torch.int32).unsqueeze(0).repeat(num_devices, 1)

        if shared_expert_ids_to_devices is not None:
            mapping_2d = map_shared_experts(mapping_2d, shared_expert_ids_to_devices, mesh_shape, cluster_axis)

        return ttnn.from_torch(
            mapping_2d,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def _device_reorder_weights(
        torch_expert_mapping: "torch.Tensor",
        torch_w0: "torch.Tensor",
        torch_w1: "torch.Tensor",
        torch_w2: "torch.Tensor",
        torch_b0: "torch.Tensor" | None,
        torch_b1: "torch.Tensor" | None,
        torch_b2: "torch.Tensor" | None,
    ):
        """Permute the expert dim so that `ShardTensorToMesh(dim=experts)` lands each
        expert on its assigned device.

        The host weights come in routed-expert-id order (`[L, num_experts, ...]`), but
        sharding by the expert dim splits contiguously — so without reordering, device 0
        gets experts [0..E/D), device 1 gets [E/D..2E/D), etc. `expert_mapping[e]` tells
        us the *target* device for expert `e`; `argsort` (stable) groups experts that
        share a target device into contiguous chunks in the right order. Same permutation
        applies to biases when present.
        """
        perm = torch.argsort(torch_expert_mapping, stable=True)
        mapped_tensors = [t[:, perm, :, :] for t in (torch_w0, torch_w1, torch_w2)]

        if torch_b0 is not None:
            mapped_tensors += [t[:, perm, :] for t in (torch_b0, torch_b1, torch_b2)]
        else:
            mapped_tensors += [None] * 3

        return tuple(mapped_tensors)

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        torch_w0: "torch.Tensor",
        torch_w1: "torch.Tensor",
        torch_w2: "torch.Tensor",
        *,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
        has_bias: bool,
        num_routed_experts: int,
        expert_mapping: list[int],
        num_shared_experts: int,
        shared_expert_ids_to_devices: dict[int, list[int]] | None = None,
        shared_id_to_torch_w0: dict[int, "torch.Tensor"] | None = None,
        shared_id_to_torch_w1: dict[int, "torch.Tensor"] | None = None,
        shared_id_to_torch_w2: dict[int, "torch.Tensor"] | None = None,
        torch_b0: "torch.Tensor" | None = None,
        torch_b1: "torch.Tensor" | None = None,
        torch_b2: "torch.Tensor" | None = None,
    ):
        """Prepare and upload all expert state to the mesh.

        Pipeline: argsort-permute routed weights on host to match the device assignment
        (`_device_reorder_weights`), upload them to the mesh sharded on the experts dim
        (dim 1) as bf16, optionally splice in shared experts on-device so each device holds
        `routed_per_device + shared_per_device` slots (`ttnn.experimental.add_shared_expert_weights`),
        then run the C++ on-device packers (`ttnn.experimental.prepare_*`) and quantize to
        `bfloat4_b` (`ttnn.experimental.quantize_weights_via_host`) — see
        `_init_total_expert_weights_impl`.

        Routed weight shapes (host, post-permute): `w0/w1 = [L, num_routed, H, N]`,
        `w2 = [L, num_routed, N, H]`. Shared weights are dicts keyed by global expert id;
        each value has shape `[L, 1, ...]` matching the routed layout.

        Biases match the routed shape minus the matmul-output dim (`[L, num_routed, N]`
        for `b0/b1`, `[L, num_routed, H]` for `b2`). Shared experts + bias is not yet
        supported because the shared splice doesn't carry bias rows — would need a parallel API.
        """
        self._validate(
            torch_w0,
            torch_w1,
            torch_w2,
            mesh_device,
            mesh_shape,
            cluster_axis,
            has_bias,
            num_routed_experts,
            expert_mapping,
            num_shared_experts,
            shared_expert_ids_to_devices,
            shared_id_to_torch_w0,
            shared_id_to_torch_w1,
            shared_id_to_torch_w2,
            torch_b0,
            torch_b1,
            torch_b2,
        )

        # An empty mapping ({} when num_shared_experts==0) means "no shared experts" —
        # normalize to None so the shared-expert paths below are skipped entirely.
        if not shared_expert_ids_to_devices:
            shared_expert_ids_to_devices = None

        num_routed = torch_w0.shape[1]
        logger.info(
            f"Initializing expert state: routed_experts={num_routed} num_shared={num_shared_experts} "
            f"mesh_shape={mesh_shape} cluster_axis={cluster_axis} has_bias={has_bias}"
        )

        torch_expert_mapping = torch.tensor(expert_mapping, dtype=torch.int)

        (
            mapped_torch_w0,
            mapped_torch_w1,
            mapped_torch_w2,
            mapped_torch_b0,
            mapped_torch_b1,
            mapped_torch_b2,
        ) = self._device_reorder_weights(
            torch_expert_mapping, torch_w0, torch_w1, torch_w2, torch_b0, torch_b1, torch_b2
        )

        num_devices = mesh_device.get_num_devices()
        num_layers = torch_w0.shape[0]
        hidden_size = torch_w0.shape[-2]
        intermediate_size = torch_w0.shape[-1]
        routed_per_device = num_routed // num_devices

        # Upload the (reordered) routed weights to the mesh sharded on the experts dim (dim 1),
        # bf16 / ROW_MAJOR, so the C++ `ttnn.experimental.prepare_*` helpers can do the layout
        # transform on-device. Each device receives its assigned contiguous expert chunk.
        def _shard_experts(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            )

        tt_w0 = _shard_experts(mapped_torch_w0)
        tt_w1 = _shard_experts(mapped_torch_w1)
        tt_w2 = _shard_experts(mapped_torch_w2)

        if shared_expert_ids_to_devices is not None:
            if has_bias:
                # add_shared_expert_weights only handles weights; extending it to bias
                # requires a parallel API and per-shared-expert bias tensors.
                raise NotImplementedError("bias + shared experts is not yet supported")

            logger.info(f"Adding shared expert weights for {len(shared_expert_ids_to_devices)} shared experts")
            # The C++ add_shared_expert_weights consumes pre-arranged shared *device* tensors
            # (per device: its assigned shared experts in sorted-id order; then concatenated
            # across devices). Arrange on host, upload sharded on dim 1, splice on-device.
            tt_shared_w0 = _shard_experts(
                self._arrange_shared(shared_id_to_torch_w0, shared_expert_ids_to_devices, num_devices)
            )
            tt_shared_w1 = _shard_experts(
                self._arrange_shared(shared_id_to_torch_w1, shared_expert_ids_to_devices, num_devices)
            )
            tt_shared_w2 = _shard_experts(
                self._arrange_shared(shared_id_to_torch_w2, shared_expert_ids_to_devices, num_devices)
            )
            routed_w0, routed_w1, routed_w2 = tt_w0, tt_w1, tt_w2
            tt_w0, tt_w1, tt_w2 = ttnn.experimental.add_shared_expert_weights(
                routed_w0,
                routed_w1,
                routed_w2,
                tt_shared_w0,
                tt_shared_w1,
                tt_shared_w2,
                cluster_axis=cluster_axis,
            )
            for t in (routed_w0, routed_w1, routed_w2, tt_shared_w0, tt_shared_w1, tt_shared_w2):
                ttnn.deallocate(t)

            total_shared_slots = sum(len(devs) for devs in shared_expert_ids_to_devices.values())
            experts_per_device = (num_routed + total_shared_slots) // num_devices
        else:
            experts_per_device = routed_per_device

        tt_b0 = tt_b1 = tt_b2 = None
        if has_bias:
            tt_b0 = _shard_experts(mapped_torch_b0)
            tt_b1 = _shard_experts(mapped_torch_b1)
            tt_b2 = _shard_experts(mapped_torch_b2)

        self.tt_expert_mapping = self._init_expert_mapping(
            torch_expert_mapping, shared_expert_ids_to_devices, mesh_device, mesh_shape, cluster_axis
        )
        logger.info("Uploaded expert mapping to mesh")

        self.tt_w0_w1, self.tt_w2 = self._init_total_expert_weights_impl(
            tt_w0,
            tt_w1,
            tt_w2,
            mesh_device,
            num_layers,
            experts_per_device,
            hidden_size,
            intermediate_size,
            has_bias,
            tt_b0,
            tt_b1,
            tt_b2,
        )

    @staticmethod
    def _arrange_shared(
        shared_dict: dict[int, "torch.Tensor"],
        shared_expert_ids_to_devices: dict[int, list[int]],
        num_devices: int,
    ) -> "torch.Tensor":
        """Arrange a `{shared_expert_id: tensor}` dict into the per-device-stacked layout the
        C++ `add_shared_expert_weights` expects.

        For each device (in order), concatenate its assigned shared experts (in sorted global-id
        order) along the experts dim (dim 1), then concatenate across devices. Sharding the
        result on dim 1 then lands each device's shared experts on it, in slot order — matching
        how the device-side splice appends shared experts after routed ones per device.
        """
        device_to_shared: list[list[int]] = [[] for _ in range(num_devices)]
        for sid in sorted(shared_expert_ids_to_devices):
            for d in shared_expert_ids_to_devices[sid]:
                device_to_shared[d].append(sid)
        per_device = [torch.cat([shared_dict[sid] for sid in device_to_shared[d]], dim=1) for d in range(num_devices)]
        return torch.cat(per_device, dim=1)

    @staticmethod
    def _init_total_expert_weights_impl(
        tt_w0: ttnn.Tensor,
        tt_w1: ttnn.Tensor,
        tt_w2: ttnn.Tensor,
        mesh_device: ttnn.MeshDevice,
        num_layers: int,
        experts_per_device: int,
        hidden_size: int,
        intermediate_size: int,
        has_bias: bool,
        tt_b0: ttnn.Tensor | None = None,
        tt_b1: ttnn.Tensor | None = None,
        tt_b2: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Pack and quantize the combined routed+shared weight device tensors to `bfloat4_b`.

        Inputs are multi-device tensors sharded on the experts dim (dim 1), bf16 / ROW_MAJOR.
        Runs the C++ on-device packers (`ttnn.experimental.prepare_*`, with the `_with_bias`
        variants when bias rows are folded in), fetches the DRAM-sharded mem configs from
        `ttnn.experimental.get_weight_mem_configs`, then quantizes to `bfloat4_b` onto those
        configs via `ttnn.experimental.quantize_weights_via_host` (host round-trip).

        `experts_per_device` is the *combined* (routed + shared) per-device expert count — the
        experts dim of the inputs, divided across the mesh. Returns the two device tensors
        `(tt_w0_w1, tt_w2)` ready to feed `moe_compute`.
        """
        logger.info(
            f"Preparing expert weights on device: per_device={experts_per_device} "
            f"hidden={hidden_size} intermediate={intermediate_size} has_bias={has_bias}"
        )

        if has_bias:
            tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_with_bias(
                tt_w0, tt_w1, tt_b0, tt_b1, L=num_layers, E=experts_per_device, K=hidden_size, N=intermediate_size
            )
            tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_with_bias(
                tt_w2, tt_b2, L=num_layers, E=experts_per_device, N=intermediate_size, K=hidden_size
            )
        else:
            tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(
                tt_w0, tt_w1, L=num_layers, E=experts_per_device, K=hidden_size, N=intermediate_size
            )
            tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_for_moe_compute(
                tt_w2, L=num_layers, E=experts_per_device, N=intermediate_size, K=hidden_size
            )

        # Raw uploaded inputs are consumed by the packers; free them before the host round-trip.
        for t in (tt_w0, tt_w1, tt_w2):
            ttnn.deallocate(t)
        if has_bias:
            for t in (tt_b0, tt_b1, tt_b2):
                ttnn.deallocate(t)

        # has_bias grows the padded K / N by a tile to accommodate the bias row.
        weight_mem_configs = ttnn.experimental.get_weight_mem_configs(
            mesh_device,
            num_layers=num_layers,
            experts_per_device=experts_per_device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            has_bias=has_bias,
        )

        tt_w0_w1 = ttnn.experimental.quantize_weights_via_host(
            tt_w0_w1_prepped, dtype=ttnn.bfloat4_b, memory_config=weight_mem_configs.w0_w1
        )
        ttnn.deallocate(tt_w0_w1_prepped)
        tt_w2 = ttnn.experimental.quantize_weights_via_host(
            tt_w2_prepped, dtype=ttnn.bfloat4_b, memory_config=weight_mem_configs.w2
        )
        ttnn.deallocate(tt_w2_prepped)
        logger.info("Prepared and quantized w0/w1 and w2 to bfloat4_b on mesh")

        return tt_w0_w1, tt_w2


class _TTMoEDecodeBuffers:
    """Persistent buffers and semaphores for the MoE decode pipeline.

    Allocates the dispatch output triple (sparse buffer, expert indices, expert
    scores), the dispatch/combine cross-device semaphores, and the combine
    output buffer once in __init__, for reuse across forward() calls.

    Shapes and memory configs mirror those used in test_optimized_moe_decode_block.py.
    """

    SPARSE_BUFFER_DTYPE = ttnn.bfloat16
    INDICES_DTYPE = ttnn.uint16
    SCORES_DTYPE = ttnn.bfloat16
    COMBINE_OUTPUT_DTYPE = ttnn.bfloat16

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        mesh_shape: tuple[int, int],
        cluster_axis: int,
        batch_per_device: int,
        hidden_size: int,
        effective_experts_k: int,
        shard_dim: int,
        compute_tilize_drain_core: ttnn.CoreCoord,
    ):
        """Allocate the persistent buffers and semaphores reused across `forward()` calls.

        Allocates four ttnn tensors and two global semaphores:
        - `dispatch_global_semaphore` / `combine_global_semaphore`: cross-device sync
          points. Single-use per forward (no double buffering needed — combine syncs
          after fully reading the dispatch output, dispatch syncs at end of pipeline).
        - `tt_dispatch_output_tensors` triple: sparse buffer (DRAM, hidden-wide token slots),
          indices and scores (both L1 height-sharded on the drain core, narrow).
        - `tt_combine_output`: DRAM `[effective_experts_k, batch_per_device, hidden_size]`
          intermediate after `moe_compute`, before the post-combine tilize.

        `effective_experts_k = select_experts_k + num_shared_experts` — the K dimension of
        the per-token expert-output stack. Sized from config; passed in to keep this class
        agnostic of where the value came from.
        """
        # --- derived sizes (seq=1 for decode) ---
        num_dispatch_devices = mesh_shape[cluster_axis]
        total_tokens = batch_per_device * num_dispatch_devices
        tokens_per_device = batch_per_device

        shard_dims = (shard_dim, None) if cluster_axis == 0 else (None, shard_dim)

        # --- global semaphores (one each, no double buffering required —
        # combine syncs after reading dispatch output, dispatch syncs at end) ---
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        self.dispatch_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)
        self.combine_global_semaphore = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)

        # --- dispatch output buffers ---
        # Sparse buffer: DRAM, row-major, sharded along cluster axis
        sparse_buffer = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, hidden_size],
                dtype=_tt_to_torch_dtype(self.SPARSE_BUFFER_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.SPARSE_BUFFER_DTYPE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        # Indices / scores share an L1 height-sharded mem config on the drain core
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
            [total_tokens, effective_experts_k],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        l1_height_sharded = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        indices = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, effective_experts_k],
                dtype=_tt_to_torch_dtype(self.INDICES_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.INDICES_DTYPE,
            memory_config=l1_height_sharded,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        scores = ttnn.from_torch(
            torch.zeros(
                [num_dispatch_devices, total_tokens, effective_experts_k],
                dtype=_tt_to_torch_dtype(self.SCORES_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.SCORES_DTYPE,
            memory_config=l1_height_sharded,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        self.tt_dispatch_output_tensors = (sparse_buffer, indices, scores)
        self.tt_combine_output = ttnn.from_torch(
            torch.zeros(
                [effective_experts_k, tokens_per_device, hidden_size],
                dtype=_tt_to_torch_dtype(self.COMBINE_OUTPUT_DTYPE),
            ),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.COMBINE_OUTPUT_DTYPE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class TTMoEDecode:
    """MoE decode block: token dispatch → expert compute → score-weighted combine → reduce-scatter.

    Constructed once per layer (or per shared layer slot); `forward()` drives one
    decode step. All shape / topology / memory-config decisions live in `config`
    (`TTMoEDecodeConfig`) — this class only orchestrates the ttnn op sequence.

    Two RS branches are auto-selected from `mesh_shape[1 - cluster_axis]`:
    - `== DEEPSEEK_RS_DP_DIM (8)`: fused `deepseek_moe_reduce_scatter` consuming the
      pre-split list of N outputs from `fast_reduce_nc_fused`.
    - `== SKIP_RS_DP_DIM (1)`: no replication, RS is a no-op.
    - else: generic `ttnn.reduce_scatter` over the single fast-reduce output.
    """

    DEEPSEEK_RS_DP_DIM: int = 8
    SKIP_RS_DP_DIM: int = 1

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: TTMoEDecodeConfig,
        torch_w0: torch.Tensor,
        torch_w1: torch.Tensor,
        torch_w2: torch.Tensor,
        shared_id_to_torch_w0: dict[int, torch.Tensor] | None = None,
        shared_id_to_torch_w1: dict[int, torch.Tensor] | None = None,
        shared_id_to_torch_w2: dict[int, torch.Tensor] | None = None,
        torch_b0: torch.Tensor | None = None,
        torch_b1: torch.Tensor | None = None,
        torch_b2: torch.Tensor | None = None,
    ):
        """Upload weights / biases / shared experts to the mesh and allocate scratch buffers.

        Routed weight shapes: `w0/w1 = [L, num_routed_experts, hidden_size, intermediate_size]`,
        `w2 = [L, num_routed_experts, intermediate_size, hidden_size]`. Shared weights are
        dicts keyed by global expert id (in `[num_routed, num_routed + num_shared)`), each
        value `[L, 1, ...]` matching the routed layout.

        Bias shapes (only when `config.has_bias`): `b0/b1 = [L, num_routed_experts, intermediate_size]`,
        `b2 = [L, num_routed_experts, hidden_size]`. Bias + shared experts together raises
        `NotImplementedError`.
        """
        self.config = config
        self.expert_state = _TTMoEDecodeExpertState(
            mesh_device,
            torch_w0,
            torch_w1,
            torch_w2,
            **config.experts.model_dump(),
            shared_id_to_torch_w0=shared_id_to_torch_w0,
            shared_id_to_torch_w1=shared_id_to_torch_w1,
            shared_id_to_torch_w2=shared_id_to_torch_w2,
            torch_b0=torch_b0,
            torch_b1=torch_b1,
            torch_b2=torch_b2,
        )
        buffers_dict = config.buffers.model_dump()
        if buffers_dict.get("compute_tilize_drain_core") is None:
            matmul_ring_size = effective_matmul_ring_size(mesh_device)
            buffers_dict["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
                mesh_device,
                config.compute.output_height_shard_dim,
                auto_output_width_shard_dim(config.hidden_size, matmul_ring_size=matmul_ring_size),
                config.hidden_size,
                mux_core_range_set=config.compute.mux_core_range_set,
            )
        else:
            raise ValueError(
                "compute_tilize_drain_core is not user-configurable; omit it to resolve dynamically at runtime"
            )
        self.buffers = _TTMoEDecodeBuffers(mesh_device, **buffers_dict)

    @property
    def _num_fast_reduce_outputs(self) -> int:
        """Number of outputs fast_reduce_nc_fused will produce — N for the deepseek
        RS-list path, 1 otherwise (downstream ttnn.reduce_scatter takes a single tensor)."""
        return self.config.num_fast_reduce_outputs

    @property
    def _pre_split_chunk(self) -> int:
        """Logical per-fast-reduce-output width (hidden_size / num_fast_reduce_outputs).
        For the single-output path this is just hidden_size."""
        return self.config.hidden_size // self._num_fast_reduce_outputs

    @property
    def _padded_pre_split_chunk(self) -> int:
        """Aligned per-fast-reduce-output width = config.reduce.split_size."""
        return self.config.reduce.split_size

    @property
    def _post_rs_logical_chunk(self) -> int:
        """Logical per-device width after RS — what the model expects downstream."""
        return self.config.hidden_size // self.config.mesh_shape[1 - self.config.cluster_axis]

    @property
    def _needs_fast_reduce_padding(self) -> bool:
        return self._pre_split_chunk != self._padded_pre_split_chunk

    def _pad_for_fast_reduce(self, tt_x: ttnn.Tensor) -> ttnn.Tensor:
        """Interleave-pad the H dim so each post-RS per-device chunk is tile-aligned.

        Downstream RS splits the H dim evenly into `num_replicated` chunks, so padding
        must be inserted at each device-chunk boundary — not just appended at the end —
        or device d>0 ends up with data shifted by `d * (padded - logical)` positions.

        Layout produced: `[chunk_0, pad_0, ..., chunk_{R-1}, pad_{R-1}]`, R=num_replicated.
        Each `chunk_d` is `hidden / R` wide; each `pad_d` brings it up to TILE_SIZE alignment.
        """
        if not self._needs_fast_reduce_padding:
            return tt_x
        num_replicated = self.config.mesh_shape[1 - self.config.cluster_axis]
        chunk = self._post_rs_logical_chunk
        padded_chunk = self._padded_pre_split_chunk * self._num_fast_reduce_outputs // num_replicated
        shape = list(tt_x.shape)
        reshaped = ttnn.reshape(tt_x, shape[:-1] + [num_replicated, chunk])
        padded = ttnn.pad(
            reshaped,
            padding=[(0, 0)] * len(shape) + [(0, padded_chunk - chunk)],
            value=0.0,
        )
        return ttnn.reshape(padded, shape[:-1] + [num_replicated * padded_chunk])

    def _unpad_after_reduce_scatter(self, tt_final: ttnn.Tensor) -> ttnn.Tensor:
        """Slice the trailing padding off each device's post-RS tensor.

        After RS each device holds at least `_post_rs_logical_chunk` valid columns followed
        by zero padding (inserted before tilize). No-op when the original hidden splits
        evenly without padding.
        """
        if not self._needs_fast_reduce_padding:
            return tt_final
        chunk = self._post_rs_logical_chunk
        shape = list(tt_final.shape)
        start = [0] * len(shape)
        end = shape[:-1] + [chunk]
        return ttnn.slice(tt_final, start, end)

    def _format_dispatch_inputs(
        self,
        tt_x: ttnn.Tensor,
        tt_indices: ttnn.Tensor,
        tt_scores: ttnn.Tensor,
    ):
        """Coerce each dispatch input into the memory config the dispatch op needs.

        For each of (`tt_x`, `tt_indices`, `tt_scores`) returns a `(tensor, dealloc_flag)`
        pair: if a `to_memory_config` was necessary, the returned tensor is a fresh
        allocation that `forward()` should deallocate after the dispatch op; if the input
        already matched, the original is returned with `dealloc=False` to leave caller
        ownership intact.
        """
        if tt_x.memory_config() != self.config.dispatch_input_memory_config:
            tt_dispatch_input_tensor_bundle = (
                ttnn.to_memory_config(tt_x, memory_config=self.config.dispatch_input_memory_config),
                True,
            )
        else:
            tt_dispatch_input_tensor_bundle = tt_x, False

        if tt_indices.memory_config() != self.config.dispatch_input_memory_config:
            tt_dispatch_input_expert_indices_tensor_bundle = (
                ttnn.to_memory_config(
                    tt_indices,
                    memory_config=self.config.dispatch_input_memory_config,
                ),
                True,
            )
        else:
            tt_dispatch_input_expert_indices_tensor_bundle = tt_indices, False

        if tt_scores.memory_config() != self.config.dispatch_input_expert_scores_memory_config:
            tt_dispatch_input_expert_scores_tensor_bundle = (
                ttnn.to_memory_config(
                    tt_scores,
                    memory_config=self.config.dispatch_input_expert_scores_memory_config,
                ),
                True,
            )
        else:
            tt_dispatch_input_expert_scores_tensor_bundle = tt_scores, False

        return (
            tt_dispatch_input_tensor_bundle,
            tt_dispatch_input_expert_indices_tensor_bundle,
            tt_dispatch_input_expert_scores_tensor_bundle,
        )

    def forward(
        self, tt_x: ttnn.Tensor, tt_scores: ttnn.Tensor, tt_indices: ttnn.Tensor, layer_id: int = 0
    ) -> ttnn.Tensor:
        """Run one decode-step MoE forward.

        Inputs (sharded along the dispatch axis):
        - `tt_x`: `[1, batch_per_device, 1, hidden_size]` activations per device.
        - `tt_indices`: `[batch_per_device, 1, 1, select_experts_k]` chosen routed experts
          per token (uint16).
        - `tt_scores`: `[batch_per_device, 1, 1, select_experts_k]` per-(token, k) score
          for the routed combine; shared experts use `config.reduce.shared_expert_scale`
          uniformly, not this tensor.
        - `layer_id`: which slice of the layered weight tensors to use (currently
          assumed `0` since the rest of the test/module pipeline is `num_layers=1`).

        Output: `[1, 1, batch_per_device, hidden_size / num_replicated]` per device,
        i.e. each device holds its post-reduce-scatter chunk of the combined hidden dim.

        Pipeline matches the reference test_optimized_moe_decode_block:
        1. `all_to_all_dispatch_metadata` → per-device sparse buffer of inbound tokens.
        2. `moe_compute` → per-(k, token) expert output stack, optionally with bias.
        3. Tilize (`deepseek_moe_post_combine_tilize` when batch_per_device == TILE_SIZE
           and an NdShard config is available; else `tilize_with_val_padding` fallback).
        4. `deepseek_moe_fast_reduce_nc_fused` → score-weighted sum over k, with shared
           experts scaled by `shared_expert_scale`.
        5. Reduce-scatter across the replicated axis (3 variants — see class docstring).
        6. Strip the per-device-chunk padding inserted before tilize, if any.
        """
        (
            (tt_dispatch_input_tensor, dealloc_input),
            (tt_dispatch_input_expert_indices_tensor, dealloc_indices),
            (tt_dispatch_input_expert_scores_tensor, dealloc_scores),
        ) = self._format_dispatch_inputs(tt_x, tt_indices, tt_scores)

        (
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
        ) = ttnn.experimental.all_to_all_dispatch_metadata(
            tt_dispatch_input_tensor,
            tt_dispatch_input_expert_indices_tensor,
            tt_dispatch_input_expert_scores_tensor,
            self.expert_state.tt_expert_mapping,
            **self.config.dispatch.model_dump(),
            # shared_expert_ids
            # cluster_axi
            # num_links
            # drain_sync_tilizer_core
            # worker_mode
            # dispatch_algorithm
            output_tensors=self.buffers.tt_dispatch_output_tensors,
            cross_device_semaphore=self.buffers.dispatch_global_semaphore,
        )

        if dealloc_input:
            ttnn.deallocate(tt_dispatch_input_tensor)

        if dealloc_scores:
            ttnn.deallocate(tt_dispatch_input_expert_scores_tensor)

        (
            _,
            _,
            _,
            tt_l1_compute_output,
            _,
            tt_combine_output,
        ) = ttnn.experimental.moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            self.expert_state.tt_expert_mapping,
            self.expert_state.tt_w0_w1,
            self.expert_state.tt_w2,
            layer_id=layer_id,
            # output_height_shard_dim
            # cluster_axis
            # mux_core_range_set
            # has_bias
            # activation_type
            **self.config.compute.model_dump(),
            optional_output_tensor=self.buffers.tt_combine_output,
            optional_cross_device_semaphore=self.buffers.combine_global_semaphore,
        )
        ttnn.deallocate(tt_l1_compute_output)

        # unsqueeze
        # [select_experts_k, tokens_per_device, hidden_size] -> [select_experts_k, 1, tokens_per_device, hidden_size]
        # Note: this does not reallocate, aliases tt_combine_output so don't dealloc
        tt_unsqueezed_output = ttnn.unsqueeze(tt_combine_output, dim=1)

        # When hidden / num_replicated isn't tile-aligned, interleave-pad each per-device
        # chunk up to split_size so fast_reduce_nc_fused's 128-divisibility check passes.
        # No-op when already aligned.
        tt_unsqueezed_output = self._pad_for_fast_reduce(tt_unsqueezed_output)

        if self.config.use_post_combine_tilize:
            tt_tilized_compute_output = ttnn.experimental.deepseek_moe_post_combine_tilize(
                tt_unsqueezed_output,
                # output_memory_config,
                **self.config.post_combine_tilize.model_dump(),
            )

        else:
            output_tensor_shape = list(tt_unsqueezed_output.shape)
            output_tensor_shape[2] = ((output_tensor_shape[2] + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            tt_tilized_compute_output = ttnn.tilize_with_val_padding(
                tt_unsqueezed_output,
                output_tensor_shape=output_tensor_shape,
                pad_value=0.0,
                # memory_config
                **self.config.tilize_with_val_padding.model_dump(),
            )

        # scale with scores and accumulate
        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            tt_tilized_compute_output,
            tt_dispatch_input_expert_indices_tensor,
            self.expert_state.tt_expert_mapping,
            # reduce_dim
            # cluster_axis
            # split_size
            # output_memory_config
            # num_shared_experts
            # shared_expert_scale
            **self.config.reduce.model_dump(),
            scores_tensor=tt_scores,
        )
        ttnn.deallocate(tt_tilized_compute_output)

        if dealloc_indices:
            ttnn.deallocate(tt_dispatch_input_expert_indices_tensor)

        # [select_experts_k, tokens_per_device, hidden_size // num_replicated_devices] final per device shape
        if (
            self.config.mesh_shape[1 - self.config.cluster_axis] == self.DEEPSEEK_RS_DP_DIM
            and self.config.topology == ttnn.Topology.Ring
        ):
            tt_final_output = ttnn.experimental.deepseek_moe_reduce_scatter(
                tt_fast_reduce_output_tensors,
                # output_memory_config
                # dim
                # num_links
                # topology
                # cluster_axis
                **self.config.deepseek_moe_reduce_scatter.model_dump(),
            )
            for t in tt_fast_reduce_output_tensors:
                ttnn.deallocate(t)

        # note: in this path the output is L1 sharded as if set up for deepseek_moe_reduce_scatter. Likely better to
        # switch this to something more generic.
        elif self.config.mesh_shape[1 - self.config.cluster_axis] == self.SKIP_RS_DP_DIM:
            tt_final_output = tt_fast_reduce_output_tensors[0]
        else:
            tt_final_output = ttnn.reduce_scatter(
                tt_fast_reduce_output_tensors[0], **self.config.reduce_scatter.model_dump()
            )
            for t in tt_fast_reduce_output_tensors:
                ttnn.deallocate(t)

        # Strip the per-chunk padding we inserted before tilize. No-op when no padding was applied.
        return self._unpad_after_reduce_scatter(tt_final_output)
