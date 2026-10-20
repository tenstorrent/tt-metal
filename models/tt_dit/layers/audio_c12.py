# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Default-off C12 experiment; no custom kernel, activation or BWE changes."""
from __future__ import annotations

import json
import os

from loguru import logger

import ttnn

from ..models.audio_vae.vocoder_ltx import DilatedConv1d
from ..parallel.config import ParallelFactor
from ..utils.c12 import (
    cache_suffix,
    checkpoint_reason,
    claim_cache_root,
    digest,
    parse_mode,
    shape_reason,
    transform_weight,
)
from .audio_ops import _AlignedOutConv1d, _t_neighbor_pad


class C12Conv1d(_AlignedOutConv1d):
    """C24 logical input/output; private packed weights at the original parameter path.

    All temporaries live through the call. No force-deallocation of borrowed input or
    views; the packed path returns a clone with its own storage. Layout/reshape copies are included in
    the experimental interval, not assumed to be free. Auto modes load a separate
    original module solely for explicitly recorded unsupported-shape fallback.
    """

    def __init__(self, baseline, *, cell, automatic, identity):
        self._cell = cell
        self._automatic = automatic
        self._identity_json = json.dumps(identity, sort_keys=True)
        super().__init__(
            24 * cell.pack,
            24 * cell.pack,
            kernel_size=7 if cell.pack == 1 else 3,
            bias=baseline.bias_enabled,
            mesh_device=baseline.mesh_device,
            dtype=ttnn.float32,
            parallel_config=baseline.parallel_config,
            ccl_manager=baseline.ccl_manager,
            split_mode="off",
        )
        self.conv_config = ttnn.Conv3dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            C_in_block=32,
            C_out_block=32,
            T_out_block=cell.time_block,
            H_out_block=1,
            W_out_block=1,
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )
        self.baseline = baseline if automatic else None
        self.execution_manifests = {}
        self._input_contract = None

    @property
    def experiment_identity(self):
        return json.loads(self._identity_json)

    def _validate_precision(self):
        kernel = self.compute_kernel_config
        expected = (
            ("math_fidelity", ttnn.MathFidelity.HiFi4),
            ("math_approx_mode", False),
            ("fp32_dest_acc_en", True),
            ("packer_l1_acc", True),
        )
        if self.dtype != ttnn.float32 or self.split_mode != "off":
            raise ValueError("C12 precision/split mode changed after construction")
        if any(getattr(kernel, name) != value for name, value in expected):
            raise ValueError("C12 compute precision changed after construction")
        if self.conv_config.weights_dtype != ttnn.float32 or self.conv_config.output_layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("C12 weight precision/layout changed after construction")
        if any(getattr(self.conv_config, name) != value for name, value in self._cell.identity()["blocking"].items()):
            raise ValueError("C12 fixed cell blocking changed after construction")

    def _prepare_torch_state(self, state):
        import torch

        self._validate_precision()
        for key in ("weight", "bias"):
            if "conv." + key in state:
                state[key] = state.pop("conv." + key)
        weight, bias = state.get("weight"), state.get("bias")
        if weight is None or weight.dtype != torch.float32 or (bias is not None and bias.dtype != torch.float32):
            raise ValueError("C12 requires checkpoint FP32 weights/bias")
        if (bias is not None) != self.bias_enabled:
            raise ValueError("C12 checkpoint bias policy mismatch")
        # Bind the actual tensors, including manually supplied state, to the evidence.
        import hashlib

        evidence = self.experiment_identity["checkpoint"]
        for key, value in (("weight", weight), ("bias", bias)):
            if value is not None:
                actual = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                if actual != evidence[key + "_sha256"]:
                    raise ValueError("C12 loaded checkpoint tensor identity mismatch: " + key)
        if self.baseline is not None:
            state["baseline.weight"] = weight.clone()
            if bias is not None:
                state["baseline.bias"] = bias.clone()
        state["weight"], packed_bias = transform_weight(weight, bias, self._cell.pack)
        if packed_bias is not None:
            state["bias"] = packed_bias
        super()._prepare_torch_state(state)

    def bind_input(self, *, batch, local_t, tail_samples):
        """Host preflight from the actual vocoder upload and equal zero-origin partition.

        Called before upload, even on trace replay. Tail units remain original samples.
        """
        starts = tuple(i * local_t for i in range(self.parallel_config.factor))
        shape = (batch, local_t, 24)
        reason = shape_reason(self._cell, shape, starts=starts, time_factor=self.parallel_config.factor)
        if reason and not self._automatic:
            raise ValueError("C12 incompatible explicit cell: " + reason)
        self._input_contract = {
            "shape": shape,
            "starts": starts,
            "tail_samples": tail_samples,
            "partition": "vocoder equal contiguous T partition, global origin0",
        }

    def forward(self, x):
        self._validate_precision()
        b, t, c = x.shape
        pc = self.parallel_config
        contract = self._input_contract
        starts = () if contract is None else contract["starts"]
        reason = "missing/mismatched upload partition evidence"
        if contract is not None and tuple(x.shape) == contract["shape"]:
            reason = shape_reason(self._cell, tuple(x.shape), starts=starts, time_factor=pc.factor)
        if x.get_dtype() != ttnn.float32 or x.layout != ttnn.ROW_MAJOR_LAYOUT:
            reason = "requires FP32 ROW_MAJOR activation"
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            reason = "requires baseline interleaved DRAM placement"
        if reason and not self._automatic:
            raise ValueError("C12 incompatible explicit cell: " + reason)
        packed_t = t // self._cell.pack
        time_blocks = (packed_t + self._cell.time_block - 1) // self._cell.time_block
        manifest = {
            **self.experiment_identity,
            "logical_shape": list(x.shape),
            "packed_local_t": packed_t,
            "time_blocks": time_blocks,
            "estimated_candidate_tile_products": time_blocks * (7 if self._cell.pack == 1 else 27),
            "cost_classification": "source-derived block estimate, not measured work",
            "padded_shape": list(x.padded_shape),
            "upload_partition": contract,
            "global_shard_starts": list(starts),
            "selected": "baseline" if reason else self._cell.mode,
            "fallback_reason": reason,
            "placement": str(x.memory_config()),
            "borrowed_input": True,
            "effective_native_packer_accumulation": "unverified; native attestation required",
            "operations": ["baseline module"]
            if reason or self._cell.pack == 1
            else [
                "phase slices",
                "channel concat/copy",
                "one-row neighbor halo" if self._cell.pack == 4 else "baseline halo",
                "reshape (may copy)",
                "conv3d",
                "trim/unpack",
                "owned output clone",
            ],
            "native_ownership_and_page_sizes": "unverified; retain runtime allocation evidence",
        }
        key = digest(manifest)
        if key not in self.execution_manifests:
            self.execution_manifests[key] = manifest
            logger.info("C12 execution manifest: {}", json.dumps(manifest, sort_keys=True))
        if reason:
            return self.baseline(x)
        if self._cell.pack == 1:
            return super().forward(x)

        # Logical slices remove physical C24->C32 padding before concatenation.
        phases = [
            ttnn.slice(x, [0, phase, 0], [b, t, 24], [1, 4, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for phase in range(4)
        ]
        packed = ttnn.concat(phases, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        halo = _t_neighbor_pad(
            packed, pad_left=1, pad_right=1, parallel_config=pc, ccl_manager=self.ccl_manager, padding_mode="zeros"
        )
        padded = ttnn.reshape(halo, (b, t // 4 + 2, 1, 1, 96))
        output = ttnn.experimental.conv3d(
            input_tensor=padded,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=self.conv_config,
            output_channels=96,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            padding_mode="zeros",
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Conv consumes both halo rows; trim explicitly to the original local span.
        rows = ttnn.reshape(output, (b, output.shape[1], 96))
        trimmed = ttnn.slice(rows, [0, 0, 0], [b, t // 4, 96])
        unpacked = ttnn.reshape(trimmed, (b, t, 24))
        return ttnn.clone(unpacked, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def install_c12(vocoder, *, mode, config, checkpoint):
    """Called only for the adapter's main vocoder, before loading any weights."""
    cell, automatic = parse_mode(mode)
    if cell is None:
        return None
    reason = checkpoint_reason(config, checkpoint)
    pc = vocoder.parallel_config
    if reason is None and (type(pc) is not ParallelFactor or pc.factor < 2 or pc.mesh_axis not in (0, 1)):
        reason = "requires channel-factor1 and one sharded time axis"
    if reason is None and vocoder.mesh_device.shape[pc.mesh_axis] != pc.factor:
        reason = "time factor does not match mesh axis"
    if reason is None and (vocoder.dtype != ttnn.float32 or vocoder.num_upsamples != 6 or vocoder.num_kernels != 3):
        reason = "main-vocoder module structure/precision mismatch"
    target = None
    if reason is None:
        block = vocoder.resblocks[16]
        target = block.convs2[0]
        attrs = {
            "unpadded_in_channels": 24,
            "unpadded_out_channels": 24,
            "kernel_size": (7, 1, 1),
            "stride": (1, 1, 1),
            "dilation": 1,
            "padding_mode": "zeros",
            "split_mode": "off",
            "dtype": ttnn.float32,
            "parallel_config": pc,
            "same_pad": 3,
            "eff_k": 7,
            "internal_padding": (0, 0, 0),
            "halo_pad_left": 3,
            "halo_pad_right": 3,
        }
        if type(target) is not DilatedConv1d:
            reason = "requires the original DilatedConv1d module type"
        elif block.channels != 24 or block.kernel_size != 7 or block.num_branches != 3:
            reason = "semantic late-stage AMP block mismatch"
        elif any(getattr(target, key, None) != value for key, value in attrs.items()):
            reason = "exact convolution identity mismatch"
        elif any(
            getattr(target.compute_kernel_config, key) != value
            for key, value in (
                ("math_fidelity", ttnn.MathFidelity.HiFi4),
                ("math_approx_mode", False),
                ("fp32_dest_acc_en", True),
                ("packer_l1_acc", True),
            )
        ):
            reason = "baseline compute precision mismatch"
        elif target.bias_enabled != (checkpoint["bias_shape"] is not None):
            reason = "checkpoint/module bias policy mismatch"
        elif target.is_loaded() or vocoder.is_loaded():
            reason = "C12 must be selected before weight loading"
    if reason:
        if not automatic:
            raise ValueError("C12 incompatible explicit cell: " + reason)
        vocoder.c12_selection = {"requested": mode, "selected": "baseline", "fallback_reason": reason}
        logger.info("C12 selection: {}", json.dumps(vocoder.c12_selection, sort_keys=True))
        return None
    identity = {
        **cell.identity(),
        "requested": mode,
        "checkpoint": checkpoint,
        "time_factor": pc.factor,
        "time_axis": pc.mesh_axis,
        "channel_factor": 1,
        "mesh_shape": list(vocoder.mesh_device.shape),
        "placement": "interleaved DRAM",
    }
    root = os.environ.get("TT_DIT_CACHE_DIR")
    if not root:
        raise ValueError("C12 requires a separate empty TT_DIT_CACHE_DIR")
    vocoder.c12_cache_root = claim_cache_root(root, identity)
    replacement = C12Conv1d(target, cell=cell, automatic=automatic, identity=identity)
    replacement._validate_precision()
    block.convs2.add_module("0", replacement)
    vocoder.c12_identity = identity
    vocoder.c12_selection = {"requested": mode, "selected": cell.mode, "fallback_reason": None}
    vocoder.c12_cache_suffix = cache_suffix(identity)
    return identity
