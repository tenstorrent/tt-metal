# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 vision tower on the prefill mesh: ViT, per-patch projector and 2x2 patch merge.

Patches are sharded over the SP mesh axis and heads over the TP axis, as in the text prefill. Each layer all-gathers
K and V over SP and runs windowed SDPA, so attention stays inside one image or one video segment; padding rows form
windows of their own. The out-projection and fc2 are all-reduced over TP, so the residual stays TP-replicated. The
projector and the merge leave each output row's hidden dim sharded over TP, the token embedding's layout, and a
final SP all-gather puts every row on every device.

Head dims are padded 80 -> 96 for SDPA. q/k head dims are permuted so the 39 rotated pairs of the partial 3D RoPE
are adjacent (rotary_embedding_llama's interleaved form); the permutation is applied to both q and k, so q.k is
unchanged. cos = 1 and sin = 0 on the pass-through and padded dims.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.reference.vision import VisionArgs, rope_frequencies, vision_segments

TILE = 32
# windowed SDPA loads cu_window_seqlens into one uint32 tile
CU_WINDOW_ENTRIES = TILE * TILE
# Per-device patch counts the tower is compiled for; a batch pads up to the next one.
LOCAL_PATCH_BUCKETS = (128, 256, 384, 640, 1024, 1536, 2048, 3072, 4096, 5248, 6144, 8192, 10368, 12288, 16384)


def padded_head_dim(head_dim: int) -> int:
    return math.ceil(head_dim / TILE) * TILE


def qk_dim_permutation(args: VisionArgs) -> torch.Tensor:
    """Head-dim order putting rotated pair (i, i + rot/2) at (2i, 2i + 1); the pass-through dims follow."""
    rot = 3 * args.axis_dim
    half = rot // 2
    perm = [d for i in range(half) for d in (i, i + half)]
    return torch.tensor(perm + list(range(rot, args.head_dim)), dtype=torch.long)


def rope_tables(segments, args: VisionArgs, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [rows, padded head_dim] in the permuted order; rows past the segments rotate by nothing."""
    freqs = rope_frequencies(segments, args)
    ang = torch.repeat_interleave(freqs, 2, dim=-1)
    hd = padded_head_dim(args.head_dim)
    cos = torch.ones(rows, hd)
    sin = torch.zeros(rows, hd)
    cos[: ang.shape[0], : ang.shape[1]] = ang.cos()
    sin[: ang.shape[0], : ang.shape[1]] = ang.sin()
    return cos, sin


@dataclass(frozen=True)
class VisionBatch:
    """Host plan of one tower call: segments, padded size, attention windows."""

    segments: tuple[tuple[int, int, int], ...]
    patches: int  # real patches
    local: int  # patches per SP device, padded
    sp: int

    @property
    def padded(self) -> int:
        return self.local * self.sp

    @property
    def rows(self) -> int:
        """Merged output rows, real ones first."""
        return self.patches // 4

    def cu_windows(self) -> tuple[int, ...]:
        bounds = [0]
        for t, h, w in self.segments:
            bounds.append(bounds[-1] + t * h * w)
        # Padding goes in windows of at most sp * TILE rows: a window costs its length squared.
        step = self.sp * TILE
        bounds += list(range(self.patches + step, self.padded, step))
        if bounds[-1] != self.padded:
            bounds.append(self.padded)
        if len(bounds) > CU_WINDOW_ENTRIES:
            raise ValueError(f"{len(bounds) - 1} attention windows exceed the {CU_WINDOW_ENTRIES - 1} one call holds")
        return tuple(bounds) + (self.padded,) * (CU_WINDOW_ENTRIES - len(bounds))


def plan_batch(grid_thw, args: VisionArgs, sp: int) -> VisionBatch:
    segments = tuple(vision_segments(grid_thw, args.segment_max_frames))
    patches = sum(t * h * w for t, h, w in segments)
    need = -(-patches // sp)
    local = next((b for b in LOCAL_PATCH_BUCKETS if b >= need), None)
    if local is None:
        local = -(-need // 128) * 128
    return VisionBatch(segments, patches, local, sp)


def max_batch_patches(sp: int) -> int:
    return LOCAL_PATCH_BUCKETS[-1] * sp


def prepare_weights(state: dict[str, torch.Tensor], args: VisionArgs, tp: int) -> dict:
    """Device layout of every weight: name -> (tensor [in, out] or [1, 1, 1, n], TP shard dim or None).

    Tensors are None when `state` lacks them (a cache-only load). Column-parallel weights shard their output dim,
    row-parallel ones their input dim; biases added after a TP reduce-scatter shard with it, the rest replicate.
    """
    a = args
    pre = "vision_tower.vision_model."
    g = lambda name: state[name].float() if name in state else None
    gt = lambda name: None if g(name) is None else g(name).T
    vec = lambda t: None if t is None else t.reshape(1, 1, 1, -1)
    hd = padded_head_dim(a.head_dim)
    out = {}

    w = g(pre + "embeddings.patch_embedding.weight")
    if w is not None:
        patch_in = math.ceil(a.patch_elems / TILE) * TILE
        w = torch.nn.functional.pad(w.reshape(a.hidden, -1).T, (0, 0, 0, patch_in - a.patch_elems))
    out["patch.weight"] = (w, None)
    out["pre_ln.weight"] = (vec(g(pre + "pre_layrnorm.weight")), None)
    out["pre_ln.bias"] = (vec(g(pre + "pre_layrnorm.bias")), None)

    perm = qk_dim_permutation(a)
    local = a.heads // tp
    for i in range(a.layers):
        p = f"{pre}encoder.layers.{i}."
        qkv_w = qkv_b = o_w = None
        if (p + "self_attn.q_proj.weight") in state:
            parts_w, parts_b = [], []
            for n in "qkv":
                pw = g(p + f"self_attn.{n}_proj.weight").reshape(a.heads, a.head_dim, a.hidden)
                pb = g(p + f"self_attn.{n}_proj.bias").reshape(a.heads, a.head_dim)
                if n != "v":
                    pw, pb = pw[:, perm], pb[:, perm]
                parts_w.append(torch.nn.functional.pad(pw, (0, 0, 0, hd - a.head_dim)))
                parts_b.append(torch.nn.functional.pad(pb, (0, hd - a.head_dim)))
            # Device c's slice is [q | k | v] of its own heads.
            qkv_w = torch.cat([torch.cat([t[c * local : (c + 1) * local] for t in parts_w]) for c in range(tp)])
            qkv_b = torch.cat([torch.cat([t[c * local : (c + 1) * local] for t in parts_b]) for c in range(tp)])
            qkv_w, qkv_b = qkv_w.reshape(-1, a.hidden).T, vec(qkv_b)
            o = g(p + "self_attn.out_proj.weight").reshape(a.hidden, a.heads, a.head_dim)
            o_w = torch.nn.functional.pad(o, (0, hd - a.head_dim)).reshape(a.hidden, -1).T
        out[f"l{i}.ln1.weight"] = (vec(g(p + "layer_norm1.weight")), None)
        out[f"l{i}.ln1.bias"] = (vec(g(p + "layer_norm1.bias")), None)
        out[f"l{i}.qkv.weight"] = (qkv_w, -1)
        out[f"l{i}.qkv.bias"] = (qkv_b, -1)
        out[f"l{i}.o.weight"] = (o_w, -2)
        out[f"l{i}.o.bias"] = (vec(g(p + "self_attn.out_proj.bias")), None)
        out[f"l{i}.ln2.weight"] = (vec(g(p + "layer_norm2.weight")), None)
        out[f"l{i}.ln2.bias"] = (vec(g(p + "layer_norm2.bias")), None)
        out[f"l{i}.fc1.weight"] = (gt(p + "mlp.fc1.weight"), -1)
        out[f"l{i}.fc1.bias"] = (vec(g(p + "mlp.fc1.bias")), -1)
        out[f"l{i}.fc2.weight"] = (gt(p + "mlp.fc2.weight"), -2)
        out[f"l{i}.fc2.bias"] = (vec(g(p + "mlp.fc2.bias")), None)

    out["proj1.weight"] = (gt("multi_modal_projector.linear_1.weight"), -1)
    out["proj1.bias"] = (vec(g("multi_modal_projector.linear_1.bias")), -1)
    out["proj2.weight"] = (gt("multi_modal_projector.linear_2.weight"), -2)
    out["proj2.bias"] = (vec(g("multi_modal_projector.linear_2.bias")), -1)
    m1 = gt("patch_merge_mlp.linear_1.weight")
    if m1 is not None:
        # Input index j*text_hidden + d (patch j of the merge group, hidden d). Device c holds its TP slice of d for
        # every j, so its rows are the four slices back to back.
        th, s_ = a.text_hidden, a.text_hidden // tp
        rows = [torch.arange(j * th + c * s_, j * th + (c + 1) * s_) for c in range(tp) for j in range(a.merge**2)]
        m1 = m1[torch.cat(rows)]
    out["merge1.weight"] = (m1, -2)
    out["merge1.bias"] = (vec(g("patch_merge_mlp.linear_1.bias")), -1)
    out["merge2.weight"] = (gt("patch_merge_mlp.linear_2.weight"), -2)
    out["merge2.bias"] = (vec(g("patch_merge_mlp.linear_2.bias")), -1)
    return out


class TtVisionTower:
    """The vision tower, projector and patch merge, sharded over a (sp, tp) prefill mesh."""

    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        state: dict[str, torch.Tensor] | None,
        args: VisionArgs,
        *,
        cache_dir: str | None = None,
        weight_dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        # The tower's own semaphore pools, one per mesh axis: a device can run ahead into a collective on one axis
        # while a peer is still in a collective on the other, and a shared pool would let their signals collide.
        make = lambda: type(ccl_manager)(mesh_device, num_links=ccl_manager.num_links, topology=ccl_manager.topology)
        self.ccl_sp, self.ccl_tp = make(), make()
        self.args = args
        self.sp_axis = mesh_config.sp_axis
        self.tp_axis = mesh_config.tp_axis
        self.sp = mesh_device.shape[self.sp_axis]
        self.tp = mesh_device.shape[self.tp_axis]
        assert args.heads % self.tp == 0 and args.intermediate % self.tp == 0 and args.text_hidden % self.tp == 0
        self.local_heads = args.heads // self.tp
        self.hd = padded_head_dim(args.head_dim)
        self.patch_in = math.ceil(args.patch_elems / TILE) * TILE
        self.scale = args.head_dim**-0.5
        self.cache_dir = cache_dir
        self.weight_dtype = weight_dtype

        self.mm_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.trans_mat = ttnn.from_torch(
            _rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._load(state or {})

    # --- weights ------------------------------------------------------------------------------------------------

    def _put(self, name: str, t: torch.Tensor | None, shard_dim: int | None, dtype=None, layout=ttnn.TILE_LAYOUT):
        """Upload `t`, sharded over TP on `shard_dim` or replicated; cached under cache_dir when given."""
        dims = [None, None]
        if shard_dim is not None:
            dims[self.tp_axis] = shard_dim
        cache = os.path.join(self.cache_dir, f"vision.{name}") if self.cache_dir else None
        return ttnn.as_tensor(
            t,
            device=self.mesh_device,
            dtype=dtype or self.weight_dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_device.shape, dims=tuple(dims)),
            cache_file_name=cache,
        )

    def _load(self, state: dict[str, torch.Tensor]) -> None:
        w = prepare_weights(state, self.args, self.tp)
        put = lambda name: self._put(name, *w[name])
        self.patch_w = put("patch.weight")
        self.pre_ln = (put("pre_ln.weight"), put("pre_ln.bias"))
        self.layers = [
            dict(
                ln1=(put(f"l{i}.ln1.weight"), put(f"l{i}.ln1.bias")),
                qkv_w=put(f"l{i}.qkv.weight"),
                qkv_b=put(f"l{i}.qkv.bias"),
                o_w=put(f"l{i}.o.weight"),
                o_b=put(f"l{i}.o.bias"),
                ln2=(put(f"l{i}.ln2.weight"), put(f"l{i}.ln2.bias")),
                fc1_w=put(f"l{i}.fc1.weight"),
                fc1_b=put(f"l{i}.fc1.bias"),
                fc2_w=put(f"l{i}.fc2.weight"),
                fc2_b=put(f"l{i}.fc2.bias"),
            )
            for i in range(self.args.layers)
        ]
        self.proj1 = (put("proj1.weight"), put("proj1.bias"))
        self.proj2 = (put("proj2.weight"), put("proj2.bias"))
        self.merge1 = (put("merge1.weight"), put("merge1.bias"))
        self.merge2 = (put("merge2.weight"), put("merge2.bias"))

    # --- inputs --------------------------------------------------------------------------------------------------

    def _shard_rows(self, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
        """[rows, cols] -> [1, 1, rows/sp, cols] per device, SP-sharded on rows, TP-replicated."""
        dims = [None, None]
        dims[self.sp_axis] = 2
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_device.shape, dims=tuple(dims)),
        )

    def prepare_inputs(self, pixel_values: torch.Tensor, grid_thw) -> tuple[VisionBatch, dict]:
        """Plan the batch and upload its patches, rotary tables, windows and per-device query offsets."""
        a = self.args
        batch = plan_batch(grid_thw, a, self.sp)
        assert pixel_values.shape[0] == batch.patches, (pixel_values.shape, batch.patches)
        x = torch.zeros(batch.padded, self.patch_in, dtype=torch.bfloat16)
        x[: batch.patches, : a.patch_elems] = pixel_values.to(torch.bfloat16)
        cos, sin = rope_tables(batch.segments, a, batch.padded)
        dims = [None, None]
        dims[self.sp_axis] = 0
        inputs = dict(
            patches=self._shard_rows(x, ttnn.bfloat16),
            cos=self._shard_rows(cos, ttnn.bfloat16),
            sin=self._shard_rows(sin, ttnn.bfloat16),
            cu=ttnn.from_torch(
                torch.tensor(batch.cu_windows(), dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
            q_offset=ttnn.from_torch(
                torch.arange(self.sp, dtype=torch.int32) * batch.local,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=self.mesh_device.shape, dims=tuple(dims)
                ),
            ),
        )
        return batch, inputs

    # --- forward -------------------------------------------------------------------------------------------------

    def _linear(self, x, w, b=None):
        return ttnn.linear(x, w, bias=b, compute_kernel_config=self.mm_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _layer_norm(self, x, wb):
        return ttnn.layer_norm(x, epsilon=self.args.eps, weight=wb[0], bias=wb[1], compute_kernel_config=self.mm_config)

    def _tp_allreduce(self, x):
        if self.tp == 1:
            return x
        return self.mesh_config.allreduce(x, self.ccl_tp, axis=self.tp_axis)

    def _tp_reduce_scatter(self, x):
        if self.tp == 1:
            return x
        out = self.mesh_config.reduce_scatter(x, self.ccl_tp, dim=3, axis=self.tp_axis)
        x.deallocate(True)
        return out

    def _sp_gather(self, x, dim):
        if self.sp == 1:
            return x
        return self.mesh_config.allgather(x, self.ccl_sp, axis=self.sp_axis, dim=dim)

    def _attention(self, x, L, inputs, local: int):
        a = self.args
        qkv = self._linear(x, L["qkv_w"], L["qkv_b"])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.local_heads,
            num_kv_heads=self.local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)
        rope = lambda t: ttnn.experimental.rotary_embedding_llama(
            t,
            inputs["cos"],
            inputs["sin"],
            self.trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self.sdpa_config,
        )
        q_r, k_r = rope(q), rope(k)
        q.deallocate(True)
        k.deallocate(True)
        k_all = self._sp_gather(k_r, dim=2)
        v_all = self._sp_gather(v, dim=2)
        k_r.deallocate(True)
        v.deallocate(True)
        grid = self.mesh_device.compute_with_storage_grid_size()
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_all,
            v_all,
            is_causal=False,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=min(local, 128),
                k_chunk_size=min(local * self.sp, 512),
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.sdpa_config,
            cu_window_seqlens=inputs["cu"],
            windowed_q_token_offset_tensor=inputs["q_offset"],
        )
        for t in (q_r, k_all, v_all):
            t.deallocate(True)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = self._linear(attn, L["o_w"])
        attn.deallocate(True)
        return self._tp_allreduce(out)

    def _block(self, x, L, inputs, local: int):
        h = self._layer_norm(x, L["ln1"])
        attn = self._attention(h, L, inputs, local)
        h.deallocate(True)
        x2 = ttnn.add(x, ttnn.add(attn, L["o_b"]))
        attn.deallocate(True)
        x.deallocate(True)
        h = self._layer_norm(x2, L["ln2"])
        f = ttnn.gelu(self._linear(h, L["fc1_w"], L["fc1_b"]), fast_and_approximate_mode=False)
        h.deallocate(True)
        f2 = self._tp_allreduce(self._linear(f, L["fc2_w"]))
        f.deallocate(True)
        out = ttnn.add(x2, ttnn.add(f2, L["fc2_b"]))
        f2.deallocate(True)
        x2.deallocate(True)
        return out

    def _merge_rows(self, x):
        """[1, 1, n, d] -> [1, 1, n/4, 4d]: each output row is four consecutive input rows side by side."""
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        view = ttnn.reshape(rm, (1, 1, x.shape[-2] // self.args.merge**2, x.shape[-1] * self.args.merge**2))
        out = ttnn.to_layout(view, ttnn.TILE_LAYOUT)
        rm.deallocate(True)
        if view.is_allocated():
            view.deallocate(True)
        return out

    def forward(self, batch: VisionBatch, inputs: dict):
        """Merged rows [1, 1, batch.padded / 4, text_hidden / tp] on every device; rows past batch.rows are padding."""
        x = self._linear(inputs["patches"], self.patch_w)
        x = self._layer_norm(x, self.pre_ln)
        for L in self.layers:
            x = self._block(x, L, inputs, batch.local)

        y = ttnn.gelu(self._linear(x, *self.proj1), fast_and_approximate_mode=False)
        x.deallocate(True)
        y = ttnn.add(self._tp_reduce_scatter(self._linear(y, self.proj2[0])), self.proj2[1])
        y = self._merge_rows(y)
        y = ttnn.add(self._tp_reduce_scatter(self._linear(y, self.merge1[0])), self.merge1[1])
        y = ttnn.gelu(y, fast_and_approximate_mode=False)
        y = ttnn.add(self._tp_reduce_scatter(self._linear(y, self.merge2[0])), self.merge2[1])
        out = self._sp_gather(y, dim=2)
        if out is not y:
            y.deallocate(True)
        return out

    def __call__(self, pixel_values: torch.Tensor, grid_thw):
        batch, inputs = self.prepare_inputs(pixel_values, grid_thw)
        out = self.forward(batch, inputs)
        for t in inputs.values():
            t.deallocate(True)
        return batch, out

    def warmup(self) -> None:
        """Compile every bucket's programs, so no request pays the JIT."""
        a = self.args
        for local in LOCAL_PATCH_BUCKETS:
            grid = [(1, 2, local * self.sp // 2)]
            pv = torch.zeros(local * self.sp, a.patch_elems)
            batch, out = self(pv, grid)
            out.deallocate(True)
            logger.info(f"vision tower warm: {batch.local} patches per device")


def _rot_transformation_mat() -> torch.Tensor:
    """rotary_embedding_llama's interleaved rotation, one tile."""
    m = torch.zeros(1, 1, TILE, TILE)
    m[..., torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1
    m[..., torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1
    return m
