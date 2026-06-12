# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B full (softmax) attention block — 1D tensor-parallel port.

This is the ``tt_transformers``-resident port of the 32-chip BH-Galaxy
``models/demos/qwen3_6_galaxy_v2/tt/llama_attention.TtLlamaAttention``
(``is_qwen36=True`` path), retargeted for an 8-chip **1D tensor-parallel** mesh
(``MESH_DEVICE=P150x8`` → mesh shape ``(1, 8)``).

It is the sibling of ``qwen36_gdn_attention.TtQwen36GDNAttention`` (the linear /
DeltaNet block); the two classes share the decoder dispatch surface, the
mesh-axis derivation, the ``_slice_layer_weights`` / ``_build_weights`` shape,
and the single-all_reduce output-projection pattern.

Qwen3.6 full-attention quirks preserved (all proven correct on the galaxy 2D
mesh — this port keeps the math byte-for-byte and only redoes the sharding/CCL)
------------------------------------------------------------------------------
  * ``attn_output_gate=True``: ``q_proj`` outputs an interleaved ``[q | gate]``
    PER HEAD (HF ``qg.view(B,T,n_q,2*hd)`` → ``[..,:hd]``=q, ``[..,hd:]``=gate).
    De-interleaved at ``__init__`` via ``reshape(n_q, 2, hd, H)``. The gate
    multiplies the attention output (``out * sigmoid(gate)``) just before the
    output projection.
  * ``partial_rotary_factor=0.25`` → ``rope_dim = int(256*0.25) = 64``. Only the
    first 64 dims of each head are rotated; the remaining 192 pass through. See
    ``_partial_rope_apply`` (slice → rotate_half → addcmul → concat), ported
    verbatim from ``qwen3_6_galaxy_v2/tt/llama_rope.py::partial_rope_apply``.
  * Per-head ``qk_norm`` (zero-centered RMSNorm over each head's 256 dims)
    applied BEFORE RoPE. The ``+1`` zero-centered offset (HF
    ``Qwen3NextRMSNorm``) is baked into the on-device weight at ``__init__``.
  * mRoPE: the host cos/sin math lives in
    ``qwen3_6_galaxy_v2/tt/qwen36_mrope.py`` (mrope_section=[11,11,10],
    interleaved); the caller builds ``[1,1,T,rope_dim]`` cos/sin and passes them
    as ``rot_mats``. (For text-only input mRoPE reduces to standard 1D RoPE.)
  * KV heads: HF ships ``n_kv_heads=4`` which does NOT divide 8 chips. Pad
    4 → 8 via ``repeat_interleave(2, dim=0)`` on the K/V weights (GQA-preserving:
    galaxy V2-TP-1 proved ``q_i//3`` of replicated == ``q_i//6`` of original),
    then **1 KV head/chip**. We do NOT replicate KV; padding keeps the per-chip
    SDPA a clean 3 Q-heads : 1 KV-head GQA.

Topology: galaxy 2D-TP vs this 1D-TP
------------------------------------
The galaxy class is a **2D-TP** layout on an ``(8, 4)`` mesh:
  * heads split 8-way across mesh-ROWS (3 Q / 3 gate / 1 KV per row),
  * hidden H split 4-way across COLS; the QKVG projection produces a PARTIAL
    inner-product sum that a ``cluster_axis=1`` (4-way col) all_reduce completes,
  * the WO projection produces a partial sum over each row's heads that a
    ``cluster_axis=0`` (8-way row) all_reduce completes.

For 8-chip 1D-TP we split **heads 8-way and DO NOT split H** (full H replicated
per chip). Per-chip head counts are IDENTICAL to galaxy's per-row counts
(3 Q / 3 gate / 1 KV per chip, head_dim=256), so the per-head SDPA / QK-norm /
partial-RoPE math sees the same per-chip tensors as the galaxy path.

Consequences for the two collectives:
  * **Input projection** ``x[full H] @ w_qkvg_chip[5120, 2048] -> [.., 2048]``:
    K is the full hidden dim ⇒ the per-chip output is the COMPLETE result, NOT
    a partial sum. So there is **NO input-side all_reduce** (the galaxy col-axis
    reduce is dropped — 2 fewer CCLs / layer than galaxy).
  * **Output projection** ``gated[n_q*hd=768] @ w_out_chip[768, H=5120] -> [.., H]``:
    each chip owns its 3 Q-heads ⇒ the per-chip output is a PARTIAL sum over the
    chip's heads. **One** ``ttnn.all_reduce`` over the single TP axis completes
    it. (``tt_all_reduce(cluster_axis=1)`` short-circuits to a no-op on a
    ``(1, 8)`` mesh — see ``ccl.py`` ``cluster_axis==1 and 1 in shape`` — so we
    call stock ``ttnn.all_reduce`` on the real TP axis, exactly like the GDN
    sibling's ``_output_proj_and_reduce``.)

Mesh mappers used (per weight)
------------------------------
  * ``w_qkvg`` [H=5120, n_q*2*hd + 2*n_kv*hd = 16384] → split the 16384 output
    dim 8-way with ``ShardTensorToMesh(dim=1)``. Built from a per-chip
    ``[Q_i(768) | Gate_i(768) | K_i(256) | V_i(256)]`` (2048-wide) interleave so
    each chip's contiguous 2048 chunk matches the forward split offsets. dim-0
    (H) is NOT split.
  * ``w_out`` [n_q*hd=6144, H=5120] → ``ShardTensorToMesh(dim=0)`` (768
    rows/chip — the chip's 3 Q-heads). Partial-sum output, reduced by the one
    out-proj all_reduce.
  * ``q_norm`` / ``k_norm`` weights → ``ReplicateTensorToMesh`` (per-head,
    head-local norm — replicated, not sharded).

Single-layer PCC test (analogue of ``test_qwen36_gdn_1d_pcc.py``)
-----------------------------------------------------------------
Build ONE ``TtQwen36FullAttention`` (a ``full_attention`` layer index, e.g. 3)
on a ``(1, 8)`` mesh, feed it a replicated full-H hidden state + ``[1,1,T,64]``
cos/sin, and compare to the reference ``Qwen36Attention`` block (float32). Run::

    export TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
        && export MESH_DEVICE=P150x8 \\
        && export HF_MODEL=Qwen/Qwen3.6-27B \\
        && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest \\
            models/tt_transformers/tests/test_qwen36_full_attention_1d_pcc.py -v -s
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.rope import HfRotarySetup

# qwen3.6 per-head RMSNorm tile size (head_dim must be tile-aligned for rms_norm).
_QWEN36_TILE = 32


def _qwen36_qknorm_flat_to_heads(
    x_flat,
    weight,
    eps: float,
    B: int,
    n_heads: int,
    T: int,
    hd: int,
    compute_kernel_config,
):
    """Per-head RMSNorm of ``[B, T, n_heads*hd]`` → ``[B, n_heads, T, hd]``.

    Ported from the galaxy ``llama_attention._qwen36_qknorm_flat_to_heads``.
    The flat layout packs heads in the INNER (last) dim, so the correct
    transform is ``reshape([B, T, n, hd]) -> rms_norm(last dim hd) ->
    permute(0,2,1,3)``. At ``T == 1`` this is byte-identical to a bare reshape
    fast path (the galaxy "row 0 valid, rows 1..N garbage" bug came from using
    the bare reshape at ``T > 1``).
    """
    if n_heads == 1:
        x_normed_3d = ttnn.rms_norm(
            x_flat,
            weight=weight,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        view_4d = ttnn.reshape(x_normed_3d, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        x_normed_3d.deallocate(True)
        return out

    x_thd = ttnn.reshape(x_flat, [B, T, n_heads, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_normed = ttnn.rms_norm(
        x_thd,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    x_thd.deallocate(True)
    out = ttnn.permute(x_normed, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_normed.deallocate(True)
    return out


def _qwen36_flat_to_heads(x_flat, B: int, n_heads: int, T: int, hd: int):
    """Reshape ``[B, T, n_heads*hd]`` → ``[B, n_heads, T, hd]`` (no norm).

    Ported from galaxy ``llama_attention._qwen36_flat_to_heads``.
    """
    if n_heads == 1:
        view_4d = ttnn.reshape(x_flat, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        return out
    x_thd = ttnn.reshape(x_flat, [B, T, n_heads, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.permute(x_thd, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_thd.deallocate(True)
    return out


def _qwen36_heads_to_flat(x_heads, B: int, n_heads: int, T: int, hd: int):
    """Reshape ``[B, n_heads, T, hd]`` → ``[B, T, n_heads*hd]``.

    Ported from galaxy ``llama_attention._qwen36_heads_to_flat``.
    """
    if n_heads == 1:
        view_3d = ttnn.reshape(x_heads, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_3d.deallocate(True)
        return out
    if T == 1:
        out = ttnn.reshape(x_heads, [B, T, n_heads * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out
    slice_list = []
    time_slice_tensors = []
    for h in range(n_heads):
        head_h = ttnn.slice(x_heads, [0, h, 0, 0], [B, h + 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_h_3d = ttnn.reshape(head_h, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slice_list.append(head_h)
        time_slice_tensors.append(head_h_3d)
    out = ttnn.concat(time_slice_tensors, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in time_slice_tensors:
        t.deallocate(True)
    for s in slice_list:
        s.deallocate(True)
    return out


class TtQwen36FullAttention(LightweightModule):
    """1D tensor-parallel full (softmax) attention for Qwen3.6-27B (8-chip).

    Constructor signature matches the kwargs the ``tt_transformers`` decoder
    passes any attention class (``models/tt_transformers/tt/decoder.py`` :53-66):
    ``mesh_device``, ``tt_ccl``, ``args``, ``state_dict``, ``weight_cache_path``,
    ``layer_num``, ``dtype``, ``transformation_mats``, ``configuration``,
    ``paged_attention_config``, ``use_paged_kv_cache``, ``prefetcher``.

    The forward dispatch surface matches ``DefaultAttention.forward`` /
    ``TtLlamaAttention.forward``:
    ``forward(x, current_pos, rot_mats, user_id, mode, page_table=..., ...,
    kv_cache=..., **kwargs)``. ``rot_mats`` is the ``(cos_tt, sin_tt)`` pair of
    partial-RoPE tables (last dim ``rope_dim=64``).
    """

    # Sliding-window attention flag (the decoder reads ``attention.is_sliding``
    # to pick the local vs global rot_mats). Qwen3.6 full-attn layers are global.
    is_sliding = False

    def __init__(
        self,
        mesh_device,
        args,
        layer_num,
        dtype=ttnn.bfloat16,
        state_dict=None,
        weight_cache_path=None,
        transformation_mats=None,
        configuration=None,
        tt_ccl=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
        **kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args if args is not None else configuration
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl
        self.transformation_mats = transformation_mats
        self.paged_attention_config = paged_attention_config
        self.use_paged_kv_cache = use_paged_kv_cache
        # qwen3.6 forces bf16 attention weights (wqkvg / wo): galaxy V2-7b layer-3
        # forward PCC dropped 0.999 -> 0.77 at bf8 once composed inside the full
        # TtTransformer (residual+norm+MLP amplify the bf8 quant noise). Keep bf16
        # for the matmul weights regardless of the requested ``dtype`` (the GDN
        # sibling keeps the requested dtype because its recurrent term needs the
        # bf8-quantised inputs in range — full-attn has no such constraint).
        if os.environ.get("QWEN36_FP32_WEIGHTS", "0") == "1":
            self.weight_dtype = dtype  # VLM fp32 escape hatch (matches galaxy)
        elif os.environ.get("QWEN36_ATTN_WEIGHTS_BF8", "0") == "1":
            self.weight_dtype = ttnn.bfloat8_b
        else:
            self.weight_dtype = ttnn.bfloat16
        self.dtype = dtype

        # --- Mesh topology: 1D-TP on a (1, 8) mesh. Derive the real TP axis. ---
        self.cluster_shape = list(mesh_device.shape)  # [1, 8]
        assert 1 in self.cluster_shape, (
            f"TtQwen36FullAttention is the 1D-TP port; expected a (1, N) or (N, 1) mesh, "
            f"got {self.cluster_shape}. Use the galaxy TtLlamaAttention for 2D meshes."
        )
        self.tp_axis = 1 if self.cluster_shape[1] != 1 else 0
        self.tp_size = self.cluster_shape[self.tp_axis]

        # --- Model dimensions (read off ModelArgs; populated for qwen3.6) ----
        self.hidden_size = self.args.dim  # 5120
        self.n_heads = self.args.n_heads  # 24
        self.head_dim = self.args.head_dim  # 256
        # HF ships n_kv_heads=4 (unpadded). ModelArgs may already report it as
        # the HF value; pad to the next multiple of tp_size that divides it.
        self.n_kv_heads_unpadded = getattr(self.args, "n_kv_heads_unpadded", self.args.n_kv_heads)  # 4
        self.eps = self.args.norm_eps
        self.max_batch_size = self.args.max_batch_size
        self.max_seq_len = self.args.max_seq_len
        self.scale = self.head_dim**-0.5

        # Partial-RoPE config.
        self.partial_rotary_factor = getattr(self.args, "partial_rotary_factor", 1.0)  # 0.25
        self.rope_dim = getattr(self.args, "rope_dim", int(self.head_dim * self.partial_rotary_factor))  # 64
        self.attn_output_gate = getattr(self.args, "attn_output_gate", True)
        # qwen3.6 q/k norm is zero-centered (HF Qwen3NextRMSNorm: w' = w + 1).
        # The framework ModelArgs defaults rms_norm_add_unit_offset=False and does
        # not set zero_centered_norm; for qwen3.6 the FA qk_norm REQUIRES the +1.
        self.zero_centered_norm = bool(
            getattr(self.args, "zero_centered_norm", getattr(self.args, "rms_norm_add_unit_offset", True))
        )

        # --- KV padding 4 -> tp_size, then per-chip head counts -------------
        # Pad n_kv to the TP size so it divides 8 chips cleanly (1 KV head/chip).
        # repeat_interleave(rep, dim=0) is GQA-preserving (see galaxy V2-TP-1).
        assert self.n_heads % self.tp_size == 0, f"n_heads={self.n_heads} % tp_size={self.tp_size}"
        assert self.tp_size % self.n_kv_heads_unpadded == 0, (
            f"tp_size={self.tp_size} must be a multiple of n_kv_heads_unpadded="
            f"{self.n_kv_heads_unpadded} for the GQA-preserving KV pad"
        )
        self.n_kv_heads = self.tp_size  # 8 (padded)
        self.kv_repeat = self.n_kv_heads // self.n_kv_heads_unpadded  # 2
        self.n_q_per_chip = self.n_heads // self.tp_size  # 3
        self.n_kv_per_chip = self.n_kv_heads // self.tp_size  # 1
        self.q_dim_per_chip = self.n_q_per_chip * self.head_dim  # 768
        self.gate_dim_per_chip = self.n_q_per_chip * self.head_dim  # 768
        self.k_dim_per_chip = self.n_kv_per_chip * self.head_dim  # 256
        self.v_dim_per_chip = self.n_kv_per_chip * self.head_dim  # 256
        self.total_per_chip = (
            self.q_dim_per_chip + self.gate_dim_per_chip + self.k_dim_per_chip + self.v_dim_per_chip
        )  # 2048
        self.gqa_per_chip = self.n_q_per_chip // self.n_kv_per_chip  # 3

        # --- Compute kernel: HiFi4 + fp32 dest accumulation (galaxy FA default).
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        try:
            self._out_num_links = int(os.environ.get("QWEN36_CCL_NUM_LINKS", "1"))
        except ValueError:
            self._out_num_links = 1

        # --- Weights (1D shard) ----------------------------------------------
        weights_dict = self._slice_layer_weights(state_dict, layer_num)
        self._build_weights(weights_dict)

        # --- KV cache (skip when vLLM supplies its own paged cache) ----------
        if not use_paged_kv_cache:
            self._init_kv_cache()

    # ------------------------------------------------------------------
    # Weight slicing + helpers
    # ------------------------------------------------------------------

    def _slice_layer_weights(self, state_dict, layer_num):
        """Pull this layer's self-attention weights out of the full state_dict.

        Accepts either the raw HF prefix ``model.language_model.layers.{n}.self_attn.``
        (what the raw-safetensors PCC test loader uses) or the tt_transformers
        meta-mapped form ``layers.{n}.self_attn.`` / ``layers.{n}.attention.``
        (what ``load_checkpoints`` may emit). Returns a flat dict keyed
        ``self_attn.<rest>`` — the form ``_resolve_weight`` expects.
        """
        if state_dict is None:
            return {}
        candidates = [
            f"model.language_model.layers.{layer_num}.self_attn.",
            f"layers.{layer_num}.self_attn.",
            f"model.layers.{layer_num}.self_attn.",
            f"layers.{layer_num}.attention.",
        ]
        out = {}
        for k, v in state_dict.items():
            for pfx in candidates:
                if k.startswith(pfx):
                    out["self_attn." + k[len(pfx) :]] = v
                    break
        return out

    def _resolve_weight(self, weights_dict, *candidate_keys):
        for k in candidate_keys:
            if k in weights_dict:
                return weights_dict[k]
        raise KeyError(
            f"FullAttention weight not found. Tried: {candidate_keys}. "
            f"Available keys: {sorted(weights_dict.keys())[:20]}..."
        )

    def _to_device(self, t, mapper, dtype=None, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=dtype or self.weight_dtype,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=mapper,
        )

    def _build_qknorm_weight(self, weight_torch):
        """Upload a per-head RMSNorm weight as ``[1, hd//32, 32]`` (replicated).

        Mirrors galaxy ``_build_qwen36_qknorm_weight``: the 3-D layout satisfies
        ``ttnn.rms_norm``'s gamma-volume constraint for a ``[..., hd]`` input,
        and the zero-centered ``w' = w + 1`` (HF Qwen3NextRMSNorm) is baked in
        here so the forward uses a plain ``ttnn.rms_norm`` with no offset op.
        """
        w = weight_torch.float()
        if self.zero_centered_norm:
            w = w + 1.0
        dim = w.numel()
        assert dim == self.head_dim, f"qk_norm weight dim {dim} != head_dim {self.head_dim}"
        assert dim % _QWEN36_TILE == 0, f"head_dim={dim} must be tile-aligned"
        w_3d = w.reshape(1, dim // _QWEN36_TILE, _QWEN36_TILE)
        return ttnn.from_torch(
            w_3d,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_weights(self, sd):
        H = self.hidden_size
        tp = self.tp_size
        hd = self.head_dim
        n_q = self.n_heads  # 24
        n_kv_unp = self.n_kv_heads_unpadded  # 4
        n_kv = self.n_kv_heads  # 8 (padded)
        q_dim_pc = self.q_dim_per_chip  # 768
        k_dim_pc = self.k_dim_per_chip  # 256

        # ---- 1. De-interleave Q and gate from q_proj.weight ------------------
        # HF q_proj.weight is [n_q*2*hd, H] = [12288, 5120], laid out per head as
        # [head0_q(hd) | head0_gate(hd) | head1_q | head1_gate | ...]. The
        # reference splits via reshape(B,T,n_q,2*hd) -> [:hd]=q, [hd:]=gate, i.e.
        # reshape the weight (n_q, 2, hd, H): [:,0]=Q, [:,1]=Gate.
        q_proj_w = self._resolve_weight(sd, "self_attn.q_proj.weight", "self_attn.wq.weight")
        if self.attn_output_gate:
            expected_q = (n_q * 2 * hd, H)
            assert q_proj_w.shape == expected_q, f"q_proj.weight: expected {expected_q}, got {q_proj_w.shape}"
            q_2hd = q_proj_w.reshape(n_q, 2, hd, H)
            wq_native = q_2hd[:, 0, :, :].reshape(n_q * hd, H).contiguous()  # [6144, H]
            wgate_native = q_2hd[:, 1, :, :].reshape(n_q * hd, H).contiguous()  # [6144, H]
        else:
            wq_native = q_proj_w  # [6144, H]
            wgate_native = None

        # ---- 1a. K, V: pad n_kv_unp -> n_kv via GQA-preserving repeat_interleave.
        wk_unp = self._resolve_weight(sd, "self_attn.k_proj.weight", "self_attn.wk.weight")  # [n_kv_unp*hd, H]
        wv_unp = self._resolve_weight(sd, "self_attn.v_proj.weight", "self_attn.wv.weight")
        assert wk_unp.shape == (n_kv_unp * hd, H), f"k_proj.weight: expected {(n_kv_unp*hd, H)}, got {wk_unp.shape}"
        assert wv_unp.shape == (n_kv_unp * hd, H), f"v_proj.weight: expected {(n_kv_unp*hd, H)}, got {wv_unp.shape}"
        # [k0,k1,k2,k3] -> [k0,k0,k1,k1,k2,k2,k3,k3] (rep=2). q_i//3 of the padded
        # head map equals q_i//6 of the original (verified in galaxy V2-TP-1).
        wk_native = wk_unp.view(n_kv_unp, hd, H).repeat_interleave(self.kv_repeat, dim=0).reshape(n_kv * hd, H)
        wv_native = wv_unp.view(n_kv_unp, hd, H).repeat_interleave(self.kv_repeat, dim=0).reshape(n_kv * hd, H)

        # Test-introspection (per-chip weight shapes).
        self.q_proj_weight_shape = tuple(wq_native.shape)
        self.gate_proj_weight_shape = tuple(wgate_native.shape) if wgate_native is not None else None
        self.k_proj_weight_shape = tuple(wk_native.shape)
        self.v_proj_weight_shape = tuple(wv_native.shape)

        # ---- 2. Build the head-sharded QKVG fused weight ---------------------
        # Per chip i: [Q_i(768) | Gate_i(768) | K_i(256) | V_i(256)] = 2048 cols.
        # Concatenate chip blocks along the output dim → [H, tp*2048 = 16384];
        # ShardTensorToMesh(dim=1) carves it into tp contiguous 2048 chunks, one
        # per chip. dim-0 (H) is NOT split (full K per chip → complete matmul).
        wq_T = wq_native.T.contiguous()  # [H, 6144]
        wk_T = wk_native.T.contiguous()  # [H, 2048]
        wv_T = wv_native.T.contiguous()  # [H, 2048]
        wg_T = wgate_native.T.contiguous() if wgate_native is not None else None  # [H, 6144]

        chip_blocks = []
        for i in range(tp):
            q_i = wq_T[:, i * q_dim_pc : (i + 1) * q_dim_pc]  # [H, 768]
            k_i = wk_T[:, i * k_dim_pc : (i + 1) * k_dim_pc]  # [H, 256]
            v_i = wv_T[:, i * k_dim_pc : (i + 1) * k_dim_pc]  # [H, 256]
            if wg_T is not None:
                g_i = wg_T[:, i * q_dim_pc : (i + 1) * q_dim_pc]  # [H, 768]
                chip_blocks.append(torch.cat([q_i, g_i, k_i, v_i], dim=-1))  # [H, 2048]
            else:
                chip_blocks.append(torch.cat([q_i, k_i, v_i], dim=-1))  # [H, 1280]
        wqkvg_T = torch.cat(chip_blocks, dim=-1)  # [H, tp*total_per_chip]
        shard_out_dim1 = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        self.w_qkvg = self._to_device(wqkvg_T, shard_out_dim1)  # per-chip [H, 2048]

        # ---- 3. Output projection: shard the INPUT (head) dim 8-way ----------
        # HF o_proj.weight is [H, n_q*hd=6144]; transpose to [6144, H] then shard
        # dim 0 (the 6144 = 24*256 input/head dim) 8-way → each chip owns its 3
        # Q-heads' 768-row slice. The per-chip matmul gated[768] @ w_out[768, H]
        # is a PARTIAL sum over the chip's heads → completed by the one all_reduce.
        wo_native = self._resolve_weight(sd, "self_attn.o_proj.weight", "self_attn.wo.weight")
        expected_wo = (H, n_q * hd)
        assert wo_native.shape == expected_wo, f"o_proj.weight: expected {expected_wo}, got {wo_native.shape}"
        wo_T = wo_native.T.contiguous()  # [6144, H]
        shard_in_dim0 = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        self.w_out = self._to_device(wo_T, shard_in_dim0)  # per-chip [768, H]
        self.wo_proj_weight_shape = tuple(wo_native.shape)

        # ---- 4. QK-norm weights (replicated; zero-centered baked in) ---------
        q_norm_w = self._resolve_weight(sd, "self_attn.q_norm.weight")
        k_norm_w = self._resolve_weight(sd, "self_attn.k_norm.weight")
        self.q_norm_w = self._build_qknorm_weight(q_norm_w)
        self.k_norm_w = self._build_qknorm_weight(k_norm_w)

    # ------------------------------------------------------------------
    # KV cache constructor (1D head-sharded)
    # ------------------------------------------------------------------

    def _kv_cache_dtype(self):
        # bf16 default. QWEN36_KV_BF8=1 halves the per-step KV read but compounds
        # over the full-attn layers and garbles decode even at ISL-128 (galaxy
        # lesson); keep OFF unless it clears the 128k coherence gate.
        return ttnn.bfloat8_b if os.environ.get("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16

    def _init_kv_cache(self):
        """Allocate the per-chip KV cache: one (padded) KV head per chip.

        n_kv padded 4 → 8; sharded on dim=1 (the KV-head dim) across the TP axis,
        so each chip holds exactly its 1 local KV head. (Paged: dim-0 is blocks.)
        """
        shard_kv = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        n_kv = self.n_kv_heads  # 8
        _kv_dtype = self._kv_cache_dtype()
        if self.paged_attention_config:
            shape = (
                self.paged_attention_config.max_num_blocks,
                n_kv,
                self.paged_attention_config.block_size,
                self.head_dim,
            )
        else:
            shape = (self.max_batch_size, n_kv, self.max_seq_len, self.head_dim)
        cache_k = torch.zeros(*shape)
        cache_v = torch.zeros(*shape)
        self.layer_past = [
            ttnn.from_torch(
                kv,
                device=self.mesh_device,
                dtype=_kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard_kv,
            )
            for kv in (cache_k, cache_v)
        ]

    # ------------------------------------------------------------------
    # Forward-stage helpers
    # ------------------------------------------------------------------

    def _partial_rope_apply(self, x_tt, cos_tt, sin_tt):
        """Apply Qwen3.6 partial RoPE to a ``[..., head_dim]`` device tensor.

        Only the first ``self.rope_dim`` channels of the last dim are rotated;
        the rest pass through. Ported verbatim from
        ``qwen3_6_galaxy_v2/tt/llama_rope.py::partial_rope_apply`` (PCC>0.99 in
        the galaxy single-block tests). ``cos_tt`` / ``sin_tt`` are
        ``[1, 1, T, rope_dim]`` (prefill) or ``[1, 1, 1, rope_dim]`` (decode).
        """
        rd = self.rope_dim  # 64
        hd = self.head_dim  # 256
        shape = list(x_tt.shape)
        assert shape[-1] == hd, f"partial_rope expected last dim={hd}, got {shape[-1]}"
        ndim = len(shape)

        begins_rot = [0] * ndim
        ends_rot = shape[:]
        ends_rot[-1] = rd
        begins_pass = [0] * ndim
        ends_pass = shape[:]
        begins_pass[-1] = rd
        x_rot = ttnn.slice(x_tt, begins_rot, ends_rot)  # [..., rd]
        x_pass = ttnn.slice(x_tt, begins_pass, ends_pass)  # [..., hd-rd]

        half = rd // 2
        shape_rot = list(x_rot.shape)
        begins_x1 = [0] * ndim
        ends_x1 = shape_rot[:]
        ends_x1[-1] = half
        begins_x2 = [0] * ndim
        ends_x2 = shape_rot[:]
        begins_x2[-1] = half
        x1 = ttnn.slice(x_rot, begins_x1, ends_x1)
        x2 = ttnn.slice(x_rot, begins_x2, ends_x2)
        neg_x2 = ttnn.neg(x2)
        rotate_half = ttnn.concat([neg_x2, x1], dim=-1)  # [..., rd]
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)

        x_rot_cos = ttnn.multiply(x_rot, cos_tt)
        x_rotated = ttnn.addcmul(x_rot_cos, rotate_half, sin_tt, value=1.0)
        x_rot.deallocate(True)
        rotate_half.deallocate(True)
        x_rot_cos.deallocate(True)

        out = ttnn.concat([x_rotated, x_pass], dim=-1)  # [..., hd]
        x_rotated.deallocate(True)
        x_pass.deallocate(True)
        return out

    def _project_and_split(self, x_3d, B, T):
        """QKVG projection (complete per chip) + split into Q/Gate/K/V flats.

        ``x_3d`` is the full-H residual (replicated across TP chips). The per-chip
        QKVG matmul has K == full H ⇒ the per-chip output is COMPLETE (no
        input-side all_reduce, unlike galaxy's col-axis reduce). Per-chip layout
        of the 2048-wide output: ``[Q(768) | Gate(768) | K(256) | V(256)]``.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        xqkvg = ttnn.linear(
            x_3d, self.w_qkvg, dtype=ttnn.bfloat16, memory_config=mem, compute_kernel_config=self.compute_kernel
        )
        if len(list(xqkvg.shape)) == 4:
            _, _, _Tq, _Nq = list(xqkvg.shape)
            xqkvg = ttnn.reshape(xqkvg, [B, _Tq, _Nq])

        q_dim = self.q_dim_per_chip
        g_dim = self.gate_dim_per_chip
        k_dim = self.k_dim_per_chip
        total = self.total_per_chip

        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim], memory_config=mem)
        if self.attn_output_gate:
            gate_flat = ttnn.slice(xqkvg, [0, 0, q_dim], [B, T, q_dim + g_dim], memory_config=mem)
            k_start = q_dim + g_dim
        else:
            gate_flat = None
            k_start = q_dim
        k_flat = ttnn.slice(xqkvg, [0, 0, k_start], [B, T, k_start + k_dim], memory_config=mem)
        v_flat = ttnn.slice(xqkvg, [0, 0, k_start + k_dim], [B, T, total], memory_config=mem)
        xqkvg.deallocate(True)
        return q_flat, gate_flat, k_flat, v_flat

    def _qknorm_and_rope(self, q_flat, k_flat, v_flat, cos_tt, sin_tt, B, T):
        """QK-norm (per head) → partial RoPE on Q,K; V to heads (no norm)."""
        n_q_pc = self.n_q_per_chip
        n_kv_pc = self.n_kv_per_chip
        hd = self.head_dim

        q_normed = _qwen36_qknorm_flat_to_heads(q_flat, self.q_norm_w, self.eps, B, n_q_pc, T, hd, self.compute_kernel)
        k_normed = _qwen36_qknorm_flat_to_heads(k_flat, self.k_norm_w, self.eps, B, n_kv_pc, T, hd, self.compute_kernel)
        v_t = _qwen36_flat_to_heads(v_flat, B, n_kv_pc, T, hd)  # [B, n_kv_pc, T, hd]

        # cos/sin built at this window's seq length; slice if the cached table is
        # wider (matches galaxy: model.forward slices to [start:start+T]).
        if list(cos_tt.shape)[-2] != T:
            _cs = list(cos_tt.shape)
            _ss = list(sin_tt.shape)
            cos_tt = ttnn.slice(cos_tt, [0, 0, 0, 0], [_cs[0], _cs[1], T, _cs[3]])
            sin_tt = ttnn.slice(sin_tt, [0, 0, 0, 0], [_ss[0], _ss[1], T, _ss[3]])

        q_rot = self._partial_rope_apply(q_normed, cos_tt, sin_tt)
        k_rot = self._partial_rope_apply(k_normed, cos_tt, sin_tt)
        q_normed.deallocate(True)
        k_normed.deallocate(True)
        return q_rot, k_rot, v_t

    def _gate_and_output(self, attn_out, gate_flat, B, T):
        """Output gate (sigmoid(gate)*attn) → WO projection → single all_reduce."""
        mem = ttnn.DRAM_MEMORY_CONFIG
        attn_flat = _qwen36_heads_to_flat(attn_out, B, self.n_q_per_chip, T, self.head_dim)  # [B, T, 768]
        if self.attn_output_gate and gate_flat is not None:
            gate_sig = ttnn.sigmoid(gate_flat, memory_config=mem)
            gated = ttnn.multiply(attn_flat, gate_sig, memory_config=mem)
            attn_flat.deallocate(True)
            gate_sig.deallocate(True)
        else:
            gated = attn_flat

        partial = ttnn.linear(
            gated, self.w_out, dtype=ttnn.bfloat16, memory_config=mem, compute_kernel_config=self.compute_kernel
        )
        gated.deallocate(True)
        # 1D-TP reduce_scatter (framework tt_all_reduce on a (1,tp) mesh): SUM the
        # tp partials and SCATTER H → fractured [.,.,T,H/tp] matching the framework
        # residual stream (DefaultAttention uses tt_all_reduce(cluster_axis=0, dim=3)).
        # tt_all_reduce reduce_scatters dim=3 → input must be 4D [1,1,M,H].
        if len(partial.shape) == 3:
            _b, _t, _h = list(partial.shape)
            partial = ttnn.reshape(partial, [1, 1, _b * _t, _h])
        reduced = tt_all_reduce(
            partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self._out_num_links,
            memory_config=mem,
            dtype=ttnn.bfloat16,
        )
        partial.deallocate(True)
        return reduced

    # ------------------------------------------------------------------
    # Public API — matches the tt_transformers attention surface
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        **kwargs,
    ):
        # ``mode`` may be the string "prefill"/"decode" OR the tt_transformers Mode enum.
        mode_val = mode.value if hasattr(mode, "value") else mode
        if mode_val == "prefill":
            return self.forward_prefill(x, rot_mats, user_id=user_id, page_table=page_table, kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def forward_prefill(self, x, rot_mats, user_id=0, page_table=None, kv_cache=None):
        """Prefill (T>1): QKVG → qk-norm → partial RoPE → KV fill → causal SDPA
        → output gate → WO + single all_reduce.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        # 1. QKVG projection + split (complete per chip; no input all_reduce).
        q_flat, gate_flat, k_flat, v_flat = self._project_and_split(x_3d, B, T)

        # 2. QK-norm + partial RoPE; V to heads.
        q_rot, k_rot, v_t = self._qknorm_and_rope(q_flat, k_flat, v_flat, cos_tt, sin_tt, B, T)
        q_flat.deallocate(True)
        k_flat.deallocate(True)
        v_flat.deallocate(True)

        # 3. KV cache fill (per chip, 1 KV head). bf16 producer into bf16/bf8
        #    cache (kernel quantizes on write for bf8).
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]
        if page_table is not None:
            # The compute seq (T) is tile/chunk-padded, but the page table covers only
            # page_len = num_blocks * block_size valid tokens (the unpadded prompt rounded
            # up to a block). Slice K/V to page_len before the fill, exactly as
            # DefaultAttention does (attention.py:1059-1062) — else paged_fill_cache
            # asserts input_shape[2] <= block_size * page_table_shape[1].
            block_size = keys_cache.shape[2]
            page_len = page_table.shape[1] * block_size
            k_fill = k_rot[:, :, :page_len, :] if page_len < k_rot.shape[2] else k_rot
            v_fill = v_t[:, :, :page_len, :] if page_len < v_t.shape[2] else v_t
            if isinstance(user_id, ttnn.Tensor):
                ttnn.experimental.paged_fill_cache(keys_cache, k_fill, page_table, batch_idx_tensor=user_id)
                ttnn.experimental.paged_fill_cache(values_cache, v_fill, page_table, batch_idx_tensor=user_id)
            else:
                ttnn.experimental.paged_fill_cache(keys_cache, k_fill, page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(values_cache, v_fill, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys_cache, k_rot, user_id % max(self.max_batch_size, 1))
            ttnn.fill_cache(values_cache, v_t, user_id % max(self.max_batch_size, 1))

        # 4. GQA expand (1 KV head -> 3 Q heads) + causal SDPA.
        k_exp = ttnn.repeat_interleave(k_rot, self.gqa_per_chip, dim=1, memory_config=mem)
        v_exp = ttnn.repeat_interleave(v_t, self.gqa_per_chip, dim=1, memory_config=mem)
        k_rot.deallocate(True)
        v_t.deallocate(True)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_exp,
            v_exp,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel,
            memory_config=mem,
        )
        k_exp.deallocate(True)
        v_exp.deallocate(True)
        q_rot.deallocate(True)

        # 5. Output gate + WO + single all_reduce.
        output = self._gate_and_output(attn_out, gate_flat, B, T)
        attn_out.deallocate(True)
        if gate_flat is not None:
            gate_flat.deallocate(True)
        # Match the decoder/DefaultAttention prefill contract: [1, 1, B*T, H].
        if len(output.shape) == 3:
            _Bo, _To, _Ho = list(output.shape)
            output = ttnn.reshape(output, [1, 1, _Bo * _To, _Ho])
        return output

    def forward_decode(self, x, current_pos, rot_mats, page_table=None, kv_cache=None):
        """Decode (T=1): QKVG → qk-norm → partial RoPE → KV update → paged SDPA
        decode → output gate → WO + single all_reduce.

        Single-user (max_batch_size==1) path. ``current_pos`` is the per-user
        position tensor; ``page_table`` selects paged KV blocks. (Batch-N decode
        — the galaxy height-sharded per-user KV-write/SDPA routing — is NOT
        ported here; this 1D bring-up targets single-user decode. Extend the
        KV-write/SDPA section with the galaxy ``paged_fused_update_cache`` +
        SCORES_BATCHED_MM_OUTPUT_MEMCFG path when batch-N is needed.)
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        H = orig_shape[-1]
        R = orig_shape[-2]  # tile-padded decode row slot (=32 even at batch-1)
        Bn = self.max_batch_size
        # The decode batch (and the paged page_table batch) is the REAL batch Bn,
        # NOT the padded row slot R. Extract the Bn valid users into dim-1 so the
        # KV-update / SDPA-decode batch matches the page_table (DefaultAttention's
        # convention; R is only tile padding). Output is padded back to R rows.
        x_3d = ttnn.reshape(x, [1, R, H])
        if R != Bn:
            x_3d = ttnn.slice(x_3d, [0, 0, 0], [1, Bn, H], memory_config=mem)
        # B=1 outer, T carries the Bn users (the SDPA-decode batch dim).
        B, T = 1, Bn
        N = T

        # 1. QKVG projection + split (complete per chip).
        q_flat, gate_flat, k_flat, v_flat = self._project_and_split(x_3d, B, T)

        # 2. QK-norm + partial RoPE; V to heads. q_rot/k_rot/v_t: [B, n*pc, N, hd].
        q_rot, k_rot, v_t = self._qknorm_and_rope(q_flat, k_flat, v_flat, cos_tt, sin_tt, B, T)
        q_flat.deallocate(True)
        k_flat.deallocate(True)
        v_flat.deallocate(True)

        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        # 3. KV update.
        if page_table is not None:
            # Paged update: the op reads batch from dim-1 and routes per-user via
            # page_table + current_pos. Permute the N users (dim-2) into dim-1:
            # [B, n_kv_pc, N, hd] -> [B, N, n_kv_pc, hd]. At N==1 this is
            # [1,1,n_kv_pc,hd] (single-user). bf16 producer (op rejects bf8).
            k_kv = ttnn.permute(k_rot, (0, 2, 1, 3), memory_config=mem)
            v_kv = ttnn.permute(v_t, (0, 2, 1, 3), memory_config=mem)
            k_rot.deallocate(True)
            v_t.deallocate(True)
            # paged_update_cache requires the producer HEIGHT_SHARDED (one batch
            # row per core, [TILE, head_dim] shard) — the layout
            # nlp_create_qkv_heads_decode emits and get_attn_create_head_output_mem_config
            # describes (we build it directly since that helper computes
            # n_local_kv_heads = n_kv(4)//num_devices(8) = 0 for qwen3.6).
            _kv_shard = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.head_dim),
                core_grid=ttnn.CoreGrid(y=1, x=max(self.max_batch_size, 1)),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k_kv = ttnn.to_memory_config(k_kv, _kv_shard)
            v_kv = ttnn.to_memory_config(v_kv, _kv_shard)
            ttnn.experimental.paged_update_cache(
                keys_cache, k_kv, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values_cache, v_kv, update_idxs_tensor=current_pos, page_table=page_table
            )
            k_kv.deallocate(True)
            v_kv.deallocate(True)
        else:
            if isinstance(current_pos, int):
                _pos = current_pos
            else:
                _pos = int(
                    ttnn.to_torch(current_pos, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0].item()
                )
            ttnn.update_cache(keys_cache, k_rot, _pos, batch_offset=0)
            ttnn.update_cache(values_cache, v_t, _pos, batch_offset=0)
            k_rot.deallocate(True)
            v_t.deallocate(True)

        # 4. SDPA decode. q must be [1, B, n_q_pc, hd] (batch in dim-1); q_rot is
        #    [B, n_q_pc, N, hd], permute (0,2,1,3) -> [B, N, n_q_pc, hd].
        q_1bnd = ttnn.permute(q_rot, (0, 2, 1, 3), memory_config=mem)
        q_rot.deallocate(True)
        # SDPA-decode needs an explicit program config: without it the kernel
        # picks a KV-reduction split that overflows num_tree_reduction_rounds
        # (esp. at a 32k paged cache). Reuse the framework's decode prog cfg.
        _sdpa_pc = self.args.get_attn_sdpa_decode_program_config(None)
        if page_table is not None:
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_1bnd,
                keys_cache,
                values_cache,
                cur_pos_tensor=current_pos,
                page_table_tensor=page_table,
                scale=self.scale,
                program_config=_sdpa_pc,
                compute_kernel_config=self.compute_kernel,
                memory_config=mem,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_1bnd,
                keys_cache,
                values_cache,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=_sdpa_pc,
                compute_kernel_config=self.compute_kernel,
                memory_config=mem,
            )
        q_1bnd.deallocate(True)

        # SDPA decode returns [1, B, n_q_pc, hd]; bring it back to [B, n_q_pc, N, hd]
        # so _qwen36_heads_to_flat (and the gate, which is [B, N, n_q_pc*hd]) line up.
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3), memory_config=mem)  # [1, n_q_pc, B, hd] == [B', n_q_pc, N, hd]

        # 5. Output gate + WO + single all_reduce → [1, 1, Bn, H/tp].
        output = self._gate_and_output(attn_out, gate_flat, B, T)
        attn_out.deallocate(True)
        if gate_flat is not None:
            gate_flat.deallocate(True)
        # Pad the Bn user rows back to the tile-padded R-row decode slot (rows
        # Bn..R-1 zero, unused) so it matches the [1, 1, R, H/tp] residual stream.
        if len(output.shape) == 3:
            _bo, _to, _ho = list(output.shape)
            output = ttnn.reshape(output, [1, 1, _bo * _to, _ho])
        if R != T:
            output = ttnn.pad(output, [(0, 0), (0, 0), (0, R - T), (0, 0)], value=0.0)
        return output


class Qwen36RopeSetup(HfRotarySetup):
    """RoPE setup for Qwen3.6 full-attention: PARTIAL rotary (rope_dim = head_dim *
    0.25 = 64), HF/NeoX cos-sin layout (matches ``_partial_rope_apply``).

    The framework builds the rope_setup with ``head_dim=args.head_dim`` (256); we
    rebuild the cos/sin tables at ``rope_dim`` so ``rot_mats`` arrive as
    ``[1, 1, T, 64]`` — exactly what the full-attn block's partial-RoPE consumes.

    mRoPE note: Qwen3.6 uses interleaved mRoPE (mrope_section [11,11,10]) over 3
    position axes. For TEXT input all three axes carry the same positions, so mRoPE
    collapses to standard 1D RoPE — which is what this (single-axis) setup produces.
    Text-only ``simple_text_demo`` is therefore exact; true multi-axis mRoPE (vision)
    would need the per-axis section split (port qwen36_mrope.py into get_rot_mats).
    """

    _PARTIAL_ROTARY_FACTOR = 0.25

    def __init__(self, device, batch_size, head_dim, max_seq_len, rope_theta, *args, **kwargs):
        rope_dim = int(head_dim * self._PARTIAL_ROTARY_FACTOR)
        super().__init__(device, batch_size, rope_dim, max_seq_len, rope_theta, *args, **kwargs)
