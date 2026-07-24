# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 transformer backbone.
# Mirrors HunyuanImage3Model in
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#
#     hidden = wte(input_ids)                # or caller-supplied inputs_embeds
#     for layer in layers:                   # 32 identical MoE decoder layers
#         hidden = layer(hidden, rope, mask)
#     hidden = ln_f(hidden)                  # OPTIONAL — see note below
#
# Note on ln_f: upstream applies the final RMSNorm *outside* the model in the
# image-generation path (see HunyuanImage3Model.forward — the ln_f call is
# commented out). We mirror that by making the final norm opt-in via
# `apply_final_norm` (default True for a standalone LM backbone; pass False to
# match the image-gen call site).
#
# Memory: with stream_experts=True each MoE rebuilds experts from host weights
# every forward. Retaining those tensors for all 32 layers pins ~150–200GB and
# gets OOM-killed mid-load; HunyuanTtModel therefore binds an on-demand disk
# expert loader per layer and drops the retained host tensors after upload of
# gate/shared/attention weights. `layer_loader(i)` returns the state_dict for
# layer i, keyed `model.layers.{i}.*`.

import gc
import os
import time

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.hunyuan_image_3_0.ref.model_config import BACKBONE_KWARGS
from models.experimental.hunyuan_image_3_0.ref.weights import load_tensors, resolve_base_model_dir

from .transformer_layer import HunyuanTtDecoderLayer
from .attention.rms_norm import HunyuanTtRMSNorm
from .parallel_utils import resid_mem_config, sp_gather, sp_shard
from .cache import cache_file, resolve_transformer_cache

_BD = BACKBONE_KWARGS


def _disk_expert_loader(layer_idx: int, moe_prefix: str, model_dir=None):
    """Return ``loader(expert_idx) -> state_dict`` reading one expert from safetensors.

    Cast to float32 to match ``load_prefixed_state_dict`` (the path used for
    gate/shared/attention upload); native bf16→as_tensor can diverge enough to
    flip MoE top-k on single-token decode. ``model_dir`` must be the SAME checkpoint
    the rest of the backbone (layer_loader) was built from — resolving it
    independently risks picking a different (e.g. base vs. instruct) checkpoint when
    both are cached, silently corrupting MoE output.
    """
    src = model_dir if model_dir is not None else resolve_base_model_dir()

    def load_one(expert_idx: int) -> dict:
        keys = [
            f"{moe_prefix}.experts.{expert_idx}.gate_and_up_proj.weight",
            f"{moe_prefix}.experts.{expert_idx}.down_proj.weight",
        ]
        return {k: v.to(torch.float32) for k, v in load_tensors(src, keys).items()}

    return load_one


def default_bf16_layers(num_layers: int) -> set[int]:
    """Return the default mixed-precision boundary for a resident backbone.

    Keep at most 3 bf16 layers at each end. A 4th *consecutive* bf16 layer during
    device upload exhausts per-chip DRAM and hangs (bf8 layers are slow ~20s but OK).
    """
    n = max(0, num_layers)
    if n <= 3:
        return set(range(n))
    head = {0, 1, 2}
    if n <= 6:
        return head
    return head | {n - 3, n - 2, n - 1}


class HunyuanTtModel(LightweightModule):
    def __init__(
        self,
        device,
        *,
        num_layers: int = _BD["num_layers"],
        hidden_size: int = _BD["hidden_size"],
        num_heads: int = _BD["num_heads"],
        num_kv_heads: int = _BD["num_kv_heads"],
        head_dim: int = _BD["head_dim"],
        num_experts: int = _BD["num_experts"],
        moe_topk: int = _BD["moe_topk"],
        use_qk_norm: bool = _BD["use_qk_norm"],
        use_mixed_mlp_moe: bool = _BD["use_mixed_mlp_moe"],
        norm_topk_prob: bool = _BD["norm_topk_prob"],
        rms_norm_eps: float = _BD["rms_norm_eps"],
        weight_dtype=ttnn.bfloat16,
        stream_experts: bool = True,
        layer_loader=None,
        embed_state_dict: dict = None,
        norm_state_dict: dict = None,
        apply_final_norm: bool = True,
        ccl_manager=None,
        expert_mesh_axis: int = 1,
        tp_axis: int = 1,
        tp_factor: int = 1,
        sp_axis: int = 0,
        sp_factor: int = 1,
        bf16_layers=None,
        weight_cache_path=None,
        model_cache_name: str = "hunyuan-image-3.0",
        model_dir=None,
    ):
        """
        Args:
            device:           TTNN device.
            num_layers:       number of decoder layers to stack (1..32).
            layer_loader:     callable(layer_idx) -> state_dict for that layer,
                              keyed `model.layers.{layer_idx}.*`. Called once per
                              layer at construction.
            embed_state_dict: dict containing `model.wte.weight` ([V, H]). Required
                              only if forward() is called with input_ids.
            norm_state_dict:  dict containing `model.ln_f.weight`. Required only if
                              apply_final_norm is True.
            apply_final_norm: apply ln_f at the end (LM backbone). Pass False to
                              match the image-generation call site.
            weight_cache_path: Optional explicit cache directory for pre-tilized
                              ``.tensorbin`` weights. When ``None`` and
                              ``TT_DIT_CACHE_DIR`` is set, a path is derived from
                              mesh shape, parallelism, dtype, and ``bf16_layers``.
            model_cache_name: Subdirectory under ``TT_DIT_CACHE_DIR`` for this
                              checkpoint variant (e.g. ``hunyuan-image-3.0-instruct``).
        """
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.apply_final_norm = apply_final_norm
        # Sequence parallel: split the token sequence across `sp_axis`. All SP
        # plumbing is contained in forward() — inputs arrive replicated (full S) and
        # outputs are returned replicated, so the pipeline/demo are unaffected.
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor

        bf16_layers = set(bf16_layers or [])
        self.weight_cache_path = resolve_transformer_cache(
            model_name=model_cache_name,
            device=device,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
            weight_dtype=weight_dtype,
            num_layers=num_layers,
            bf16_layers=bf16_layers,
            weight_cache_path=weight_cache_path,
        )
        if self.weight_cache_path is not None:
            self.weight_cache_path.mkdir(parents=True, exist_ok=True)
            if os.environ.get("HY_VERBOSE", "1") != "0":
                print(f"[backbone] TT_DIT cache dir: {self.weight_cache_path}", flush=True)

        # Token embedding table (ROW_MAJOR weight; ttnn.embedding emits TILE).
        # bf8/bf4 require TILE layout and cannot back a ROW_MAJOR embedding table.
        self.embed_weight = None
        if embed_state_dict is not None:
            w = embed_state_dict["model.wte.weight"]  # [V, H]
            embed_dtype = weight_dtype
            if embed_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                embed_dtype = ttnn.bfloat16
            is_mesh = device.__class__.__name__ == "MeshDevice"
            self.embed_weight = ttnn.as_tensor(
                w,
                dtype=embed_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
                cache_file_name=cache_file(self.weight_cache_path, "model.wte.weight"),
            )

        if layer_loader is None:
            raise ValueError("layer_loader is required")

        self.layers = []
        verbose = os.environ.get("HY_VERBOSE", "1") != "0"
        for i in range(num_layers):
            if verbose:
                print(f"[backbone] loading layer {i + 1}/{num_layers} ...", flush=True)
            t_layer = time.time()
            sd = layer_loader(i)
            layer_dtype = ttnn.bfloat16 if i in bf16_layers else weight_dtype
            layer = HunyuanTtDecoderLayer(
                device,
                sd,
                layer_num=i,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                moe_topk=moe_topk,
                use_qk_norm=use_qk_norm,
                use_mixed_mlp_moe=use_mixed_mlp_moe,
                norm_topk_prob=norm_topk_prob,
                rms_norm_eps=rms_norm_eps,
                weight_dtype=layer_dtype,
                stream_experts=stream_experts,
                ccl_manager=ccl_manager,
                expert_mesh_axis=expert_mesh_axis,
                tp_axis=tp_axis,
                tp_factor=tp_factor,
                sp_axis=sp_axis,
                sp_factor=sp_factor,
                weight_cache_path=self.weight_cache_path,
            )
            # Single-device streaming MoE: drop ~layer-sized host expert pack and
            # reload one expert at a time from disk on forward (avoids ~200GB RSS).
            if stream_experts and ccl_manager is None and hasattr(layer.mlp, "bind_expert_loader"):
                moe_prefix = f"model.layers.{i}.mlp"
                layer.mlp.bind_expert_loader(_disk_expert_loader(i, moe_prefix, model_dir))
            self.layers.append(layer)
            del sd
            gc.collect()
            if verbose:
                dt = time.time() - t_layer
                note = " (bf8 upload is slow; not stuck)" if layer_dtype != ttnn.bfloat16 and dt > 5 else ""
                print(f"[backbone] layer {i + 1}/{num_layers} ready ({dt:.1f}s){note}", flush=True)

        self.ln_f = None
        if apply_final_norm:
            if norm_state_dict is None:
                raise ValueError("norm_state_dict with 'model.ln_f.weight' is required when apply_final_norm=True")
            self.ln_f = HunyuanTtRMSNorm(
                device,
                hidden_size,
                norm_state_dict,
                "model.ln_f",
                eps=rms_norm_eps,
                weight_cache_path=self.weight_cache_path,
            )

    def embed(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Embed input_ids ([B, S] uint32, ROW_MAJOR) -> [B, S, H] TILE."""
        if self.embed_weight is None:
            raise ValueError("model was built without embed_state_dict; pass inputs_embeds to forward() instead")
        # Normalise to rank-3 [B, S, H] (ttnn.embedding may emit a leading 1-dim);
        # downstream attention assumes a 3-D hidden tensor.
        bsz = input_ids.shape[0]
        seq = input_ids.shape[-1]
        # Gate with resid_mem_config(S): an L1 residual at long S pins the buffer
        # floor under SDPA/matmul static CBs (CB end > lowest L1 addr → clash).
        # Matches the SP entry path and RESID_L1_MAX_SEQ in parallel_utils.
        emb = ttnn.embedding(input_ids, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=resid_mem_config(seq))
        return ttnn.reshape(emb, [bsz, seq, self.hidden_size])

    def forward(
        self,
        input_ids: ttnn.Tensor = None,
        *,
        inputs_embeds: ttnn.Tensor = None,
        seq_len: int,
        image_infos=None,
        attention_mask=None,
        kv_cache=None,
        use_cache: bool = False,
        decode_step: bool = False,
        cos_sin=None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids:      [B, S] uint32 ROW_MAJOR token ids (embedded on device).
            inputs_embeds:  [B, S, H] TILE — pre-embedded hidden states; takes
                            precedence over input_ids (matches the image-gen path).
            seq_len:        S — used to build the shared 2D RoPE tables.
            image_infos:    per-batch image span info for 2D RoPE (None => text-only).
            attention_mask: optional additive mask [B,1,S,S]; None => causal SDPA.
        Returns:
            [B, S, H] TTNN tensor (final hidden states, ln_f applied iff
            apply_final_norm).
        """
        # Track whether `hidden` is caller-owned (inputs_embeds) — if so, don't
        # free it after the first layer.
        if inputs_embeds is not None:
            hidden = inputs_embeds
            caller_owns_hidden = True
            # Spill L1 inputs_embeds when S is above the residual CB-clash bound so
            # the live residual does not sit under layer-0 SDPA/matmul CBs.
            target_mc = resid_mem_config(int(hidden.shape[1]))
            if (
                hidden.memory_config().buffer_type == ttnn.BufferType.L1
                and target_mc.buffer_type == ttnn.BufferType.DRAM
            ):
                spilled = ttnn.to_memory_config(hidden, target_mc)
                if spilled is not hidden:
                    # Take ownership of the DRAM copy; leave caller tensor alone.
                    hidden = spilled
                    caller_owns_hidden = False
        elif input_ids is not None:
            hidden = self.embed(input_ids)
            caller_owns_hidden = False
        else:
            raise ValueError("provide either input_ids or inputs_embeds")

        # Build the 2D RoPE tables once and share them across all layers.
        owns_cos_sin = cos_sin is None
        if cos_sin is not None:
            cos_tt, sin_tt = cos_sin
        else:
            cos_tt, sin_tt = self.layers[0].self_attn.rope.prepare_cos_sin(seq_len, image_infos=image_infos)

        # --- Sequence-parallel entry reshard --------------------------------
        # Split the replicated inputs across sp_axis: hidden + cos/sin on the seq
        # dim, the mask on its QUERY dim (keys stay full so each device attends to
        # the whole sequence). Outputs are gathered back to full S before return.
        sp = self.sp_factor > 1
        sp_owned = []
        sp_pad = 0  # tokens padded so each shard is tile-aligned (sliced off at exit)
        if sp:
            n, ax, ccl = self.sp_factor, self.sp_axis, self.ccl_manager
            # Each shard must be tile-aligned, so pad S up to a multiple of n*TILE.
            # The real gen sequence (e.g. 4107) is neither even nor tile-aligned.
            TILE = 32
            mult = n * TILE
            S_pad = ((seq_len + mult - 1) // mult) * mult
            sp_pad = S_pad - seq_len
            mask_already_sp_sharded = False
            if attention_mask is not None:
                mshape = list(attention_mask.shape)
                # Fast path: caller provided a query-sharded mask [B,1,S_pad/n,S_pad]
                # (uploaded with a mesh mapper), so skip full-mask pad/shard.
                mask_already_sp_sharded = len(mshape) >= 4 and mshape[2] == (S_pad // n) and mshape[3] == S_pad
            if sp_pad:
                # hidden/cos/sin: zero-pad the seq dim (padded query outputs are
                # discarded at exit; padded keys are masked out below).
                hidden = ttnn.pad(hidden, [(0, 0), (0, sp_pad), (0, 0)], value=0.0)
                cos_tt = ttnn.pad(cos_tt, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
                sin_tt = ttnn.pad(sin_tt, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
                if attention_mask is not None and not mask_already_sp_sharded:
                    # Mask the padded KEY columns (-1e30) so real queries ignore the
                    # padding; padded query ROWS can be anything (sliced off later).
                    attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, 0), (0, 0), (0, sp_pad)], value=-1.0e30)
                    attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
            # hidden feeds the input_layernorm next: land it L1-resident to avoid a
            # DRAM->L1 copy — but only up to the measured residual-stream CB-clash bound
            # (per-device seq S_pad/n); above it the layer input must be DRAM too, else
            # the input_layernorm's CBs collide with this resident input. cos/sin/mask
            # keep the DRAM default (the SDPA mask MUST be DRAM).
            hidden = sp_shard(ccl, hidden, dim=1, mesh_axis=ax, n=n, out_memory_config=resid_mem_config(S_pad // n))
            caller_owns_hidden = False  # we created a fresh sharded tensor
            cos_tt = sp_shard(ccl, cos_tt, dim=2, mesh_axis=ax, n=n, out_memory_config=ttnn.L1_MEMORY_CONFIG)
            sin_tt = sp_shard(ccl, sin_tt, dim=2, mesh_axis=ax, n=n, out_memory_config=ttnn.L1_MEMORY_CONFIG)
            if attention_mask is not None:
                if not mask_already_sp_sharded:
                    attention_mask = sp_shard(ccl, attention_mask, dim=2, mesh_axis=ax, n=n)  # [B,1,S_pad/n,S_pad]
                    sp_owned.append(attention_mask)

        for layer in self.layers:
            nxt = layer(
                hidden,
                seq_len=seq_len,
                image_infos=image_infos,
                attention_mask=attention_mask,
                cos_sin=(cos_tt, sin_tt),
                kv_cache=kv_cache,
                use_cache=use_cache,
                decode_step=decode_step,
            )
            if not caller_owns_hidden:
                ttnn.deallocate(hidden)
            caller_owns_hidden = False
            hidden = nxt

        if owns_cos_sin or sp:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)

        # --- Sequence-parallel exit gather ----------------------------------
        # Re-assemble the full (replicated) sequence so the caller/pipeline sees the
        # same [B, S, H] contract as the non-SP path.
        if sp:
            full = sp_gather(self.ccl_manager, hidden, dim=1, mesh_axis=self.sp_axis, n=self.sp_factor)
            ttnn.deallocate(hidden)
            hidden = full  # [B, S_pad, H]
            if sp_pad:
                # Drop the padding rows -> back to the real [B, S, H] contract.
                shp = list(hidden.shape)
                unpadded = ttnn.slice(hidden, [0, 0, 0], [shp[0], shp[1] - sp_pad, shp[2]])
                ttnn.deallocate(hidden)
                hidden = unpadded
            for t in sp_owned:
                ttnn.deallocate(t)

        if self.apply_final_norm and self.ln_f is not None:
            normed = self.ln_f(hidden)
            ttnn.deallocate(hidden)
            hidden = normed

        return hidden
