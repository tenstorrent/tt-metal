# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared chained TTNN pipeline for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

ONE forward path, imported by BOTH demo/ and tests/e2e/ so a green test
guarantees a working demo (no wiring drift).

Task head: causal text generation (NemotronHForCausalLM). The backbone is the
graduated `nemotron_h_model` stub (52-layer hybrid Mamba2 / GQA-attn / MoE with
lazy per-layer weight load+evict — 30B does not fit on device at once). On top
we add the untied lm_head and a greedy decode loop.

Two modes, SAME outer code:
  * compose=False : backbone = nemotron_h_model.__call__ (monolith; fastest;
                    invokes the nemotron_h_model graduated stub).
  * compose=True  : drive the 52-layer loop here and route each layer's mixer
                    through the graduated CHILD stubs so EVERY graduated module
                    is invoked (Gate 2):
                      - M-layers -> nemotron_h_mamba2_mixer (-> mamba_r_m_s_norm_gated)
                        and at least one M-layer -> nemotron_h_block
                      - E-layers -> nemotron_h_m_o_e (-> nemotron_h_topk_router,
                        re_l_u_squared_activation)
                      - *-layers -> attention (REUSE, via the monolith helper)
                    The monolith (nemotron_h_model) still provides the
                    embedding, the pre/final RMSNorm and the residual scaffold,
                    so it too is invoked.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16._stubs import nemotron_h_block as _block_stub
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16._stubs import nemotron_h_m_o_e as _moe_stub
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16._stubs import nemotron_h_mamba2_mixer as _mamba_stub
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16._stubs import nemotron_h_model as _model_stub
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import _invocation

# NOTE: QKV-attention-reshard optimization (ViT pattern) does not apply to NemotronH.
# This model uses Mamba2 SSD mixers (not QKV-based attention) and conventional
# attention layers are handled via REUSE (not synthesized). No grid resharding needed.

HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# graduated modules that MUST be invoked in a composed run (Gate 2)
GRADUATED_MODULES = (
    "nemotron_h_model",
    "nemotron_h_block",
    "nemotron_h_mamba2_mixer",
    "mamba_r_m_s_norm_gated",
    "nemotron_h_m_o_e",
    "nemotron_h_topk_router",
    "re_l_u_squared_activation",
)


def _ckcfg():
    try:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:
        try:
            return ttnn.GrayskullComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except Exception:
            return None


def open_pipeline_mesh(l1_small_size=24576, rows=2, cols=2):
    """Open the 4-chip DPxTP mesh (rows=DP, cols=TP) with the inter-chip fabric
    enabled and the shard runner active, so the graduated Phase-2 shard stubs
    shard the MoE experts on the TP axis and all_reduce. Returns (device, is_mesh).
    Falls back to a single device (TP=1, everything replicated == native)."""
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=l1_small_size)
        os.environ["TT_HW_PLANNER_SHARD_RUN"] = "1"
        print(f"[pipeline] opened MeshDevice shape={list(dev.shape)} DP={rows} TP={cols} FABRIC_1D shard_active=True", flush=True)
        return dev, True
    except Exception as e:
        print(f"[pipeline] mesh open failed ({e}); falling back to single device (TP=1, replicated)", flush=True)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
        os.environ.pop("TT_HW_PLANNER_SHARD_RUN", None)
        dev = ttnn.open_device(device_id=0)
        return dev, False


def close_pipeline_mesh(dev, is_mesh):
    if is_mesh:
        ttnn.close_mesh_device(dev)
    else:
        ttnn.close_device(dev)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass


class NemotronHPipeline:
    def __init__(self, device, hf_model, compose=True):
        self.device = device
        self.hf = hf_model
        self.cfg = hf_model.config
        self.backbone = hf_model.backbone
        self.compose = compose
        self.invoked = set()
        self._ckc = _ckcfg()
        try:
            self._is_mesh = isinstance(device, ttnn.MeshDevice)
        except AttributeError:
            self._is_mesh = False
        self.shard_active = bool(os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and self._is_mesh

        # Backbone driver = graduated nemotron_h_model stub. The compose path
        # uses M's embedding, RMSNorm, attention helper and residual scaffold,
        # so the nemotron_h_model stub is genuinely part of every run.
        self.M = _model_stub.build(device, self.backbone)
        self.invoked.add("nemotron_h_model")
        _invocation.record("nemotron_h_model")

        # Untied lm_head: (vocab, hidden) -> store transposed (hidden, vocab) bf16.
        lm_w = hf_model.lm_head.weight.detach().t().contiguous()
        _lm_kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if self._is_mesh:
            _lm_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        self._lm_head = ttnn.from_torch(lm_w.to(torch.bfloat16), **_lm_kw)
        self.vocab = int(hf_model.lm_head.weight.shape[0])
        self.hidden = int(self.cfg.hidden_size)

        # per-layer child instance caches (built lazily, evicted after each layer)
        self._mixer_cache = {}

    # ------------------------------------------------------------------ #
    # Backbone forward
    # ------------------------------------------------------------------ #
    def _embed_to_fp32(self, ids_ttnn):
        """Replicate the nemotron_h_model.__call__ embedding preamble: token ids
        -> embedding -> fp32 residual stream."""
        M = self.M
        M._gap()
        ids = ttnn.to_layout(ids_ttnn, ttnn.ROW_MAJOR_LAYOUT)
        if ids.dtype != ttnn.uint32:
            ids = ttnn.typecast(ids, ttnn.uint32)
        h = M._apply_embeddings(ids)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        if h.dtype != ttnn.bfloat16:
            h = ttnn.typecast(h, ttnn.bfloat16)
        return ttnn.typecast(h, ttnn.float32)

    def _mixer_child(self, i):
        """Lazily build + cache the graduated child mixer stub for layer i."""
        if i in self._mixer_cache:
            return self._mixer_cache[i]
        layer = self.backbone.layers[i]
        if i in self.M._MAMBA_LAYERS:
            # Use nemotron_h_block for the first mamba layer (so the block stub
            # is genuinely invoked on real data), the fp32 mamba mixer for the
            # rest. Both are graduated implementations of the M-layer.
            if i == self.M._MAMBA_LAYERS[0]:
                inst = ("block", _block_stub.build(self.device, layer))
            else:
                inst = ("mamba", _mamba_stub.build(self.device, layer.mixer))
        else:
            inst = ("moe", _moe_stub.build(self.device, layer.mixer))
        self._mixer_cache[i] = inst
        return inst

    def _evict_mixer_child(self, i):
        self._mixer_cache.pop(i, None)

    def _backbone(self, ids_ttnn):
        # The nemotron_h_model stub is genuinely invoked on EVERY forward: in the
        # monolith path it drives the whole loop; in the compose path it provides
        # the embedding, the pre/final RMSNorm, the attention helper and the
        # residual scaffold. Record it here (not just at build) so the Gate-2
        # registry reflects the actual per-run execution even after reset().
        self.invoked.add("nemotron_h_model")
        _invocation.record("nemotron_h_model")
        if not self.compose:
            # Monolith path: the graduated nemotron_h_model does the whole loop.
            return self.M(ids_ttnn)

        M = self.M
        h = self._embed_to_fp32(ids_ttnn)
        for i in range(M._N_LAYERS):
            if i in self.M._MAMBA_LAYERS and i == self.M._MAMBA_LAYERS[0]:
                # The block stub does its OWN pre-norm + residual; feed it the
                # bf16 residual stream and let it return the updated stream.
                kind, inst = self._mixer_child(i)
                self.invoked.add("nemotron_h_block")
                out = inst(ttnn.typecast(h, ttnn.bfloat16))
                _invocation.record("nemotron_h_block")
                new_h = ttnn.typecast(out, ttnn.float32)
                ttnn.deallocate(h)
                h = new_h
            else:
                hn = M._rmsnorm(h, M._g_norm_f32[i])
                if i in M._MAMBA_LAYERS:
                    kind, inst = self._mixer_child(i)
                    self.invoked.add("nemotron_h_mamba2_mixer")
                    self.invoked.add("mamba_r_m_s_norm_gated")
                    m = inst(hn)
                    _invocation.record("nemotron_h_mamba2_mixer")
                    _invocation.record("mamba_r_m_s_norm_gated")
                elif i in M._ATTN_LAYERS:
                    m = M._attn(i, hn)  # attention is REUSE, not a graduated work product
                else:
                    kind, inst = self._mixer_child(i)
                    self.invoked.add("nemotron_h_m_o_e")
                    self.invoked.add("nemotron_h_topk_router")
                    self.invoked.add("re_l_u_squared_activation")
                    m = inst(hn)
                    _invocation.record("nemotron_h_m_o_e")
                    _invocation.record("nemotron_h_topk_router")
                    _invocation.record("re_l_u_squared_activation")
                ttnn.deallocate(hn)
                mf = ttnn.typecast(m, ttnn.float32)
                try:
                    ttnn.deallocate(m)
                except Exception:
                    pass
                new_h = ttnn.add(h, mf)
                ttnn.deallocate(h)
                ttnn.deallocate(mf)
                h = new_h
            self._evict_mixer_child(i)
            # Evict any monolith lazily-loaded per-layer weights.
            _pfx = "w_layers_%d_" % i
            for _k in [_kk for _kk in list(M.__dict__) if _kk.startswith(_pfx)]:
                try:
                    ttnn.deallocate(M.__dict__[_k])
                except Exception:
                    pass
                M.__dict__.pop(_k, None)
        h_f32 = M._rmsnorm(h, M._g_norm_f_f32)
        ttnn.deallocate(h)
        return ttnn.typecast(h_f32, ttnn.bfloat16)

    # ------------------------------------------------------------------ #
    # Logits + generation
    # ------------------------------------------------------------------ #
    def forward_logits(self, input_ids):
        """input_ids: torch.LongTensor (1, L). Returns last-position logits
        (torch fp32, shape (vocab,))."""
        _id_kw = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        if self._is_mesh:
            _id_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        ids_ttnn = ttnn.from_torch(input_ids.to(torch.int32), **_id_kw)
        h = self._backbone(ids_ttnn)  # (1, L, hidden) bf16, post final-norm
        L = int(h.shape[1])
        last = ttnn.slice(h, [0, L - 1, 0], [1, L, self.hidden])  # (1,1,hidden)
        if self._ckc is not None:
            logits = ttnn.linear(last, self._lm_head, compute_kernel_config=self._ckc)
        else:
            logits = ttnn.linear(last, self._lm_head)
        # Outputs are replicated across the mesh; read back one replica.
        if self._is_mesh:
            out = ttnn.to_torch(
                logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
            ).to(torch.float32).reshape(-1)
        else:
            out = ttnn.to_torch(logits).to(torch.float32).reshape(-1)
        return out[: self.vocab]

    def generate(self, input_ids, max_new_tokens, eos_token_id=2, verbose=True):
        """Greedy decode. Returns (new_ids list, per_step_logits tensor (N,vocab))."""
        ids = input_ids.clone()
        new_ids = []
        step_logits = []
        for step in range(max_new_tokens):
            logits = self.forward_logits(ids)
            nxt = int(torch.argmax(logits).item())
            step_logits.append(logits)
            new_ids.append(nxt)
            if verbose:
                print(f"[pipeline] step {step}: next_token={nxt}", flush=True)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=ids.dtype)], dim=1)
            if nxt == eos_token_id:
                break
        return new_ids, torch.stack(step_logits, dim=0)


def build_pipeline(device, hf_model, compose=True):
    """Single entry point used by BOTH demo/ and tests/e2e/."""
    return NemotronHPipeline(device, hf_model, compose=compose)
