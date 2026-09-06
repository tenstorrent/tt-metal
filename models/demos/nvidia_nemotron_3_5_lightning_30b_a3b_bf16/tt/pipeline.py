# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN pipeline for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (NemotronHForCausalLM).

Both `demo/demo_text_generation.py` and `tests/e2e/test_e2e_pipeline.py` import
`build_pipeline` + `NemotronHPipeline.run_text_generation` from HERE, so a green
test guarantees a working demo -- there is no second copy of the wiring.

WHAT THIS IS
------------
An EXPLICIT forward pass written out below: token ids -> ttnn embedding -> 52
(capped: 7) hybrid Mamba2 / attention / MoE blocks, each routed through the
GRADUATED stubs under `_stubs/` -> final RMSNorm -> lm_head -> on-device argmax
-> next token -> repeat. No `model.generate()`, no HF submodule call, no torch
compute op anywhere in the hot path. HF appears only at BUILD time (weight
extraction) and inside `_hf_reference_text_generation` (the golden).

WHERE EACH GRADUATED STUB LIVES (Gate 2)
----------------------------------------
Two graduated stubs sometimes cover overlapping math -- a whole-block stub AND
its constituents. Rather than call one of them redundantly, DIFFERENT layers use
DIFFERENT variants, so every stub sits on a real load-bearing edge and its
output feeds the next layer:

  MAMBA_A  (1st mamba block)  nemotron_h_block            -- owns norm+mixer+residual
  MAMBA_B  (2nd mamba block)  nemotron_h_r_m_s_norm
                            + nemotron_h_mamba2_mixer     -- with its gated grouped
                            + zamba2_r_m_s_norm_gated        norm supplied by the stub
  MAMBA_C  (rest)             nemotron_h_r_m_s_norm + nemotron_h_mamba2_mixer
  ATTN     (attention blocks) nemotron_h_r_m_s_norm + nemotron_h_attention
  MOE_A    (1st moe block)    nemotron_h_r_m_s_norm + nemotron_h_mo_e
  MOE_B    (2nd moe block)    nemotron_h_r_m_s_norm + nemotron_h_topk_router
                            + nemotron_h_experts + nemotron_h_m_l_p (shared expert)
  MOE_C    (3rd moe block)    nemotron_h_r_m_s_norm + nemotron_h_topk_router
                            + nemotron_h_experts + re_l_u_squared_activation
                              (shared expert built from the relu2 stub)

`nemotron_h_r_m_s_norm` also serves the model's final norm. There is no
coverage sweep anywhere in this package.

MESH (4 chips, TP=2 x DP=2)
---------------------------
`ttnn.set_fabric_config(FABRIC_1D)` then `open_mesh_device(MeshShape(2, 2))`;
rows = DP, cols = TP. Every graduated stub computes `_tp_axis = len(shape)-1 = 1`
and all_reduces on that axis, so they agree on which physical axis is TP. The
sharded bodies are gated on `TT_HW_PLANNER_SHARD_RUN`, which this module sets
before building.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_attention as _attn_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_block as _block_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_experts as _experts_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_m_l_p as _mlp_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_mamba2_mixer as _mamba_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_mo_e as _moe_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_r_m_s_norm as _rmsnorm_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_topk_router as _router_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import re_l_u_squared_activation as _relu2_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import zamba2_r_m_s_norm_gated as _gnorm_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref, _invocation

HF_MODEL_ID = _hf_ref.HF_MODEL_ID

# NemotronHForCausalLM, is_encoder_decoder == False  ->  [prefill, decode].
PIPELINE_STAGES = ["prefill", "decode"]

# The ten graduated modules (every _stubs/<n>.py with a .last_good_native or
# .last_good_sharded snapshot). Gate 2 asserts all ten actually executed.
GRADUATED_MODULES = (
    "nemotron_h_attention",
    "nemotron_h_block",
    "nemotron_h_experts",
    "nemotron_h_m_l_p",
    "nemotron_h_mamba2_mixer",
    "nemotron_h_mo_e",
    "nemotron_h_r_m_s_norm",
    "nemotron_h_topk_router",
    "re_l_u_squared_activation",
    "zamba2_r_m_s_norm_gated",
)

# BATCH: 32 independent samples per call. One program per step feeds all 32.
BATCH = 32

# Depth cap forced by DRAM -- see tt/_hf_ref.py for the arithmetic.
DEFAULT_LAYERS = _hf_ref.DEFAULT_GATE_LAYERS

# Fixed trace capacity (the pinned sequence axis). Bound is
# config.max_position_embeddings; 64 is what fits alongside a depth-7 resident
# build and is printed if it has to shrink.
DEFAULT_TRACE_CAPACITY = 64

_DEMO_DIR = Path(__file__).resolve().parents[1]
_GOLDEN_DIR = _DEMO_DIR / "_captured" / "_e2e_golden"


def _ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _is_mesh(device):
    try:
        return isinstance(device, ttnn.MeshDevice)
    except AttributeError:
        return False


def _mesh_shape(device):
    return list(device.shape) if _is_mesh(device) else [1, 1]


def _replicate(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if _is_mesh(device) else {}
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, **kw)


def _shard(device, t, dim, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=_mesh_shape(device), dims=(None, dim)),
    )


def _reshape_rm(t, shape):
    """Reshape through ROW_MAJOR.

    A TILE-layout tensor pads its second-to-last dim up to the 32-row tile
    height, so reshaping ACROSS that dim (e.g. merging batch into sequence when
    T=10) reinterprets padding as data and silently corrupts every row but the
    first. Round-tripping through ROW_MAJOR moves the data properly. The
    graduated stubs use this same pattern for their head reshapes.
    """
    was_tile = t.layout == ttnn.TILE_LAYOUT
    if was_tile:
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, list(shape))
    if was_tile:
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    return t


def _dup(t):
    """A fresh device copy of `t` (several graduated stubs deallocate the tensor
    they were handed once no upcast copy was needed)."""
    try:
        return ttnn.clone(t)
    except (AttributeError, RuntimeError, TypeError):
        return ttnn.add(t, 0.0)


def _first_shard(t):
    """Read a mesh tensor back as its chip-0 shard.

    Every value the pipeline reads back (hidden states, logits, sampled ids) is
    REPLICATED across the mesh by construction -- each TP column all_reduces its
    partial sums, and the DP rows run identical data -- so chip 0's copy IS the
    full result. `ttnn.to_torch` refuses a multi-device tensor without a
    composer, so take the per-device tensor explicitly.
    """
    try:
        shards = ttnn.get_device_tensors(t)
    except (AttributeError, RuntimeError, TypeError):
        return ttnn.to_torch(t)
    return ttnn.to_torch(shards[0]) if shards else ttnn.to_torch(t)


# --------------------------------------------------------------------------- #
#  one layer of the hybrid stack
# --------------------------------------------------------------------------- #
class TtNemotronHLayer:
    """One block of the NemotronH stack.

    Every element of `NemotronHPipeline.layers` is an instance of THIS class --
    a plain, same-typed python list of them -- so the stack is discoverable by a
    structural walk regardless of what base class the stubs happen to use.
    """

    def __init__(self, pipeline, idx, block_type, variant, hf_block):
        self.pipeline = pipeline
        self.idx = idx
        self.block_type = block_type
        self.variant = variant
        self.hf_block = hf_block
        self.stub_names = []

        device = pipeline.device
        self.device = device

        if variant == "MAMBA_A":
            self.block = _block_stub.build(device, hf_block)
            self.stub_names = ["nemotron_h_block"]
            return

        # every non-MAMBA_A variant owns its own pre-norm
        self.norm = _rmsnorm_stub.build(device, hf_block.norm)
        self.stub_names = ["nemotron_h_r_m_s_norm"]
        mixer = hf_block.mixer

        if variant in ("MAMBA_B", "MAMBA_C"):
            self.mixer = _mamba_stub.build(device, mixer)
            self.stub_names.append("nemotron_h_mamba2_mixer")
            if variant == "MAMBA_B":
                # supply the mixer's gated grouped norm from the graduated stub.
                # Its weight must be split on the same TP axis as the mixer's
                # head-sharded intermediate width.
                self.gated_norm = _gnorm_stub.build(device, mixer.norm, tp_shard=self.mixer._shard)
                self.mixer.gated_norm = self._gated_norm_call
                self.stub_names.append("zamba2_r_m_s_norm_gated")

        elif variant == "ATTN":
            self.mixer = _attn_stub.build(device, mixer)
            self.stub_names.append("nemotron_h_attention")

        elif variant == "MOE_A":
            self.mixer = _moe_stub.build(device, mixer)
            self.stub_names.append("nemotron_h_mo_e")

        elif variant in ("MOE_B", "MOE_C"):
            self.router = _router_stub.build(device, mixer.gate)
            self.experts = _experts_stub.build(device, mixer.experts)
            self.stub_names += ["nemotron_h_topk_router", "nemotron_h_experts"]
            self.top_k = int(mixer.top_k)
            self.n_experts = int(mixer.n_routed_experts)
            self.norm_topk_prob = bool(mixer.norm_topk_prob)
            self.routed_scaling_factor = float(mixer.routed_scaling_factor)
            bias = mixer.gate.e_score_correction_bias.detach().float().reshape(1, 1, self.n_experts)
            self._bias = _replicate(device, bias)
            if variant == "MOE_B":
                self.shared = _mlp_stub.build(device, mixer.shared_experts)
                self.stub_names.append("nemotron_h_m_l_p")
            else:
                # shared expert assembled here so the relu2 stub sits between
                # its two projections and its output feeds down_proj.
                self.relu2 = _relu2_stub.build(device, mixer.shared_experts.act_fn)
                self.stub_names.append("re_l_u_squared_activation")
                up = mixer.shared_experts.up_proj.weight.detach().float().t().contiguous()
                dn = mixer.shared_experts.down_proj.weight.detach().float().t().contiguous()
                tp = _mesh_shape(device)[-1] if pipeline.sharded else 1
                self._sh_tp = pipeline.sharded and tp > 1 and up.shape[1] % tp == 0
                if self._sh_tp:
                    self._sh_up = _shard(device, up, 1)  # column-parallel
                    self._sh_dn = _shard(device, dn, 0)  # row-parallel
                else:
                    self._sh_up = _replicate(device, up)
                    self._sh_dn = _replicate(device, dn)
        else:
            raise ValueError(f"unknown variant {variant}")

    # ---- the gated-norm hook handed to the mamba mixer (MAMBA_B) ---------- #
    def _gated_norm_call(self, y, gate):
        _invocation.record("zamba2_r_m_s_norm_gated")
        return self.gated_norm(y, gate)

    # ---- on-device top-k router (n_group == topk_group == 1) --------------- #
    def _route(self, h_flat):
        """h_flat: (tokens, hidden) -> dense (tokens, n_experts) routing weights.

        Takes an already-flattened token matrix: routing is per token, and
        flattening in the caller keeps the one tile-crossing reshape in a single
        place (see `_reshape_rm`). The (1, T, ...) shapes used inside only ever
        add/remove a LEADING 1, which is metadata-safe in TILE layout.

        Mirrors NemotronHMoE.route_tokens_to_experts. n_group == 1 makes the
        group machinery a no-op. Selection uses a FULLY-FP32 all-pairs rank
        rather than bf16 ttnn.topk: the 6th/7th-of-128 boundary scores round
        equal in bf16 and the router over/under-selects (this is the same
        reason the graduated `nemotron_h_mo_e` stub does it this way).
        """
        ckc = self.pipeline.ckc
        E, top_k = self.n_experts, self.top_k

        _invocation.record("nemotron_h_topk_router")
        logits = self.router(h_flat)  # (tokens, E) fp32
        T = int(logits.shape[0])
        logits = ttnn.reshape(logits, [1, T, E])

        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self._bias)

        bb = ttnn.reshape(choice, [1, T, 1, E])
        b_full = ttnn.repeat(bb, [1, 1, E, 1])  # [.,.,i,j] = choice_j
        ttnn.deallocate(bb)
        a_full = ttnn.transpose(b_full, 2, 3)  # [.,.,i,j] = choice_i
        gt = ttnn.gt(b_full, a_full)
        ttnn.deallocate(b_full)
        ttnn.deallocate(a_full)
        gt_f = ttnn.typecast(gt, ttnn.float32)
        ttnn.deallocate(gt)
        rank = ttnn.sum(gt_f, dim=3)
        ttnn.deallocate(gt_f)
        if list(rank.shape)[-1] != E:
            rank = ttnn.reshape(rank, [1, T, E])
        mask = ttnn.lt(rank, float(top_k))
        ttnn.deallocate(rank)
        ttnn.deallocate(choice)
        if mask.dtype != ttnn.float32:
            mask = ttnn.typecast(mask, ttnn.float32)

        W = ttnn.multiply(scores, mask)
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        if self.norm_topk_prob:
            denom = ttnn.add(ttnn.sum(W, dim=-1, keepdim=True), 1e-20)
            W2 = ttnn.multiply(W, ttnn.reciprocal(denom))
            ttnn.deallocate(W)
            ttnn.deallocate(denom)
            W = W2
        W = ttnn.multiply(W, self.routed_scaling_factor)
        return ttnn.reshape(W, [T, E]), T

    # ---- forward ---------------------------------------------------------- #
    def __call__(self, x):
        """x: (B, T, hidden) on device -> (B, T, hidden) on device."""
        ckc = self.pipeline.ckc

        if self.variant == "MAMBA_A":
            _invocation.record("nemotron_h_block")
            return self.block(x)  # stub owns norm + mixer + residual

        # The norm stub deallocates its own working tensor, which IS the caller's
        # tensor when the input already arrives as fp32 (its `_fp32` is a no-op
        # then). The residual stream is fp32, so hand it a copy -- otherwise the
        # residual `x` is freed under us before the add below.
        _invocation.record("nemotron_h_r_m_s_norm")
        h = self.norm(_dup(x))

        if self.variant in ("MAMBA_B", "MAMBA_C"):
            _invocation.record("nemotron_h_mamba2_mixer")
            y = self.mixer(h)
        elif self.variant == "ATTN":
            _invocation.record("nemotron_h_attention")
            y = self.mixer(h)
        elif self.variant == "MOE_A":
            _invocation.record("nemotron_h_mo_e")
            y = self.mixer(h)
        else:  # MOE_B / MOE_C
            B, T = int(h.shape[0]), int(h.shape[1])
            hid = int(h.shape[2])
            h_flat = _reshape_rm(h, [B * T, hid])  # tokens are independent here
            W, ntok = self._route(h_flat)

            _invocation.record("nemotron_h_experts")
            routed = self.experts(h_flat, routing_dense=W)  # (tokens, hidden)
            routed = _reshape_rm(routed, [B, T, hid])

            if self.variant == "MOE_B":
                _invocation.record("nemotron_h_m_l_p")
                shared = self.shared(h)
            else:
                # upcast first: the norm stub hands back bf16, and feeding a
                # bf16 activation into these fp32 weights is what made this
                # variant the worst layer in the chain (0.971 vs 0.990 for the
                # MLP-stub variant, which upcasts internally).
                hh = ttnn.typecast(h, ttnn.float32) if h.dtype != ttnn.float32 else h
                up = ttnn.matmul(hh, self._sh_up, compute_kernel_config=ckc)
                _invocation.record("re_l_u_squared_activation")
                act = self.relu2(up)
                ttnn.deallocate(up)
                shared = ttnn.matmul(act, self._sh_dn, compute_kernel_config=ckc)
                ttnn.deallocate(act)
                if self._sh_tp:
                    shared = ttnn.all_reduce(shared, cluster_axis=1, topology=ttnn.Topology.Linear)

            y = ttnn.add(ttnn.typecast(routed, ttnn.float32), ttnn.typecast(shared, ttnn.float32))

        # Keep the residual stream in fp32. The graduated stubs each return
        # bf16, but truncating the RESIDUAL too costs ~3 decimal digits per
        # layer and compounds: measured 2026-09-06, a bf16 residual gave e2e
        # PCC 0.947 at depth 7 where the fp32 residual clears the gate. The HF
        # reference runs the whole block in fp32, so this matches it.
        return ttnn.add(ttnn.typecast(x, ttnn.float32), ttnn.typecast(y, ttnn.float32))


# --------------------------------------------------------------------------- #
#  the pipeline
# --------------------------------------------------------------------------- #
def _assign_variants(block_types):
    """Map each block index to a layer variant, for ANY depth.

    The first occurrence of each block type takes the whole-block stub; the
    next occurrences take the decomposed variants, so every graduated stub is
    reached as soon as the depth contains it (7 blocks suffices for this model).
    """
    variants, seen = [], {"mamba": 0, "moe": 0, "attention": 0, "mlp": 0}
    for bt in block_types:
        n = seen[bt]
        seen[bt] = n + 1
        if bt == "mamba":
            variants.append("MAMBA_A" if n == 0 else ("MAMBA_B" if n == 1 else "MAMBA_C"))
        elif bt == "moe":
            variants.append("MOE_A" if n == 0 else ("MOE_B" if n == 1 else ("MOE_C" if n == 2 else "MOE_A")))
        elif bt == "attention":
            variants.append("ATTN")
        else:
            raise ValueError(f"unsupported block type {bt!r} for this pipeline")
    return variants


class NemotronHPipeline:
    """Resident TT pipeline: weights are built ONCE and stay on device."""

    PIPELINE_STAGES = PIPELINE_STAGES
    GRADUATED_MODULES = GRADUATED_MODULES

    def __init__(self, device, model, layers=None, trace_capacity=DEFAULT_TRACE_CAPACITY, batch=BATCH):
        self.device = device
        self.hf = model  # HF reference stays reachable: ground truth for section structure
        self.config = model.config
        self.ckc = _ckc()
        self.batch = batch
        self.trace_capacity = trace_capacity
        self.sharded = bool(os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh(device)
        self.saw_shard_upload = False
        self.saw_collective = False

        hf_blocks = list(model.model.layers)
        if layers is not None:
            hf_blocks = hf_blocks[:layers]
        self.block_types = [b.block_type for b in hf_blocks]
        self.variants = _assign_variants(self.block_types)

        # ---- the repeated stack: a plain list of same-typed elements ------- #
        import time as _time

        self.layers = []
        for i, (bt, v, blk) in enumerate(zip(self.block_types, self.variants, hf_blocks)):
            _t = _time.time()
            self.layers.append(TtNemotronHLayer(self, i, bt, v, blk))
            print(f"[build] layer {i} {bt}/{v} in {_time.time() - _t:.1f}s", flush=True)

        # ---- everything outside the stack --------------------------------- #
        emb = model.model.embeddings.weight.detach().to(torch.bfloat16).contiguous()
        self.embed_w = ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            **({"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if _is_mesh(device) else {}),
        )
        self.final_norm = _rmsnorm_stub.build(device, model.model.norm_f)
        # lm_head in fp32: HF computes `self.lm_head(...).float()`, and a bf16
        # head visibly costs final-logit PCC.
        lm = model.lm_head.weight.detach().float().t().contiguous()  # (hidden, vocab)
        self.lm_head_w = _replicate(device, lm, dtype=ttnn.float32)
        self.vocab_size = int(lm.shape[1])
        self.hidden_size = int(lm.shape[0])

        # trace state
        self._trace = {}
        self._persistent = {}

    # -- introspection the perf harness uses -------------------------------- #
    @property
    def n_layers(self):
        return len(self.layers)

    def describe(self):
        return {
            "layers": self.n_layers,
            "block_types": self.block_types,
            "variants": self.variants,
            "batch": self.batch,
            "sharded": self.sharded,
            "mesh_shape": _mesh_shape(self.device),
            "stubs_per_layer": {i: l.stub_names for i, l in enumerate(self.layers)},
        }

    def sharding_evidence(self):
        """Which built stubs actually took their TP-sharded branch.

        Gate 1's runtime half: a TP=2 result must contain ShardTensor2dMesh
        uploads AND a collective, not pure replication.
        """
        ev = {}
        for l in self.layers:
            for attr in ("block", "mixer", "experts", "shared"):
                m = getattr(l, attr, None)
                if m is not None and getattr(m, "_shard", False):
                    ev.setdefault(type(m).__name__, 0)
                    ev[type(m).__name__] += 1
            if getattr(l, "_sh_tp", False):
                ev.setdefault("moe_c_shared_expert", 0)
                ev["moe_c_shared_expert"] += 1
            gn = getattr(l, "gated_norm", None)
            if gn is not None and getattr(gn, "_tp_shard", False):
                ev.setdefault("MambaRMSNormGated", 0)
                ev["MambaRMSNormGated"] += 1
        return ev

    # ----------------------------------------------------------------- #
    #  the real forward pass
    # ----------------------------------------------------------------- #
    def embed(self, ids_tt):
        """token ids (B, T) uint32 ROW_MAJOR -> hidden (B, T, H) bf16 TILE."""
        h = ttnn.embedding(ids_tt, self.embed_w)
        return ttnn.to_layout(h, ttnn.TILE_LAYOUT)

    def forward_hidden(self, ids_tt):
        """ids (B, T) -> final-normed hidden states (B, T, H). Pure ttnn."""
        h = self.embed(ids_tt)
        for layer in self.layers:
            h = layer(h)
        _invocation.record("nemotron_h_r_m_s_norm")
        return self.final_norm(h)

    def forward_logits(self, ids_tt, last_only=True):
        """ids (B, T) -> lm_head logits. (B, 1, vocab) when last_only."""
        h = self.forward_hidden(ids_tt)
        if last_only:
            B, T = int(h.shape[0]), int(h.shape[1])
            h = ttnn.slice(h, [0, T - 1, 0], [B, T, self.hidden_size])
        if h.dtype != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)
        return ttnn.matmul(h, self.lm_head_w, compute_kernel_config=self.ckc)

    def _ids_to_device(self, ids):
        t = ids.to(torch.int32)
        kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(self.device)} if _is_mesh(self.device) else {}
        return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, **kw)

    # ----------------------------------------------------------------- #
    #  Call 1 -- text generation.  THE task entrypoint.
    # ----------------------------------------------------------------- #
    def run_text_generation(self, input_ids, max_new_tokens=None, stop_ids=None, progress=False):
        """Greedy autoregressive decode, entirely on device.

        input_ids : (B, T0) torch int64, produced by the Source-A tokenizer.
        Returns a dict with the generated ids, the full sequences, and the
        per-step logits (fp32, on host) for the PCC gate.

        Decode horizon: STOP-TOKEN first -- generation_config.eos_token_id
        ([2, 11] for this checkpoint) -- with a safety cap so a non-terminating
        run cannot hang. Both the TT loop and the HF golden use this same rule
        and the same cap, so the two sequences always have the same length.
        """
        stop_ids = set(stop_ids if stop_ids is not None else self.stop_ids())
        N = max_new_tokens if max_new_tokens is not None else self.decode_cap(input_ids.shape[1])

        B = int(input_ids.shape[0])
        ids_tt = self._ids_to_device(input_ids)
        seq = input_ids.clone()
        finished = torch.zeros(B, dtype=torch.bool)
        step_logits = []

        for step in range(N):
            logits = self.forward_logits(ids_tt, last_only=True)  # (B,1,vocab) on device
            nxt = ttnn.argmax(logits, dim=-1)  # on-device greedy pick
            step_logits.append(_first_shard(logits).reshape(B, -1)[:, : self.vocab_size].float())
            ttnn.deallocate(logits)

            nxt_rm = ttnn.to_layout(nxt, ttnn.ROW_MAJOR_LAYOUT)
            nxt_rm = ttnn.reshape(nxt_rm, [B, 1])
            nxt_rm = ttnn.typecast(nxt_rm, ttnn.uint32)
            ids_tt = ttnn.concat([ids_tt, nxt_rm], dim=1)

            tok = _first_shard(nxt_rm).reshape(B).to(torch.int64)
            seq = torch.cat([seq, tok.reshape(B, 1)], dim=1)
            finished |= torch.tensor([int(t) in stop_ids for t in tok.tolist()])
            if progress:
                print(f"  [tt] step {step + 1}/{N} tokens={tok.tolist()[:4]}...", flush=True)
            if bool(finished.all()):
                break

        return {
            "sequences": seq,
            "new_ids": seq[:, input_ids.shape[1] :],
            "step_logits": torch.stack(step_logits, dim=0),  # (steps, B, vocab)
            "steps": len(step_logits),
        }

    # -- decode-horizon helpers (shared by TT and the golden) --------------- #
    def stop_ids(self):
        gc = getattr(self.hf, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None) if gc is not None else None
        if eos is None:
            eos = getattr(self.config, "eos_token_id", None)
        if eos is None:
            return []
        return list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]

    def decode_cap(self, prompt_len):
        """Safety cap for the stop-token rule.

        The graduated stubs are STATELESS full-sequence bodies (no KV / SSM
        cache), so step k recomputes the whole prefix and decode is O(N^2) in
        tokens. TT_E2E_MAX_NEW_TOKENS (default 16) is that hardware-forced
        bound, clamped by the config's own context limit.
        """
        env = int(os.environ.get("TT_E2E_MAX_NEW_TOKENS", "16"))
        ctx = int(self.config.max_position_embeddings) - int(prompt_len)
        return max(1, min(env, ctx))

    # ----------------------------------------------------------------- #
    #  the GOLDEN (Source A).  Never called from the TT path.
    # ----------------------------------------------------------------- #
    def _hf_reference_text_generation(self, input_ids, max_new_tokens=None, stop_ids=None):
        N = max_new_tokens if max_new_tokens is not None else self.decode_cap(input_ids.shape[1])
        eos = list(stop_ids) if stop_ids is not None else self.stop_ids()
        with torch.no_grad():
            gen = self.hf.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=N,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=int(getattr(self.config, "pad_token_id", 0) or 0),
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
            )
        return {
            "sequences": gen.sequences,
            "new_ids": gen.sequences[:, input_ids.shape[1] :],
            "step_logits": torch.stack([l.float() for l in gen.logits], dim=0),  # (steps, B, vocab)
            "steps": len(gen.logits),
        }

    def _hf_reference_teacher_forced(self, sequences, prompt_len, steps):
        """The golden for the prefixes the TT pipeline ACTUALLY produced.

        Free-running `generate()` (above) answers "does TT walk the same path?".
        This answers "for the context TT was in at step k, is TT's next-token
        distribution right?" -- which is what a port is responsible for. One HF
        forward over the whole TT sequence yields every step's logits: position
        `prompt_len + k - 1` predicts the token TT emitted at step k.

        The TT path is untouched: nothing is fed back into it. Only the
        REFERENCE is evaluated on TT's own history.
        """
        with torch.no_grad():
            out = self.hf(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
                use_cache=False,
            )
        pos = [prompt_len + k - 1 for k in range(steps)]
        return out.logits[:, pos, :].float().permute(1, 0, 2).contiguous()  # (steps, B, vocab)

    # ----------------------------------------------------------------- #
    #  COMMAND 3 -- per-stage trace contract
    # ----------------------------------------------------------------- #
    def _captured_input_ids(self):
        """The tokenized 32-prompt batch the e2e test and demo both drive."""
        p = _GOLDEN_DIR / "input_ids.pt"
        if p.exists():
            return torch.load(p)
        from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tests.e2e import make_golden

        return make_golden.build_input_ids(self.batch)

    def _pin(self, ids, C):
        """Pad / truncate the sequence axis to the fixed capacity C.

        Padding uses pad_token_id at the END, so output over [0:real_len] is the
        same computation the unpadded run performs for those positions (the
        model is causal, so trailing positions cannot influence earlier ones).
        """
        B, T = ids.shape
        if T >= C:
            return ids[:, :C], C
        pad = int(getattr(self.config, "pad_token_id", 0) or 0)
        tail = torch.full((B, C - T), pad, dtype=ids.dtype)
        return torch.cat([ids, tail], dim=1), T

    # ---- prefill ---------------------------------------------------------- #
    def prefill_trace_inputs(self):
        """ZERO-ARG: exactly what prefill_trace_setup takes."""
        return {"input_ids": self._captured_input_ids()[: self.batch]}

    def prefill_trace_items(self):
        """One traced prefill retires B x C positions through the repeated blocks."""
        return int(self.batch * self.trace_capacity)

    def prefill_trace_setup(self, inputs):
        C = self.trace_capacity
        ids, real = self._pin(inputs["input_ids"], C)
        buf = self._ids_to_device(ids)
        self._persistent["prefill_ids"] = buf
        self._persistent["prefill_real_len"] = real
        # Warm the shape-dependent constants (causal mask, mamba shift/tril/neg
        # matrices, the router's rank scratch) at exactly this capacity so no
        # from_torch / ttnn.zeros can fire inside the captured region.
        out = self.forward_logits(buf, last_only=True)
        self._persistent["prefill_ref"] = _first_shard(out)
        ttnn.deallocate(out)
        return buf

    def prefill_trace_step(self):
        return self.forward_logits(self._persistent["prefill_ids"], last_only=True)

    # ---- decode ----------------------------------------------------------- #
    def decode_trace_inputs(self):
        return {"input_ids": self._captured_input_ids()[: self.batch]}

    def decode_trace_items(self):
        """One decode step retires one token per batch row."""
        return int(self.batch)

    def decode_prefill(self, inputs):
        """AR contract: seed the resident decode state.

        NOTE, honestly: the graduated stubs are stateless full-sequence bodies
        with no KV / SSM cache to seed, so "resident state" here is the pinned
        capacity-C id buffer that a decode step reads and never rebuilds. There
        is no cache to recompute, so the decode contract's "reads them, never
        recomputes" holds trivially.
        """
        return self.decode_trace_setup(inputs)

    def decode_trace_setup(self, inputs):
        C = self.trace_capacity
        ids, real = self._pin(inputs["input_ids"], C)
        buf = self._ids_to_device(ids)
        self._persistent["decode_ids"] = buf
        self._persistent["decode_real_len"] = real
        out = self.forward_logits(buf, last_only=True)
        self._persistent["decode_ref"] = _first_shard(out)
        ttnn.deallocate(out)
        return buf

    def decode_step(self):
        return self.decode_trace_step()

    def decode_trace_step(self):
        return self.forward_logits(self._persistent["decode_ids"], last_only=True)

    # ---- selftests -------------------------------------------------------- #
    def trace_capture_selftest(self, device=None):
        """Capture ONE step per stage, execute it, PCC it, RELEASE before the
        next stage. Returns True only if every stage captured host-free AND its
        trace output matches the eager reference."""
        device = device or self.device
        ok = True
        for stage in self.PIPELINE_STAGES:
            inputs = getattr(self, f"{stage}_trace_inputs")()
            getattr(self, f"{stage}_trace_setup")(inputs)
            ref = self._persistent[f"{stage}_ref"]
            tid = None
            try:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = getattr(self, f"{stage}_trace_step")()
                ttnn.end_trace_capture(device, tid, cq_id=0)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                got = _first_shard(out)
                pcc = _pcc(got, ref)
                print(f"[trace] stage={stage} C={self.trace_capacity} pcc={pcc:.6f}", flush=True)
                ok = ok and pcc >= 0.99
            except Exception as e:  # capture overflow / host op inside the region
                print(f"[trace] stage={stage} FAILED to capture: {type(e).__name__}: {e}", flush=True)
                ok = False
            finally:
                if tid is not None:
                    try:
                        ttnn.release_trace(device, tid)
                    except Exception:
                        pass
        return ok

    def host_op_selftest(self):
        """AUTHORITATIVE fully-on-device check.

        Input encoding (tokenization) and the one-time weight build happen
        OUTSIDE the observed region; only encoded-ids -> logits (embedding,
        every block, final norm, lm_head, argmax) is inside it.
        """
        from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

        ids = self._captured_input_ids()[: self.batch]
        ids, _ = self._pin(ids, self.trace_capacity)
        buf = self._ids_to_device(ids)  # encoding/upload: outside the observed region
        with observe_host_ops() as ops:
            logits = self.forward_logits(buf, last_only=True)
            nxt = ttnn.argmax(logits, dim=-1)
            ttnn.synchronize_device(self.device)
        ttnn.deallocate(logits)
        del nxt
        return verdict(ops)


# --------------------------------------------------------------------------- #
#  factory + PCC helper
# --------------------------------------------------------------------------- #
def _pcc(a, b):
    # float64: in fp32 the normalised dot product rounds to slightly above 1.0
    # on near-identical tensors, which reads as a nonsense PCC in the report.
    a = torch.as_tensor(a).double().flatten()
    b = torch.as_tensor(b).double().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    if d == 0:
        return 1.0 if float((a - b).abs().max()) == 0 else 0.0
    return float((a @ b).item() / d)


def pcc(a, b):
    return _pcc(a, b)


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """CONSTRUCT and RETURN the resident pipeline object (never runs it).

    `layers` caps the depth built; None means EVERY layer. NemotronH has ONE
    repeated stack (the hybrid decoder) shared by both stages, so the per-stage
    overrides `prefill_layers` / `decode_layers` exist for call-signature
    compatibility and simply resolve against the same stack -- the deeper of the
    two wins so neither stage is built shallower than it asked for.

    Any demo kwarg (prompt, text, language, ...) is accepted and ignored: the
    resident build derives its shapes from the config, not from a prompt.
    """
    env_layers = os.environ.get("TT_PERF_LAYERS")
    if layers is None and env_layers:
        layers = int(env_layers)
    per_stage = [v for v in (prefill_layers, decode_layers) if v is not None]
    if per_stage:
        layers = max(per_stage) if layers is None else max([layers] + per_stage)
    if layers == 0:
        raise ValueError("layers=0 is not a model; pass None for every layer or a positive depth")

    os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")

    if model is None:
        model = _hf_ref.load_reference(layers if layers is not None else None, dtype=torch.float32)

    return NemotronHPipeline(
        device,
        model,
        layers=layers,
        trace_capacity=int(kwargs.get("trace_capacity", DEFAULT_TRACE_CAPACITY)),
        batch=int(kwargs.get("batch", BATCH)),
    )


def open_mesh(rows=2, cols=2, l1_small_size=24576, trace_region_size=0):
    """Open the 4-chip TP=2 x DP=2 mesh (fabric first), falling back to whatever
    is actually available at runtime and saying so.

    NOTE: on this box the fabric only trains on the FULL mesh -- a partial
    MeshShape(1, 2) dies in `Fabric Router Sync` -- so the fallback below is a
    single device, not a smaller mesh.
    """
    want = rows * cols
    have = ttnn.get_num_devices()
    if have < want:
        print(f"[mesh] only {have} device(s) available, wanted {want} -- falling back", flush=True)
        rows, cols = 1, have
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    kw = {"l1_small_size": l1_small_size}
    if trace_region_size:
        kw["trace_region_size"] = trace_region_size
    dev = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **kw)
    print(f"[mesh] opened MeshShape({rows}, {cols})  DP={rows} TP={cols}", flush=True)
    return dev


def close_mesh(device):
    ttnn.close_mesh_device(device)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
