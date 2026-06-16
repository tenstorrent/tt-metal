# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""NemotronH-30B full model forward pass using TTNN components.

Layer pattern (52 layers):
  M = Mamba2 block   (SSM only — norm + Mamba2 mixer + residual)
  E = MoE MLP block  (norm + gate + 128 routed experts + shared expert + residual)
  * = Dense attention (norm + GQA attention + residual, no RoPE in HF source)

hidden_states is a ttnn.Tensor on device throughout (embedding → lm_head).
Only lm_head.forward brings the final result to CPU (logits).

Weight loading is lazy: weights are fetched from safetensors shards on first
access and cached in a dict so each shard is opened at most once.
"""

import json
import os

import torch

import ttnn

from .dense_attention import dense_attention_forward
from .embedding import embedding_forward, embedding_forward_tt
from .kv_cache import DecoderState
from .layer_norm import layer_norm_forward
from .lm_head import lm_head_forward, lm_head_forward_device
from .mamba2_layer import mamba2_layer_forward
from .moe_experts import moe_experts_forward
from .moe_gate import moe_gate_forward, moe_gate_forward_cpu
from .shared_expert import shared_expert_forward

# Per-layer cache for pre-stacked expert weight tensors on device.
# Column-parallel TP sharding: intermediate dim split 4 ways across devices.
# up   [1,128,2688,1856] bf16 → [1,128,2688,464]/device at shard_dim=3 (≈1.19 GiB)
# down [1,128,1856,2688] bf16 → [1,128,464,2688]/device at shard_dim=2 (≈1.19 GiB)
# 23 E-layers × 2 matrices × 1.19 GiB ≈ 54.7 GiB total, 13.7 GiB per device.
_EXPERT_DEVICE_CACHE: dict = {}  # layer_idx -> (up_tt, down_tt)

# Disk cache for pre-stacked, pre-tilized expert weight tensors.
# ttnn.as_tensor saves on first run; subsequent runs load directly — skips torch.stack
# and tile-layout conversion (the ~10-min CPU bottleneck on cold start).
# Override with env var TT_NEMOTRON_EXPERT_CACHE.
_EXPERT_DISK_CACHE_DIR = os.path.expanduser(
    os.environ.get(
        "TT_NEMOTRON_EXPERT_CACHE",
        "~/.cache/ttnn/nemotron30b_experts_bf16_tp4",
    )
)

SNAP = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/"
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
N_LAYERS = 52


class WeightCache:
    """Lazy-loading weight cache backed by safetensors shards."""

    def __init__(self, snap: str = SNAP):
        idx_path = os.path.join(snap, "model.safetensors.index.json")
        with open(idx_path) as f:
            self._idx = json.load(f)["weight_map"]
        self._snap = snap
        self._shards: dict = {}

    def _shard(self, filename: str):
        if filename not in self._shards:
            from safetensors.torch import load_file

            path = os.path.join(self._snap, filename)
            size_gb = os.path.getsize(os.path.realpath(path)) / 1e9
            print(f"  [WeightCache] loading shard {filename} ({size_gb:.1f} GB)...", flush=True)
            self._shards[filename] = load_file(path)
            print(f"  [WeightCache] shard {filename} loaded.", flush=True)
        return self._shards[filename]

    def __getitem__(self, key: str) -> torch.Tensor:
        filename = self._idx[key]
        return self._shard(filename)[key]

    def __contains__(self, key: str) -> bool:
        return key in self._idx


def _get_stacked_expert_weights(mesh_device, layer_idx: int, wc: "WeightCache"):
    """Return (up_tt, down_tt) stacked expert-weight tensors for layer_idx.

    Column-parallel TP sharding (shard intermediate dim, not expert count):
      up_tt:   [1, 128, 2688, 464] bfloat16, 1 shard per TP device (dim=3)
      down_tt: [1, 128, 464, 2688] bfloat16, 1 shard per TP device (dim=2)

    All 128 experts are present on every device so sparse_matmul always has
    active experts — avoids the noc_semaphore_wait hang from the prior
    expert-count sharding (tt-metal#45943).  Each device holds 13.7 GiB of
    expert weights (54.7 GiB total ÷ 4) — same as bfloat4_b replicated but
    at full bfloat16 precision.  moe_experts_forward adds an all_reduce to sum
    the 4 partial intermediate-column outputs.

    Disk cache (ttnn.as_tensor): first run stacks from safetensors and saves
    pre-tilized shards; subsequent runs load directly — skips torch.stack and
    tile-layout conversion.  Cache dir: _EXPERT_DISK_CACHE_DIR.
    """
    if layer_idx in _EXPERT_DEVICE_CACHE:
        return _EXPERT_DEVICE_CACHE[layer_idx]

    os.makedirs(_EXPERT_DISK_CACHE_DIR, exist_ok=True)
    up_cache = os.path.join(_EXPERT_DISK_CACHE_DIR, f"layer_{layer_idx}_up")
    down_cache = os.path.join(_EXPERT_DISK_CACHE_DIR, f"layer_{layer_idx}_down")
    from_disk = os.path.exists(up_cache + ".bin")

    # Count how many E-layers exist to show X/23 progress.
    e_indices = [i for i, t in enumerate(PATTERN) if t == "E"]
    e_num = e_indices.index(layer_idx) + 1 if layer_idx in e_indices else "?"
    tag = f"E-layer {e_num}/{len(e_indices)} (layer {layer_idx})"
    if from_disk:
        print(f"  [experts] {tag}: loading from disk cache...", flush=True)
    else:
        print(f"  [experts] {tag}: stacking from safetensors (first run — will cache)...", flush=True)

    p = f"backbone.layers.{layer_idx}"
    if not from_disk:
        up_cpu = (
            torch.stack([wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)])
            .transpose(-1, -2)  # [128, 1856, 2688] → [128, 2688, 1856]
            .unsqueeze(0)  # [1, 128, 2688, 1856]
            .bfloat16()
            .contiguous()
        )
        down_cpu = (
            torch.stack([wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)])
            .transpose(-1, -2)  # [128, 2688, 1856] → [128, 1856, 2688]
            .unsqueeze(0)  # [1, 128, 1856, 2688]
            .bfloat16()
            .contiguous()
        )
    else:
        up_cpu = None
        down_cpu = None

    # ttnn.as_tensor: saves tilized+sharded tensor on first run, loads on subsequent.
    # Column-parallel: shard intermediate dim → [1,128,2688,464] and [1,128,464,2688].
    up_tt = ttnn.as_tensor(
        up_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=up_cache,
    )
    down_tt = ttnn.as_tensor(
        down_cpu,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=down_cache,
    )
    print(f"  [experts] {tag}: done.", flush=True)
    _EXPERT_DEVICE_CACHE[layer_idx] = (up_tt, down_tt)
    return up_tt, down_tt


def _moe_layer_forward(
    mesh_device,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    layer_idx: int,
    wc: "WeightCache",
    cpu_gate: bool = False,
) -> ttnn.Tensor:
    """E-type block: pre-norm → gate + experts + shared_expert → residual.

    cpu_gate=True : gate on CPU float32 (exact HF routing, not trace-compatible).
    cpu_gate=False: fully on-device gate (F32 linear+sigmoid+bias, BF16 topk) — trace-compatible.
    """
    residual = hidden_states
    p = f"backbone.layers.{layer_idx}"

    normed_tt = layer_norm_forward(mesh_device, hidden_states, wc[f"{p}.norm.weight"])

    B = normed_tt.shape[0]
    S = normed_tt.shape[1]
    H = normed_tt.shape[2]

    # Flatten for gate/experts: [B, S, H] → [B*S, H]
    flat_tt = ttnn.reshape(normed_tt, [B * S, H])

    if cpu_gate:
        routing_weights_tt = moe_gate_forward_cpu(
            mesh_device,
            flat_tt,
            wc[f"{p}.mixer.gate.weight"],
            wc[f"{p}.mixer.gate.e_score_correction_bias"],
        )
    else:
        routing_weights_tt = moe_gate_forward(
            mesh_device,
            flat_tt,
            wc[f"{p}.mixer.gate.weight"],
            wc[f"{p}.mixer.gate.e_score_correction_bias"],
        )

    # Lazily build and cache pre-stacked expert weight tensors on device.
    up_tt, down_tt = _get_stacked_expert_weights(mesh_device, layer_idx, wc)

    # Routed experts via sparse_matmul (returns [B*S, 2688] on device).
    expert_out_tt = moe_experts_forward(mesh_device, flat_tt, routing_weights_tt, up_tt, down_tt)

    # Shared expert
    shared_out_tt = shared_expert_forward(
        mesh_device,
        normed_tt,
        w_up=wc[f"{p}.mixer.shared_experts.up_proj.weight"],
        w_down=wc[f"{p}.mixer.shared_experts.down_proj.weight"],
    )

    expert_out_reshaped = ttnn.reshape(expert_out_tt, [B, S, H])
    moe_out_tt = ttnn.add(expert_out_reshaped, shared_out_tt)
    return ttnn.add(residual, moe_out_tt)


def _layer_stack_forward(
    mesh_device,
    hidden_states: ttnn.Tensor,
    wc: WeightCache,
    num_layers: int,
    decoder_state: "DecoderState | None" = None,
    cpu_gate: bool = False,
) -> ttnn.Tensor:
    """Layer loop — stateless (decoder_state=None) or stateful (decoder_state provided).

    cpu_gate=True: MoE gate on CPU float32 (correct routing, not trace-compatible).
    cpu_gate=False: fully on-device gate (trace-compatible, ~15 tok/s).
    """
    m_idx = 0  # index within M_LAYER_INDICES
    d_idx = 0  # index within D_LAYER_INDICES

    for li in range(min(num_layers, N_LAYERS)):
        layer_type = PATTERN[li]
        p = f"backbone.layers.{li}"

        if layer_type == "M":
            ssm_state = decoder_state.ssm_states[m_idx] if decoder_state else None
            conv_state = decoder_state.conv_states[m_idx] if decoder_state else None
            hidden_states, state_new, conv_state_new = mamba2_layer_forward(
                mesh_device,
                hidden_states,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
                ssm_state=ssm_state,
                conv_state=conv_state,
            )
            if decoder_state:
                # In-place copy into pre-allocated output tensors (trace-compatible).
                # Replacing ssm_state_outs[m_idx] frees the previous tensor's DRAM
                # buffer during trace capture; later ops in the same trace can reuse
                # that freed buffer and overwrite the SSM state before advance() reads it.
                ttnn.assign(state_new, decoder_state.ssm_state_outs[m_idx])
                state_new.deallocate(True)
                h_tm2_out, h_tm1_out, hBC_out = conv_state_new
                ttnn.assign(h_tm2_out, decoder_state.conv_state_outs[m_idx][0])
                ttnn.assign(h_tm1_out, decoder_state.conv_state_outs[m_idx][1])
                ttnn.assign(hBC_out, decoder_state.conv_state_outs[m_idx][2])
                hBC_out.deallocate(True)
            m_idx += 1
        elif layer_type == "E":
            hidden_states = _moe_layer_forward(
                mesh_device,
                hidden_states,
                li,
                wc,
                cpu_gate=cpu_gate,
            )
        else:
            kv_cache = decoder_state.kv_caches[d_idx] if decoder_state else None
            page_table = decoder_state.page_tables[d_idx] if decoder_state else None
            current_pos = decoder_state.current_pos if decoder_state else None
            hidden_states = dense_attention_forward(
                mesh_device,
                hidden_states,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
                kv_cache=kv_cache,
                page_table=page_table,
                current_pos=current_pos,
            )
            d_idx += 1
    return hidden_states


def nemotron_h_forward(
    mesh_device,
    input_ids: torch.Tensor,  # [B, S] int64
    wc: WeightCache | None = None,
    num_layers: int = N_LAYERS,
) -> torch.Tensor:
    """Full NemotronH forward returning logits [B, S, vocab_size] on CPU.

    Args:
        mesh_device:  Open TTNN MeshDevice.
        input_ids:    Token ids [B, S].
        wc:           WeightCache (created internally if None).
        num_layers:   Run only the first N layers (default: all 52).

    Returns:
        Logits [B, S, 131072] bfloat16 on CPU.
    """
    if wc is None:
        wc = WeightCache()

    hidden_states = embedding_forward(mesh_device, input_ids, wc["backbone.embeddings.weight"])
    hidden_states = _layer_stack_forward(mesh_device, hidden_states, wc, num_layers)

    return lm_head_forward(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )


def nemotron_h_forward_device(
    mesh_device,
    ids_tt: ttnn.Tensor,  # pre-allocated uint32 device tensor [B, S]
    wc: WeightCache | None = None,
    num_layers: int = N_LAYERS,
) -> ttnn.Tensor:
    """Full NemotronH forward returning logits as ttnn.Tensor (no D2H).

    Stateless variant — used by test_decode_traced and benchmarks.

    Returns:
        Logits [B, S, 131072] bfloat16 on device.
    """
    if wc is None:
        wc = WeightCache()

    hidden_states = embedding_forward_tt(mesh_device, ids_tt, wc["backbone.embeddings.weight"])
    hidden_states = _layer_stack_forward(mesh_device, hidden_states, wc, num_layers)

    return lm_head_forward_device(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )


def nemotron_h_forward_stateful(
    mesh_device,
    ids_tt: ttnn.Tensor,  # pre-allocated uint32 device tensor [B, S]
    wc: WeightCache,
    decoder_state: DecoderState,
    num_layers: int = N_LAYERS,
    cpu_gate: bool = False,
) -> ttnn.Tensor:
    """Stateful NemotronH forward for generation.

    cpu_gate=False (default): MoE gate on device — trace-compatible (~15 tok/s).
    cpu_gate=True: gate on CPU float32 (exact routing, ~7 tok/s eager).

    After this call:
      - decoder_state.ssm_state_outs[i] has been written in-place with the new SSM state
      - decoder_state.kv_caches[j] have been updated in-place at current_pos

    Call decoder_state.advance() after each forward to copy outputs to inputs.

    Returns:
        Logits [B, S, 131072] bfloat16 on device.
    """
    hidden_states = embedding_forward_tt(mesh_device, ids_tt, wc["backbone.embeddings.weight"])
    hidden_states = _layer_stack_forward(
        mesh_device,
        hidden_states,
        wc,
        num_layers,
        decoder_state,
        cpu_gate=cpu_gate,
    )
    return lm_head_forward_device(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )
