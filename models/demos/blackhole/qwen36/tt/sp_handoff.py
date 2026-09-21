# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host round-trip: inject sequence-parallel (SP) prefill caches into the TP=4 decode model.

SP prefill runs full-attention KV and GDN recurrent/conv state in an UNSHARDED (full) layout
on the last die. This module reshapes that full layout into the per-device shards the TP=4
`Qwen36Model` (tt/model.py) expects in `layer.attention.{k_caches,v_caches,rec_state,
conv_states}`, uploads them, and writes them in place so `model.decode_tp(token, pos)` can
continue the sequence. v1 goes through host torch tensors (no on-device transfer).

Per-device shard conventions (verified against the CURRENT tree):
  * full attention: device d holds KV head ``(d * n_kv_heads) // num_devices``
    (attention/tp.py `reset_state`/`forward_prefill`; tp_common.py `replicate_kv_weight`).
  * GDN: device d holds value heads ``[d*Nv_tp, (d+1)*Nv_tp)`` of `rec_state`
    (gdn/tp.py `reset_state`; tp_common.py `prepare_gdn_qkv`), and `conv_states[m]`
    (m=1..K-1) row m-1 of the per-device [q|k|v] conv carry (gdn/tp.py `forward_prefill`
    capture_state writeback); `conv_states[0]` is always zero (the shifted-out tap).

All shapes below assume B=1 (max_batch_size=1) — the leading axis these helpers stack
devices on coincides with the model's batch axis, which is only valid at B=1.
"""
import torch

import ttnn

# --------------------------------------------------------------------------- #
# Full (unsharded) <-> per-device-stacked host tensor mappings.
# "stacked" = torch.Tensor with a leading axis of size num_devices, one slice per device,
# ready to upload with mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) (per-device shard
# shape == the tensor's shape with dim 0 divided by num_devices == 1, i.e. that axis
# disappears into the per-device physical shard).
# --------------------------------------------------------------------------- #


def kv_full_to_tp_host(k_full, v_full, num_devices=4, n_kv_heads=2):
    """Full KV [1, n_kv_heads, S, head_dim] -> per-device stacked [num_devices, 1, S, head_dim].

    Device d gets KV head ``(d * n_kv_heads) // num_devices`` (replicate_kv_weight order)."""

    def _stack(full):
        parts = [
            full[:, (d * n_kv_heads) // num_devices : (d * n_kv_heads) // num_devices + 1] for d in range(num_devices)
        ]
        return torch.cat(parts, dim=0)

    return _stack(k_full), _stack(v_full)


def kv_tp_host_to_full(k_stacked, v_stacked, num_devices=4, n_kv_heads=2):
    """Inverse of ``kv_full_to_tp_host``: per-device stacked [num_devices,1,S,HD] -> full
    [1, n_kv_heads, S, HD]. Devices sharing a head hold identical data; last writer wins."""
    S, HD = k_stacked.shape[-2], k_stacked.shape[-1]

    def _unstack(stacked):
        full = torch.zeros(1, n_kv_heads, S, HD, dtype=stacked.dtype)
        for d in range(num_devices):
            full[:, (d * n_kv_heads) // num_devices] = stacked[d]
        return full

    return _unstack(k_stacked), _unstack(v_stacked)


def gdn_rec_full_to_tp_host(rec_full, num_devices=4):
    """Full recurrent state [1, Nv, Dk, Dv] -> per-device stacked [num_devices, Nv/num_devices, Dk, Dv].

    Device d gets value heads [d*Nv_tp, (d+1)*Nv_tp)."""
    nv = rec_full.shape[1]
    assert nv % num_devices == 0, f"Nv={nv} not divisible by num_devices={num_devices}"
    nv_tp = nv // num_devices
    parts = [rec_full[:, d * nv_tp : (d + 1) * nv_tp] for d in range(num_devices)]
    return torch.cat(parts, dim=0)


def gdn_rec_tp_host_to_full(rec_stacked, num_devices=4):
    """Inverse of ``gdn_rec_full_to_tp_host``: [num_devices, Nv_tp, Dk, Dv] -> [1, Nv, Dk, Dv]."""
    parts = [rec_stacked[d : d + 1] for d in range(num_devices)]
    return torch.cat(parts, dim=1)


def gdn_conv_full_to_tp_host(conv_full, num_devices=4, key_dim=2048, value_dim=2048):
    """Full conv carry [1, K-1, key_dim*2+value_dim] (channels ordered [q|k|v]) -> list over
    m=0..K-1 of per-device stacked [num_devices, 1, D_tp] (D_tp = 2*key_dim/num_devices +
    value_dim/num_devices). m=0 is zeros (the shifted-out tap, per gdn/tp.py reset writeback);
    m=1..K-1 is row m-1 of the carry, gathered per device as
    ``[q[d*kp:(d+1)*kp] | k[key_dim+d*kp:key_dim+(d+1)*kp] | v[2*key_dim+d*vp:2*key_dim+(d+1)*vp]]``
    (NOT a contiguous slice — matches tp_common.py ``prepare_gdn_qkv`` per-device grouping)."""
    K_minus_1 = conv_full.shape[1]
    kp, vp = key_dim // num_devices, value_dim // num_devices
    D = 2 * kp + vp
    result = [torch.zeros(num_devices, 1, D, dtype=conv_full.dtype)]
    for m in range(1, K_minus_1 + 1):
        row = conv_full[:, m - 1, :]  # [1, key_dim*2+value_dim]
        parts = []
        for d in range(num_devices):
            q_d = row[:, d * kp : (d + 1) * kp]
            k_d = row[:, key_dim + d * kp : key_dim + (d + 1) * kp]
            v_d = row[:, 2 * key_dim + d * vp : 2 * key_dim + (d + 1) * vp]
            parts.append(torch.cat([q_d, k_d, v_d], dim=-1))  # [1, D]
        # stack (not cat): each part already carries the batch=1 axis at dim0 -- a NEW leading
        # axis is needed for the device dim (target per-device shape is [1, B, D], 3D).
        result.append(torch.stack(parts, dim=0))  # [num_devices, 1, D]
    return result


def gdn_conv_tp_host_to_full(conv_states_list, num_devices=4, key_dim=2048, value_dim=2048):
    """Inverse of ``gdn_conv_full_to_tp_host``: list of K tensors [num_devices,1,D_tp]
    (index 0 discarded — the zeroed/shifted-out tap) -> full [1, K-1, key_dim*2+value_dim]."""
    kp, vp = key_dim // num_devices, value_dim // num_devices
    rows = []
    for m in range(1, len(conv_states_list)):
        cs = conv_states_list[m]  # [num_devices, 1, D_tp]
        q_full = torch.cat([cs[d, 0, :kp] for d in range(num_devices)], dim=-1)
        k_full = torch.cat([cs[d, 0, kp : 2 * kp] for d in range(num_devices)], dim=-1)
        v_full = torch.cat([cs[d, 0, 2 * kp : 2 * kp + vp] for d in range(num_devices)], dim=-1)
        rows.append(torch.cat([q_full, k_full, v_full], dim=-1))
    return torch.stack(rows, dim=0).unsqueeze(0)  # [1, K-1, key_dim*2+value_dim]


# --------------------------------------------------------------------------- #
# Device upload / write-in-place.
# --------------------------------------------------------------------------- #
def _upload_sharded(mesh, host_stacked, dtype, layout=ttnn.TILE_LAYOUT):
    """torch [num_devices, ...] -> ttnn tensor sharded dim=0 (per-device shard drops that axis)."""
    return ttnn.from_torch(
        host_stacked,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def inject_into_tp_model(model, kv_per_attn_layer, gdn_per_layer):
    """Write SP-prefill's full-layout caches into `model`'s TP=4 per-device state in place.

    kv_per_attn_layer: {layer_idx -> (K, V)} host torch, K/V [1, n_kv_heads, S, head_dim] bf16.
    gdn_per_layer: {layer_idx -> (rec, conv)} host torch, rec [1, Nv, Dk, Dv] fp32,
    conv [1, K-1, gdn_key_dim*2+gdn_value_dim] bf16.

    `model.reset_tp()` should be called first (this function will lazily reset any layer
    whose state buffers are still unallocated, but does not zero already-allocated ones)."""
    mesh = model.device
    args = model.args
    num_devices = mesh.get_num_devices()
    n_kv_heads = args.n_kv_heads

    for layer_idx, (K, V) in kv_per_attn_layer.items():
        layer = model.layers[layer_idx]
        assert layer.is_full_attention, f"layer {layer_idx} is not a full-attention layer"
        attn = layer.attention
        if attn.k_caches is None:
            attn.reset_state()
        k_stacked, v_stacked = kv_full_to_tp_host(K, V, num_devices=num_devices, n_kv_heads=n_kv_heads)
        k_dev = _upload_sharded(mesh, k_stacked.to(torch.bfloat16), attn.k_caches[0].dtype)
        v_dev = _upload_sharded(mesh, v_stacked.to(torch.bfloat16), attn.v_caches[0].dtype)
        ttnn.fill_cache(attn.k_caches[0], k_dev, 0)
        ttnn.fill_cache(attn.v_caches[0], v_dev, 0)
        ttnn.deallocate(k_dev)
        ttnn.deallocate(v_dev)

    for layer_idx, (rec, conv) in gdn_per_layer.items():
        layer = model.layers[layer_idx]
        assert not layer.is_full_attention, f"layer {layer_idx} is not a GDN layer"
        attn = layer.attention
        if attn.rec_state is None or attn.conv_states is None:
            attn.reset_state()
        rec_stacked = gdn_rec_full_to_tp_host(rec, num_devices=num_devices)
        rec_dev = _upload_sharded(mesh, rec_stacked.to(torch.float32), attn.rec_state.dtype)
        ttnn.copy(rec_dev, attn.rec_state)
        ttnn.deallocate(rec_dev)

        conv_list = gdn_conv_full_to_tp_host(
            conv, num_devices=num_devices, key_dim=args.gdn_key_dim, value_dim=args.gdn_value_dim
        )
        for m, cm in enumerate(conv_list):
            cm_dev = _upload_sharded(mesh, cm.to(torch.bfloat16), attn.conv_states[m].dtype)
            ttnn.copy(cm_dev, attn.conv_states[m])
            ttnn.deallocate(cm_dev)


def snapshot_tp_model_to_full_host(model):
    """Inverse of ``inject_into_tp_model``: read `model`'s per-device TP state back into the
    full (unsharded) layout. Used as the round-trip oracle path in the test.

    Returns (kv_per_attn_layer, gdn_per_layer) in the same format ``inject_into_tp_model`` takes."""
    mesh = model.device
    args = model.args
    num_devices = mesh.get_num_devices()
    n_kv_heads = args.n_kv_heads
    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)

    kv_per_attn_layer, gdn_per_layer = {}, {}
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        if layer.is_full_attention:
            assert attn.k_caches is not None and attn.v_caches is not None, f"layer {layer_idx} KV cache unallocated"
            k_stacked = ttnn.to_torch(attn.k_caches[0], mesh_composer=comp0)
            v_stacked = ttnn.to_torch(attn.v_caches[0], mesh_composer=comp0)
            K_full, V_full = kv_tp_host_to_full(k_stacked, v_stacked, num_devices=num_devices, n_kv_heads=n_kv_heads)
            kv_per_attn_layer[layer_idx] = (K_full, V_full)
        else:
            assert (
                attn.rec_state is not None and attn.conv_states is not None
            ), f"layer {layer_idx} GDN state unallocated"
            rec_stacked = ttnn.to_torch(attn.rec_state, mesh_composer=comp0)
            rec_full = gdn_rec_tp_host_to_full(rec_stacked, num_devices=num_devices)
            conv_stacked_list = [ttnn.to_torch(cs, mesh_composer=comp0) for cs in attn.conv_states]
            conv_full = gdn_conv_tp_host_to_full(
                conv_stacked_list, num_devices=num_devices, key_dim=args.gdn_key_dim, value_dim=args.gdn_value_dim
            )
            gdn_per_layer[layer_idx] = (rec_full, conv_full)
    return kv_per_attn_layer, gdn_per_layer
